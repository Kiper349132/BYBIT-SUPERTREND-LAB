"""Crash-safety of the research store: chunk files are the primary source,
nothing computed is ever deleted, metadata writes are fsync'ed with backups."""
import gzip
import json
import os

import pytest

import lab_core as core
import lab_storage
from lab_storage import ResearchStore
from synthetic import synthetic_market

CFG = {"symbol": "ST", "months": 12, "tfs": ["60"], "fee": 0.00055, "slippage": 0.0002, "depth": "Тест"}


@pytest.fixture()
def rows():
    m = synthetic_market(9000, "60")
    plan = core.build_walk_forward_plan_arrays(m["timestamp"], m["open"], m["close"])
    core.worker_set_inline("60", m, plan)
    return core._worker_eval_period(("60", 14, [(2.0, ("BOTH", "LONG", "SHORT")), (3.0, ("BOTH",))], 0.00055, 0.0002, 26, "broad"))


def _store(tmp_path):
    return ResearchStore.create(tmp_path, CFG, {"60": synthetic_market(9000, "60")})


def _commit_three(st, rows):
    for i in range(3):
        st.write_unit("60", "broad", f"u{i}", rows, 1000)
        st.flush(reason="test")
    return sorted(p.name for p in st.chunks_dir.glob("chunk_*.jsonl.gz"))


def test_fsync_and_backup(tmp_path, rows, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(lab_storage.os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    st = _store(tmp_path)
    n0 = len(calls)
    _commit_three(st, rows)
    assert len(calls) - n0 >= 3 * 3            # chunk + index + index backup per commit
    bak = json.loads((st.root / "index.json.bak").read_text())
    cur = json.loads((st.root / "index.json").read_text())
    assert bak["generation"] == cur["generation"] - 1   # previous generation kept


@pytest.mark.parametrize("damage", ["empty", "garbage", "both", "deleted"])
def test_corrupt_index_never_deletes_chunks(tmp_path, rows, damage):
    st = _store(tmp_path)
    names = _commit_three(st, rows)
    useful, nrows = st.committed_useful, st.committed_rows
    if damage in ("empty", "both"):
        st.index_path.write_bytes(b"")
    if damage == "garbage":
        st.index_path.write_bytes(b"{not json")
    if damage == "both":
        (st.root / "index.json.bak").write_bytes(b"\x00\x01")
    if damage == "deleted":
        st.index_path.unlink(); (st.root / "index.json.bak").unlink()
    st2 = ResearchStore.open(st.root)
    assert sorted(p.name for p in st2.chunks_dir.glob("chunk_*.jsonl.gz")) == names
    assert st2.committed_useful == useful and st2.committed_rows == nrows
    assert set(st2.committed_units("60", "broad")) == {"u0", "u1", "u2"}
    assert json.loads(st2.index_path.read_text())["chunks"]  # index rebuilt from chunks


def test_chunk_committed_but_not_indexed_is_recovered(tmp_path, rows, monkeypatch):
    st = _store(tmp_path)
    st.write_unit("60", "broad", "u0", rows, 1000)

    def crash(point):
        if point == "chunk_renamed":
            raise KeyboardInterrupt("simulated power loss")
    monkeypatch.setattr(lab_storage, "CRASH_HOOK", crash)
    with pytest.raises(KeyboardInterrupt):
        st.flush()
    monkeypatch.setattr(lab_storage, "CRASH_HOOK", None)
    st2 = ResearchStore.open(st.root)
    assert set(st2.committed_units("60", "broad")) == {"u0"}
    assert any("missing from index" in n for n in st2.recovery_notes)


def test_unfinished_and_truncated_chunks_are_quarantined_not_deleted(tmp_path, rows):
    st = _store(tmp_path)
    _commit_three(st, rows)
    # truncated committed chunk (disk error) + an unfinished .tmp chunk (process killed)
    victim = st.chunks_dir / "chunk_000002.jsonl.gz"
    data = victim.read_bytes()
    victim.write_bytes(data[: len(data) // 2])
    st.write_unit("60", "broad", "u9", rows, 1000)   # opens chunk_000004 .tmp, never committed
    st.abandon_open_chunk()
    st2 = ResearchStore.open(st.root)
    q = sorted(p.name for p in st2.quarantine_dir.iterdir())
    assert any(n.endswith("chunk_000002.jsonl.gz") for n in q)
    assert any(n.endswith("chunk_000004.jsonl.gz.tmp") for n in q)
    assert set(st2.committed_units("60", "broad")) == {"u0", "u2"}   # u1 and u9 will be recomputed
    # new commits never reuse a quarantined sequence number in a confusing way
    st2.write_unit("60", "broad", "u1", rows, 1000); st2.flush()
    assert set(st2.committed_units("60", "broad")) == {"u0", "u1", "u2"}


def test_rows_roundtrip_exactly(tmp_path, rows):
    st = _store(tmp_path)
    st.write_unit("60", "broad", "u0", rows, 1000)
    st.write_unit("60", "fine", "f0", rows[:1], 10)
    st.flush()
    back = list(st.iter_rows("60", "broad"))
    assert [core.autoresult_to_dict(r) for r in back] == [core.autoresult_to_dict(r) for r in rows]
    assert len(list(st.iter_rows("60", "fine"))) == 1
    assert list(st.iter_rows("60", "broad", allowed_units={"zzz"})) == []


def test_chunk_footer_required(tmp_path, rows):
    st = _store(tmp_path)
    st.write_unit("60", "broad", "u0", rows, 1000); st.flush()
    fp = next(st.chunks_dir.glob("chunk_*.jsonl.gz"))
    lines = gzip.decompress(fp.read_bytes()).splitlines()
    fp.write_bytes(gzip.compress(b"\n".join(lines[:-1]) + b"\n"))  # valid gzip, footer missing
    st.index_path.unlink(); (st.root / "index.json.bak").unlink()
    st2 = ResearchStore.open(st.root)
    assert st2.committed_units("60", "broad") == {}
    assert any(p.name.endswith(fp.name) for p in st2.quarantine_dir.iterdir())

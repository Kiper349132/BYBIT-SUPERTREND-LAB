"""REGRESSION: uninterrupted run == interrupted run + restart + resume.

Compared objects (exact equality, every float bit-for-bit):
  * final TOP-100 (all fields),
  * the full retained report pool,
  * per timeframe: stable regions chosen for fine / cluster_wide / cluster_deep,
    the candidate sets of fine-search, cluster-search (wide and deep),
    the Adaptive V2 LONG/SHORT components and every stage's retained pool.

Interruptions covered: STOP button / window close (same code path),
Ctrl+C (SIGINT to the headless runner), BrokenProcessPool (a worker is
SIGKILLed), memory guard stop, hard process death at random moments and at
the worst points of a checkpoint commit, lost/corrupt index and caches.
"""
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lab_search import ResearchRunner
from lab_storage import ResearchStore
from resume_helpers import SYNTH_SPEC, canonical, diff_summary, make_cfg, synthetic_set

ROOT = Path(__file__).resolve().parents[1]


def _run(root_or_new, *, workers=2, tmp=None, **kw):
    if isinstance(root_or_new, Path) and (root_or_new / "run.json").exists():
        store = ResearchStore.open(root_or_new)
    else:
        store = ResearchStore.create(tmp, make_cfg(), synthetic_set())
    res = ResearchRunner(make_cfg(), store, workers=workers, **kw).run()
    return store, res


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("baseline")
    count = {"n": 0}
    store, res = _run(None, tmp=tmp, unit_hook=lambda *a: count.__setitem__("n", count["n"] + 1))
    assert res.status == "complete"
    return {"canon": canonical(res.top, res.report_pool, res.trace), "units": count["n"]}


def _assert_same(baseline, res):
    assert res.status == "complete"
    got = canonical(res.top, res.report_pool, res.trace)
    assert got == baseline["canon"], diff_summary(baseline["canon"], got)


def _stopper(at):
    count = {"n": 0}

    def hook(runner, tf, stage, uid):
        count["n"] += 1
        if count["n"] == at:
            runner.stop_event.set()
    return hook


def test_baseline_is_meaningful(baseline):
    c = baseline["canon"]
    assert len(c["top"]) == 100 and baseline["units"] > 40
    for tf in ("30", "60"):
        t = c["trace"][tf]
        assert all(t["candidates"][s]["count"] > 0 for s in ("fine", "cluster_wide", "cluster_deep"))
        assert t["regions"]["fine"] and t["components"]["LONG"] and t["components"]["SHORT"]


@pytest.mark.parametrize("frac", [0.02, 0.3, 0.55, 0.8, 0.97])
def test_stop_then_resume(baseline, tmp_path, frac):
    """STOP button and window close both set the same stop_event."""
    at = max(1, int(baseline["units"] * frac))
    store, res = _run(None, tmp=tmp_path, unit_hook=_stopper(at))
    assert res.status == "stopped"
    assert not store.is_complete
    store2, res2 = _run(store.root, workers=3)      # different worker count on purpose
    _assert_same(baseline, res2)


def test_many_stops_until_done(baseline, tmp_path):
    root = None
    for attempt in range(60):
        store, res = _run(root, tmp=tmp_path, workers=1 + attempt % 3, unit_hook=_stopper(7))
        root = store.root
        if res.status == "complete":
            break
    assert attempt > 3
    _assert_same(baseline, res)


def test_memory_guard_stop_then_resume(baseline, tmp_path):
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return {"phys_pct": 96.0 if calls["n"] >= baseline["units"] // 2 else 40.0, "commit_pct": 40.0}
    store, res = _run(None, tmp=tmp_path, memory_probe=probe)
    assert res.status == "memory_stop" and "Checkpoint" in res.message
    _store2, res2 = _run(store.root)
    _assert_same(baseline, res2)


def _killer(at):
    count = {"n": 0}

    def hook(runner, tf, stage, uid):
        count["n"] += 1
        if count["n"] == at:
            pids = runner.pool.worker_pids()
            os.kill(pids[0], signal.SIGKILL)
    return hook


@pytest.mark.parametrize("frac", [0.1, 0.6])
def test_broken_process_pool_auto_recovers(baseline, tmp_path, frac):
    store, res = _run(None, tmp=tmp_path, workers=3, unit_hook=_killer(max(1, int(baseline["units"] * frac))))
    assert any(s.get("pool_restarts", 0) >= 1 for s in res.search_stats) or res.status == "complete"
    _assert_same(baseline, res)


def test_broken_process_pool_without_restarts_then_resume(baseline, tmp_path):
    from lab_search import PoolFailure
    store = ResearchStore.create(tmp_path, make_cfg(), synthetic_set())
    with pytest.raises(PoolFailure):
        ResearchRunner(make_cfg(), store, workers=2, pool_max_restarts=0,
                       unit_hook=_killer(int(baseline["units"] * 0.4))).run()
    _s, res = _run(store.root)
    _assert_same(baseline, res)


def test_resume_with_lost_index_and_caches(baseline, tmp_path):
    store, res = _run(None, tmp=tmp_path, unit_hook=_stopper(int(baseline["units"] * 0.7)))
    assert res.status == "stopped"
    for p in (store.index_path, store.root / "index.json.bak"):
        p.write_bytes(b"garbage")
    shutil.rmtree(store.cache_dir)
    chunks_before = sorted(p.name for p in store.chunks_dir.iterdir())
    store2 = ResearchStore.open(store.root)
    assert sorted(p.name for p in store2.chunks_dir.iterdir()) == chunks_before   # nothing deleted
    res2 = ResearchRunner(make_cfg(), store2, workers=2, use_stage_cache=False).run()
    _assert_same(baseline, res2)


def test_stage_cache_and_chunk_rebuild_agree(baseline, tmp_path):
    """Completed stages restored from cache vs re-read from chunks."""
    store, res = _run(None, tmp=tmp_path, unit_hook=_stopper(int(baseline["units"] * 0.9)))
    _s, with_cache = _run(store.root)
    _assert_same(baseline, with_cache)


# ---------------------------------------------------------------------------
# Separate OS processes: Ctrl+C and hard kills
# ---------------------------------------------------------------------------

def _cli(args, env_extra=None, **popen):
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.Popen([sys.executable, str(ROOT / "lab_cli.py"), "run", *args], cwd=str(ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True, **popen)


def _orphans_exit(pgid, timeout=15.0) -> bool:
    """True when no process of the group survives (worker watchdog works)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.2)
    return False


def _drain(p, timeout=20.0) -> str:
    """Read remaining output after the main process exited. Workers orphaned by
    a hard parent death must exit by themselves (parent watchdog); the test
    fails if they survive."""
    alive = not _orphans_exit(p.pid, timeout)
    _kill_group(p)
    out = p.stdout.read()
    assert not alive, "worker processes survived the death of their parent"
    return out


def _kill_group(p):
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        pass


def _root_from_output(lines):
    for line in lines:
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("event") == "store":
            return Path(d["root"])
    raise AssertionError("no store line:\n" + "\n".join(lines[-20:]))


def _finish_with_cli(root: Path, result_json: Path, attempts=12):
    for _ in range(attempts):
        p = _cli(["--research-dir", str(root.parent), "--resume", str(root), "--workers", "2",
                  "--checkpoint-every", "400", "--result-json", str(result_json)])
        p.wait(timeout=600)
        out = _drain(p)
        if p.returncode == 0:
            return out
    raise AssertionError(out)


def _cli_canon(path: Path) -> dict:
    d = json.loads(path.read_text())
    return canonical(d["top"], d["report_pool"], d["trace"])


def test_cli_uninterrupted_matches_in_process_baseline(baseline, tmp_path):
    """Different process, small checkpoint interval (many chunks) -> same answer."""
    rj = tmp_path / "r.json"
    p = _cli(["--research-dir", str(tmp_path), "--synthetic", SYNTH_SPEC, "--workers", "3",
              "--checkpoint-every", "400", "--result-json", str(rj)])
    p.wait(timeout=600)
    out = _drain(p)
    assert p.returncode == 0, out
    got = _cli_canon(rj)
    assert got == baseline["canon"], diff_summary(baseline["canon"], got)


def test_ctrl_c_then_resume(baseline, tmp_path):
    rj = tmp_path / "r.json"
    p = _cli(["--research-dir", str(tmp_path), "--synthetic", SYNTH_SPEC, "--workers", "2", "--checkpoint-every", "400"])
    lines = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 120:
        line = p.stdout.readline()
        if not line:
            break
        lines.append(line.strip())
        if "CHECKPOINT" in line and len(lines) > 3:
            # A console Ctrl+C reaches the whole process group (workers too).
            os.killpg(p.pid, signal.SIGINT)
            break
    p.wait(timeout=120)
    lines += _drain(p).splitlines()
    assert p.returncode == 2, "\n".join(lines[-15:])
    assert any('"status": "stopped"' in x for x in lines)
    root = _root_from_output(lines)
    assert not (root / "COMPLETE.json").exists()
    _finish_with_cli(root, rj)
    got = _cli_canon(rj)
    assert got == baseline["canon"], diff_summary(baseline["canon"], got)


@pytest.mark.parametrize("env", [
    {"LAB_TEST_CRASH_AT": "chunk_written:2"},     # chunk fsync'ed but not renamed
    {"LAB_TEST_CRASH_AT": "chunk_renamed:3"},     # chunk committed, index not updated
    {"LAB_TEST_CRASH_AT": "index_written:5"},     # just after a complete commit
    {"LAB_TEST_CRASH_AFTER_UNITS": "23"},         # mid-chunk: uncommitted units lost
])
def test_hard_crash_at_commit_points_then_resume(baseline, tmp_path, env):
    rj = tmp_path / "r.json"
    p = _cli(["--research-dir", str(tmp_path), "--synthetic", SYNTH_SPEC, "--workers", "2", "--checkpoint-every", "400"], env)
    p.wait(timeout=600)
    out = _drain(p)
    assert p.returncode == 77, out
    root = _root_from_output(out.splitlines())
    _finish_with_cli(root, rj)
    got = _cli_canon(rj)
    assert got == baseline["canon"], diff_summary(baseline["canon"], got)


def test_random_sigkill_until_done(baseline, tmp_path):
    rng = random.Random(1234)
    rj = tmp_path / "r.json"
    root = None
    kills = 0
    for attempt in range(25):
        args = ["--research-dir", str(tmp_path), "--workers", "2", "--checkpoint-every", "300", "--result-json", str(rj)]
        args += ["--resume", str(root)] if root else ["--synthetic", SYNTH_SPEC]
        p = _cli(args)
        delay = rng.uniform(0.6, 2.2)
        try:
            p.wait(timeout=delay)
        except subprocess.TimeoutExpired:
            if attempt % 2:
                _kill_group(p)                   # power cut: parent AND workers die
            else:
                p.send_signal(signal.SIGKILL)    # only the parent dies (Task Manager)
            p.wait()
            kills += 1
        out = _drain(p)
        if root is None:
            root = _root_from_output(out.splitlines())
        if p.returncode == 0:
            break
    assert kills >= 2, "test did not actually interrupt the run"
    got = _cli_canon(rj)
    assert got == baseline["canon"], diff_summary(baseline["canon"], got)

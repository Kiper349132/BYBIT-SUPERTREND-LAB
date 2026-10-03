"""Crash-safe, append-only research store (v0.8).

Design rules
------------
* chunk files are the PRIMARY source of truth. Every committed chunk
  `chunks/chunk_NNNNNN.jsonl.gz` ends with a footer that lists the work units
  it completes. A unit counts as done if and only if a valid chunk says so.
* `index.json` is only a cache of chunk manifests. If it is missing or
  corrupt (power loss, disk error) the index is rebuilt by scanning the chunks.
  Nothing computed is ever deleted because of a bad index.
* Every metadata file is written with tmp file -> flush -> fsync -> atomic
  replace, and the previous generation is kept as `.bak`.
* Unreadable / unfinished files are MOVED to `quarantine/`, never deleted.
* Results stream straight into the open chunk file instead of being buffered
  as Python objects in RAM.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

from lab_core import (
    MARKET_FIELDS, SEARCH_SCHEMA, VERSION, AutoResult, autoresult_from_dict, autoresult_to_dict, market_file,
)
from lab_log import file_log

CHECKPOINT_USEFUL_EVERY = 1_000_000
FOOTER_KEY = "__chunk_footer__"

# Test hook: called with a stage name ("chunk_written", "chunk_renamed",
# "index_written") so tests can simulate a crash at the worst moment.
CRASH_HOOK: Optional[Callable[[str], None]] = None


def _hook(point: str) -> None:
    if CRASH_HOOK is not None:
        CRASH_HOOK(point)


# --------------------------------------------------------------------------
# Durable file primitives
# --------------------------------------------------------------------------

def fsync_dir(path: Path) -> None:
    """Make a rename durable on POSIX. Windows cannot open directories; NTFS
    journals the rename metadata itself."""
    if os.name == "nt":
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_write_json(path: Path, obj, keep_backup: bool = True) -> None:
    """tmp + flush + fsync + atomic replace; previous version kept as .bak."""
    path = Path(path)
    if keep_backup and path.exists():
        try:
            prev = path.read_bytes()
            json.loads(prev.decode("utf-8"))  # only back up a valid generation
            atomic_write_bytes(path.with_name(path.name + ".bak"), prev)
        except Exception:
            pass
    atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8"))


def read_json_candidates(path: Path) -> list:
    """Valid JSON documents among path and path.bak, newest generation first."""
    found = []
    for p in (path, path.with_name(path.name + ".bak")):
        try:
            found.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    found.sort(key=lambda d: int(d.get("generation", 0) or 0) if isinstance(d, dict) else 0, reverse=True)
    return found


def save_npy_durable(path: Path, arr: np.ndarray) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, np.ascontiguousarray(arr), allow_pickle=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


# --------------------------------------------------------------------------
# Signatures
# --------------------------------------------------------------------------

def config_hash(cfg: dict) -> str:
    material = {
        "schema": SEARCH_SCHEMA,
        "symbol": cfg.get("symbol"), "months": cfg.get("months"), "tfs": [str(x) for x in cfg.get("tfs", [])],
        "fee": cfg.get("fee"), "slippage": cfg.get("slippage"), "depth": cfg.get("depth"),
    }
    raw = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def data_hash(market: dict) -> str:
    """Hash of the FULL candle arrays (v0.7.1 only fingerprinted row count and
    the first/last candle)."""
    h = hashlib.sha256()
    for tf in sorted(market, key=lambda x: int(x)):
        h.update(str(tf).encode())
        for name in ("timestamp", "open", "high", "low", "close"):
            h.update(np.ascontiguousarray(market[tf][name]).tobytes())
    return h.hexdigest()[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Research store
# --------------------------------------------------------------------------

class ResearchStore:
    """One research run on disk.

    Layout::

        run.json            immutable: config, hashes, creation time
        market/             candle arrays used by this run (.npy, memory-mapped by workers)
        chunks/             committed result chunks (PRIMARY data)
        index.json(.bak)    cache of chunk manifests
        stage_cache/        accelerators: retained pools / landscapes of finished stages
        regions/            explainability: which regions/seeds each stage chose and why
        quarantine/         files that could not be verified (never deleted automatically)
        COMPLETE.json       written once the run finished
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.chunks_dir = self.root / "chunks"
        self.market_dir = self.root / "market"
        self.cache_dir = self.root / "stage_cache"
        self.regions_dir = self.root / "regions"
        self.quarantine_dir = self.root / "quarantine"
        self.index_path = self.root / "index.json"
        self.run_path = self.root / "run.json"
        self.complete_path = self.root / "COMPLETE.json"
        for d in (self.chunks_dir, self.market_dir, self.cache_dir, self.regions_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.run_meta: dict = {}
        self.chunks: dict[str, dict] = {}       # chunk name -> manifest
        self.generation = 0
        self.recovery_notes: list[str] = []
        self.resumed = False
        # open (uncommitted) chunk state
        self._fh = None
        self._raw = None
        self._tmp_path: Optional[Path] = None
        self._pending_units: list[list] = []
        self._pending_rows = 0
        self._pending_useful = 0

    # ----- creation / discovery -------------------------------------------------

    @classmethod
    def create(cls, research_dir: Path, cfg: dict, market: dict) -> "ResearchStore":
        ch = config_hash(cfg)
        dh = data_hash(market)
        stem = f"{cfg.get('symbol', 'UNKNOWN')}_{ch}_{dh}"
        root = Path(research_dir) / stem
        if root.exists():
            root = Path(research_dir) / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        store = cls(root)
        for tf, arrs in market.items():
            for name in MARKET_FIELDS:
                save_npy_durable(market_file(store.market_dir, tf, name), arrs[name])
        meta = {
            "version": VERSION, "schema": SEARCH_SCHEMA, "config_hash": ch, "data_hash": dh,
            "created_utc": _now(),
            "config": {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in cfg.items() if k != "workers"},
            "tfs": [str(t) for t in market],
            "rows": {str(t): int(len(market[t]["close"])) for t in market},
        }
        atomic_write_json(store.run_path, meta)
        store.run_meta = meta
        store._recover()
        file_log(f"STORE created {store.root}")
        return store

    @classmethod
    def open(cls, root: Path) -> "ResearchStore":
        store = cls(root)
        cands = read_json_candidates(store.run_path)
        if not cands:
            raise RuntimeError(f"run.json в {root} не читается — папка не будет изменена")
        store.run_meta = cands[0]
        store._recover()
        store.resumed = True
        file_log(f"STORE opened {store.root}: chunks={len(store.chunks)} notes={store.recovery_notes}")
        return store

    @staticmethod
    def find_incomplete(research_dir: Path, cfg: dict) -> list[Path]:
        """Incomplete runs created with the same configuration and search schema,
        newest first. Used to offer a resume with the run's own stored candles."""
        ch = config_hash(cfg)
        out = []
        base = Path(research_dir)
        if not base.exists():
            return out
        for d in base.iterdir():
            if not d.is_dir() or (d / "COMPLETE.json").exists():
                continue
            cands = read_json_candidates(d / "run.json")
            if cands and cands[0].get("config_hash") == ch and cands[0].get("schema") == SEARCH_SCHEMA:
                out.append(d)
        out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return out

    def load_market(self) -> dict:
        """Read-only memory-mapped candle arrays of this run."""
        out = {}
        for tf in self.run_meta.get("tfs", []):
            out[str(tf)] = {name: np.load(market_file(self.market_dir, tf, name), mmap_mode="r", allow_pickle=False).view(np.ndarray)
                            for name in MARKET_FIELDS}
        return out

    @property
    def is_complete(self) -> bool:
        return self.complete_path.exists()

    # ----- recovery ------------------------------------------------------------

    def _quarantine(self, path: Path, why: str) -> None:
        self.quarantine_dir.mkdir(exist_ok=True)
        dst = self.quarantine_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{path.name}"
        try:
            shutil.move(str(path), str(dst))
            self.recovery_notes.append(f"quarantined {path.name}: {why}")
            file_log(f"STORE quarantine {path} -> {dst}: {why}")
        except Exception as exc:
            self.recovery_notes.append(f"could not quarantine {path.name}: {exc}")

    @staticmethod
    def scan_chunk(path: Path) -> dict:
        """Read a whole chunk, verify gzip CRC and footer, return its manifest."""
        rows = 0
        footer = None
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                if footer is not None:
                    raise ValueError("data after footer")
                d = json.loads(line)
                if FOOTER_KEY in d:
                    footer = d[FOOTER_KEY]
                else:
                    rows += 1
        if footer is None:
            raise ValueError("footer missing (unfinished chunk)")
        if int(footer.get("rows", -1)) != rows:
            raise ValueError(f"row count mismatch {rows} != {footer.get('rows')}")
        return {"rows": rows, "units": footer.get("units", []), "useful": int(footer.get("useful", 0)),
                "bytes": path.stat().st_size, "seq": int(footer.get("seq", 0))}

    def _recover(self) -> None:
        index = None
        for cand in read_json_candidates(self.index_path):
            if isinstance(cand, dict) and isinstance(cand.get("chunks"), dict):
                index = cand
                break
        if index is None and self.index_path.exists():
            self.recovery_notes.append("index.json and backup unreadable: rebuilding from chunks")
        cached = (index or {}).get("chunks", {})
        self.generation = int((index or {}).get("generation", 0) or 0)
        # unfinished chunks from an interrupted process
        for fp in sorted(self.chunks_dir.glob("*.tmp")):
            self._quarantine(fp, "unfinished chunk (process stopped before commit)")
        chunks = {}
        for fp in sorted(self.chunks_dir.glob("chunk_*.jsonl.gz")):
            man = cached.get(fp.name)
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            if man is not None and int(man.get("bytes", -1)) == size:
                chunks[fp.name] = man
                continue
            try:
                chunks[fp.name] = self.scan_chunk(fp)
                if man is None:
                    self.recovery_notes.append(f"chunk {fp.name} was committed but missing from index: recovered")
            except (OSError, EOFError, ValueError, zlib.error, json.JSONDecodeError, gzip.BadGzipFile) as exc:
                self._quarantine(fp, f"chunk failed verification: {exc}")
        missing = sorted(set(cached) - set(chunks))
        for name in missing:
            self.recovery_notes.append(f"chunk {name} listed in index but absent: its units will be recomputed")
        self.chunks = chunks
        if index is None or missing or set(chunks) != set(cached):
            self._write_index()

    def _write_index(self) -> None:
        self.generation += 1
        atomic_write_json(self.index_path, {
            "generation": self.generation, "version": VERSION, "updated_utc": _now(),
            "note": "Cache only. Chunk files are authoritative; this file is rebuilt from them if lost.",
            "chunks": self.chunks,
        })

    # ----- queries ---------------------------------------------------------------

    def committed_units(self, tf: str, stage: str) -> dict:
        out = {}
        for man in self.chunks.values():
            for u in man.get("units", []):
                if str(u[0]) == str(tf) and str(u[1]) == str(stage):
                    out[str(u[2])] = int(u[3])
        for u in self._pending_units:
            if str(u[0]) == str(tf) and str(u[1]) == str(stage):
                out[str(u[2])] = int(u[3])
        return out

    def stage_chunk_fingerprint(self, tf: str, stage: str) -> list:
        fp = []
        for name in sorted(self.chunks):
            man = self.chunks[name]
            if any(str(u[0]) == str(tf) and str(u[1]) == str(stage) for u in man.get("units", [])):
                fp.append([name, int(man.get("bytes", 0))])
        return fp

    @property
    def committed_useful(self) -> int:
        return sum(int(m.get("useful", 0)) for m in self.chunks.values())

    @property
    def committed_rows(self) -> int:
        return sum(int(m.get("rows", 0)) for m in self.chunks.values())

    @property
    def pending_useful(self) -> int:
        return self._pending_useful

    def iter_rows(self, tf: str, stage: str, allowed_units: Optional[set] = None) -> Iterator[AutoResult]:
        """Committed rows of (tf, stage), read back from the chunks.

        Rows are stored unit after unit and the footer records each unit's row
        count, so rows can be attributed to their unit exactly; with
        `allowed_units` only rows of those units are returned.
        """
        tf, stage = str(tf), str(stage)
        for name in sorted(self.chunks):
            man = self.chunks[name]
            units = man.get("units", [])
            if not any(str(u[0]) == tf and str(u[1]) == stage for u in units):
                continue
            spans = []  # per row: include?
            for u in units:
                keep = str(u[0]) == tf and str(u[1]) == stage and (allowed_units is None or str(u[2]) in allowed_units)
                spans.append((int(u[4]) if len(u) > 4 else -1, keep))
            exact = all(n >= 0 for n, _ in spans)
            ui = 0; left = spans[0][0] if spans else 0
            with gzip.open(self.chunks_dir / name, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if FOOTER_KEY in d:
                        continue
                    if exact:
                        while left == 0 and ui + 1 < len(spans):
                            ui += 1; left = spans[ui][0]
                        left -= 1
                        if not spans[ui][1]:
                            continue
                    elif not (str(d.get("interval")) == tf and str(d.get("stage")) == stage):
                        continue
                    yield autoresult_from_dict(d)

    # ----- writing ----------------------------------------------------------------

    def _next_seq(self) -> int:
        seqs = [int(n.split("_")[1].split(".")[0]) for n in self.chunks]
        return (max(seqs) if seqs else 0) + 1

    def write_unit(self, tf: str, stage: str, unit_id: str, rows: Iterable[AutoResult], useful_checks: int) -> None:
        """Stream one finished unit into the open chunk (no RAM buffering)."""
        if self._fh is None:
            seq = self._next_seq()
            self._tmp_path = self.chunks_dir / f"chunk_{seq:06d}.jsonl.gz.tmp"
            self._raw = open(self._tmp_path, "wb")
            self._fh = gzip.GzipFile(fileobj=self._raw, mode="wb", compresslevel=3, mtime=0)
            self._seq = seq
        n = 0
        for row in rows:
            self._fh.write(json.dumps(autoresult_to_dict(row), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            self._fh.write(b"\n")
            n += 1
        self._pending_units.append([str(tf), str(stage), str(unit_id), int(useful_checks), n])
        self._pending_rows += n
        self._pending_useful += int(useful_checks)

    def should_flush(self) -> bool:
        return self._pending_useful >= CHECKPOINT_USEFUL_EVERY

    def flush(self, reason: str = "") -> bool:
        """Commit the open chunk: footer -> fsync -> atomic rename -> index."""
        if self._fh is None:
            return False
        footer = {FOOTER_KEY: {"seq": self._seq, "rows": self._pending_rows, "useful": self._pending_useful,
                               "units": self._pending_units, "reason": reason, "utc": _now(), "version": VERSION}}
        self._fh.write(json.dumps(footer, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
        self._fh.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()
        _hook("chunk_written")
        final = self._tmp_path.with_name(self._tmp_path.name[:-4])
        os.replace(self._tmp_path, final)
        fsync_dir(self.chunks_dir)
        _hook("chunk_renamed")
        self.chunks[final.name] = {"rows": self._pending_rows, "units": self._pending_units, "useful": self._pending_useful,
                                   "bytes": final.stat().st_size, "seq": self._seq}
        self._write_index()
        _hook("index_written")
        file_log(f"CHECKPOINT {final.name}: rows={self._pending_rows:,} useful={self._pending_useful:,} reason={reason}")
        self._fh = self._raw = self._tmp_path = None
        self._pending_units = []
        self._pending_rows = 0
        self._pending_useful = 0
        return True

    def abandon_open_chunk(self) -> None:
        """Close the uncommitted chunk without committing (it is quarantined on next open)."""
        try:
            if self._fh is not None:
                self._fh.close()
            if self._raw is not None:
                self._raw.close()
        except Exception:
            pass
        self._fh = self._raw = None

    # ----- stage caches (accelerators, never authoritative) ----------------------

    def _cache_path(self, tf: str, stage: str) -> Path:
        return self.cache_dir / f"{tf}_{stage}.json.gz"

    def save_stage_cache(self, tf: str, stage: str, payload: dict) -> None:
        """Streamed (no giant in-memory JSON string), fsync'ed, atomic."""
        payload = dict(payload)
        payload["chunk_fingerprint"] = self.stage_chunk_fingerprint(tf, stage)
        path = self._cache_path(tf, stage)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=3, mtime=0) as gz:
                with io.TextIOWrapper(gz, encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)

    def load_stage_cache(self, tf: str, stage: str) -> Optional[dict]:
        p = self._cache_path(tf, stage)
        if not p.exists():
            return None
        try:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            self._quarantine(p, f"stage cache unreadable: {exc}")
            return None
        if payload.get("chunk_fingerprint") != self.stage_chunk_fingerprint(tf, stage):
            return None
        return payload

    def save_regions(self, tf: str, stage: str, info: dict) -> None:
        atomic_write_json(self.regions_dir / f"{tf}_{stage}.json", info, keep_backup=False)

    # ----- completion ------------------------------------------------------------

    def finalize(self, extra: Optional[dict] = None) -> None:
        self.flush(reason="finalize")
        atomic_write_json(self.complete_path, {"completed_utc": _now(), "extra": extra or {}}, keep_backup=False)

    def summary(self) -> dict:
        stage_counts: dict = {}
        for man in self.chunks.values():
            for u in man.get("units", []):
                k = f"{u[0]}|{u[1]}"
                stage_counts[k] = stage_counts.get(k, 0) + int(u[4] if len(u) > 4 else 0)
        return {
            "version": VERSION, "schema": SEARCH_SCHEMA, "session_dir": str(self.root),
            "status": "complete" if self.is_complete else "incomplete", "resumed": self.resumed,
            "config_hash": self.run_meta.get("config_hash"), "data_hash": self.run_meta.get("data_hash"),
            "checkpoint_useful_interval": CHECKPOINT_USEFUL_EVERY,
            "committed_useful_checks": self.committed_useful, "committed_result_rows": self.committed_rows,
            "chunk_count": len(self.chunks), "chunk_bytes": sum(int(m.get("bytes", 0)) for m in self.chunks.values()),
            "stage_counts": stage_counts, "recovery_notes": list(self.recovery_notes),
            "index_path": str(self.index_path), "run_path": str(self.run_path),
        }

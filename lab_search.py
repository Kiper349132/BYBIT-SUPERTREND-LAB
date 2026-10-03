"""Deterministic, resumable hierarchical Supertrend search (v0.8).

Guarantees
----------
* Every decision (seeds, regions, candidate sets, retained pools, final TOP)
  is a pure function of the SET of evaluated results. The order in which
  worker results arrive does not matter (exact top-K with total-order tie
  breaks), so an interrupted + resumed run gives the same answer as an
  uninterrupted one. tests/test_resume_equivalence.py enforces this.
* After a restart the already computed part of a stage is rebuilt from the
  chunk files (the primary data); caches only speed this up.
* Search expansion uses SELECTION data only. VALIDATION and the development
  holdout never influence which parameters are explored.

Hierarchy per timeframe
-----------------------
1. broad / breadth  wide coarse grid + neutral breadth probes
2. regions          stable-region analysis on the coarse landscape: a point is
                    good if the LOWER QUARTILE of selection scores in its
                    neighbourhood is high (a plateau, not a lonely spike)
3. fine             local deepening around the best stable regions
4. cluster_wide     regions re-evaluated with fine results, wider local grids
5. cluster_deep     very fine grid ONLY around the most stable local points
6. adaptive_v2      regime-switching combinations of the best LONG/SHORT parts
"""
from __future__ import annotations

import ctypes
from array import array
import gc
import hashlib
import heapq
import json
import math
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import lab_core as core
from lab_core import AutoResult, result_key
from lab_log import file_log
from lab_storage import ResearchStore

RETAIN_ROWS_PER_STAGE = 3500
ADAPTIVE_REGIME_CHUNK = 8
REPORT_POOL_LIMIT = 25000
BASE_STAGES = ("broad", "breadth", "fine", "cluster_wide", "cluster_deep")
ALL_STAGES = BASE_STAGES + ("adaptive_v2",)
MODES = ("BOTH", "LONG", "SHORT")


# --------------------------------------------------------------------------
# Search profiles (grids unchanged from v0.7.1; region radii are new)
# --------------------------------------------------------------------------

def _arange(a, b, step):
    return np.arange(a, b, step)


DEPTH_PROFILES = {
    "Стандартный": dict(
        coarse_periods=list(range(2, 181, 4)), coarse_mults=[round(x, 3) for x in _arange(0.4, 9.0001, 0.20)],
        coarse_p_step=4, coarse_m_step=0.20,
        fine=(28, 4, 0.40, 0.05), cluster=(12, 10, 0.70, 0.05), micro=(8, 3, 0.12, 0.01), component_limit=5,
        breadth=dict(p_long=range(30, 141, 5), m_long=(1.5, 4.0001, 0.10), p_fast=range(2, 13), m_fast=(6.0, 10.0001, 0.10),
                     p_edge=range(150, 241, 10), m_edge=(0.5, 10.0001, 0.50)),
        adaptive=dict(ema_pairs=[(8, 21), (12, 26), (20, 50)], adx_periods=[14], adx_thresholds=[18, 25],
                      separations=[0.10, 0.25], confirms=[2], holds=[3, 6]),
    ),
    "Глубокий": dict(
        coarse_periods=list(range(2, 242, 6)), coarse_mults=[round(x, 3) for x in _arange(0.30, 12.0001, 0.10)],
        coarse_p_step=6, coarse_m_step=0.10,
        fine=(30, 3, 0.30, 0.025), cluster=(12, 6, 0.45, 0.025), micro=(8, 2, 0.08, 0.005), component_limit=4,
        breadth=dict(p_long=range(30, 181, 2), m_long=(1.2, 4.5001, 0.05), p_fast=range(2, 19), m_fast=(5.5, 11.0001, 0.05),
                     p_edge=range(180, 301, 5), m_edge=(0.5, 12.0001, 0.25)),
        adaptive=dict(ema_pairs=[(6, 18), (8, 21), (12, 26), (20, 50)], adx_periods=[10, 14], adx_thresholds=[15, 20, 25],
                      separations=[0.10, 0.25], confirms=[2, 3], holds=[4, 8]),
    ),
    "Максимальный": dict(
        coarse_periods=list(range(2, 321, 2)), coarse_mults=[round(x, 3) for x in _arange(0.25, 14.0001, 0.05)],
        coarse_p_step=2, coarse_m_step=0.05,
        fine=(180, 8, 0.70, 0.015), cluster=(70, 24, 1.30, 0.015), micro=(45, 7, 0.28, 0.0025), component_limit=12,
        breadth=dict(p_long=range(20, 241), m_long=(1.0, 5.0001, 0.015), p_fast=range(2, 21), m_fast=(5.0, 12.5001, 0.015),
                     p_edge=range(220, 401, 3), m_edge=(0.3, 14.0001, 0.10)),
        adaptive=dict(ema_pairs=[(5, 13), (6, 18), (8, 21), (12, 26), (20, 50), (30, 80)], adx_periods=[8, 10, 14, 20],
                      adx_thresholds=[12, 15, 18, 20, 25, 30], separations=[0.0, 0.08, 0.15, 0.25, 0.40],
                      confirms=[2, 3, 4], holds=[3, 6, 12]),
    ),
    # Small profiles for automated tests / load tests (not shown in the GUI).
    "Тест": dict(
        coarse_periods=list(range(4, 41, 6)), coarse_mults=[round(x, 3) for x in _arange(1.0, 5.0001, 0.5)],
        coarse_p_step=6, coarse_m_step=0.5,
        fine=(4, 2, 0.30, 0.10), cluster=(3, 3, 0.40, 0.10), micro=(2, 1, 0.06, 0.02), component_limit=2,
        breadth=dict(p_long=range(10, 31, 10), m_long=(1.5, 3.0001, 0.5), p_fast=range(2, 4), m_fast=(6.0, 7.0001, 0.5),
                     p_edge=range(45, 51, 5), m_edge=(1.0, 3.0001, 1.0)),
        adaptive=dict(ema_pairs=[(8, 21), (12, 26)], adx_periods=[14], adx_thresholds=[18],
                      separations=[0.10, 0.25], confirms=[2], holds=[3]),
    ),
}


def profile_for(depth: str) -> dict:
    return DEPTH_PROFILES.get(depth, DEPTH_PROFILES["Глубокий"])


# --------------------------------------------------------------------------
# System memory
# --------------------------------------------------------------------------

def memory_status() -> dict:
    """Physical RAM load and commit-charge load (percent).

    On Windows a process is killed / a worker dies when the COMMIT limit
    (RAM + page file) is exhausted, which can happen while physical RAM still
    looks fine. v0.7.1 only watched physical RAM.
    """
    out = {"phys_pct": 0.0, "commit_pct": 0.0, "avail_phys_mb": 0.0, "avail_commit_mb": 0.0, "total_phys_mb": 0.0}
    try:
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            st = MEMORYSTATUSEX(); st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                out["phys_pct"] = float(st.dwMemoryLoad)
                out["total_phys_mb"] = st.ullTotalPhys / 1048576.0
                out["avail_phys_mb"] = st.ullAvailPhys / 1048576.0
                out["avail_commit_mb"] = st.ullAvailPageFile / 1048576.0
                if st.ullTotalPageFile:
                    out["commit_pct"] = 100.0 * (1.0 - st.ullAvailPageFile / st.ullTotalPageFile)
                return out
        vals = {}
        with open("/proc/meminfo", "r", encoding="ascii", errors="ignore") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                try:
                    vals[k] = float(v.strip().split()[0]) / 1024.0
                except Exception:
                    pass
        total = vals.get("MemTotal", 0.0); avail = vals.get("MemAvailable", vals.get("MemFree", 0.0))
        if total > 0:
            out["phys_pct"] = max(0.0, min(100.0, (1.0 - avail / total) * 100.0))
            out["total_phys_mb"] = total; out["avail_phys_mb"] = avail
        limit = vals.get("CommitLimit", 0.0); committed = vals.get("Committed_AS", 0.0)
        out["avail_commit_mb"] = max(0.0, limit - committed) if limit else avail
        # Linux overcommits; physical availability is the meaningful limit there.
        out["commit_pct"] = out["phys_pct"]
    except Exception:
        pass
    return out


# Measured by tools/loadtest.py (docs/stage01): a v0.8 worker needs ~140 MB
# (python + numpy + numba runtime) plus ~195 bytes per candle of the largest
# timeframe for its temporary arrays (262,800 candles -> 191 MB peak,
# 790,000 candles -> 296 MB peak). 260 bytes/candle keeps a safety margin.
WORKER_BASE_MB = 150.0
WORKER_BYTES_PER_CANDLE = 260.0
PARENT_RESERVE_MB = 1200.0


def recommended_workers(max_candles: int = 0, cpu_count: Optional[int] = None, mem: Optional[dict] = None) -> tuple[int, str]:
    cpu = max(1, cpu_count or os.cpu_count() or 1)
    mem = mem or memory_status()
    per_worker = WORKER_BASE_MB + max_candles * WORKER_BYTES_PER_CANDLE / 1048576.0
    budget = min(mem.get("avail_phys_mb", 0.0) or 0.0, mem.get("avail_commit_mb", 0.0) or 1e12) - PARENT_RESERVE_MB
    by_mem = int(max(1, budget // per_worker)) if budget > 0 else 1
    by_cpu = cpu - 1 if cpu > 2 else cpu
    n = max(1, min(32, by_cpu, by_mem))
    why = f"CPU {cpu} → {by_cpu}; RAM доступно {mem.get('avail_phys_mb', 0):.0f} МБ, ~{per_worker:.0f} МБ/процесс → {by_mem}"
    return n, why


# --------------------------------------------------------------------------
# Order-independent retained pools
# --------------------------------------------------------------------------

def _score(x) -> float:
    try:
        x = float(x)
    except Exception:
        return -1e9
    return x if math.isfinite(x) else -1e9


class _TopK:
    """Exact top-k by (score, key). Independent of insertion order."""
    __slots__ = ("k", "heap", "rows")

    def __init__(self, k: int):
        self.k = int(k); self.heap = []; self.rows = {}

    def add(self, score: float, key: tuple, row) -> None:
        if self.k <= 0 or key in self.rows:
            return
        item = (score, key)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item); self.rows[key] = row
        elif item > self.heap[0]:
            _s, old = heapq.heapreplace(self.heap, item)
            del self.rows[old]
            self.rows[key] = row


class TopPool:
    """Retained working set of one stage / timeframe.

    The `selection` heap and the LONG/SHORT `component` heaps use SELECTION
    information only and are the only ones the search may read. `validation`
    and `robust` heaps exist for the report/GUI.
    """

    def __init__(self, limit: int = RETAIN_ROWS_PER_STAGE):
        self.limit = int(limit)
        self.h = {
            "selection": _TopK(limit // 2),
            "validation": _TopK(limit // 4),
            "robust": _TopK(limit // 4),
        }
        self.per_family = max(25, limit // 24)
        self.families: dict = {}
        self.components = {"LONG": _TopK(max(50, limit // 8)), "SHORT": _TopK(max(50, limit // 8))}

    @staticmethod
    def component_score(row: AutoResult) -> float:
        # v0.8: selection windows only (v0.7.1 used up_/down_ stats of ALL
        # research windows including VALIDATION -> leakage).
        if row.mode == "LONG":
            return _score(row.selection_score + 0.6 * row.sel_up_positive_ratio + row.sel_up_avg_return_pct / 35.0)
        return _score(row.selection_score + 0.6 * row.sel_down_positive_ratio + row.sel_down_avg_return_pct / 35.0)

    def add(self, row: AutoResult) -> None:
        key = result_key(row)
        self.h["selection"].add(_score(row.selection_score), key, row)
        self.h["validation"].add(_score(row.validation_score), key, row)
        self.h["robust"].add(_score(row.robust_score), key, row)
        fam = (row.strategy_type, row.mode)
        tk = self.families.get(fam)
        if tk is None:
            tk = self.families[fam] = _TopK(self.per_family)
        tk.add(_score(row.selection_score), key, row)
        if row.strategy_type == "BASE" and row.mode in self.components and row.selection_score > -1e8:
            self.components[row.mode].add(self.component_score(row), key, row)

    def add_many(self, rows) -> None:
        for r in rows:
            self.add(r)

    def all_rows(self) -> list[AutoResult]:
        merged = {}
        for tk in list(self.h.values()) + list(self.families.values()) + list(self.components.values()):
            merged.update(tk.rows)
        return sorted(merged.values(), key=lambda r: (_score(r.robust_score), _score(r.validation_score),
                                                      _score(r.selection_score), result_key(r)), reverse=True)

    def component_rows(self, mode: str) -> list[AutoResult]:
        tk = self.components[mode]
        return sorted(tk.rows.values(), key=lambda r: (self.component_score(r), result_key(r)), reverse=True)


def merge_pools(pools, limit: int) -> TopPool:
    out = TopPool(limit)
    for p in pools:
        out.add_many(p.all_rows())
    return out


# --------------------------------------------------------------------------
# Landscape and stable regions (selection scores only)
# --------------------------------------------------------------------------

class Landscape:
    """Compact (mode, period, multiplier, selection_score) table of BASE results."""

    # typed arrays: ~21 bytes per point instead of ~130 bytes with Python lists
    def __init__(self):
        self.mode = array("b"); self.period = array("i"); self.mult = array("d"); self.score = array("d")

    def add(self, row: AutoResult) -> None:
        if row.strategy_type != "BASE":
            return
        self.mode.append(MODES.index(row.mode)); self.period.append(int(row.atr_period))
        self.mult.append(round(float(row.multiplier), 4)); self.score.append(_score(row.selection_score))

    def add_many(self, rows) -> None:
        for r in rows:
            self.add(r)

    def extend(self, other: "Landscape") -> None:
        self.mode.extend(other.mode); self.period.extend(other.period); self.mult.extend(other.mult); self.score.extend(other.score)

    def __len__(self) -> int:
        return len(self.mode)

    def to_payload(self) -> dict:
        return {"mode": list(self.mode), "period": list(self.period), "mult": list(self.mult), "score": list(self.score)}

    @classmethod
    def from_payload(cls, d: dict) -> "Landscape":
        x = cls()
        x.mode.extend(d["mode"]); x.period.extend(d["period"]); x.mult.extend(d["mult"]); x.score.extend(d["score"])
        return x

    def arrays(self):
        """Deduplicated arrays sorted by (mode, period, mult) - order independent."""
        if not len(self.mode):
            e = np.array([], dtype=np.int64)
            return e, e, np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        mo = np.frombuffer(self.mode, dtype=np.int8).astype(np.int64); pe = np.frombuffer(self.period, dtype=np.intc).astype(np.int64)
        mu = np.frombuffer(self.mult, dtype=np.float64).copy(); sc = np.frombuffer(self.score, dtype=np.float64).copy()
        order = np.lexsort((sc, mu, pe, mo))
        mo, pe, mu, sc = mo[order], pe[order], mu[order], sc[order]
        keep = np.r_[True, (mo[1:] != mo[:-1]) | (pe[1:] != pe[:-1]) | (mu[1:] != mu[:-1])]
        return mo[keep], pe[keep], mu[keep], sc[keep]


def stable_regions(land: Landscape, *, limit: int, p_radius: float, m_radius: float, min_members: int = 9,
                   max_candidates: int = 4000, extra_both: int = 0) -> dict:
    """Pick up to `limit` non-overlapping stable regions.

    Region score of a point = lower quartile (q25) of the selection scores of
    all evaluated points within ±p_radius ATR periods and ±m_radius multiplier
    (same mode). Ranked by (q25, median, positive share, own score) with a
    deterministic tie break. Returns explainable JSON-ready info.
    """
    mo, pe, mu, sc = land.arrays()
    info = {"p_radius": p_radius, "m_radius": m_radius, "min_members": min_members, "landscape_points": int(len(sc)),
            "fallback": False, "regions": []}
    if len(sc) == 0:
        return info
    valid = sc > -1e8
    cand_idx = np.flatnonzero(valid)
    # candidates: best points by own score, deterministic order
    cand_idx = cand_idx[np.lexsort((mu[cand_idx], pe[cand_idx], mo[cand_idx], -sc[cand_idx]))][:max_candidates]
    by_mode = {}
    for m in range(len(MODES)):
        sel = np.flatnonzero(mo == m)
        by_mode[m] = (pe[sel], mu[sel], sc[sel])  # already sorted by (period, mult)
    scored = []
    for i in cand_idx:
        m = int(mo[i]); p = int(pe[i]); x = float(mu[i])
        P, M, S = by_mode[m]
        a = int(np.searchsorted(P, p - p_radius, side="left")); b = int(np.searchsorted(P, p + p_radius, side="right"))
        mm = M[a:b]; ss = S[a:b]
        nb = ss[np.abs(mm - x) <= m_radius + 1e-9]
        n = int(nb.size)
        if n < min_members:
            continue
        q25 = float(np.quantile(nb, 0.25)); med = float(np.median(nb)); pos = float(np.mean(nb > 0.0))
        scored.append((q25, med, pos, float(sc[i]), m, p, x, n))
    scored.sort(key=lambda t: (-t[0], -t[1], -t[2], -t[3], t[4], t[5], t[6]))
    if not scored:
        info["fallback"] = True
        scored = [(float(sc[i]), float(sc[i]), float(sc[i] > 0), float(sc[i]), int(mo[i]), int(pe[i]), float(mu[i]), 1)
                  for i in cand_idx]

    def overlaps(t, chosen):
        return any(c[4] == t[4] and abs(c[5] - t[5]) <= p_radius and abs(c[6] - t[6]) <= m_radius + 1e-9 for c in chosen)

    chosen = []
    for t in scored:
        if len(chosen) >= limit:
            break
        if not overlaps(t, chosen):
            chosen.append(t)
    if extra_both:
        # v0.7.1 rule kept: make sure BOTH-mode regions are represented.
        target = len(chosen) + extra_both
        for t in scored:
            if len(chosen) >= target:
                break
            if t[4] == MODES.index("BOTH") and not overlaps(t, chosen):
                chosen.append(t)
    for rank, t in enumerate(chosen, 1):
        info["regions"].append({"rank": rank, "mode": MODES[t[4]], "atr_period": t[5], "multiplier": t[6],
                                "q25_selection_score": t[0], "median_selection_score": t[1], "positive_share": t[2],
                                "center_selection_score": t[3], "members": t[7]})
    return info


def local_grid(regions: list[dict], p_radius: int, m_radius: float, m_step: float, m_lo: float, m_hi: float, digits: int = 4) -> set:
    out = set()
    for r in regions:
        p0, m0, mode = int(r["atr_period"]), float(r["multiplier"]), r["mode"]
        for p in range(max(2, p0 - p_radius), p0 + p_radius + 1):
            lo = max(m_lo, m0 - m_radius); hi = min(m_hi, m0 + m_radius); m = lo
            while m <= hi + 1e-12:
                out.add((p, round(m, digits), mode)); m += m_step
    return out


def group_specs(candidates: set) -> dict:
    grouped: dict = {}
    for period, mult, mode in candidates:
        grouped.setdefault(int(period), {}).setdefault(round(float(mult), 4), set()).add(mode)
    return {p: [(m, tuple(sorted(md))) for m, md in sorted(mm.items())] for p, mm in sorted(grouped.items())}


def _digest(items) -> dict:
    """Exact, order-independent fingerprint of a set/list of tuples."""
    seq = sorted(items)
    raw = json.dumps(seq, separators=(",", ":"), default=str).encode("utf-8")
    return {"count": len(seq), "sha256": hashlib.sha256(raw).hexdigest()}


def _unit_id(prefix: str, payload) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:10]}"


def adaptive_component_rows(pool: TopPool, mode: str, limit: int) -> list[AutoResult]:
    ranked = pool.component_rows(mode)
    chosen: list[AutoResult] = []
    for row in ranked:
        if any(abs(row.atr_period - c.atr_period) <= 2 and abs(row.multiplier - c.multiplier) <= 0.18 for c in chosen):
            continue
        chosen.append(row)
        if len(chosen) >= limit:
            break
    return chosen or ranked[:limit]


def diversified_top(rows: list[AutoResult], limit: int = 100) -> list[AutoResult]:
    """Research-only TOP that preserves different strategy families (v0.7.1 logic,
    with a deterministic total order)."""
    ranked = sorted(rows, key=lambda r: (_score(r.robust_score), _score(r.validation_score), _score(r.selection_score), result_key(r)), reverse=True)
    chosen: list[AutoResult] = []
    used_keys = set()

    def is_near(row):
        if row.strategy_type != "BASE":
            return any(
                c.strategy_type == row.strategy_type and row.interval == c.interval
                and abs(row.long_atr_period - c.long_atr_period) <= 1 and abs(row.long_multiplier - c.long_multiplier) <= 0.10
                and abs(row.short_atr_period - c.short_atr_period) <= 1 and abs(row.short_multiplier - c.short_multiplier) <= 0.10
                and (row.strategy_type != "ADAPTIVE_V2" or (
                    row.ema_fast == c.ema_fast and row.ema_slow == c.ema_slow
                    and abs(row.adx_threshold - c.adx_threshold) <= 5
                    and abs(row.regime_separation_atr - c.regime_separation_atr) <= 0.15))
                for c in chosen)
        return any(c.strategy_type == "BASE" and row.interval == c.interval and row.mode == c.mode
                   and abs(row.atr_period - c.atr_period) <= 1 and abs(row.multiplier - c.multiplier) <= 0.12 for c in chosen)

    def final():
        return sorted(chosen, key=lambda x: (_score(x.robust_score), result_key(x)), reverse=True)

    families = {}
    for r in ranked:
        if r.robust_score <= -1e8:
            continue
        families.setdefault((r.strategy_type, r.interval, r.mode), []).append(r)
    for fam in sorted(families):
        added = 0
        for r in families[fam]:
            k = result_key(r)
            if k in used_keys or is_near(r):
                continue
            chosen.append(r); used_keys.add(k); added += 1
            if len(chosen) >= limit:
                return final()
            if added >= 2:
                break
    for r in ranked:
        k = result_key(r)
        if k in used_keys or r.robust_score <= -1e8 or is_near(r):
            continue
        chosen.append(r); used_keys.add(k)
        if len(chosen) >= limit:
            break
    if len(chosen) < limit:
        for r in ranked:
            k = result_key(r)
            if k in used_keys or r.robust_score <= -1e8:
                continue
            chosen.append(r); used_keys.add(k)
            if len(chosen) >= limit:
                break
    return final()


# --------------------------------------------------------------------------
# Worker pool: ONE pool for the whole run, auto-recovery after a worker crash
# --------------------------------------------------------------------------

class StopRequested(Exception):
    pass


class MemoryStop(Exception):
    pass


class PoolFailure(Exception):
    pass


class WorkerPool:
    def __init__(self, workers: int, market_dir: Path, plans: dict, max_tasks_per_child: Optional[int] = 400,
                 max_restarts: int = 3, on_event: Optional[Callable[[str], None]] = None):
        self.workers = max(1, int(workers))
        self.market_dir = market_dir
        self.plans = plans
        self.max_tasks_per_child = max_tasks_per_child
        self.max_restarts = max_restarts
        self.restarts = 0
        self.on_event = on_event or (lambda s: None)
        self.ctx = mp.get_context("spawn")   # identical behaviour on Windows and Linux
        self.executor: Optional[ProcessPoolExecutor] = None
        self._start()

    def _start(self):
        kw = dict(max_workers=self.workers, mp_context=self.ctx, initializer=core._worker_init,
                  initargs=(str(self.market_dir), self.plans))
        if self.max_tasks_per_child:
            kw["max_tasks_per_child"] = int(self.max_tasks_per_child)
        self.executor = ProcessPoolExecutor(**kw)

    def worker_pids(self) -> list[int]:
        try:
            return sorted(int(p) for p in (self.executor._processes or {}).keys())
        except Exception:
            return []

    def terminate(self):
        ex = self.executor
        self.executor = None
        if ex is None:
            return
        procs = []
        try:
            procs = list((ex._processes or {}).values())
        except Exception:
            pass
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        for p in procs:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.join(timeout=5)
            except Exception:
                pass

    def close(self):
        ex = self.executor
        self.executor = None
        if ex is not None:
            ex.shutdown(wait=True, cancel_futures=True)

    def run(self, fn, items: list, stop_check: Callable[[], None]):
        """Yield (item, result) for every item in completion order.

        At most 2 tasks per worker are in flight. If a worker process dies
        (BrokenProcessPool) the pool is recreated - with fewer workers, since
        memory pressure is the usual cause - and unfinished items are resubmitted.
        Results are deterministic functions of the item, so recomputation is
        harmless.
        """
        pending = list(items)
        pending.reverse()
        while True:
            inflight = {}
            broken = None
            try:
                while pending or inflight:
                    while pending and len(inflight) < max(2, self.workers * 2):
                        item = pending.pop()
                        inflight[self.executor.submit(fn, item[1])] = item
                    done, _ = wait(tuple(inflight), timeout=0.5, return_when=FIRST_COMPLETED)
                    stop_check()
                    for fut in done:
                        item = inflight.pop(fut)
                        try:
                            res = fut.result()
                        except BrokenProcessPool:
                            pending.append(item)
                            raise
                        yield item, res
                        del res
                return
            except BrokenProcessPool as exc:
                broken = exc
                for item in inflight.values():
                    pending.append(item)
            # recover
            self.terminate()
            self.restarts += 1
            msg = f"Расчётный процесс аварийно завершился ({broken}). Перезапуск пула #{self.restarts}"
            file_log("POOL " + msg)
            if self.restarts > self.max_restarts:
                raise PoolFailure(msg)
            self.workers = max(1, int(self.workers * 0.75))
            self.on_event(msg + f", процессов: {self.workers}")
            self._start()


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

@dataclass
class RunResult:
    status: str                       # complete | stopped | memory_stop
    report_pool: list = field(default_factory=list)
    top: list = field(default_factory=list)
    search_stats: list = field(default_factory=list)
    trace: dict = field(default_factory=dict)   # deterministic decisions (for tests / report)
    message: str = ""


class ResearchRunner:
    def __init__(self, cfg: dict, store: ResearchStore, *, workers: int,
                 on_status: Optional[Callable[[dict], None]] = None,
                 stop_event: Optional[threading.Event] = None,
                 memory_probe: Callable[[], dict] = memory_status,
                 memory_stop_pct: float = 94.0,
                 max_tasks_per_child: Optional[int] = 400,
                 use_stage_cache: bool = True,
                 pool_max_restarts: int = 3,
                 unit_hook: Optional[Callable[["ResearchRunner", str, str, str], None]] = None):
        self.cfg = dict(cfg)
        self.store = store
        self.workers = max(1, int(workers))
        self.on_status = on_status or (lambda d: None)
        self.stop_event = stop_event or threading.Event()
        self.memory_probe = memory_probe
        self.memory_stop_pct = memory_stop_pct
        self.max_tasks_per_child = max_tasks_per_child
        self.use_stage_cache = use_stage_cache
        self.pool_max_restarts = pool_max_restarts
        self.unit_hook = unit_hook
        self.profile = profile_for(cfg.get("depth", "Глубокий"))
        self.market = store.load_market()
        self.plans = {tf: core.build_walk_forward_plan_arrays(a["timestamp"], a["open"], a["close"]) for tf, a in self.market.items()}
        self.pool: Optional[WorkerPool] = None
        self.trace: dict = {}
        self.search_stats: list = []
        self.started = time.monotonic()
        self.session_base_useful = store.committed_useful
        self.est_total = 0
        self.last_mem = {}
        self.units_done_session = 0

    # ----- status ---------------------------------------------------------------

    def _status(self, phase: str, detail: str = "", log: str = "", progress: Optional[float] = None):
        checked = self.store.committed_useful + self.store.pending_useful
        elapsed = max(0.001, time.monotonic() - self.started)
        session = max(0, checked - self.session_base_useful)
        speed = session / elapsed if session else 0.0
        remain = max(0, self.est_total - checked)
        self.on_status({
            "phase": phase, "checked": checked, "total": max(self.est_total, checked), "speed": speed,
            "eta": (remain / speed) if speed > 0 else None, "workers": self.pool.workers if self.pool else 0,
            "detail": detail, "log": log, "progress": progress,
            "ram_pct": self.last_mem.get("phys_pct", 0.0), "commit_pct": self.last_mem.get("commit_pct", 0.0),
        })

    def _check_stop(self):
        if self.stop_event.is_set():
            raise StopRequested()

    def _check_memory(self):
        mem = self.memory_probe()
        self.last_mem = mem
        load = max(float(mem.get("phys_pct", 0.0)), float(mem.get("commit_pct", 0.0)))
        if load >= self.memory_stop_pct:
            raise MemoryStop(f"занято {load:.1f}% памяти (RAM/commit)")

    # ----- one stage ---------------------------------------------------------------

    def _rebuild(self, tf: str, stage: str, pool: TopPool, land: Landscape, n_units: int, committed: dict) -> bool:
        """Fill pool/landscape with already committed results of the stage.
        Returns True when a valid stage cache was used."""
        if self.use_stage_cache and len(committed) == n_units:
            cache = self.store.load_stage_cache(tf, stage)
            if cache is not None:
                pool.add_many(core.autoresult_from_dict(d) for d in cache["pool"])
                land.extend(Landscape.from_payload(cache["landscape"]))
                return True
        if committed:
            for row in self.store.iter_rows(tf, stage, allowed_units=set(committed)):
                pool.add(row); land.add(row)
        return False

    def _run_units(self, tf: str, stage: str, label: str, units: list, fn, pool: TopPool, land: Landscape,
                   checks_per_config: int, progress_span: tuple) -> dict:
        """units: list of (unit_id, task, n_configs). Deterministic regardless of
        completion order; resumes from committed chunks."""
        committed = self.store.committed_units(tf, stage)
        unit_ids = {u[0] for u in units}
        committed = {k: v for k, v in committed.items() if k in unit_ids}
        from_cache = self._rebuild(tf, stage, pool, land, len(units), committed)
        todo = [u for u in units if u[0] not in committed]
        total_cfg = sum(u[2] for u in units)
        done_cfg = sum(u[2] for u in units if u[0] in committed)
        if committed:
            self._status(f"{tf}м · {label}", log=f"{tf}м · {label}: восстановлено из checkpoint {len(committed)}/{len(units)} блоков"
                         + (" (кэш этапа)" if from_cache else " (из chunk-файлов)"))
        if todo:
            for (uid, _task, ncfg), batch in self.pool.run(fn, todo, self._check_stop):
                pool.add_many(batch); land.add_many(batch)
                self.store.write_unit(tf, stage, uid, batch, ncfg * checks_per_config)
                del batch
                done_cfg += ncfg
                self.units_done_session += 1
                if self.store.should_flush():
                    self.store.flush(reason=f"interval_{tf}_{stage}")
                    self._status(f"{tf}м · {label}", log=f"CHECKPOINT: на диске {self.store.committed_useful:,} проверок")
                if self.unit_hook:
                    self.unit_hook(self, tf, stage, uid)
                self._check_memory()
                self._check_stop()
                a, b = progress_span
                self._status(f"{tf}м · {label}", detail=(
                    f"{done_cfg:,}/{total_cfg:,} · диск {self.store.committed_useful:,} · буфер {self.store.pending_useful:,} · "
                    f"RAM {self.last_mem.get('phys_pct', 0):.0f}% · commit {self.last_mem.get('commit_pct', 0):.0f}%"),
                    progress=a + (b - a) * done_cfg / max(1, total_cfg))
            self.store.flush(reason=f"stage_complete_{tf}_{stage}")
        if not from_cache:
            self.store.save_stage_cache(tf, stage, {"pool": [core.autoresult_to_dict(r) for r in pool.all_rows()],
                                                    "landscape": land.to_payload()})
        return {"units": len(units), "configs": total_cfg, "resumed_units": len(committed)}

    def _base_stage(self, tf, stage, label, candidates, min_trades, checks, span):
        grouped = group_specs(candidates)
        fee, slip = self.cfg["fee"], self.cfg["slippage"]
        units = []
        for period, specs in grouped.items():
            uid = _unit_id(f"p{period}", specs)
            units.append((uid, (tf, period, specs, fee, slip, min_trades, stage), sum(len(md) for _m, md in specs)))
        pool = TopPool(RETAIN_ROWS_PER_STAGE); land = Landscape()
        self.est_total += sum(u[2] for u in units) * checks
        meta = self._run_units(tf, stage, label, units, core._worker_eval_period, pool, land, checks, span)
        meta["candidates"] = len(candidates)
        return pool, land, meta

    # ----- main ---------------------------------------------------------------------

    def run(self) -> RunResult:
        try:
            core.warmup_kernels()
            self.pool = WorkerPool(self.workers, self.store.market_dir, self.plans, self.max_tasks_per_child,
                                   max_restarts=self.pool_max_restarts, on_event=lambda m: self._status("ПУЛ ПРОЦЕССОВ", log=m))
            report_pool = TopPool(REPORT_POOL_LIMIT)
            tfs = [str(t) for t in self.cfg["tfs"] if str(t) in self.market]
            for tf_index, tf in enumerate(tfs):
                self._check_stop()
                tf_pool = self._run_timeframe(tf, tf_index, len(tfs))
                if tf_pool is not None:
                    report_pool.add_many(tf_pool.all_rows())
                gc.collect()
            self.pool.close()
            self.pool = None
            rows = report_pool.all_rows()
            if not rows:
                raise RuntimeError("Не удалось получить ни одного результата")
            top = diversified_top(rows, 100)
            return RunResult("complete", rows, top, self.search_stats, self.trace)
        except StopRequested:
            self._shutdown("user_stop")
            return RunResult("stopped", trace=self.trace, message="Остановлено; checkpoint сохранён")
        except MemoryStop as exc:
            self._shutdown("memory_stop")
            return RunResult("memory_stop", trace=self.trace,
                             message=f"Защитная остановка: {exc}. Checkpoint сохранён; повторный запуск продолжит расчёт.")
        except BaseException:
            self._shutdown("error")
            raise

    def _shutdown(self, reason: str):
        if self.pool is not None:
            self.pool.terminate()
            self.pool = None
        try:
            self.store.flush(reason=reason)
        except Exception as exc:
            file_log(f"FLUSH on {reason} failed: {exc}")
            self.store.abandon_open_chunk()

    def _run_timeframe(self, tf: str, tf_index: int, n_tfs: int) -> Optional[TopPool]:
        prof = self.profile
        plan = self.plans[tf]
        if len(plan["windows"]) < 4:
            self._status(f"ПРОПУСК {tf}м", log=f"{tf}м: только {len(plan['windows'])} walk-forward окон")
            return None
        t0 = time.monotonic()
        checks = len(plan["windows"]) + 4
        min_trades = max(24, int(self.cfg["months"] * 2.2))
        hs = plan["holdout_start"]
        span = lambda a, b: ((tf_index + a / 100) / n_tfs * 100, (tf_index + b / 100) / n_tfs * 100)
        trace = self.trace.setdefault(tf, {})

        coarse = {(p, m, mode) for p in prof["coarse_periods"] if p < hs for m in prof["coarse_mults"] for mode in MODES}
        br = prof["breadth"]; breadth = set()
        for p0 in br["p_long"]:
            for m in _arange(*br["m_long"]):
                breadth.add((p0, round(float(m), 4), "BOTH"))
        for p0 in br["p_fast"]:
            for m in _arange(*br["m_fast"]):
                breadth.add((p0, round(float(m), 4), "BOTH"))
        for p0 in br["p_edge"]:
            for m in _arange(*br["m_edge"]):
                for mode in MODES:
                    breadth.add((p0, round(float(m), 4), mode))
        breadth = {x for x in breadth if x[0] < hs} - coarse

        stats = {}
        pools = {}
        land = Landscape()   # cumulative selection landscape of all BASE stages so far

        def stage(name, label, cands, a, b):
            pools[name], stage_land, stats[name] = self._base_stage(tf, name, label, cands, min_trades, checks, span(a, b))
            land.extend(stage_land)

        stage("broad", "широкий грубый поиск", coarse, 0, 18)
        stage("breadth", "поиск в ширину", breadth, 18, 30)

        # 2. stable regions on the coarse landscape -> 3. local deepening
        rp, rm = 2 * prof["coarse_p_step"], 2 * prof["coarse_m_step"]
        fs, fpr, fmr, fms = prof["fine"]
        reg_fine = stable_regions(land, limit=fs, p_radius=rp, m_radius=rm)
        self.store.save_regions(tf, "fine", reg_fine)
        fine = local_grid(reg_fine["regions"], fpr, fmr, fms, 0.15, 16.0) - coarse - breadth
        stage("fine", "углубление вокруг устойчивых областей", fine, 30, 48)

        cs, cpr, cmr, cms = prof["cluster"]
        reg_cluster = stable_regions(land, limit=cs, p_radius=rp, m_radius=rm, extra_both=max(6, cs // 3))
        self.store.save_regions(tf, "cluster_wide", reg_cluster)
        cluster = local_grid(reg_cluster["regions"], cpr, cmr, cms, 0.12, 18.0) - coarse - breadth - fine
        stage("cluster_wide", "кластеры в ширину", cluster, 48, 68)

        ms, mpr, mmr, mms = prof["micro"]
        reg_micro = stable_regions(land, limit=ms, p_radius=max(1, fpr), m_radius=fmr / 2.0)
        self.store.save_regions(tf, "cluster_deep", reg_micro)
        micro = local_grid(reg_micro["regions"], mpr, mmr, mms, 0.10, 20.0, digits=5) - coarse - breadth - fine - cluster
        stage("cluster_deep", "мелкая сетка только локально", micro, 68, 82)
        land = None  # release the cumulative landscape before Adaptive V2

        # Trace = exact fingerprints of every search decision (tests compare them
        # between interrupted and uninterrupted runs). Digests instead of the
        # sets themselves keep RAM small in Maximum mode.
        trace["candidates"] = {"fine": _digest(fine), "cluster_wide": _digest(cluster), "cluster_deep": _digest(micro)}
        trace["regions"] = {"fine": reg_fine["regions"], "cluster_wide": reg_cluster["regions"], "cluster_deep": reg_micro["regions"]}
        del coarse, breadth, fine, cluster, micro

        # 6. Adaptive V2 (components chosen with SELECTION information only)
        base_pool = merge_pools([pools[s] for s in BASE_STAGES], RETAIN_ROWS_PER_STAGE * 2)
        longs = adaptive_component_rows(base_pool, "LONG", prof["component_limit"])
        shorts = adaptive_component_rows(base_pool, "SHORT", prof["component_limit"])
        trace["components"] = {"LONG": [result_key(r) for r in longs], "SHORT": [result_key(r) for r in shorts]}
        ad = prof["adaptive"]
        regime_configs = [
            {"ema_fast": ef, "ema_slow": es, "adx_period": ap, "adx_threshold": at, "separation_atr": sep, "confirm_bars": cb, "min_hold_bars": hb}
            for ef, es in ad["ema_pairs"] for ap in ad["adx_periods"] for at in ad["adx_thresholds"]
            for sep in ad["separations"] for cb in ad["confirms"] for hb in ad["holds"] if ef < es
        ]
        ad_pool = TopPool(RETAIN_ROWS_PER_STAGE)
        ad_total = len(longs) * len(shorts) * len(regime_configs)
        if longs and shorts and ad_total:
            self._prepare_features(tf, ad["ema_pairs"], ad["adx_periods"])
            comp_sig = [result_key(r) for r in longs] + [result_key(r) for r in shorts]
            units = []
            for i in range(0, len(regime_configs), ADAPTIVE_REGIME_CHUNK):
                chunk = regime_configs[i:i + ADAPTIVE_REGIME_CHUNK]
                uid = _unit_id(f"r{i // ADAPTIVE_REGIME_CHUNK:05d}", [chunk, comp_sig])
                units.append((uid, (tf, chunk, longs, shorts, self.cfg["fee"], self.cfg["slippage"], min_trades),
                              len(chunk) * len(longs) * len(shorts)))
            self.est_total += ad_total * checks
            stats["adaptive_v2"] = self._run_units(tf, "adaptive_v2", "Adaptive V2", units, core._worker_eval_adaptive_v2_chunk,
                                                   ad_pool, Landscape(), checks, span(82, 100))
        else:
            stats["adaptive_v2"] = {"units": 0, "configs": 0, "resumed_units": 0}

        tf_pool = merge_pools([pools[s] for s in BASE_STAGES] + [ad_pool], RETAIN_ROWS_PER_STAGE * 3)
        trace["stage_pools"] = {s: _digest([result_key(r) for r in pools[s].all_rows()]) for s in BASE_STAGES}
        trace["stage_pools"]["adaptive_v2"] = _digest([result_key(r) for r in ad_pool.all_rows()])
        cfg_total = sum(v.get("configs", 0) for v in stats.values())
        self.search_stats.append({
            "timeframe_min": tf, "candles": int(len(self.market[tf]["close"])), "workers": self.pool.workers if self.pool else 0,
            "depth": self.cfg.get("depth"), "numba": core.NUMBA_AVAILABLE,
            "walk_forward_windows": len(plan["windows"]), "selection_windows": len(plan["selection_windows"]),
            "validation_windows": len(plan["validation_windows"]), "walk_forward_test_days": plan["test_days"],
            "warmup_days": plan["warmup_days"], "development_holdout_days": plan["holdout_days"],
            **{f"{s}_candidates": stats[s].get("configs", 0) for s in ALL_STAGES},
            **{f"{s}_resumed_units": stats[s].get("resumed_units", 0) for s in ALL_STAGES},
            "adaptive_v2_long_components": len(longs), "adaptive_v2_short_components": len(shorts),
            "adaptive_v2_regime_configs": len(regime_configs),
            "evaluated_strategy_configs": cfg_total, "useful_backtest_evaluations": cfg_total * checks,
            "retained_in_memory": len(tf_pool.all_rows()), "seconds_this_session": round(time.monotonic() - t0, 3),
            "pool_restarts": self.pool.restarts if self.pool else 0,
        })
        self._status(f"{tf}м ГОТОВО", log=f"{tf}м: {cfg_total:,} стратегий · всё сохранено на диск ({self.store.committed_useful:,} проверок)")
        return tf_pool

    def _prepare_features(self, tf: str, ema_pairs, adx_periods):
        """Precompute regime features once in the parent and share them through
        memory-mapped files (identical numerics to v0.7.1)."""
        from lab_storage import save_npy_durable
        a = self.market[tf]
        h = np.asarray(a["high"]); l = np.asarray(a["low"]); c = np.asarray(a["close"])
        for name in core.feature_names([x for pair in ema_pairs for x in pair], adx_periods):
            path = core.market_file(self.store.market_dir, tf, name)
            arr = core.compute_feature(name, h, l, c)
            if path.exists():
                try:
                    if np.array_equal(np.load(path, allow_pickle=False), arr):
                        continue
                except Exception:
                    pass
            save_npy_durable(path, arr)

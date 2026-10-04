"""Exhaustive dense grid of BASE Supertrend configs on real candles, evaluated
with the program's own per-config function (lab_core.evaluate_base_config via
the worker). Saves one compact .npz table used by analyse_search.py.

    python audit/dense_grid.py --tf 60 --pmin 2 --pmax 240 --pstep 1 --mmin 0.3 --mmax 12.0 --mstep 0.05 --out grid_60.npz
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from common import core, load_binance
from lab_search import MODES, WorkerPool
from lab_storage import ResearchStore

FIELDS = ("selection_score", "validation_score", "robust_score", "wf_positive_ratio", "wf_best_return_pct",
          "wf_median_return_pct", "selection_positive_ratio", "validation_positive_ratio")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="60")
    ap.add_argument("--pmin", type=int, default=2); ap.add_argument("--pmax", type=int, default=240); ap.add_argument("--pstep", type=int, default=1)
    ap.add_argument("--mmin", type=float, default=0.3); ap.add_argument("--mmax", type=float, default=12.0); ap.add_argument("--mstep", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = load_binance(a.tf)
    tmp = Path(tempfile.mkdtemp(prefix="audit_grid_"))
    cfg = {"symbol": "GRID", "months": 18, "tfs": [a.tf], "fee": 0.00055, "slippage": 0.0002, "depth": "Глубокий"}
    store = ResearchStore.create(tmp, cfg, {a.tf: m})
    plan = core.build_walk_forward_plan_arrays(m["timestamp"], m["open"], m["close"])
    min_trades = max(24, int(18 * 2.2))
    mults = [round(x, 4) for x in np.arange(a.mmin, a.mmax + 1e-9, a.mstep)]
    specs = [(x, MODES) for x in mults]
    items = [(p, (a.tf, p, specs, 0.00055, 0.0002, min_trades, "grid")) for p in range(a.pmin, a.pmax + 1, a.pstep)]
    cols = {k: [] for k in ("period", "mult", "mode", *FIELDS, "research_ret", "research_dd", "trades", "research_pf",
                            "holdout_ret", "holdout_dd", "holdout_trades")}
    t0 = time.monotonic()
    pool = WorkerPool(a.workers, store.market_dir, {a.tf: plan}, max_tasks_per_child=None)
    try:
        for _item, rows in pool.run(core._worker_eval_period, items, lambda: None):
            for r in rows:
                cols["period"].append(r.atr_period); cols["mult"].append(r.multiplier); cols["mode"].append(MODES.index(r.mode))
                for f in FIELDS:
                    cols[f].append(getattr(r, f))
                cols["research_ret"].append(r.train.total_return_pct); cols["research_dd"].append(r.train.max_drawdown_pct)
                cols["trades"].append(r.train.trades); cols["research_pf"].append(r.train.profit_factor)
                cols["holdout_ret"].append(r.test.total_return_pct); cols["holdout_dd"].append(r.test.max_drawdown_pct)
                cols["holdout_trades"].append(r.test.trades)
    finally:
        pool.close()
        shutil.rmtree(tmp, ignore_errors=True)
    np.savez_compressed(a.out, **{k: np.asarray(v) for k, v in cols.items()},
                        meta=np.array([a.tf, len(cols["period"]), round(time.monotonic() - t0, 1)], dtype=object))
    print(a.tf, len(cols["period"]), "configs", round(time.monotonic() - t0, 1), "s")


if __name__ == "__main__":
    main()

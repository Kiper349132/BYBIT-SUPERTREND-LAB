"""Look-ahead invariants on real candles.

Invariant 1 (point-in-time): for random N, replacing ALL candles after N with a
different path must not change any indicator, signal or regime value at
candles <= N, nor the return of any walk-forward window that ends at <= N.

Invariant 2 (phase isolation, full program search): rewriting everything from
the first VALIDATION candle on must not change any search decision (regions,
fine/cluster candidate sets, Adaptive V2 components) - only validation-based
numbers may change. Rewriting everything from the DEVELOPMENT HOLDOUT on must
not change ANY ranking (TOP order, selection/validation/robust scores) - only
the holdout ("test") metrics may change.

Also documents: the split boundaries are anchored at the LAST candle, so
appending new candles moves them.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from common import core, load_binance
from lab_search import ResearchRunner
from lab_storage import ResearchStore
from null_tests import null_sign


def replace_after(m, n_from, seed):
    """Candles >= n_from replaced by a direction-randomised path continuing from close[n_from-1]."""
    alt = null_sign(m, seed)
    out = {k: np.array(v, copy=True) for k, v in m.items()}
    scale = m["close"][n_from - 1] / alt["close"][n_from - 1]
    for k in ("open", "high", "low", "close"):
        out[k][n_from:] = alt[k][n_from:] * scale
    return out


def invariant_1(m, plan):
    rng = np.random.default_rng(7)
    n = len(m["close"])
    Ns = sorted(set(int(x) for x in rng.integers(3000, n - 50, size=6)) | {3500, 5000, 6500, plan["selection_end"], plan["holdout_start"] - 1})
    fails = []
    for N in Ns:
        p = replace_after(m, N + 1, seed=N)
        h, l, c, o = m["high"], m["low"], m["close"], m["open"]
        h2, l2, c2, o2 = p["high"], p["low"], p["close"], p["open"]
        for per, mult in [(3, 1.5), (14, 2.5), (60, 4.0), (240, 8.0)]:
            a1 = core._fast_atr(h, l, c, per); a2 = core._fast_atr(h2, l2, c2, per)
            t1 = core._fast_trend(h, l, c, a1, mult); t2 = core._fast_trend(h2, l2, c2, a2, mult)
            if not (np.array_equal(a1[:N + 1], a2[:N + 1], equal_nan=True) and np.array_equal(t1[:N + 1], t2[:N + 1])):
                fails.append(("supertrend", N, per, mult))
            for mode in ("BOTH", "LONG", "SHORT"):
                for a, b, *_ in plan["windows"]:
                    if b > N:
                        continue
                    r1 = core.backtest_fast_arrays(o, c, m["months"], t1, max(a, per), b, 0.00055, 0.0002, mode, 1)
                    r2 = core.backtest_fast_arrays(o2, c2, p["months"], t2, max(a, per), b, 0.00055, 0.0002, mode, 1)
                    if r1 != r2:
                        fails.append(("window", N, per, mult, mode, a, b))
        r1 = core.online_regime_v2(h, l, c, 8, 21, 14, 20.0, 0.15, 3, 6)
        r2 = core.online_regime_v2(h2, l2, c2, 8, 21, 14, 20.0, 0.15, 3, 6)
        if not np.array_equal(r1[:N + 1], r2[:N + 1]):
            fails.append(("regime_v2", N))
        for p_ in (6, 21, 50):
            if not np.array_equal(core._ema_array(c, p_)[:N + 1], core._ema_array(c2, p_)[:N + 1]):
                fails.append(("ema", N, p_))
        for p_ in (10, 14, 20):
            if not np.array_equal(core._adx_array(h, l, c, p_)[:N + 1], core._adx_array(h2, l2, c2, p_)[:N + 1]):
                fails.append(("adx", N, p_))
    return Ns, fails


def search(m, tf, label, workers=1):
    tmp = Path(tempfile.mkdtemp(prefix="audit_la_"))
    try:
        cfg = {"symbol": label, "months": 18, "tfs": [tf], "fee": 0.00055, "slippage": 0.0002, "depth": "Глубокий"}
        store = ResearchStore.create(tmp, cfg, {tf: m})
        return ResearchRunner(cfg, store, workers=workers, max_tasks_per_child=None).run()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tf = "60"
    m = load_binance(tf)
    plan = core.build_walk_forward_plan_arrays(m["timestamp"], m["open"], m["close"])
    report = {}
    Ns, fails = invariant_1(m, plan)
    report["invariant_1"] = {"N_tested": Ns, "failures": fails}
    print("Invariant 1 (point-in-time):", "PASS" if not fails else fails[:5], "N =", Ns)

    base = search(m, tf, "LA0")
    after_val = search(replace_after(m, plan["validation_start"], 11), tf, "LA1")
    after_hold = search(replace_after(m, plan["holdout_start"], 12), tf, "LA2")
    t0, t1, t2 = base.trace[tf], after_val.trace[tf], after_hold.trace[tf]
    dec = lambda t: (t["regions"], t["candidates"], t["components"])
    report["validation_rewritten_search_decisions_identical"] = dec(t0) == dec(t1)
    report["validation_rewritten_top_changed"] = [r.robust_score for r in base.top] != [r.robust_score for r in after_val.top]
    strip = lambda rows: [(core.result_key(r), r.selection_score, r.validation_score, r.robust_score, r.train.total_return_pct) for r in rows]
    report["holdout_rewritten_search_decisions_identical"] = dec(t0) == dec(t2)
    report["holdout_rewritten_ranking_identical"] = strip(base.top) == strip(after_hold.top) and strip(base.report_pool) == strip(after_hold.report_pool)
    report["holdout_rewritten_test_metrics_changed"] = [r.test.total_return_pct for r in base.top] != [r.test.total_return_pct for r in after_hold.top]
    print(json.dumps({k: v for k, v in report.items() if k != "invariant_1"}, indent=1))

    # split anchored at the end: appending 30 days moves every boundary
    extra = 30 * 24
    ts2 = np.r_[m["timestamp"], m["timestamp"][-1] + 3_600_000 * np.arange(1, extra + 1)]
    pr = np.r_[m["close"], np.full(extra, m["close"][-1])]
    plan2 = core.build_walk_forward_plan_arrays(ts2, pr, pr)
    report["appending_30_days_moves_boundaries"] = {
        k: [plan[k], plan2[k]] for k in ("research_start", "selection_end", "validation_start", "holdout_start")}
    print("Boundaries before/after appending 30 days:", report["appending_30_days_moves_boundaries"])
    Path("lookahead_report.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

"""How much more optimistic is the backtest than real trading? Measured on the
real TOP-20 of a real-candle search:

  * program drawdown (closed trades) vs mark-to-market drawdown using every
    candle's adverse extreme (low for longs, high for shorts);
  * funding actually paid/received on Binance BTCUSDT (8h) while in position;
  * fee charged on entry notional vs exact fee on entry + exit notional;
  * walk-forward window statistics from flat-start windows vs the same windows
    cut from one continuous position path.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import core, load_binance, load_funding
from lab_search import ResearchRunner
from lab_storage import ResearchStore

FEE, SLIP = 0.00055, 0.0002


def desired_for(row, m):
    h, l, c = m["high"], m["low"], m["close"]
    atr = core._fast_atr(h, l, c, row.atr_period)
    d = core._fast_trend(h, l, c, atr, row.multiplier).astype(np.int8)
    if row.mode == "LONG":
        d[d != 1] = 0
    elif row.mode == "SHORT":
        d[d != -1] = 0
    return d


def path_stats(m, d, a, b, fund_t, fund_r):
    """Bar-by-bar position path: position during candle j is d[j-1] (decided at
    close j-1, executed at open j), exactly the program's convention."""
    o, h, l, c, ts = m["open"], m["high"], m["low"], m["close"], m["timestamp"]
    eq = 1.0; peak = 1.0; mtm_dd = 0.0; funding = 0.0; fee_exact_extra = 0.0
    pos = 0; entry = 0.0; units = 0.0
    trades = []
    for j in range(a + 1, b + 1):
        want = int(d[j - 1]) if j - 1 < b else pos
        if want != pos:
            if pos != 0:
                px = o[j] * (1 - SLIP) if pos == 1 else o[j] * (1 + SLIP)
                gross = (px - entry) / entry if pos == 1 else (entry - px) / entry
                net = max(gross - 2 * FEE, -0.999999)
                fee_exact_extra += FEE * (px / entry - 1)          # exit fee on exit notional
                eq_trade_start = eq_at_entry
                eq = eq_trade_start * (1 + net)
                trades.append(net)
            pos = want
            if pos != 0:
                entry = o[j] * (1 + SLIP) if pos == 1 else o[j] * (1 - SLIP)
                eq_at_entry = eq
        if pos != 0:
            worst = l[j] if pos == 1 else h[j]
            r_worst = (worst - entry) / entry if pos == 1 else (entry - worst) / entry
            eq_worst = eq_at_entry * (1 + r_worst - FEE)
            peak = max(peak, eq)
            mtm_dd = max(mtm_dd, 1 - eq_worst / peak)
            r_close = (c[j] - entry) / entry if pos == 1 else (entry - c[j]) / entry
            peak = max(peak, eq_at_entry * (1 + r_close - FEE))
            # funding events during candle j (Binance funding at 00/08/16 UTC)
            k0 = np.searchsorted(fund_t, ts[j], "left"); k1 = np.searchsorted(fund_t, ts[j] + (ts[1] - ts[0]), "left")
            for k in range(k0, k1):
                notional = eq_at_entry * (1 + r_close)
                funding += -pos * fund_r[k] * notional / eq_at_entry if eq_at_entry else 0.0
    return mtm_dd * 100, funding * 100, fee_exact_extra * 100


def main(out_json):
    tf = "60"
    m = load_binance(tf)
    ft, fr = load_funding()
    tmp = Path(tempfile.mkdtemp(prefix="audit_real_"))
    try:
        cfg = {"symbol": "REAL", "months": 18, "tfs": [tf], "fee": FEE, "slippage": SLIP, "depth": "Глубокий"}
        store = ResearchStore.create(tmp, cfg, {tf: m})
        res = ResearchRunner(cfg, store, workers=1, max_tasks_per_child=None).run()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    plan = core.build_walk_forward_plan_arrays(m["timestamp"], m["open"], m["close"])
    rs, hs = plan["research_start"], plan["holdout_start"]
    out = []
    base_top = [r for r in res.top if r.strategy_type == "BASE"][:20]
    for r in base_top:
        d = desired_for(r, m)
        mtm_dd, fund, fee_extra = path_stats(m, d, rs, hs - 1, ft, fr)
        # windows: flat-start (program) vs cut from continuous path
        flat = [core.backtest_fast_arrays(m["open"], m["close"], m["months"], d, a, b, FEE, SLIP, "BOTH", 1).total_return_pct
                for a, b, *_ in plan["windows"]]
        cont = core.backtest_fast_arrays(m["open"], m["close"], m["months"], d, rs, hs - 1, FEE, SLIP, "BOTH", 1)
        out.append({"key": core.result_key(r), "robust": r.robust_score, "research_ret": r.train.total_return_pct,
                    "trades": r.train.trades, "closed_dd": r.train.max_drawdown_pct, "mtm_dd": mtm_dd,
                    "funding_pct_of_equity": fund, "fee_exact_extra_pct": fee_extra,
                    "sum_window_returns": float(np.sum(flat)), "continuous_return": cont.total_return_pct,
                    "holdout_ret": r.test.total_return_pct})
    rows = [core.autoresult_to_dict(x) for x in res.top]
    Path(out_json).write_text(json.dumps({"top20_realism": out, "top": rows,
                                          "trace": res.trace}, default=list, indent=1))
    for x in out:
        print(f"{str(x['key']):38s} rob {x['robust']:.3f} ret {x['research_ret']:7.1f}% trades {x['trades']:4d} "
              f"DD closed {x['closed_dd']:5.1f}% MTM {x['mtm_dd']:5.1f}% funding {x['funding_pct_of_equity']:+6.2f}% feeΔ {x['fee_exact_extra_pct']:+.3f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "realism_60.json")

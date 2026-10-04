"""Hand-calculated PnL scenarios vs the program's backtest functions.

Expected values are written out as plain arithmetic (not via program code).
Program convention (v0.7.1 = v0.8):
  * signal = Supertrend direction on the CLOSED candle i;
  * the position change is executed at OPEN of candle i+1;
  * a position still open at the end of a segment is closed at CLOSE of the
    last candle of the segment;
  * net return per trade = gross(after slippage) - 2 * fee (fee on entry notional).
"""
from __future__ import annotations

import math

import numpy as np

from common import core

FEE = 0.001
results = []


def check(name, got, exp, tol=1e-12):
    ok = abs(got - exp) <= tol * max(1.0, abs(exp))
    results.append((name, got, exp, ok))
    return ok


def run(o, c, trend, start, end, mode, fee=FEE, slip=0.0, months=None):
    o = np.asarray(o, float); c = np.asarray(c, float)
    months = np.full(len(o), 202601, np.int32) if months is None else np.asarray(months, np.int32)
    return core.backtest_fast_arrays(o, c, months, np.asarray(trend, np.int8), start, end, fee, slip, mode, 1)


# ---------------------------------------------------------------- S1 LONG
o = [100, 101, 102, 103, 104, 105, 110, 108, 107, 106]
c = [101, 102, 103, 104, 105, 109, 108, 107, 106, 105]
t = [-1, -1, 1, 1, 1, -1, -1, -1, -1, -1]
# trend flips to +1 on candle 2 -> buy at open[3]=103; flips back on candle 5 -> sell at open[6]=110
m = run(o, c, t, 0, 9, "LONG")
exp = (110 / 103 - 1) - 2 * FEE
check("S1 long: trades", m.trades, 1)
check("S1 long: net return %", m.total_return_pct, exp * 100)

# ---------------------------------------------------------------- S2 SHORT
m = run(o, c, [-x for x in t], 0, 9, "SHORT")
# short from candle 2 signal: sell at open[3]=103, buy back at open[6]=110
check("S2 short: net return %", m.total_return_pct, ((103 - 110) / 103 - 2 * FEE) * 100)

# ---------------------------------------------------------------- S3 BOTH with reversals
t3 = [1, 1, -1, -1, 1, 1, 1, 1, 1, 1]
m = run(o, c, t3, 0, 9, "BOTH")
# long: open[1]=101 -> open[3]=103 ; short: open[3]=103 -> open[5]=105 ; long: open[5]=105 -> close[9]=105 (segment end)
r = [(103 / 101 - 1) - 2 * FEE, ((103 - 105) / 103) - 2 * FEE, (105 / 105 - 1) - 2 * FEE]
check("S3 reversal: trades", m.trades, 3)
check("S3 reversal: compounded return %", m.total_return_pct, ((1 + r[0]) * (1 + r[1]) * (1 + r[2]) - 1) * 100)
check("S3 reversal: long trades", m.long_trades, 2)
check("S3 reversal: short trades", m.short_trades, 1)
wins = [x for x in r if x > 0]; losses = [x for x in r if x <= 0]
check("S3 win rate %", m.win_rate, len(wins) / 3 * 100)
check("S3 profit factor", m.profit_factor, sum(wins) / abs(sum(losses)))
eq = np.cumprod([1.0] + [1 + x for x in r]); dd = max(1 - eq[i] / eq[: i + 1].max() for i in range(len(eq)))
check("S3 max DD % (closed trades)", m.max_drawdown_pct, dd * 100)

# ---------------------------------------------------------------- S4 slippage
m = run(o, c, t, 0, 9, "LONG", slip=0.0005)
check("S4 long with slippage %", m.total_return_pct, ((110 * 0.9995 - 103 * 1.0005) / (103 * 1.0005) - 2 * FEE) * 100)
m = run(o, c, [-x for x in t], 0, 9, "SHORT", slip=0.0005)
check("S4 short with slippage %", m.total_return_pct, ((103 * 0.9995 - 110 * 1.0005) / (103 * 0.9995) - 2 * FEE) * 100)

# ---------------------------------------------------------------- S5 several consecutive trades + months
o5 = [100, 100, 102, 104, 103, 101, 100, 99, 101, 104, 106, 108, 107, 106]
c5 = [100, 102, 104, 103, 101, 100, 99, 101, 104, 106, 108, 107, 106, 105]
t5 = [1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1]
mo = [202601] * 6 + [202602] * 8
m = run(o5, c5, t5, 0, 13, "BOTH", months=mo)
# long open[1]=100 -> open[4]=103 ; short open[4]=103 -> open[8]=101 ; long open[8]=101 -> open[12]=107 ; short open[12]=107 -> close[13]=105
r5 = [(103 / 100 - 1) - 2 * FEE, (103 - 101) / 103 - 2 * FEE, (107 / 101 - 1) - 2 * FEE, (107 - 105) / 107 - 2 * FEE]
check("S5 trades", m.trades, 4)
check("S5 return %", m.total_return_pct, (np.prod([1 + x for x in r5]) - 1) * 100)
check("S5 months (by exit candle)", m.months, 2)   # exits at candles 4 (Jan), 8, 12, 13 (Feb)

# ---------------------------------------------------------------- S6 position open across a 21-day window boundary
# continuous segment 0..13 vs two windows 0..6 and 7..13 (how walk-forward windows are evaluated)
cont = run(o5, c5, [1] * 14, 0, 13, "LONG")
w1 = run(o5, c5, [1] * 14, 0, 6, "LONG")
w2 = run(o5, c5, [1] * 14, 7, 13, "LONG")
check("S6 continuous: 1 trade open[1]->close[13]", cont.total_return_pct, ((105 / 100 - 1) - 2 * FEE) * 100)
check("S6 window 1: forced exit at close[6]=99", w1.total_return_pct, ((99 / 100 - 1) - 2 * FEE) * 100)
check("S6 window 2: re-entry at open[8]=101 (candle 7 move missed)", w2.total_return_pct, ((105 / 101 - 1) - 2 * FEE) * 100)
win_chain = (1 + w1.total_return_pct / 100) * (1 + w2.total_return_pct / 100) - 1
results.append(("S6 windows chained vs continuous (difference, %)", win_chain * 100, cont.total_return_pct, None))

# ---------------------------------------------------------------- S7 closed-trade drawdown hides intrabar pain
o7 = [100, 100, 90, 75, 70, 85, 100, 110]
h7 = [101, 101, 91, 76, 72, 90, 105, 112]
l7 = [99, 89, 74, 69, 68, 80, 95, 108]
c7 = [100, 90, 75, 70, 85, 100, 110, 111]
m = run(o7, c7, [1] * 8, 0, 7, "LONG")
mtm = 1 - min(l7[1:]) / 100.0     # entry at open[1]=100, worst low 68 while holding
results.append(("S7 program max DD % (closed trades)", m.max_drawdown_pct, 0.0, abs(m.max_drawdown_pct) < 1e-12))
results.append(("S7 true intrabar drawdown % while holding", mtm * 100, 32.0, abs(mtm * 100 - 32.0) < 1e-9))

# ---------------------------------------------------------------- S8 fee on exit notional
gross = 110 / 103 - 1
results.append(("S8 program fee cost (2*fee on entry notional), %", 2 * FEE * 100, None, None))
results.append(("S8 exact fee cost fee*(1 + exit/entry), %", FEE * (1 + 110 / 103) * 100, None, None))

# ---------------------------------------------------------------- S9 short squeeze clamp
o9 = [100, 100, 150, 260, 300]; c9 = [100, 150, 260, 300, 310]
m = run(o9, c9, [-1] * 5, 0, 4, "SHORT")
results.append(("S9 short loss > 100% clamped to -99.9999%", m.total_return_pct, -99.9999, abs(m.total_return_pct + 99.9999) < 1e-6))

# ---------------------------------------------------------------- S10 adaptive kernel follows the same rules
lt = np.array([1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1], np.int8)
st = np.array([-1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1], np.int8)
reg = np.array([1, 1, 1, 0, 0, -1, -1, -1, 1, 1, 1, 1, -1, -1], np.int8)
desired = np.where((reg == 1) & (lt == 1), 1, np.where((reg == -1) & (st == -1), -1, 0)).astype(np.int8)
a = core._metrics_from_adaptive_tuple(core._adaptive_segment_nb(np.array(o5, float), np.array(c5, float), lt, st, reg, 0, 13, FEE, 0.0), 1)
b = core.backtest_desired_fast_arrays(np.array(o5, float), np.array(c5, float), np.full(14, 202601, np.int32), desired, 0, 13, FEE, 0.0, 1)
# desired = [1,1,1,0,0,-1,-1,0,1,1,1,0,-1,-1]: long o[1]=100->o[4]=103 (flat on candle 3), short o[6]=100->o[8]=101 (flat on candle 7),
# long o[9]=104->o[12]=107, short o[13]=106->c[13]=105
r10 = [(103 / 100 - 1) - 2 * FEE, (100 - 101) / 100 - 2 * FEE, (107 / 104 - 1) - 2 * FEE, (106 - 105) / 106 - 2 * FEE]
check("S10 adaptive kernel trades", a.trades, 4)
check("S10 adaptive kernel return %", a.total_return_pct, (np.prod([1 + x for x in r10]) - 1) * 100)
check("S10 vector backtest == kernel", b.total_return_pct, a.total_return_pct)

if __name__ == "__main__":
    print(f"{'scenario':62s} {'program':>14s} {'by hand':>14s}  result")
    for name, got, exp, ok in results:
        e = "" if exp is None else f"{exp:14.6f}"
        flag = "" if ok is None else ("OK" if ok else "MISMATCH")
        print(f"{name:62s} {got:14.6f} {e:>14s}  {flag}")
    bad = [r for r in results if r[3] is False]
    print("\nALL HAND CHECKS PASS" if not bad else f"\n{len(bad)} MISMATCHES")

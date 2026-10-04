"""Supertrend specification: program vs TradingView Pine v5 `ta.supertrend`.

The TradingView reference below is a literal transcription of the published
Pine source of ta.supertrend / ta.atr / ta.rma (not an export from
TradingView itself):

    pine_supertrend(factor, atrPeriod) =>
        src = hl2
        atr = ta.atr(atrPeriod)                      // ta.rma(ta.tr(true), atrPeriod)
        upperBand = src + factor * atr
        lowerBand = src - factor * atr
        prevLowerBand = nz(lowerBand[1])
        prevUpperBand = nz(upperBand[1])
        lowerBand := lowerBand > prevLowerBand or close[1] < prevLowerBand ? lowerBand : prevLowerBand
        upperBand := upperBand < prevUpperBand or close[1] > prevUpperBand ? upperBand : prevUpperBand
        int _direction = na
        float superTrend = na
        prevSuperTrend = superTrend[1]
        if na(atr[1])
            _direction := 1
        else if prevSuperTrend == prevUpperBand
            _direction := close > upperBand ? -1 : 1
        else
            _direction := close < lowerBand ? 1 : -1
        superTrend := _direction == -1 ? lowerBand : upperBand
        [superTrend, _direction]

    ta.rma(src, n): first value = SMA of the first n values, then alpha = 1/n
    ta.tr(true):    first bar = high - low, then max(high-low, |high-close[1]|, |low-close[1]|)

TradingView direction: -1 = UP trend, +1 = DOWN trend (opposite of the program).
"""
from __future__ import annotations

import math

import numpy as np

from common import core, load_binance


def tv_atr(h, l, c, n):
    tr = np.empty(len(c)); tr[0] = h[0] - l[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    out = np.full(len(c), np.nan)
    s = math.nan
    for i in range(len(c)):
        if i < n - 1:
            continue
        if math.isnan(s):
            s = sum(tr[i - n + 1:i + 1]) / n
        else:
            s = (1.0 / n) * tr[i] + (1.0 - 1.0 / n) * s
        out[i] = s
    return out


def tv_supertrend(h, l, c, n, factor):
    atr = tv_atr(h, l, c, n)
    N = len(c)
    up = np.full(N, np.nan); lo = np.full(N, np.nan); dirn = np.full(N, np.nan); st = np.full(N, np.nan)
    for i in range(N):
        src = (h[i] + l[i]) / 2
        if math.isnan(atr[i]):
            continue
        ub = src + factor * atr[i]; lb = src - factor * atr[i]
        plb = 0.0 if i == 0 or math.isnan(lo[i - 1]) else lo[i - 1]
        pub = 0.0 if i == 0 or math.isnan(up[i - 1]) else up[i - 1]
        lb = lb if (lb > plb or c[i - 1] < plb) else plb
        ub = ub if (ub < pub or c[i - 1] > pub) else pub
        prev_st = st[i - 1] if i > 0 else math.nan
        if i == 0 or math.isnan(atr[i - 1]):
            d = 1
        elif prev_st == pub:
            d = -1 if c[i] > ub else 1
        else:
            d = 1 if c[i] < lb else -1
        up[i], lo[i], dirn[i] = ub, lb, d
        st[i] = lb if d == -1 else ub
    return atr, dirn, st


def main():
    m = load_binance("60")
    h, l, c = m["high"], m["low"], m["close"]
    print(f"{'ATR':>4} {'mult':>5} | ATR max rel diff | bars before agreement | disagreeing bars after first agreement")
    worst_after = 0
    for n, f in [(3, 1.5), (10, 3.0), (14, 2.5), (34, 5.95), (100, 2.0), (240, 8.0)]:
        atr_tv, dir_tv, _ = tv_supertrend(h, l, c, n, f)
        atr_p = core._fast_atr(h, l, c, n)
        trend_p = core._fast_trend(h, l, c, atr_p, f)
        ok = ~np.isnan(atr_tv)
        rel = np.nanmax(np.abs(atr_tv[ok] - atr_p[ok]) / atr_tv[ok])
        mapped = np.where(dir_tv == -1, 1, np.where(dir_tv == 1, -1, 0))
        agree = mapped[ok] == trend_p[ok]
        first = int(np.argmax(agree)) if agree.any() else -1
        after = int((~agree[first:]).sum()) if first >= 0 else -1
        worst_after = max(worst_after, after)
        print(f"{n:4d} {f:5.2f} | {rel:.2e}       | {first:5d}                 | {after}")
    print("\nResult:", "IDENTICAL after the initial direction converges" if worst_after == 0 else "DIFFERENCES FOUND")


if __name__ == "__main__":
    main()

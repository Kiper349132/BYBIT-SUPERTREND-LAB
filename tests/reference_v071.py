"""FROZEN copy of the v0.7.1 numerical code (commit 3328721), used only by the
tests to prove that v0.8 did not change any backtest / score arithmetic.
Do not edit."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

@dataclass
class BacktestMetrics:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_return_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_trade_pct: float = 0.0
    median_trade_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    long_return_pct: float = 0.0
    short_return_pct: float = 0.0
    long_trades: int = 0
    short_trades: int = 0
    profitable_months: int = 0
    months: int = 0
    profitable_month_ratio: float = 0.0
    score: float = -1e9


@dataclass
class AutoResult:
    interval: str
    mode: str
    atr_period: int
    multiplier: float
    train: BacktestMetrics              # walk-forward research zone, holdout excluded
    test: BacktestMetrics               # historical development holdout; excluded from ranking
    stage: str = ""
    strategy_type: str = "BASE"        # BASE or ADAPTIVE
    # Adaptive strategy uses separate Supertrend settings for bull/bear regimes.
    long_atr_period: int = 0
    long_multiplier: float = 0.0
    short_atr_period: int = 0
    short_multiplier: float = 0.0
    regime_hours: int = 0
    regime_threshold_pct: float = 0.0
    online_up_ratio: float = 0.0
    online_down_ratio: float = 0.0
    online_flat_ratio: float = 0.0
    regime_switches: int = 0
    # v0.6 separates candidate generation from later validation. Fine/ultra/cluster
    # stages may use selection_score only; validation_score is never used to
    # decide where to search next. robust_score is the post-search research rank
    # and still never uses the development holdout.
    selection_score: float = -1e9
    validation_score: float = -1e9
    robust_score: float = -1e9
    selection_windows: int = 0
    selection_positive_ratio: float = 0.0
    selection_median_return_pct: float = 0.0
    selection_worst_return_pct: float = 0.0
    validation_windows: int = 0
    validation_positive_ratio: float = 0.0
    validation_median_return_pct: float = 0.0
    validation_worst_return_pct: float = 0.0
    # Smarter past-only regime classifier (ADAPTIVE_V2).
    regime_model: str = ""
    ema_fast: int = 0
    ema_slow: int = 0
    adx_period: int = 0
    adx_threshold: float = 0.0
    regime_separation_atr: float = 0.0
    regime_confirm_bars: int = 0
    regime_min_hold_bars: int = 0
    wf_windows: int = 0
    wf_positive_windows: int = 0
    wf_positive_ratio: float = 0.0
    wf_median_return_pct: float = 0.0
    wf_worst_return_pct: float = 0.0
    wf_best_return_pct: float = 0.0
    up_windows: int = 0
    up_positive_ratio: float = 0.0
    up_avg_return_pct: float = 0.0
    down_windows: int = 0
    down_positive_ratio: float = 0.0
    down_avg_return_pct: float = 0.0
    flat_windows: int = 0
    flat_positive_ratio: float = 0.0
    flat_avg_return_pct: float = 0.0


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    atr = np.full(n, np.nan, dtype=np.float64)
    if period < 1 or n <= period:
        return atr
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    prev_close = close[:-1]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)),
    )
    first_idx = period - 1
    atr[first_idx] = np.mean(tr[:period])
    for i in range(first_idx + 1, n):
        atr[i] = ((period - 1) * atr[i - 1] + tr[i]) / period
    return atr


@njit(cache=False)
def _wilder_atr_nb(high, low, close, period):
    n = len(close)
    atr = np.empty(n, dtype=np.float64)
    for i in range(n):
        atr[i] = np.nan
    if period < 1 or n <= period:
        return atr
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        a = high[i] - low[i]
        b = abs(high[i] - close[i-1])
        c = abs(low[i] - close[i-1])
        tr[i] = max(a, b, c)
    first_idx = period - 1
    total = 0.0
    for i in range(period):
        total += tr[i]
    atr[first_idx] = total / period
    for i in range(first_idx + 1, n):
        atr[i] = ((period - 1) * atr[i - 1] + tr[i]) / period
    return atr


@njit(cache=False)
def _supertrend_trend_nb(high, low, close, atr, multiplier):
    n = len(close)
    trend = np.zeros(n, dtype=np.int8)
    start = -1
    for i in range(n):
        if not np.isnan(atr[i]):
            start = i
            break
    if start < 0:
        return trend
    final_upper = np.empty(n, dtype=np.float64)
    final_lower = np.empty(n, dtype=np.float64)
    for i in range(n):
        final_upper[i] = np.nan
        final_lower[i] = np.nan
    hl2 = (high[start] + low[start]) * 0.5
    upper = hl2 + multiplier * atr[start]
    lower = hl2 - multiplier * atr[start]
    final_upper[start] = upper
    final_lower[start] = lower
    trend[start] = 1 if close[start] >= hl2 else -1
    for i in range(start + 1, n):
        hl2 = (high[i] + low[i]) * 0.5
        upper = hl2 + multiplier * atr[i]
        lower = hl2 - multiplier * atr[i]
        prev_u = final_upper[i-1]
        prev_l = final_lower[i-1]
        if upper < prev_u or close[i-1] > prev_u:
            final_upper[i] = upper
        else:
            final_upper[i] = prev_u
        if lower > prev_l or close[i-1] < prev_l:
            final_lower[i] = lower
        else:
            final_lower[i] = prev_l
        if trend[i-1] == -1:
            trend[i] = 1 if close[i] > final_upper[i] else -1
        else:
            trend[i] = -1 if close[i] < final_lower[i] else 1
    return trend


def _fast_atr(high, low, close, period):
    if NUMBA_AVAILABLE:
        return _wilder_atr_nb(high, low, close, int(period))
    return wilder_atr(high, low, close, int(period))


def _fast_trend(high, low, close, atr, multiplier):
    if NUMBA_AVAILABLE:
        return _supertrend_trend_nb(high, low, close, atr, float(multiplier))
    return supertrend_from_atr(high, low, close, atr, float(multiplier))[0]


def build_walk_forward_plan(df: pd.DataFrame) -> dict:
    """Build sequential research windows plus a development holdout.

    v0.6 deliberately separates the research windows into two chronological
    groups:
      SELECTION  -> may be used to decide where the optimiser searches deeper;
      VALIDATION -> is evaluated after candidate generation and must not be used
                    to generate fine/ultra/cluster candidates.

    The final historical block is called DEVELOPMENT_HOLDOUT. It is still not
    used in any search score, but after a previous report has already been read
    by a human/AI it can no longer honestly be called globally untouched. A new
    truly unseen forward test requires future candles that did not exist during
    development.

    Raw Bybit timestamp milliseconds are authoritative; pandas datetime storage
    precision is intentionally irrelevant.
    """
    n = len(df)
    if n < 400:
        raise ValueError("Слишком мало свечей для walk-forward проверки")
    if "timestamp" not in df.columns:
        raise ValueError("В данных нет timestamp Bybit для walk-forward проверки")

    times = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(np.float64)
    if not np.all(np.isfinite(times)):
        raise ValueError("В timestamp есть пустые или некорректные значения")
    times = times.astype(np.int64, copy=False)
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamp должен строго возрастать; проверьте дубли и сортировку")

    start_ms = int(times[0]); end_ms = int(times[-1]); day_ms = 86_400_000
    total_days = max(1.0, (end_ms - start_ms) / day_ms)
    holdout_days = int(max(60, min(90, round(total_days * 0.20))))
    warmup_days = int(max(90, min(180, round(total_days * 0.30))))
    test_days = 21
    holdout_ms = end_ms - holdout_days * day_ms
    first_test_ms = start_ms + warmup_days * day_ms
    if first_test_ms >= holdout_ms - test_days * day_ms:
        warmup_days = max(45, int(total_days * 0.20))
        first_test_ms = start_ms + warmup_days * day_ms

    windows = []
    cursor = first_test_ms
    while cursor + test_days * day_ms <= holdout_ms:
        nxt = cursor + test_days * day_ms
        a = int(np.searchsorted(times, cursor, side="left"))
        b = int(np.searchsorted(times, nxt, side="left")) - 1
        if b - a >= 20:
            start_price = float(df["open"].iloc[a]); end_price = float(df["close"].iloc[b])
            bench = (end_price / start_price - 1.0) * 100.0 if start_price > 0 else 0.0
            regime = "UP" if bench >= 3.0 else ("DOWN" if bench <= -3.0 else "FLAT")
            windows.append((a, b, regime, bench))
        cursor = nxt

    holdout_idx = int(np.searchsorted(times, holdout_ms, side="left"))
    holdout_idx = max(1, min(n - 2, holdout_idx))
    research_start = windows[0][0] if windows else max(1, int(n * 0.35))

    # Chronological split: roughly 65% of research windows generate candidates;
    # the later 35% validate them. Keep at least 2 validation windows whenever
    # the history is long enough.
    wcount = len(windows)
    if wcount >= 6:
        sel_n = max(4, min(wcount - 2, int(math.floor(wcount * 0.65))))
    elif wcount >= 4:
        sel_n = max(2, wcount - 2)
    else:
        sel_n = wcount
    selection_windows = windows[:sel_n]
    validation_windows = windows[sel_n:]
    validation_start = int(validation_windows[0][0]) if validation_windows else holdout_idx
    selection_end = max(research_start + 2, validation_start - 1)

    return {
        "windows": windows,
        "selection_windows": selection_windows,
        "validation_windows": validation_windows,
        "research_start": research_start,
        "selection_end": selection_end,
        "validation_start": validation_start,
        "holdout_start": holdout_idx,
        "holdout_days": holdout_days,
        "warmup_days": warmup_days,
        "test_days": test_days,
        "total_days": float(total_days),
        "methodology_note": (
            "SELECTION windows drive candidate generation; later VALIDATION windows do not. "
            "Historical final block is DEVELOPMENT_HOLDOUT because previous reports may already have exposed it."
        ),
    }

def supertrend_from_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    trend = np.zeros(n, dtype=np.int8)
    st = np.full(n, np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(atr))
    if len(valid) == 0:
        return trend, st

    start = int(valid[0])
    hl2 = (high + low) * 0.5
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    final_upper = np.full(n, np.nan, dtype=np.float64)
    final_lower = np.full(n, np.nan, dtype=np.float64)
    final_upper[start] = upper_basic[start]
    final_lower[start] = lower_basic[start]
    trend[start] = 1 if close[start] >= hl2[start] else -1
    st[start] = final_lower[start] if trend[start] == 1 else final_upper[start]

    for i in range(start + 1, n):
        if upper_basic[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_basic[i]
        else:
            final_upper[i] = final_upper[i - 1]
        if lower_basic[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_basic[i]
        else:
            final_lower[i] = final_lower[i - 1]

        if trend[i - 1] == -1:
            trend[i] = 1 if close[i] > final_upper[i] else -1
        else:
            trend[i] = -1 if close[i] < final_lower[i] else 1
        st[i] = final_lower[i] if trend[i] == 1 else final_upper[i]
    return trend, st


def _effective_entry_array(prices: np.ndarray, sides: np.ndarray, slip: float) -> np.ndarray:
    return np.where(sides == 1, prices * (1.0 + slip), prices * (1.0 - slip))


def _effective_exit_array(prices: np.ndarray, sides: np.ndarray, slip: float) -> np.ndarray:
    return np.where(sides == 1, prices * (1.0 - slip), prices * (1.0 + slip))


def metrics_from_returns(
    returns: np.ndarray,
    sides: np.ndarray,
    exit_months: np.ndarray,
    min_trades: int,
) -> BacktestMetrics:
    m = BacktestMetrics()
    if returns.size == 0:
        return m
    arr = returns.astype(np.float64, copy=False)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    m.trades = int(arr.size)
    m.wins = int(wins.size)
    m.losses = int(losses.size)
    m.win_rate = float(wins.size / arr.size * 100.0)
    equity = np.cumprod(1.0 + arr)
    m.total_return_pct = float((equity[-1] - 1.0) * 100.0)
    gp = float(wins.sum())
    gl = float(abs(losses.sum()))
    m.profit_factor = gp / gl if gl > 1e-12 else (99.0 if gp > 0 else 0.0)
    m.avg_trade_pct = float(arr.mean() * 100.0)
    m.median_trade_pct = float(np.median(arr) * 100.0)
    m.best_trade_pct = float(arr.max() * 100.0)
    m.worst_trade_pct = float(arr.min() * 100.0)

    long_arr = arr[sides == 1]
    short_arr = arr[sides == -1]
    m.long_trades = int(long_arr.size)
    m.short_trades = int(short_arr.size)
    if long_arr.size:
        m.long_return_pct = float((np.prod(1.0 + long_arr) - 1.0) * 100.0)
    if short_arr.size:
        m.short_return_pct = float((np.prod(1.0 + short_arr) - 1.0) * 100.0)

    eq = np.r_[1.0, equity]
    peaks = np.maximum.accumulate(eq)
    dd = (eq / peaks - 1.0) * 100.0
    m.max_drawdown_pct = float(abs(dd.min())) if dd.size else 0.0

    monthly = {}
    for r, month in zip(arr.tolist(), exit_months.tolist()):
        monthly[month] = (1.0 + monthly.get(month, 0.0)) * (1.0 + r) - 1.0
    m.months = len(monthly)
    m.profitable_months = sum(1 for x in monthly.values() if x > 0)
    m.profitable_month_ratio = m.profitable_months / m.months if m.months else 0.0

    if m.trades >= min_trades and m.total_return_pct > -99.9:
        dd_guard = max(m.max_drawdown_pct, 1.0)
        pf = min(max(m.profit_factor, 0.0), 3.0)
        sample_factor = math.sqrt(min(m.trades, 250) / 250.0)
        stability = 0.45 + 0.55 * m.profitable_month_ratio
        m.score = (m.total_return_pct / dd_guard) * pf * sample_factor * stability
    return m


def backtest_fast_arrays(
    opens: np.ndarray,
    closes: np.ndarray,
    month_codes: np.ndarray,
    trend: np.ndarray,
    start_idx: int,
    end_idx: int,
    fee_rate: float,
    slippage_rate: float,
    mode: str,
    min_trades: int,
) -> BacktestMetrics:
    start_idx = max(0, int(start_idx))
    end_idx = min(len(opens) - 1, int(end_idx))
    if end_idx - start_idx < 3:
        return BacktestMetrics()

    desired = trend[start_idx:end_idx].astype(np.int8, copy=True)
    if mode == "LONG":
        desired[desired != 1] = 0
    elif mode == "SHORT":
        desired[desired != -1] = 0
    if desired.size == 0:
        return BacktestMetrics()

    starts = np.flatnonzero(np.r_[True, desired[1:] != desired[:-1]])
    ends = np.r_[starts[1:] - 1, desired.size - 1]
    run_sides = desired[starts]
    mask = run_sides != 0
    starts = starts[mask]
    ends = ends[mask]
    sides = run_sides[mask]
    if starts.size == 0:
        return BacktestMetrics()

    entry_idx = start_idx + 1 + starts
    exit_idx = np.where(ends < desired.size - 1, start_idx + 2 + ends, end_idx)
    entry_raw = opens[entry_idx]
    exit_raw = np.where(ends < desired.size - 1, opens[exit_idx], closes[end_idx])
    entry_eff = _effective_entry_array(entry_raw, sides, slippage_rate)
    exit_eff = _effective_exit_array(exit_raw, sides, slippage_rate)
    gross = np.where(sides == 1, (exit_eff - entry_eff) / entry_eff, (entry_eff - exit_eff) / entry_eff)
    net = gross - 2.0 * fee_rate
    # The model uses the whole notional capital for a signal. A short move above
    # +100% would otherwise create negative equity, which is impossible in a real
    # margin account because the position would be liquidated/bankrupt first.
    net = np.maximum(net, -0.999999)
    return metrics_from_returns(net, sides, month_codes[exit_idx], min_trades)




def online_regime_from_past(
    close: np.ndarray,
    interval: str,
    lookback_hours: int,
    threshold_pct: float,
) -> np.ndarray:
    """Classify each bar using only prices that were already known at that bar.

    +1 = rising regime, -1 = falling regime, 0 = flat/uncertain.
    The signal is based on trailing return from *lookback_hours* ago to the
    current closed candle. No future candle is used.
    """
    tf_min = max(1, int(interval))
    bars = max(2, int(round(float(lookback_hours) * 60.0 / tf_min)))
    out = np.zeros(len(close), dtype=np.int8)
    if len(close) <= bars:
        return out
    prev = close[:-bars]
    cur = close[bars:]
    valid = prev > 0
    ret = np.zeros_like(cur, dtype=np.float64)
    ret[valid] = (cur[valid] / prev[valid] - 1.0) * 100.0
    thr = abs(float(threshold_pct))
    vals = np.zeros(len(ret), dtype=np.int8)
    vals[ret >= thr] = 1
    vals[ret <= -thr] = -1
    out[bars:] = vals
    return out



def _ema_array(values: np.ndarray, period: int) -> np.ndarray:
    """Past-only EMA used by the v0.6 regime classifier."""
    period = max(2, int(period))
    out = np.empty(len(values), dtype=np.float64)
    if len(values) == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    out[0] = float(values[0])
    for i in range(1, len(values)):
        out[i] = alpha * float(values[i]) + (1.0 - alpha) * out[i-1]
    return out


def _adx_array(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder ADX, calculated strictly from current/past candles."""
    period = max(2, int(period))
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    if n < period + 3:
        return out
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up = high[i] - high[i-1]
        dn = low[i-1] - low[i]
        plus_dm[i] = up if up > dn and up > 0 else 0.0
        minus_dm[i] = dn if dn > up and dn > 0 else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    # Wilder smoothing.
    atr = np.zeros(n, dtype=np.float64)
    psm = np.zeros(n, dtype=np.float64)
    msm = np.zeros(n, dtype=np.float64)
    seed = period
    atr[seed] = np.sum(tr[1:seed+1])
    psm[seed] = np.sum(plus_dm[1:seed+1])
    msm[seed] = np.sum(minus_dm[1:seed+1])
    dx = np.zeros(n, dtype=np.float64)
    for i in range(seed, n):
        if i > seed:
            atr[i] = atr[i-1] - atr[i-1] / period + tr[i]
            psm[i] = psm[i-1] - psm[i-1] / period + plus_dm[i]
            msm[i] = msm[i-1] - msm[i-1] / period + minus_dm[i]
        if atr[i] > 1e-12:
            pdi = 100.0 * psm[i] / atr[i]
            mdi = 100.0 * msm[i] / atr[i]
            den = pdi + mdi
            dx[i] = 100.0 * abs(pdi - mdi) / den if den > 1e-12 else 0.0
    adx_start = min(n - 1, seed * 2 - 1)
    if adx_start > seed:
        out[adx_start] = float(np.mean(dx[seed:adx_start+1]))
        for i in range(adx_start + 1, n):
            out[i] = ((out[i-1] * (period - 1)) + dx[i]) / period
    return out


@njit(cache=False)
def _regime_v2_nb(close, ema_fast, ema_slow, adx, atr, adx_threshold, separation_atr, confirm_bars, min_hold_bars):
    """Past-only trend state with confirmation + hysteresis.

    A regime may change only after the raw state has persisted for confirm_bars
    and the current state has been held for at least min_hold_bars. This avoids
    the rapid flip-flopping seen in the v0.5 trailing-return classifier.
    """
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    current = 0
    held = 0
    pending = 0
    pending_count = 0
    for i in range(n):
        raw = 0
        if atr[i] > 0.0 and adx[i] >= adx_threshold:
            gap = ema_fast[i] - ema_slow[i]
            need = separation_atr * atr[i]
            slope_ref = i - max(2, confirm_bars)
            slow_slope = 0.0 if slope_ref < 0 else ema_slow[i] - ema_slow[slope_ref]
            if gap > need and slow_slope >= 0.0:
                raw = 1
            elif gap < -need and slow_slope <= 0.0:
                raw = -1
        if raw == current:
            pending = 0
            pending_count = 0
            held += 1
            out[i] = current
            continue
        if raw != pending:
            pending = raw
            pending_count = 1
        else:
            pending_count += 1
        if pending_count >= confirm_bars and (current == 0 or held >= min_hold_bars):
            current = pending
            held = 0
            pending = 0
            pending_count = 0
        else:
            held += 1
        out[i] = current
    return out


def online_regime_v2(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ema_fast_period: int,
    ema_slow_period: int,
    adx_period: int,
    adx_threshold: float,
    separation_atr: float,
    confirm_bars: int,
    min_hold_bars: int,
) -> np.ndarray:
    """Smarter market regime: EMA direction + ADX + ATR separation + hysteresis.

    Every value at candle i is derived only from candle i and older candles.
    """
    ef = _ema_array(close, ema_fast_period)
    es = _ema_array(close, ema_slow_period)
    adx = _adx_array(high, low, close, adx_period)
    atr = _fast_atr(high, low, close, max(2, int(adx_period)))
    atr = np.nan_to_num(atr, nan=0.0, posinf=0.0, neginf=0.0)
    return _regime_v2_nb(
        close.astype(np.float64, copy=False), ef, es, adx, atr,
        float(adx_threshold), float(separation_atr), max(1, int(confirm_bars)), max(1, int(min_hold_bars)),
    )


def _window_return_stats(values: list[float]) -> tuple[int, float, float, float, float]:
    if not values:
        return 0, 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return (
        len(values),
        float(np.mean(arr > 0.0)),
        float(np.median(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
    )


def _phase_score(metrics: BacktestMetrics, window_returns: list[float], min_trades: int) -> float:
    """Score one chronological research phase without touching later data."""
    if metrics.trades < min_trades or len(window_returns) < 2:
        return -1e9
    n, pos, median_ret, worst_ret, _best = _window_return_stats(window_returns)
    dd = max(metrics.max_drawdown_pct, 2.0)
    pf = min(max(metrics.profit_factor, 0.0), 3.0)
    sample = math.sqrt(min(metrics.trades, 300) / 300.0)
    stability = 0.20 + 0.80 * pos
    worst_penalty = 1.0 / (1.0 + max(0.0, -worst_ret) / 10.0)
    median_factor = max(0.25, min(1.8, 1.0 + median_ret / 10.0))
    return (metrics.total_return_pct / dd) * pf * sample * stability * worst_penalty * median_factor


def _combine_research_scores(selection_score: float, validation_score: float,
                             sel_pos: float, val_pos: float,
                             sel_med: float, val_med: float) -> float:
    """Final research-only rank. Later validation gets more weight than selection."""
    if selection_score <= -1e8 or validation_score <= -1e8:
        return -1e9
    transfer = 1.0 / (1.0 + abs(sel_med - val_med) / 8.0)
    window_balance = 0.30 + 0.70 * min(sel_pos, val_pos)
    # A robust candidate must survive BOTH chronological phases. Arithmetic
    # averaging let one spectacular phase hide a nearly useless other phase.
    if selection_score <= 0.0 or validation_score <= 0.0:
        return min(selection_score, validation_score) - 0.25 * abs(selection_score - validation_score)
    geometric = math.sqrt(selection_score * validation_score)
    score_ratio = min(selection_score, validation_score) / max(selection_score, validation_score)
    phase_balance = 0.35 + 0.65 * math.sqrt(max(0.0, score_ratio))
    return geometric * transfer * window_balance * phase_balance


def backtest_desired_fast_arrays(
    opens: np.ndarray,
    closes: np.ndarray,
    month_codes: np.ndarray,
    desired_full: np.ndarray,
    start_idx: int,
    end_idx: int,
    fee_rate: float,
    slippage_rate: float,
    min_trades: int,
) -> BacktestMetrics:
    """Backtest a pre-built desired-position vector {-1, 0, +1}.

    A desired signal observed on candle i enters at open of candle i+1. This
    keeps the same no-lookahead convention as the base Supertrend backtest.
    """
    start_idx = max(0, int(start_idx))
    end_idx = min(len(opens) - 1, int(end_idx))
    if end_idx - start_idx < 3:
        return BacktestMetrics()
    desired = desired_full[start_idx:end_idx].astype(np.int8, copy=False)
    if desired.size == 0:
        return BacktestMetrics()

    starts = np.flatnonzero(np.r_[True, desired[1:] != desired[:-1]])
    ends = np.r_[starts[1:] - 1, desired.size - 1]
    run_sides = desired[starts]
    mask = run_sides != 0
    starts = starts[mask]
    ends = ends[mask]
    sides = run_sides[mask]
    if starts.size == 0:
        return BacktestMetrics()

    entry_idx = start_idx + 1 + starts
    exit_idx = np.where(ends < desired.size - 1, start_idx + 2 + ends, end_idx)
    valid = (entry_idx < len(opens)) & (exit_idx < len(opens))
    entry_idx, exit_idx, starts, ends, sides = entry_idx[valid], exit_idx[valid], starts[valid], ends[valid], sides[valid]
    if entry_idx.size == 0:
        return BacktestMetrics()

    entry_raw = opens[entry_idx]
    exit_raw = np.where(ends < desired.size - 1, opens[exit_idx], closes[end_idx])
    entry_eff = _effective_entry_array(entry_raw, sides, slippage_rate)
    exit_eff = _effective_exit_array(exit_raw, sides, slippage_rate)
    gross = np.where(sides == 1, (exit_eff - entry_eff) / entry_eff, (entry_eff - exit_eff) / entry_eff)
    net = np.maximum(gross - 2.0 * fee_rate, -0.999999)
    return metrics_from_returns(net, sides, month_codes[exit_idx], min_trades)


def _adaptive_component_rows(rows: list[AutoResult], mode: str, limit: int) -> list[AutoResult]:
    """Pick diverse research-only components; final holdout is never used."""
    pool = [r for r in rows if r.strategy_type == "BASE" and r.mode == mode and r.selection_score > -1e8]
    # Mix overall robustness with the descriptive research-window regime stats.
    # These labels are used only for historical candidate selection, never for a
    # live trading decision. The live switch uses online_regime_from_past().
    if mode == "LONG":
        key = lambda r: (r.selection_score + 0.6 * r.up_positive_ratio + r.up_avg_return_pct / 35.0)
    else:
        key = lambda r: (r.selection_score + 0.6 * r.down_positive_ratio + r.down_avg_return_pct / 35.0)
    ranked = sorted(pool, key=key, reverse=True)
    chosen: list[AutoResult] = []
    for row in ranked:
        if any(abs(row.atr_period - c.atr_period) <= 2 and abs(row.multiplier - c.multiplier) <= 0.18 for c in chosen):
            continue
        chosen.append(row)
        if len(chosen) >= limit:
            break
    return chosen or ranked[:limit]



@njit(cache=False)
def _adaptive_segment_nb(opens, closes, long_trend, short_trend, regime, start_idx, end_idx, fee, slip):
    """Fast segment backtest for the adaptive signal; no future data is used."""
    n = len(opens)
    if start_idx < 0:
        start_idx = 0
    if end_idx >= n:
        end_idx = n - 1
    if end_idx - start_idx < 3:
        return (0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0)

    current = 0
    entry = 0.0
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    gp = 0.0
    gl = 0.0
    total_net = 0.0
    best = -1e100
    worst = 1e100
    trades = 0
    wins = 0
    long_trades = 0
    short_trades = 0
    long_eq = 1.0
    short_eq = 1.0

    # desired signal at candle i is executed at open i+1
    for i in range(start_idx, end_idx):
        d = 0
        if regime[i] == 1 and long_trend[i] == 1:
            d = 1
        elif regime[i] == -1 and short_trend[i] == -1:
            d = -1
        if d != current:
            px = opens[i + 1]
            if current != 0:
                if current == 1:
                    exit_eff = px * (1.0 - slip)
                    gross = (exit_eff - entry) / entry
                else:
                    exit_eff = px * (1.0 + slip)
                    gross = (entry - exit_eff) / entry
                net = gross - 2.0 * fee
                if net < -0.999999:
                    net = -0.999999
                trades += 1
                total_net += net
                if net > 0.0:
                    wins += 1
                    gp += net
                else:
                    gl += -net
                if net > best:
                    best = net
                if net < worst:
                    worst = net
                equity *= 1.0 + net
                if equity > peak:
                    peak = equity
                dd = (1.0 - equity / peak) * 100.0 if peak > 0.0 else 100.0
                if dd > max_dd:
                    max_dd = dd
                if current == 1:
                    long_trades += 1
                    long_eq *= 1.0 + net
                else:
                    short_trades += 1
                    short_eq *= 1.0 + net
            current = d
            if current == 1:
                entry = px * (1.0 + slip)
            elif current == -1:
                entry = px * (1.0 - slip)

    if current != 0:
        px = closes[end_idx]
        if current == 1:
            exit_eff = px * (1.0 - slip)
            gross = (exit_eff - entry) / entry
        else:
            exit_eff = px * (1.0 + slip)
            gross = (entry - exit_eff) / entry
        net = gross - 2.0 * fee
        if net < -0.999999:
            net = -0.999999
        trades += 1
        total_net += net
        if net > 0.0:
            wins += 1
            gp += net
        else:
            gl += -net
        if net > best:
            best = net
        if net < worst:
            worst = net
        equity *= 1.0 + net
        if equity > peak:
            peak = equity
        dd = (1.0 - equity / peak) * 100.0 if peak > 0.0 else 100.0
        if dd > max_dd:
            max_dd = dd
        if current == 1:
            long_trades += 1
            long_eq *= 1.0 + net
        else:
            short_trades += 1
            short_eq *= 1.0 + net

    if trades == 0:
        return (0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0)
    pf = gp / gl if gl > 1e-12 else (99.0 if gp > 0.0 else 0.0)
    avg = total_net / trades * 100.0
    return (
        trades, wins, (equity - 1.0) * 100.0, pf, max_dd, avg,
        best * 100.0, worst * 100.0, long_trades, short_trades,
        (long_eq - 1.0) * 100.0, (short_eq - 1.0) * 100.0,
    )


def _metrics_from_adaptive_tuple(x, min_trades: int) -> BacktestMetrics:
    m = BacktestMetrics()
    m.trades = int(x[0]); m.wins = int(x[1]); m.losses = m.trades - m.wins
    if m.trades <= 0:
        return m
    m.win_rate = m.wins / m.trades * 100.0
    m.total_return_pct = float(x[2]); m.profit_factor = float(x[3]); m.max_drawdown_pct = float(x[4])
    m.avg_trade_pct = float(x[5]); m.best_trade_pct = float(x[6]); m.worst_trade_pct = float(x[7])
    m.long_trades = int(x[8]); m.short_trades = int(x[9]); m.long_return_pct = float(x[10]); m.short_return_pct = float(x[11])
    # median/monthly statistics are intentionally left at zero during the massive
    # search. They are recomputed trade-by-trade for the selected top candidates.
    if m.trades >= min_trades and m.total_return_pct > -99.9:
        dd_guard = max(m.max_drawdown_pct, 1.0)
        pf = min(max(m.profit_factor, 0.0), 3.0)
        sample_factor = math.sqrt(min(m.trades, 250) / 250.0)
        m.score = (m.total_return_pct / dd_guard) * pf * sample_factor
    return m

def evaluate_adaptive_candidate(
    interval: str,
    long_row: AutoResult,
    short_row: AutoResult,
    regime_hours: int,
    regime_threshold_pct: float,
    long_trend: np.ndarray,
    short_trend: np.ndarray,
    regime: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    months: np.ndarray,
    plan: dict,
    fee: float,
    slippage: float,
    min_trades: int,
) -> AutoResult:
    research_start = int(plan["research_start"])
    holdout_start = int(plan["holdout_start"])
    research = _metrics_from_adaptive_tuple(
        _adaptive_segment_nb(opens, closes, long_trend, short_trend, regime, research_start, holdout_start - 1, fee, slippage),
        min_trades,
    )
    holdout = _metrics_from_adaptive_tuple(
        _adaptive_segment_nb(opens, closes, long_trend, short_trend, regime, holdout_start, len(closes) - 1, fee, slippage),
        1,
    )

    wr: list[float] = []
    regimes = {"UP": [], "DOWN": [], "FLAT": []}
    for a, b, diagnostic_regime, _bench in plan["windows"]:
        wm = _metrics_from_adaptive_tuple(
            _adaptive_segment_nb(opens, closes, long_trend, short_trend, regime, int(a), int(b), fee, slippage),
            1,
        )
        ret = float(wm.total_return_pct)
        wr.append(ret)
        regimes[diagnostic_regime].append(ret)

    wf_n = len(wr)
    wf_pos = sum(1 for x in wr if x > 0)
    wf_ratio = wf_pos / wf_n if wf_n else 0.0
    median_ret = float(np.median(np.asarray(wr, dtype=np.float64))) if wr else 0.0
    worst_ret = float(min(wr)) if wr else 0.0
    best_ret = float(max(wr)) if wr else 0.0

    def reg_stats(name: str):
        arr = regimes[name]
        if not arr:
            return 0, 0.0, 0.0
        return len(arr), sum(1 for x in arr if x > 0) / len(arr), float(np.mean(np.asarray(arr, dtype=np.float64)))

    up_n, up_pos, up_avg = reg_stats("UP")
    dn_n, dn_pos, dn_avg = reg_stats("DOWN")
    fl_n, fl_pos, fl_avg = reg_stats("FLAT")

    robust = -1e9
    if research.trades >= min_trades and wf_n >= 4:
        dd = max(research.max_drawdown_pct, 2.0)
        pf = min(max(research.profit_factor, 0.0), 3.0)
        sample = math.sqrt(min(research.trades, 350) / 350.0)
        stability = 0.15 + 0.85 * wf_ratio
        worst_penalty = 1.0 / (1.0 + max(0.0, -worst_ret) / 10.0)
        median_factor = max(0.30, min(1.80, 1.0 + median_ret / 10.0))
        # Reward balance across descriptive market regimes in the research zone.
        available = [(up_n, up_pos), (dn_n, dn_pos), (fl_n, fl_pos)]
        pos_rates = [x[1] for x in available if x[0] > 0]
        balance = (0.55 + 0.45 * min(pos_rates)) if pos_rates else 0.55
        robust = (research.total_return_pct / dd) * pf * sample * stability * worst_penalty * median_factor * balance

    nonzero = regime[regime != 0]
    total = max(1, len(regime))
    switches = int(np.sum(regime[1:] != regime[:-1])) if len(regime) > 1 else 0
    return AutoResult(
        interval=interval, mode="ADAPTIVE", atr_period=0, multiplier=0.0,
        train=research, test=holdout, stage="adaptive", strategy_type="ADAPTIVE",
        long_atr_period=long_row.atr_period, long_multiplier=long_row.multiplier,
        short_atr_period=short_row.atr_period, short_multiplier=short_row.multiplier,
        regime_hours=int(regime_hours), regime_threshold_pct=float(regime_threshold_pct),
        online_up_ratio=float(np.sum(regime == 1) / total),
        online_down_ratio=float(np.sum(regime == -1) / total),
        online_flat_ratio=float(np.sum(regime == 0) / total),
        regime_switches=switches,
        robust_score=float(robust), wf_windows=wf_n, wf_positive_windows=wf_pos,
        wf_positive_ratio=float(wf_ratio), wf_median_return_pct=median_ret,
        wf_worst_return_pct=worst_ret, wf_best_return_pct=best_ret,
        up_windows=up_n, up_positive_ratio=float(up_pos), up_avg_return_pct=up_avg,
        down_windows=dn_n, down_positive_ratio=float(dn_pos), down_avg_return_pct=dn_avg,
        flat_windows=fl_n, flat_positive_ratio=float(fl_pos), flat_avg_return_pct=fl_avg,
    )

def evaluate_adaptive_v2_candidate(
    interval: str,
    long_row: AutoResult,
    short_row: AutoResult,
    regime_params: dict,
    long_trend: np.ndarray,
    short_trend: np.ndarray,
    regime: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    months: np.ndarray,
    plan: dict,
    fee: float,
    slippage: float,
    min_trades: int,
) -> AutoResult:
    research_start = int(plan["research_start"])
    selection_end = int(plan.get("selection_end", plan["holdout_start"] - 1))
    validation_start = int(plan.get("validation_start", plan["holdout_start"]))
    holdout_start = int(plan["holdout_start"])
    full_research = _metrics_from_adaptive_tuple(
        _adaptive_segment_nb(opens, closes, long_trend, short_trend, regime, research_start, holdout_start - 1, fee, slippage),
        min_trades,
    )
    selection_metrics = _metrics_from_adaptive_tuple(
        _adaptive_segment_nb(opens, closes, long_trend, short_trend, regime, research_start, selection_end, fee, slippage),
        max(6, min_trades // 2),
    )
    validation_metrics = _metrics_from_adaptive_tuple(
        _adaptive_segment_nb(opens, closes, long_trend, short_trend, regime, validation_start, holdout_start - 1, fee, slippage),
        1,
    ) if validation_start < holdout_start - 2 else BacktestMetrics()
    development_holdout = _metrics_from_adaptive_tuple(
        _adaptive_segment_nb(opens, closes, long_trend, short_trend, regime, holdout_start, len(closes) - 1, fee, slippage),
        1,
    )

    wr, sel_wr, val_wr = [], [], []
    regimes = {"UP": [], "DOWN": [], "FLAT": []}
    sel_keys = {(int(a), int(b)) for a,b,*_ in plan.get("selection_windows", plan["windows"])}
    val_keys = {(int(a), int(b)) for a,b,*_ in plan.get("validation_windows", [])}
    for a, b, diagnostic_regime, _bench in plan["windows"]:
        wm = _metrics_from_adaptive_tuple(
            _adaptive_segment_nb(opens, closes, long_trend, short_trend, regime, int(a), int(b), fee, slippage), 1,
        )
        ret = float(wm.total_return_pct)
        wr.append(ret); regimes[diagnostic_regime].append(ret)
        key = (int(a), int(b))
        if key in sel_keys:
            sel_wr.append(ret)
        elif key in val_keys:
            val_wr.append(ret)

    wf_n, wf_ratio, median_ret, worst_ret, best_ret = _window_return_stats(wr)
    wf_pos = int(round(wf_ratio * wf_n)) if wf_n else 0
    sel_n, sel_pos, sel_med, sel_worst, _ = _window_return_stats(sel_wr)
    val_n, val_pos, val_med, val_worst, _ = _window_return_stats(val_wr)
    selection_score = _phase_score(selection_metrics, sel_wr, max(6, min_trades // 2))
    validation_score = _phase_score(validation_metrics, val_wr, 1) if val_n >= 2 else selection_score
    robust = _combine_research_scores(selection_score, validation_score, sel_pos, val_pos if val_n else sel_pos, sel_med, val_med if val_n else sel_med)

    def reg_stats(name: str):
        arr = regimes[name]
        if not arr:
            return 0, 0.0, 0.0
        return len(arr), sum(1 for x in arr if x > 0) / len(arr), float(np.mean(np.asarray(arr, dtype=np.float64)))
    up_n, up_pos, up_avg = reg_stats("UP")
    dn_n, dn_pos, dn_avg = reg_stats("DOWN")
    fl_n, fl_pos, fl_avg = reg_stats("FLAT")
    total = max(1, len(regime))
    switches = int(np.sum(regime[1:] != regime[:-1])) if len(regime) > 1 else 0
    return AutoResult(
        interval=interval, mode="ADAPTIVE", atr_period=0, multiplier=0.0,
        train=full_research, test=development_holdout, stage="adaptive_v2", strategy_type="ADAPTIVE_V2",
        long_atr_period=long_row.atr_period, long_multiplier=long_row.multiplier,
        short_atr_period=short_row.atr_period, short_multiplier=short_row.multiplier,
        regime_model="EMA_ADX_HYSTERESIS",
        ema_fast=int(regime_params["ema_fast"]), ema_slow=int(regime_params["ema_slow"]),
        adx_period=int(regime_params["adx_period"]), adx_threshold=float(regime_params["adx_threshold"]),
        regime_separation_atr=float(regime_params["separation_atr"]),
        regime_confirm_bars=int(regime_params["confirm_bars"]), regime_min_hold_bars=int(regime_params["min_hold_bars"]),
        online_up_ratio=float(np.sum(regime == 1) / total), online_down_ratio=float(np.sum(regime == -1) / total),
        online_flat_ratio=float(np.sum(regime == 0) / total), regime_switches=switches,
        selection_score=float(selection_score), validation_score=float(validation_score), robust_score=float(robust),
        selection_windows=sel_n, selection_positive_ratio=float(sel_pos), selection_median_return_pct=sel_med,
        selection_worst_return_pct=sel_worst, validation_windows=val_n, validation_positive_ratio=float(val_pos),
        validation_median_return_pct=val_med, validation_worst_return_pct=val_worst,
        wf_windows=wf_n, wf_positive_windows=wf_pos, wf_positive_ratio=float(wf_ratio),
        wf_median_return_pct=median_ret, wf_worst_return_pct=worst_ret, wf_best_return_pct=best_ret,
        up_windows=up_n, up_positive_ratio=float(up_pos), up_avg_return_pct=up_avg,
        down_windows=dn_n, down_positive_ratio=float(dn_pos), down_avg_return_pct=dn_avg,
        flat_windows=fl_n, flat_positive_ratio=float(fl_pos), flat_avg_return_pct=fl_avg,
    )

# ----- multiprocessing worker state -----
_WORKER = {}


def _worker_init(payload: dict):
    global _WORKER
    _WORKER = payload


def _worker_eval_period(task):
    period, specs, plan, fee, slippage, min_trades, interval, stage = task
    h = _WORKER["high"]; l = _WORKER["low"]; c = _WORKER["close"]; o = _WORKER["open"]; months = _WORKER["months"]
    windows = plan["windows"]
    selection_windows = plan.get("selection_windows", windows)
    validation_windows = plan.get("validation_windows", [])
    research_start = int(plan["research_start"])
    selection_end = int(plan.get("selection_end", plan["holdout_start"] - 1))
    validation_start = int(plan.get("validation_start", plan["holdout_start"]))
    holdout_start = int(plan["holdout_start"])
    atr = _fast_atr(h, l, c, int(period))
    out = []
    for mult, modes in specs:
        trend = _fast_trend(h, l, c, atr, float(mult))
        for mode in modes:
            full_research = backtest_fast_arrays(
                o, c, months, trend, max(research_start, int(period)), holdout_start - 1,
                fee, slippage, mode, min_trades,
            )
            selection_metrics = backtest_fast_arrays(
                o, c, months, trend, max(research_start, int(period)), selection_end,
                fee, slippage, mode, max(6, min_trades // 2),
            )
            validation_metrics = backtest_fast_arrays(
                o, c, months, trend, max(validation_start, int(period)), holdout_start - 1,
                fee, slippage, mode, 1,
            ) if validation_start < holdout_start - 2 else BacktestMetrics()
            development_holdout = backtest_fast_arrays(
                o, c, months, trend, holdout_start, len(c) - 1,
                fee, slippage, mode, 1,
            )

            wr = []
            sel_wr = []
            val_wr = []
            regimes = {"UP": [], "DOWN": [], "FLAT": []}
            sel_keys = {(int(a), int(b)) for a,b,*_ in selection_windows}
            val_keys = {(int(a), int(b)) for a,b,*_ in validation_windows}
            for a, b, regime, _bench in windows:
                wm = backtest_fast_arrays(
                    o, c, months, trend, max(int(a), int(period)), int(b),
                    fee, slippage, mode, 1,
                )
                ret = float(wm.total_return_pct)
                wr.append(ret)
                regimes[regime].append(ret)
                key = (int(a), int(b))
                if key in sel_keys:
                    sel_wr.append(ret)
                elif key in val_keys:
                    val_wr.append(ret)

            wf_windows, wf_ratio, median_ret, worst_ret, best_ret = _window_return_stats(wr)
            wf_positive = int(round(wf_ratio * wf_windows)) if wf_windows else 0
            sel_n, sel_pos, sel_med, sel_worst, _ = _window_return_stats(sel_wr)
            val_n, val_pos, val_med, val_worst, _ = _window_return_stats(val_wr)

            def reg_stats(name):
                arr = regimes[name]
                if not arr:
                    return 0, 0.0, 0.0
                return len(arr), sum(1 for x in arr if x > 0) / len(arr), float(np.mean(np.asarray(arr, dtype=np.float64)))

            up_n, up_pos, up_avg = reg_stats("UP")
            dn_n, dn_pos, dn_avg = reg_stats("DOWN")
            fl_n, fl_pos, fl_avg = reg_stats("FLAT")

            selection_score = _phase_score(selection_metrics, sel_wr, max(6, min_trades // 2))
            validation_score = _phase_score(validation_metrics, val_wr, 1) if val_n >= 2 else selection_score
            robust = _combine_research_scores(selection_score, validation_score, sel_pos, val_pos if val_n else sel_pos, sel_med, val_med if val_n else sel_med)

            out.append(AutoResult(
                interval=interval, mode=mode, atr_period=int(period), multiplier=float(mult),
                train=full_research, test=development_holdout, stage=stage,
                selection_score=float(selection_score), validation_score=float(validation_score), robust_score=float(robust),
                selection_windows=sel_n, selection_positive_ratio=float(sel_pos), selection_median_return_pct=sel_med,
                selection_worst_return_pct=sel_worst, validation_windows=val_n, validation_positive_ratio=float(val_pos),
                validation_median_return_pct=val_med, validation_worst_return_pct=val_worst,
                wf_windows=wf_windows, wf_positive_windows=wf_positive, wf_positive_ratio=float(wf_ratio),
                wf_median_return_pct=median_ret, wf_worst_return_pct=worst_ret, wf_best_return_pct=best_ret,
                up_windows=up_n, up_positive_ratio=float(up_pos), up_avg_return_pct=up_avg,
                down_windows=dn_n, down_positive_ratio=float(dn_pos), down_avg_return_pct=dn_avg,
                flat_windows=fl_n, flat_positive_ratio=float(fl_pos), flat_avg_return_pct=fl_avg,
            ))
    return out


def _worker_eval_adaptive_v2_chunk(task):
    """Evaluate a chunk of Adaptive V2 regime configs in one child process.

    The input OHLC arrays live in the per-process _WORKER state initialized once
    by ProcessPoolExecutor. Each worker builds its local Supertrend/EMA/ADX caches
    once for the whole chunk, then evaluates every LONG×SHORT×regime combination.
    This is intentionally process-parallel because the Python orchestration around
    the Numba kernels would otherwise leave Adaptive V2 mostly on one CPU core.
    """
    (
        regime_chunk, long_components, short_components, ema_pairs, adx_periods,
        plan, fee, slippage, min_trades, interval,
    ) = task
    h = _WORKER["high"]; l = _WORKER["low"]; c = _WORKER["close"]
    o = _WORKER["open"]; months = _WORKER["months"]

    trend_cache = {}
    atr_cache = {}
    for comp in list(long_components) + list(short_components):
        k = (int(comp.atr_period), round(float(comp.multiplier), 6))
        if k in trend_cache:
            continue
        p = int(comp.atr_period)
        if p not in atr_cache:
            atr_cache[p] = _fast_atr(h, l, c, p)
        trend_cache[k] = _fast_trend(h, l, c, atr_cache[p], float(comp.multiplier))

    ema_periods = sorted({int(x) for pair in ema_pairs for x in pair})
    ema_cache = {p: _ema_array(c, p) for p in ema_periods}
    adx_cache = {int(p): _adx_array(h, l, c, int(p)) for p in adx_periods}
    regime_atr_cache = {
        int(p): np.nan_to_num(_fast_atr(h, l, c, int(p)), nan=0.0, posinf=0.0, neginf=0.0)
        for p in adx_periods
    }

    rows = []
    for rp in regime_chunk:
        regime = _regime_v2_nb(
            c, ema_cache[int(rp["ema_fast"])], ema_cache[int(rp["ema_slow"])],
            adx_cache[int(rp["adx_period"])], regime_atr_cache[int(rp["adx_period"])],
            float(rp["adx_threshold"]), float(rp["separation_atr"]),
            int(rp["confirm_bars"]), int(rp["min_hold_bars"]),
        )
        for lrow in long_components:
            lt = trend_cache[(int(lrow.atr_period), round(float(lrow.multiplier), 6))]
            for srow in short_components:
                st = trend_cache[(int(srow.atr_period), round(float(srow.multiplier), 6))]
                rows.append(evaluate_adaptive_v2_candidate(
                    interval, lrow, srow, rp, lt, st, regime, o, c, months,
                    plan, fee, slippage, min_trades,
                ))
    return rows

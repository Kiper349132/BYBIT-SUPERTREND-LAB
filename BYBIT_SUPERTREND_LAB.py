from __future__ import annotations

import csv
import gc
import gzip
import heapq
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import zipfile
import queue
import threading
import ctypes
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Optional

try:
    import numpy as np
    import pandas as pd
    import requests
except ImportError as exc:
    raise SystemExit(
        "Не хватает библиотек. Запустите START.bat или выполните: pip install -r requirements.txt\n"
        f"Ошибка: {exc}"
    )

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError as exc:
    raise SystemExit(
        "Не хватает matplotlib. Запустите START.bat или выполните: pip install -r requirements.txt\n"
        f"Ошибка: {exc}"
    )

APP_NAME = "Bybit Supertrend Lab"
VERSION = "0.7.1"
API_URL = "https://api.bybit.com/v5/market/kline"
SUPPORTED_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720"]
AUTO_INTERVALS = ["1", "3", "5", "15", "30", "60"]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
RESEARCH_DIR = BASE_DIR / "research"
RUNTIME_LOG = LOGS_DIR / "runtime.log"
CHECKPOINT_USEFUL_EVERY = 1_000_000
MEMORY_FLUSH_MIN_USEFUL = 200_000
RETAIN_ROWS_PER_STAGE = 3500
ADAPTIVE_REGIME_CHUNK = 8
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
RESEARCH_DIR.mkdir(exist_ok=True)

def fmt_seconds(seconds: float) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or seconds < 0:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def file_log(text: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {text}\n"
    try:
        with RUNTIME_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
    except Exception as exc:
        # Last-resort fallback. Logging must not be able to crash the program,
        # but the failure should not disappear silently either.
        try:
            fallback = Path.home() / "BYBIT_SUPERTREND_LAB_runtime.log"
            with fallback.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.write(f"[{stamp}] PRIMARY LOG ERROR: {exc}\n")
        except Exception:
            pass


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


@dataclass
class Trade:
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    gross_return_pct: float
    net_return_pct: float


class StopRequested(Exception):
    pass


def now_utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def dt_to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def interval_ms(interval: str) -> int:
    return int(interval) * 60_000


def download_bybit_klines(
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    progress: Optional[Callable[[float, str], None]] = None,
    stop_event: Optional[threading.Event] = None,
    partial_path: Optional[Path] = None,
) -> pd.DataFrame:
    symbol = symbol.upper().strip()
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Таймфрейм {interval} не поддерживается")
    if end_dt <= start_dt:
        raise ValueError("Конечная дата должна быть позже начальной")

    start_ms = dt_to_ms(start_dt)
    end_ms = min(dt_to_ms(end_dt), now_utc_ms())
    cursor_end = end_ms
    rows: list[list[str]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": f"{APP_NAME}/{VERSION}"})

    expected = max(1, math.ceil((end_ms - start_ms) / interval_ms(interval)))
    request_no = 0

    if partial_path is not None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        with partial_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        file_log(f"PARTIAL created: {partial_path}")
    if progress:
        progress(0.0, f"{interval}м: подключение к Bybit…")
    file_log(f"HTTP begin symbol={symbol} tf={interval} start={start_dt.isoformat()} end={end_dt.isoformat()}")

    while cursor_end >= start_ms:
        if stop_event and stop_event.is_set():
            raise StopRequested()

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "start": start_ms,
            "end": cursor_end,
            "limit": 1000,
        }

        payload = None
        last_error: Optional[Exception] = None
        for attempt in range(1, 5):
            try:
                response = session.get(API_URL, params=params, timeout=12)
                file_log(f"HTTP GET attempt={attempt} tf={interval} end={cursor_end}")
                response.raise_for_status()
                payload = response.json()
                if payload.get("retCode") != 0:
                    raise RuntimeError(
                        f"Bybit retCode={payload.get('retCode')}: {payload.get('retMsg', 'неизвестная ошибка')}"
                    )
                break
            except Exception as exc:
                last_error = exc
                file_log(f"HTTP ERROR tf={interval} attempt={attempt}: {exc}")
                if progress:
                    progress(
                        min(99.0, len(rows) / expected * 100.0),
                        f"{interval}м: ошибка связи, повтор {attempt}/4 — {exc}",
                    )
                if attempt == 4:
                    raise RuntimeError(f"Не удалось получить свечи Bybit: {exc}") from exc
                time.sleep(0.8 * attempt)

        if payload is None:
            raise RuntimeError(f"Нет ответа Bybit: {last_error}")

        batch = payload.get("result", {}).get("list", [])
        request_no += 1
        if not batch:
            break

        rows.extend(batch)
        if partial_path is not None:
            try:
                with partial_path.open("a", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerows(batch)
                    fh.flush()
            except Exception as exc:
                raise RuntimeError(f"Не удалось записать временный файл {partial_path.name}: {exc}") from exc

        oldest = min(int(x[0]) for x in batch)
        pct = min(100.0, len(rows) / expected * 100.0)
        oldest_dt = datetime.fromtimestamp(oldest / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if progress:
            progress(
                pct,
                f"{interval}м: запрос №{request_no} · получено {len(rows):,}/{expected:,} свечей · {pct:.1f}% · дошли до {oldest_dt}",
            )

        if oldest <= start_ms:
            break
        if oldest >= cursor_end:
            raise RuntimeError("Bybit вернул повторяющийся диапазон. Загрузка остановлена.")
        cursor_end = oldest - 1
        time.sleep(0.035)

    if not rows:
        raise RuntimeError("Bybit не вернул свечи для выбранного периода")
    file_log(f"HTTP complete tf={interval} raw_rows={len(rows)} requests={request_no}")

    frame = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").astype("Int64")
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna().copy()
    frame["timestamp"] = frame["timestamp"].astype("int64")
    frame = frame.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    frame = frame[(frame["timestamp"] >= start_ms) & (frame["timestamp"] <= end_ms)]

    # Последняя незакрытая свеча в тест не попадает.
    if not frame.empty:
        last_ts = int(frame.iloc[-1]["timestamp"])
        if last_ts + interval_ms(interval) > now_utc_ms():
            frame = frame.iloc[:-1].copy()

    frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.reset_index(drop=True)
    if len(frame) < 300:
        raise RuntimeError(f"Слишком мало данных: {len(frame)} свечей")
    if progress:
        progress(100.0, f"{interval}м: загрузка завершена · {len(frame):,} закрытых свечей")
    return frame


def save_candles(df: pd.DataFrame, symbol: str, interval: str) -> Path:
    first = df.iloc[0]["datetime"].strftime("%Y%m%d")
    last = df.iloc[-1]["datetime"].strftime("%Y%m%d")
    path = DATA_DIR / f"{symbol}_{interval}m_{first}_{last}.csv"
    df.to_csv(path, index=False)
    return path


def load_candles_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"timestamp", "open", "high", "low", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"В CSV нет колонок: {', '.join(sorted(missing))}")
    for col in ["timestamp", "open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["volume", "turnover"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df



def find_reusable_candles(symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    """Reuse a previously downloaded CSV when it covers almost all of the requested period."""
    pattern = f"{symbol.upper()}_{interval}m_*.csv"
    candidates = [p for p in DATA_DIR.glob(pattern) if "DOWNLOADING" not in p.name.upper()]
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    requested_seconds = max(1.0, (end_dt - start_dt).total_seconds())
    tolerance = max(int(interval) * 60 * 3, requested_seconds * 0.01)
    for path in candidates:
        try:
            df = load_candles_csv(path)
            if df.empty:
                continue
            first = pd.Timestamp(df["datetime"].iloc[0]).to_pydatetime()
            last = pd.Timestamp(df["datetime"].iloc[-1]).to_pydatetime()
            start_gap = max(0.0, (first - start_dt).total_seconds())
            end_gap = max(0.0, (end_dt - last).total_seconds())
            coverage = max(0.0, 1.0 - (start_gap + end_gap) / requested_seconds)
            if start_gap <= tolerance and end_gap <= tolerance and coverage >= 0.99:
                file_log(f"REUSE data {path.name}: rows={len(df)} coverage={coverage:.4f}")
                return df, path
        except Exception as exc:
            file_log(f"REUSE skip {path.name}: {exc}")
    return None, None

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


def backtest_with_trades(
    df: pd.DataFrame,
    trend: np.ndarray,
    start_idx: int,
    end_idx: int,
    fee_rate: float,
    slippage_rate: float,
    mode: str,
) -> tuple[BacktestMetrics, list[Trade], np.ndarray]:
    opens = df["open"].to_numpy(np.float64)
    closes = df["close"].to_numpy(np.float64)
    month_codes = (df["datetime"].dt.year * 100 + df["datetime"].dt.month).to_numpy(np.int32)
    start_idx = max(0, int(start_idx))
    end_idx = min(len(df) - 1, int(end_idx))
    desired = trend[start_idx:end_idx].astype(np.int8, copy=True)
    if mode == "LONG":
        desired[desired != 1] = 0
    elif mode == "SHORT":
        desired[desired != -1] = 0
    if desired.size == 0:
        return BacktestMetrics(), [], np.array([1.0])
    starts = np.flatnonzero(np.r_[True, desired[1:] != desired[:-1]])
    ends = np.r_[starts[1:] - 1, desired.size - 1]
    run_sides = desired[starts]
    mask = run_sides != 0
    starts, ends, sides = starts[mask], ends[mask], run_sides[mask]
    if starts.size == 0:
        return BacktestMetrics(), [], np.array([1.0])

    entry_idx = start_idx + 1 + starts
    exit_idx = np.where(ends < desired.size - 1, start_idx + 2 + ends, end_idx)
    entry_raw = opens[entry_idx]
    exit_raw = np.where(ends < desired.size - 1, opens[exit_idx], closes[end_idx])
    entry_eff = _effective_entry_array(entry_raw, sides, slippage_rate)
    exit_eff = _effective_exit_array(exit_raw, sides, slippage_rate)
    gross = np.where(sides == 1, (exit_eff - entry_eff) / entry_eff, (entry_eff - exit_eff) / entry_eff)
    net = gross - 2.0 * fee_rate
    net = np.maximum(net, -0.999999)
    metrics = metrics_from_returns(net, sides, month_codes[exit_idx], 1)
    equity = np.r_[1.0, np.cumprod(1.0 + net)]
    trades = []
    times = df["datetime"].tolist()
    for k in range(len(starts)):
        trades.append(Trade(
            side="LONG" if sides[k] == 1 else "SHORT",
            entry_time=times[int(entry_idx[k])],
            exit_time=times[int(exit_idx[k])],
            entry_price=float(entry_eff[k]),
            exit_price=float(exit_eff[k]),
            gross_return_pct=float(gross[k] * 100.0),
            net_return_pct=float(net[k] * 100.0),
        ))
    return metrics, trades, equity




def backtest_desired_with_trades(
    df: pd.DataFrame,
    desired_full: np.ndarray,
    start_idx: int,
    end_idx: int,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[BacktestMetrics, list[Trade], np.ndarray]:
    opens = df["open"].to_numpy(np.float64)
    closes = df["close"].to_numpy(np.float64)
    months = (df["datetime"].dt.year * 100 + df["datetime"].dt.month).to_numpy(np.int32)
    start_idx = max(0, int(start_idx))
    end_idx = min(len(df) - 1, int(end_idx))
    desired = desired_full[start_idx:end_idx].astype(np.int8, copy=False)
    if desired.size == 0:
        return BacktestMetrics(), [], np.array([1.0])
    starts = np.flatnonzero(np.r_[True, desired[1:] != desired[:-1]])
    ends = np.r_[starts[1:] - 1, desired.size - 1]
    sides = desired[starts]
    mask = sides != 0
    starts, ends, sides = starts[mask], ends[mask], sides[mask]
    if starts.size == 0:
        return BacktestMetrics(), [], np.array([1.0])
    entry_idx = start_idx + 1 + starts
    exit_idx = np.where(ends < desired.size - 1, start_idx + 2 + ends, end_idx)
    valid = (entry_idx < len(opens)) & (exit_idx < len(opens))
    entry_idx, exit_idx, starts, ends, sides = entry_idx[valid], exit_idx[valid], starts[valid], ends[valid], sides[valid]
    if entry_idx.size == 0:
        return BacktestMetrics(), [], np.array([1.0])
    entry_raw = opens[entry_idx]
    exit_raw = np.where(ends < desired.size - 1, opens[exit_idx], closes[end_idx])
    entry_eff = _effective_entry_array(entry_raw, sides, slippage_rate)
    exit_eff = _effective_exit_array(exit_raw, sides, slippage_rate)
    gross = np.where(sides == 1, (exit_eff - entry_eff) / entry_eff, (entry_eff - exit_eff) / entry_eff)
    net = np.maximum(gross - 2.0 * fee_rate, -0.999999)
    metrics = metrics_from_returns(net, sides, months[exit_idx], 1)
    equity = np.r_[1.0, np.cumprod(1.0 + net)]
    times = df["datetime"].tolist()
    trades = []
    for k in range(len(entry_idx)):
        trades.append(Trade(
            side="LONG" if sides[k] == 1 else "SHORT",
            entry_time=times[int(entry_idx[k])], exit_time=times[int(exit_idx[k])],
            entry_price=float(entry_eff[k]), exit_price=float(exit_eff[k]),
            gross_return_pct=float(gross[k] * 100.0), net_return_pct=float(net[k] * 100.0),
        ))
    return metrics, trades, equity


def build_selected_signal(df: pd.DataFrame, row: AutoResult) -> tuple[np.ndarray, str]:
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    if row.strategy_type in ("ADAPTIVE", "ADAPTIVE_V2"):
        latr = wilder_atr(h, l, c, row.long_atr_period)
        satr = wilder_atr(h, l, c, row.short_atr_period)
        ltrend, _ = supertrend_from_atr(h, l, c, latr, row.long_multiplier)
        strend, _ = supertrend_from_atr(h, l, c, satr, row.short_multiplier)
        if row.strategy_type == "ADAPTIVE_V2":
            regime = online_regime_v2(
                h, l, c, row.ema_fast, row.ema_slow, row.adx_period, row.adx_threshold,
                row.regime_separation_atr, row.regime_confirm_bars, row.regime_min_hold_bars,
            )
            regime_desc = (
                f"EMA {row.ema_fast}/{row.ema_slow} · ADX{row.adx_period}>={row.adx_threshold:g} · "
                f"sep {row.regime_separation_atr:g}ATR · confirm {row.regime_confirm_bars} · hold {row.regime_min_hold_bars}"
            )
        else:
            regime = online_regime_from_past(c, row.interval, row.regime_hours, row.regime_threshold_pct)
            regime_desc = f"режим {row.regime_hours}ч/{row.regime_threshold_pct:g}%"
        desired = np.zeros(len(c), dtype=np.int8)
        desired[(regime == 1) & (ltrend == 1)] = 1
        desired[(regime == -1) & (strend == -1)] = -1
        label = (
            f"{row.strategy_type} · UP: ATR {row.long_atr_period} ×{row.long_multiplier:g} · "
            f"DOWN: ATR {row.short_atr_period} ×{row.short_multiplier:g} · {regime_desc}"
        )
        return desired, label
    atr = wilder_atr(h, l, c, row.atr_period)
    trend, _ = supertrend_from_atr(h, l, c, atr, row.multiplier)
    desired = trend.astype(np.int8, copy=True)
    if row.mode == "LONG":
        desired[desired != 1] = 0
    elif row.mode == "SHORT":
        desired[desired != -1] = 0
    return desired, f"{row.mode} · ATR {row.atr_period} ×{row.multiplier:g}"

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


def group_specs(candidates: set[tuple[int, float, str]]) -> dict[int, list[tuple[float, tuple[str, ...]]]]:
    grouped: dict[int, dict[float, set[str]]] = {}
    for period, mult, mode in candidates:
        grouped.setdefault(period, {}).setdefault(round(float(mult), 4), set()).add(mode)
    result = {}
    for period, mm in grouped.items():
        result[period] = [(mult, tuple(sorted(modes))) for mult, modes in sorted(mm.items())]
    return result


def choose_seed_rows(rows: list[AutoResult], limit: int = 12) -> list[AutoResult]:
    # IMPORTANT: search expansion uses only the early chronological SELECTION
    # phase. Later VALIDATION windows and development holdout cannot influence
    # where the optimiser looks next.
    ranked = sorted(rows, key=lambda r: (r.selection_score, r.mode, -r.atr_period, -r.multiplier), reverse=True)
    chosen = []
    for row in ranked:
        if row.selection_score <= -1e8:
            continue
        too_close = any(
            row.mode == c.mode
            and abs(row.atr_period - c.atr_period) <= 2
            and abs(row.multiplier - c.multiplier) <= 0.5
            for c in chosen
        )
        if not too_close:
            chosen.append(row)
        if len(chosen) >= limit:
            break
    return chosen or ranked[:limit]


def _result_key(row: AutoResult):
    if row.strategy_type == "ADAPTIVE_V2":
        return (
            row.interval, "ADAPTIVE_V2", row.long_atr_period, round(row.long_multiplier, 4),
            row.short_atr_period, round(row.short_multiplier, 4), row.ema_fast, row.ema_slow,
            row.adx_period, round(row.adx_threshold, 3), round(row.regime_separation_atr, 3),
            row.regime_confirm_bars, row.regime_min_hold_bars,
        )
    if row.strategy_type == "ADAPTIVE":
        return (
            row.interval, "ADAPTIVE", row.long_atr_period, round(row.long_multiplier, 4),
            row.short_atr_period, round(row.short_multiplier, 4),
            row.regime_hours, round(row.regime_threshold_pct, 4),
        )
    return (row.interval, row.mode, row.atr_period, round(row.multiplier, 4))


def _autoresult_to_dict(row: AutoResult) -> dict:
    """Portable representation used by the disk-backed research store."""
    return asdict(row)


def _autoresult_from_dict(d: dict) -> AutoResult:
    d = dict(d)
    d["train"] = BacktestMetrics(**dict(d.get("train") or {}))
    d["test"] = BacktestMetrics(**dict(d.get("test") or {}))
    return AutoResult(**d)


def _memory_percent() -> float:
    """Best-effort total physical RAM usage percentage without extra packages."""
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
            stat = MEMORYSTATUSEX(); stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return float(stat.dwMemoryLoad)
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            vals = {}
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    try: vals[k] = float(v.strip().split()[0])
                    except Exception: pass
            total = vals.get("MemTotal", 0.0); avail = vals.get("MemAvailable", vals.get("MemFree", 0.0))
            if total > 0: return max(0.0, min(100.0, (1.0 - avail / total) * 100.0))
    except Exception:
        pass
    return 0.0


def _retain_search_rows(current: list[AutoResult], incoming: list[AutoResult], limit: int = RETAIN_ROWS_PER_STAGE) -> list[AutoResult]:
    """Keep a small but diverse in-RAM working set; the full set lives on disk."""
    if not incoming:
        return current
    merged = current + incoming
    best = {}
    for r in merged:
        k = _result_key(r)
        old = best.get(k)
        if old is None or (r.selection_score, r.validation_score, r.robust_score) > (old.selection_score, old.validation_score, old.robust_score):
            best[k] = r
    vals = list(best.values())
    if len(vals) <= limit:
        return vals

    chosen = {}
    # Search expansion is selection-only, but report/GUI also need validation/robust leaders.
    rankings = [
        (lambda x: x.selection_score, max(1, limit // 2)),
        (lambda x: x.validation_score, max(1, limit // 4)),
        (lambda x: x.robust_score, max(1, limit // 4)),
    ]
    for keyfn, n in rankings:
        for r in heapq.nlargest(n, vals, key=keyfn):
            chosen[_result_key(r)] = r
    # Preserve modes/families even if one direction dominates the market sample.
    families = {}
    for r in vals:
        families.setdefault((r.strategy_type, r.mode), []).append(r)
    per_family = max(25, limit // max(20, len(families) * 8))
    for group in families.values():
        for r in heapq.nlargest(per_family, group, key=lambda x: x.selection_score):
            chosen[_result_key(r)] = r
    result = list(chosen.values())
    if len(result) > limit:
        result = heapq.nlargest(limit, result, key=lambda x: max(x.selection_score, x.validation_score, x.robust_score))
    return result


def _research_signature_for_version(cfg: dict, data: dict[str, pd.DataFrame], version_tag: str) -> str:
    material = {
        "version": version_tag,
        "symbol": cfg.get("symbol"), "months": cfg.get("months"), "tfs": list(cfg.get("tfs", [])),
        "fee": cfg.get("fee"), "slippage": cfg.get("slippage"), "depth": cfg.get("depth"),
        "data": {},
    }
    for tf in sorted(data, key=lambda x: int(x)):
        df = data[tf]
        material["data"][str(tf)] = {
            "rows": int(len(df)),
            "first_ts": int(df["timestamp"].iloc[0]) if len(df) else 0,
            "last_ts": int(df["timestamp"].iloc[-1]) if len(df) else 0,
            "first_close": round(float(df["close"].iloc[0]), 8) if len(df) else 0.0,
            "last_close": round(float(df["close"].iloc[-1]), 8) if len(df) else 0.0,
        }
    raw = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _research_signature(cfg: dict, data: dict[str, pd.DataFrame]) -> str:
    return _research_signature_for_version(cfg, data, VERSION)



class ResearchPersistence:
    """Crash-safe append-only storage for very large optimisation runs.

    Results are flushed in compressed JSONL chunks after roughly one million useful
    backtest evaluations. Only small seed/top pools remain in RAM. A checkpoint is
    atomically updated *after* the chunk is safely on disk, so an unexpected exit
    loses at most the current unflushed batch and the run can continue.
    """
    def __init__(self, cfg: dict, data: dict[str, pd.DataFrame]):
        self.cfg = dict(cfg)
        self.signature = _research_signature(cfg, data)
        stem = f"{cfg.get('symbol','UNKNOWN')}_{self.signature}"
        base = RESEARCH_DIR / stem
        checkpoint = base / "checkpoint.json"
        self.legacy_resumed_from = ""

        # v0.7.1 changes only execution/memory management, not the search space or
        # result schema. Therefore an incomplete v0.7.0 store is safe to resume.
        # This preserves the user's already committed million-checkpoint chunks.
        if not checkpoint.exists():
            legacy_sig = _research_signature_for_version(cfg, data, "0.7.0")
            legacy_base = RESEARCH_DIR / f"{cfg.get('symbol','UNKNOWN')}_{legacy_sig}"
            legacy_cp = legacy_base / "checkpoint.json"
            if legacy_cp.exists():
                try:
                    legacy_state = json.loads(legacy_cp.read_text(encoding="utf-8"))
                except Exception:
                    legacy_state = {}
                if legacy_state.get("status") != "complete" and legacy_state.get("signature") == legacy_sig:
                    self.signature = legacy_sig
                    stem = f"{cfg.get('symbol','UNKNOWN')}_{legacy_sig}"
                    base = legacy_base
                    checkpoint = legacy_cp
                    self.legacy_resumed_from = "0.7.0"

        # An incomplete matching session is automatically resumed. A completed one
        # is preserved and a fresh timestamped run is created.
        if checkpoint.exists():
            try:
                old = json.loads(checkpoint.read_text(encoding="utf-8"))
            except Exception:
                old = {}
            if old.get("status") == "complete":
                base = RESEARCH_DIR / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.root = base
        self.chunks_dir = self.root / "chunks"
        self.seeds_dir = self.root / "stage_seeds"
        self.root.mkdir(parents=True, exist_ok=True); self.chunks_dir.mkdir(exist_ok=True); self.seeds_dir.mkdir(exist_ok=True)
        self.checkpoint_path = self.root / "checkpoint.json"
        self.index_path = self.root / "research_index.json"
        self.readme_path = self.root / "README_RESEARCH.txt"
        self.buffer_rows: list[AutoResult] = []
        self.buffer_useful = 0
        self.pending_units: list[tuple[str,str,str]] = []
        self.resumed = False
        self._load_or_init()
        # Stale temporary/orphan chunks mean the previous process died between
        # file creation and the atomic checkpoint update. Only checkpoint-listed
        # chunks are authoritative; unlisted chunks are removed so resume cannot
        # double-count them.
        for fp in self.chunks_dir.glob("*.tmp"):
            try: fp.unlink()
            except Exception: pass
        known={str(x.get("file")) for x in self.cp.get("chunks",[]) if x.get("file")}
        for fp in self.chunks_dir.glob("chunk_*.jsonl.gz"):
            if fp.name not in known:
                try: fp.unlink()
                except Exception: pass

    def _load_or_init(self):
        if self.checkpoint_path.exists():
            try:
                cp = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
                if cp.get("signature") == self.signature and cp.get("status") != "complete":
                    self.cp = cp; self.resumed = True; return
            except Exception as exc:
                file_log(f"CHECKPOINT read error: {exc}")
        self.cp = {
            "version": VERSION, "signature": self.signature, "status": "incomplete",
            "created_utc": datetime.now(timezone.utc).isoformat(), "updated_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_useful_interval": CHECKPOINT_USEFUL_EVERY,
            "committed_useful_checks": 0, "committed_result_rows": 0, "chunk_count": 0,
            "completed_units": {}, "completed_stages": {}, "stage_meta": {},
            "strategy_type_counts": {}, "timeframe_counts": {}, "stage_counts": {}, "chunks": [],
            "config": {k: (v.isoformat() if isinstance(v, datetime) else v) for k,v in self.cfg.items() if k != "workers"},
        }
        self._write_checkpoint()
        self._write_readme()

    def _write_readme(self):
        txt = (
            f"BYBIT SUPERTREND LAB v{VERSION} - DISK RESEARCH STORE\n\n"
            f"This folder is the durable result store for one optimisation run.\n"
            f"Results are committed approximately every {CHECKPOINT_USEFUL_EVERY:,} useful backtest evaluations.\n"
            "chunks/*.jsonl.gz contain the full AutoResult records. stage_seeds/*.jsonl.gz contain only small retained pools used to continue the search.\n"
            "checkpoint.json is the crash/resume state. research_index.json is the final compact index.\n"
            "Do not delete this folder while a run is incomplete. After a completed run, chunks are the full raw research archive; they can be deleted only if you intentionally no longer need full-result reanalysis.\n"
        )
        try: self.readme_path.write_text(txt, encoding="utf-8")
        except Exception: pass

    def _write_checkpoint(self):
        self.cp["updated_utc"] = datetime.now(timezone.utc).isoformat()
        tmp = self.checkpoint_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.cp, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.checkpoint_path)

    @property
    def committed_useful(self) -> int:
        return int(self.cp.get("committed_useful_checks", 0) or 0)

    @property
    def committed_rows(self) -> int:
        return int(self.cp.get("committed_result_rows", 0) or 0)

    def _stage_key(self, tf: str, stage: str) -> str:
        return f"{tf}|{stage}"

    def is_stage_complete(self, tf: str, stage: str) -> bool:
        return bool(self.cp.get("completed_stages", {}).get(self._stage_key(tf,stage), False))

    def unit_done(self, tf: str, stage: str, unit_id: str) -> bool:
        return str(unit_id) in set(self.cp.get("completed_units", {}).get(self._stage_key(tf,stage), []))

    def queue_unit(self, tf: str, stage: str, unit_id: str, rows: list[AutoResult], useful_checks: int) -> bool:
        self.buffer_rows.extend(rows)
        self.buffer_useful += int(useful_checks)
        self.pending_units.append((str(tf), str(stage), str(unit_id)))
        ram = _memory_percent()
        if self.buffer_useful >= CHECKPOINT_USEFUL_EVERY:
            self.flush(reason=f"interval_{CHECKPOINT_USEFUL_EVERY}")
            return True
        # A high system-wide RAM percentage can remain elevated for a while due to
        # Windows file cache and worker processes. Do not turn that into a flush
        # after every tiny unit. Emergency RAM flushes are allowed only after a
        # meaningful buffer has accumulated; the normal checkpoint is still 1M.
        if ram >= 88.0 and self.buffer_useful >= MEMORY_FLUSH_MIN_USEFUL:
            self.flush(reason=f"memory_guard_{ram:.1f}pct")
            return True
        return False

    def flush(self, reason: str = "manual", force: bool = False):
        if not self.pending_units and not self.buffer_rows:
            return
        seq = int(self.cp.get("chunk_count", 0) or 0) + 1
        rows = self.buffer_rows
        useful = int(self.buffer_useful)
        if rows:
            final = self.chunks_dir / f"chunk_{seq:06d}.jsonl.gz"
            tmp = Path(str(final) + ".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=2, newline="\n") as fh:
                for row in rows:
                    fh.write(json.dumps(_autoresult_to_dict(row), ensure_ascii=False, separators=(",", ":")))
                    fh.write("\n")
            os.replace(tmp, final)
            size = final.stat().st_size
            self.cp.setdefault("chunks", []).append({"file":final.name,"rows":len(rows),"useful_checks":useful,"bytes":size,"reason":reason})
            self.cp["chunk_count"] = seq
            self.cp["committed_result_rows"] = self.committed_rows + len(rows)
            for r in rows:
                st = str(r.strategy_type); tf = str(r.interval); sg = str(r.stage)
                self.cp["strategy_type_counts"][st] = int(self.cp["strategy_type_counts"].get(st,0))+1
                self.cp["timeframe_counts"][tf] = int(self.cp["timeframe_counts"].get(tf,0))+1
                sk=f"{tf}|{sg}"; self.cp["stage_counts"][sk]=int(self.cp["stage_counts"].get(sk,0))+1
        self.cp["committed_useful_checks"] = self.committed_useful + useful
        cu = self.cp.setdefault("completed_units", {})
        for tf, stage, unit_id in self.pending_units:
            k=self._stage_key(tf,stage); arr=cu.setdefault(k,[])
            if unit_id not in arr: arr.append(unit_id)
        self._write_checkpoint()
        file_log(f"CHECKPOINT flush #{seq}: rows={len(rows):,} useful={useful:,} reason={reason} RAM={_memory_percent():.1f}%")
        self.buffer_rows = []; self.buffer_useful = 0; self.pending_units = []
        gc.collect()

    def save_partial_seed(self, tf: str, stage: str, rows: list[AutoResult]):
        """Persist the small in-RAM working pool without marking the stage complete."""
        fp = self.seeds_dir / f"{tf}_{stage}.jsonl.gz"
        tmp = Path(str(fp)+".tmp")
        with gzip.open(tmp,"wt",encoding="utf-8",compresslevel=3,newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(_autoresult_to_dict(row),ensure_ascii=False,separators=(",",":"))+"\n")
        os.replace(tmp,fp)

    def save_stage_seed(self, tf: str, stage: str, rows: list[AutoResult], meta: Optional[dict] = None):
        self.flush(reason=f"stage_complete_{tf}_{stage}", force=True)
        fp = self.seeds_dir / f"{tf}_{stage}.jsonl.gz"
        tmp = Path(str(fp)+".tmp")
        with gzip.open(tmp,"wt",encoding="utf-8",compresslevel=3,newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(_autoresult_to_dict(row),ensure_ascii=False,separators=(",",":"))+"\n")
        os.replace(tmp,fp)
        k=self._stage_key(tf,stage)
        self.cp.setdefault("completed_stages",{})[k]=True
        if meta is not None: self.cp.setdefault("stage_meta",{})[k]=dict(meta)
        self._write_checkpoint()

    def load_stage_seed(self, tf: str, stage: str) -> list[AutoResult]:
        fp=self.seeds_dir/f"{tf}_{stage}.jsonl.gz"
        if not fp.exists(): return []
        out=[]
        with gzip.open(fp,"rt",encoding="utf-8") as fh:
            for line in fh:
                line=line.strip()
                if line: out.append(_autoresult_from_dict(json.loads(line)))
        return out

    def stage_meta(self, tf: str, stage: str) -> dict:
        return dict(self.cp.get("stage_meta",{}).get(self._stage_key(tf,stage),{}) or {})

    def finalize(self, extra: Optional[dict] = None):
        self.flush(reason="finalize", force=True)
        self.cp["status"]="complete"; self.cp["completed_utc"]=datetime.now(timezone.utc).isoformat()
        if extra: self.cp["final"] = extra
        self._write_checkpoint()
        idx=self.summary()
        tmp=self.index_path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(idx,ensure_ascii=False,indent=2),encoding="utf-8"); os.replace(tmp,self.index_path)

    def safe_stop(self, reason: str):
        self.flush(reason=reason, force=True)
        self.cp["status"]="incomplete"; self.cp["last_stop_reason"]=reason; self._write_checkpoint()

    def summary(self) -> dict:
        chunk_bytes=sum(int(x.get("bytes",0) or 0) for x in self.cp.get("chunks",[]))
        return {
            "version":VERSION,"signature":self.signature,"session_dir":str(self.root),"status":self.cp.get("status"),
            "resumed":self.resumed,"legacy_resumed_from":self.legacy_resumed_from,"checkpoint_useful_interval":CHECKPOINT_USEFUL_EVERY,
            "committed_useful_checks":self.committed_useful,"committed_result_rows":self.committed_rows,
            "chunk_count":int(self.cp.get("chunk_count",0) or 0),"chunk_bytes":chunk_bytes,
            "strategy_type_counts":dict(self.cp.get("strategy_type_counts",{})),
            "timeframe_counts":dict(self.cp.get("timeframe_counts",{})),"stage_counts":dict(self.cp.get("stage_counts",{})),
            "completed_stages":dict(self.cp.get("completed_stages",{})),"stage_meta":dict(self.cp.get("stage_meta",{})),
            "checkpoint_path":str(self.checkpoint_path),"index_path":str(self.index_path),
        }


def diversified_top(rows: list[AutoResult], limit: int = 100) -> list[AutoResult]:
    """Research-only TOP that deliberately preserves different strategy families.

    v0.5 could fill almost the whole visible TOP with one spectacular SHORT
    family. v0.6 first reserves a few representatives from each
    (strategy_type, timeframe, mode) family, then fills the rest globally.
    Development holdout is never consulted.
    """
    ranked = sorted(rows, key=lambda r: (r.robust_score, r.validation_score, r.selection_score), reverse=True)
    chosen: list[AutoResult] = []
    used_keys = set()

    def is_near(row: AutoResult) -> bool:
        if row.strategy_type != "BASE":
            return any(
                c.strategy_type == row.strategy_type and row.interval == c.interval
                and abs(row.long_atr_period-c.long_atr_period) <= 1 and abs(row.long_multiplier-c.long_multiplier) <= 0.10
                and abs(row.short_atr_period-c.short_atr_period) <= 1 and abs(row.short_multiplier-c.short_multiplier) <= 0.10
                and (row.strategy_type != "ADAPTIVE_V2" or (
                    row.ema_fast == c.ema_fast and row.ema_slow == c.ema_slow
                    and abs(row.adx_threshold-c.adx_threshold) <= 5
                    and abs(row.regime_separation_atr-c.regime_separation_atr) <= 0.15
                ))
                for c in chosen
            )
        return any(
            c.strategy_type == "BASE" and row.interval == c.interval and row.mode == c.mode
            and abs(row.atr_period-c.atr_period) <= 1 and abs(row.multiplier-c.multiplier) <= 0.12
            for c in chosen
        )

    # Family representatives.
    families = {}
    for r in ranked:
        if r.robust_score <= -1e8:
            continue
        fam = (r.strategy_type, r.interval, r.mode)
        families.setdefault(fam, []).append(r)
    for fam in sorted(families):
        added = 0
        for r in families[fam]:
            k = _result_key(r)
            if k in used_keys or is_near(r):
                continue
            chosen.append(r); used_keys.add(k); added += 1
            if len(chosen) >= limit:
                return sorted(chosen, key=lambda x:x.robust_score, reverse=True)
            if added >= 2:
                break

    for r in ranked:
        k = _result_key(r)
        if k in used_keys or r.robust_score <= -1e8 or is_near(r):
            continue
        chosen.append(r); used_keys.add(k)
        if len(chosen) >= limit:
            break
    # If diversity filters are too strict, fill with remaining unique rows.
    if len(chosen) < limit:
        for r in ranked:
            k=_result_key(r)
            if k in used_keys or r.robust_score <= -1e8:
                continue
            chosen.append(r); used_keys.add(k)
            if len(chosen) >= limit:
                break
    return sorted(chosen, key=lambda x:x.robust_score, reverse=True)

def results_dataframe(rows: list[AutoResult]) -> pd.DataFrame:
    records = []
    for row in rows:
        rec = {
            "timeframe_min": row.interval,
            "mode": row.mode,
            "atr_period": row.atr_period,
            "multiplier": row.multiplier,
            "stage": row.stage,
            "strategy_type": row.strategy_type,
            "long_atr_period": row.long_atr_period,
            "long_multiplier": row.long_multiplier,
            "short_atr_period": row.short_atr_period,
            "short_multiplier": row.short_multiplier,
            "regime_hours": row.regime_hours,
            "regime_threshold_pct": row.regime_threshold_pct,
            "online_up_ratio": row.online_up_ratio,
            "online_down_ratio": row.online_down_ratio,
            "online_flat_ratio": row.online_flat_ratio,
            "regime_switches": row.regime_switches,
            "regime_model": row.regime_model,
            "ema_fast": row.ema_fast, "ema_slow": row.ema_slow,
            "adx_period": row.adx_period, "adx_threshold": row.adx_threshold,
            "regime_separation_atr": row.regime_separation_atr,
            "regime_confirm_bars": row.regime_confirm_bars,
            "regime_min_hold_bars": row.regime_min_hold_bars,
            "selection_score": row.selection_score, "validation_score": row.validation_score,
            "robust_score": row.robust_score,
            "selection_windows": row.selection_windows, "selection_positive_ratio": row.selection_positive_ratio,
            "selection_median_return_pct": row.selection_median_return_pct,
            "selection_worst_return_pct": row.selection_worst_return_pct,
            "validation_windows": row.validation_windows, "validation_positive_ratio": row.validation_positive_ratio,
            "validation_median_return_pct": row.validation_median_return_pct,
            "validation_worst_return_pct": row.validation_worst_return_pct,
            "wf_windows": row.wf_windows,
            "wf_positive_windows": row.wf_positive_windows,
            "wf_positive_ratio": row.wf_positive_ratio,
            "wf_median_return_pct": row.wf_median_return_pct,
            "wf_worst_return_pct": row.wf_worst_return_pct,
            "wf_best_return_pct": row.wf_best_return_pct,
            "up_windows": row.up_windows,
            "up_positive_ratio": row.up_positive_ratio,
            "up_avg_return_pct": row.up_avg_return_pct,
            "down_windows": row.down_windows,
            "down_positive_ratio": row.down_positive_ratio,
            "down_avg_return_pct": row.down_avg_return_pct,
            "flat_windows": row.flat_windows,
            "flat_positive_ratio": row.flat_positive_ratio,
            "flat_avg_return_pct": row.flat_avg_return_pct,
        }
        for prefix, metrics in (("train", row.train), ("test", row.test)):
            for k, v in asdict(metrics).items():
                rec[f"{prefix}_{k}"] = v
        records.append(rec)
    return pd.DataFrame(records)


def save_results(all_rows: list[AutoResult], top_rows: list[AutoResult], symbol: str, research_summary: Optional[dict] = None) -> tuple[Path, Path]:
    """Save human-sized extracts. The full result stream lives in research/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    retained_path = RESULTS_DIR / f"RETAINED_RESULTS_{symbol}_{ts}.csv"
    top_path = RESULTS_DIR / f"TOP100_{symbol}_{ts}.csv"
    results_dataframe(all_rows).to_csv(retained_path, index=False)
    results_dataframe(top_rows).to_csv(top_path, index=False)
    if research_summary:
        info = RESULTS_DIR / f"RESEARCH_STORE_{symbol}_{ts}.txt"
        info.write_text(
            "BYBIT SUPERTREND LAB - FULL RESULT STORE\n\n"
            f"Full raw research session: {research_summary.get('session_dir','')}\n"
            f"Committed useful checks: {int(research_summary.get('committed_useful_checks',0)):,}\n"
            f"Committed result rows: {int(research_summary.get('committed_result_rows',0)):,}\n"
            f"Chunks: {int(research_summary.get('chunk_count',0)):,}\n"
            f"Checkpoint interval: {int(research_summary.get('checkpoint_useful_interval',CHECKPOINT_USEFUL_EVERY)):,} useful checks\n\n"
            "RETAINED_RESULTS is only the small working/report sample. The complete raw results are in research/chunks.\n",
            encoding="utf-8",
        )
    return retained_path, top_path


def _json_safe(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def diagnose_dataframe(df: pd.DataFrame, interval: str) -> dict:
    if df is None or df.empty:
        return {"timeframe_min": interval, "rows": 0, "error": "empty dataframe"}
    ts = pd.to_datetime(df["datetime"], utc=True)
    ms = int(interval) * 60 * 1000
    # Use the raw exchange timestamp when available. Pandas may keep datetime
    # internally at ms/us/ns resolution; dividing astype(int64) by a hardcoded
    # factor caused false duplicate reports in v0.3.0.
    if "timestamp" in df.columns:
        vals = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(np.int64)
    else:
        vals = np.asarray([int(x.timestamp() * 1000) for x in ts], dtype=np.int64)
    diffs = np.diff(vals) if len(vals) > 1 else np.array([], dtype=np.int64)
    gap_mask = diffs > ms
    missing_estimate = int(np.sum(np.maximum((diffs[gap_mask] // ms) - 1, 0))) if gap_mask.any() else 0
    duplicates = int(pd.Series(vals).duplicated().sum())
    non_monotonic = int(np.sum(diffs <= 0)) if diffs.size else 0
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    invalid_ohlc = int(np.sum((h < np.maximum.reduce([o, c, l])) | (l > np.minimum.reduce([o, c, h]))))
    nonpositive = int(np.sum((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)))
    return {
        "timeframe_min": interval,
        "rows": int(len(df)),
        "start_utc": ts.iloc[0].isoformat(),
        "end_utc": ts.iloc[-1].isoformat(),
        "duplicate_timestamps": duplicates,
        "non_monotonic_steps": non_monotonic,
        "gap_events": int(gap_mask.sum()),
        "missing_candles_estimate": missing_estimate,
        "invalid_ohlc_rows": invalid_ohlc,
        "nonpositive_price_rows": nonpositive,
    }


def create_chatgpt_report(
    cfg: dict,
    search_stats: list[dict],
    all_rows: list[AutoResult],
    top_rows: list[AutoResult],
    data_by_tf: dict[str, pd.DataFrame],
    search_seconds: float = 0.0,
    research_summary: Optional[dict] = None,
) -> Path:
    """Create a reproducible v0.7 audit package.

    The report intentionally separates SELECTION, VALIDATION and the historical
    DEVELOPMENT_HOLDOUT. Since earlier reports have already exposed the last
    historical block, v0.7 never calls it a fresh/untouched forward test.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol = (cfg or {}).get("symbol", "UNKNOWN")
    out = REPORTS_DIR / f"CHATGPT_REPORT_{symbol}_{ts}.zip"

    data_diag=[]; plan_rows=[]
    for tf, df in sorted(data_by_tf.items(), key=lambda x:int(x[0])):
        try:
            data_diag.append(diagnose_dataframe(df,tf)); plan=build_walk_forward_plan(df)
            sel_keys={(int(a),int(b)) for a,b,*_ in plan.get("selection_windows",[])}
            val_keys={(int(a),int(b)) for a,b,*_ in plan.get("validation_windows",[])}
            for idx,(a,b,regime,bench) in enumerate(plan["windows"],1):
                k=(int(a),int(b)); phase="SELECTION" if k in sel_keys else ("VALIDATION" if k in val_keys else "RESEARCH")
                plan_rows.append({
                    "timeframe_min":tf,"window_no":idx,"phase":phase,
                    "start_utc":str(df["datetime"].iloc[int(a)]),"end_utc":str(df["datetime"].iloc[int(b)]),
                    "diagnostic_regime_after_window":regime,"btc_move_pct_in_window":float(bench),
                    "used_to_expand_search":phase=="SELECTION",
                })
            hs=int(plan["holdout_start"])
            plan_rows.append({
                "timeframe_min":tf,"window_no":"DEVELOPMENT_HOLDOUT","phase":"DEVELOPMENT_HOLDOUT",
                "start_utc":str(df["datetime"].iloc[hs]),"end_utc":str(df["datetime"].iloc[-1]),
                "diagnostic_regime_after_window":"DEVELOPMENT_ONLY",
                "btc_move_pct_in_window":float((df["close"].iloc[-1]/df["open"].iloc[hs]-1)*100),
                "used_to_expand_search":False,
            })
        except Exception as exc:
            data_diag.append({"timeframe_min":tf,"error":str(exc)})

    rdf=results_dataframe(all_rows) if all_rows else pd.DataFrame()
    topdf=results_dataframe(top_rows) if top_rows else pd.DataFrame()

    top_snapshot=[]
    for rank,row in enumerate(top_rows,1):
        top_snapshot.append({
            "rank":rank,"strategy_type":row.strategy_type,"timeframe_min":row.interval,"mode":row.mode,"stage":row.stage,
            "base":{"atr_period":row.atr_period,"multiplier":row.multiplier},
            "adaptive_components":{"long_atr":row.long_atr_period,"long_mult":row.long_multiplier,"short_atr":row.short_atr_period,"short_mult":row.short_multiplier},
            "adaptive_v2_regime":{"model":row.regime_model,"ema_fast":row.ema_fast,"ema_slow":row.ema_slow,"adx_period":row.adx_period,
                "adx_threshold":row.adx_threshold,"separation_atr":row.regime_separation_atr,"confirm_bars":row.regime_confirm_bars,"min_hold_bars":row.regime_min_hold_bars,
                "up_ratio":row.online_up_ratio,"down_ratio":row.online_down_ratio,"flat_ratio":row.online_flat_ratio,"switches":row.regime_switches},
            "scores":{"selection":row.selection_score,"validation":row.validation_score,"research_combined":row.robust_score},
            "selection":{"windows":row.selection_windows,"positive_ratio":row.selection_positive_ratio,"median_pct":row.selection_median_return_pct,"worst_pct":row.selection_worst_return_pct},
            "validation":{"windows":row.validation_windows,"positive_ratio":row.validation_positive_ratio,"median_pct":row.validation_median_return_pct,"worst_pct":row.validation_worst_return_pct},
            "research":asdict(row.train),"development_holdout":asdict(row.test),
        })

    # Strategy class summary.
    strategy_summary=[]
    if not rdf.empty:
        for (stype,tf),g in rdf.groupby(["strategy_type","timeframe_min"],dropna=False):
            valid=g[g["robust_score"]>-1e8]
            dev=valid[(valid["train_total_return_pct"]>0)&(valid["test_total_return_pct"]>0)]
            strict=dev[(dev["test_profit_factor"]>1)&(dev["test_max_drawdown_pct"]<=20)&(dev["test_trades"]>=5)]
            strategy_summary.append({
                "strategy_type":stype,"timeframe_min":tf,"candidates":len(g),"valid_candidates":len(valid),
                "selection_score_positive_ratio":float((valid["selection_score"]>0).mean()) if len(valid) else 0,
                "validation_score_positive_ratio":float((valid["validation_score"]>0).mean()) if len(valid) else 0,
                "development_holdout_positive_ratio":float((valid["test_total_return_pct"]>0).mean()) if len(valid) else 0,
                "development_survivors":len(dev),"strict_development_survivors":len(strict),
                "median_validation_window_pct":float(valid["validation_median_return_pct"].median()) if len(valid) else 0,
                "median_development_holdout_pct":float(valid["test_total_return_pct"].median()) if len(valid) else 0,
                "best_development_holdout_pct":float(valid["test_total_return_pct"].max()) if len(valid) else 0,
            })

    survivors=pd.DataFrame()
    if not rdf.empty:
        survivors=rdf[(rdf["robust_score"]>-1e8)&(rdf["train_total_return_pct"]>0)&(rdf["test_total_return_pct"]>0)].copy()
        if not survivors.empty:
            survivors["strict_development_survivor"]=(survivors["test_profit_factor"]>1)&(survivors["test_max_drawdown_pct"]<=20)&(survivors["test_trades"]>=5)
            survivors=survivors.sort_values(["strict_development_survivor","validation_score","test_total_return_pct"],ascending=[False,False,False])

    # Parameter coverage: proves that width/depth actually happened.
    coverage=[]
    if not rdf.empty:
        for (stype,tf,stage),g in rdf.groupby(["strategy_type","timeframe_min","stage"],dropna=False):
            rec={"strategy_type":stype,"timeframe_min":tf,"stage":stage,"candidates":len(g)}
            if stype=="BASE":
                rec.update({"atr_min":int(g["atr_period"].min()),"atr_max":int(g["atr_period"].max()),"atr_unique":int(g["atr_period"].nunique()),
                            "mult_min":float(g["multiplier"].min()),"mult_max":float(g["multiplier"].max()),"mult_unique":int(g["multiplier"].nunique()),"modes":",".join(sorted(set(g["mode"].astype(str))))})
            else:
                rec.update({"ema_pairs":int(g[["ema_fast","ema_slow"]].drop_duplicates().shape[0]),"adx_configs":int(g[["adx_period","adx_threshold"]].drop_duplicates().shape[0]),
                            "regime_configs":int(g[["ema_fast","ema_slow","adx_period","adx_threshold","regime_separation_atr","regime_confirm_bars","regime_min_hold_bars"]].drop_duplicates().shape[0])})
            coverage.append(rec)

    # Base parameter clusters binned without using development holdout.
    clusters=[]
    if not rdf.empty:
        base=rdf[(rdf["strategy_type"]=="BASE")&(rdf["selection_score"]>-1e8)].copy()
        if not base.empty:
            base["atr_bucket"]=(base["atr_period"]//10)*10
            base["mult_bucket"]=(base["multiplier"]/0.25).round()*0.25
            for (tf,mode,ab,mb),g in base.groupby(["timeframe_min","mode","atr_bucket","mult_bucket"]):
                if len(g)<3: continue
                clusters.append({
                    "timeframe_min":tf,"mode":mode,"atr_bucket":ab,"mult_bucket":mb,"members":len(g),
                    "selection_positive_ratio":float((g["selection_score"]>0).mean()),
                    "validation_positive_ratio":float((g["validation_score"]>0).mean()),
                    "median_selection_score":float(g["selection_score"].median()),"median_validation_score":float(g["validation_score"].median()),
                    "median_validation_window_pct":float(g["validation_median_return_pct"].median()),
                    "development_holdout_positive_ratio":float((g["test_total_return_pct"]>0).mean()),
                    "median_development_holdout_pct":float(g["test_total_return_pct"].median()),
                })
    clusters_df=pd.DataFrame(clusters)
    if not clusters_df.empty:
        clusters_df=clusters_df.sort_values(["validation_positive_ratio","median_validation_score","members"],ascending=[False,False,False])

    # Neighbourhood stability around top research-ranked candidates.
    neighbourhood=[]
    for rank,row in enumerate(top_rows[:50],1):
        if row.strategy_type=="BASE":
            neigh=[r for r in all_rows if r.strategy_type=="BASE" and r.interval==row.interval and r.mode==row.mode and abs(r.atr_period-row.atr_period)<=3 and abs(r.multiplier-row.multiplier)<=0.25]
        else:
            neigh=[r for r in all_rows if r.strategy_type==row.strategy_type and r.interval==row.interval and abs(r.long_atr_period-row.long_atr_period)<=2 and abs(r.long_multiplier-row.long_multiplier)<=0.15 and abs(r.short_atr_period-row.short_atr_period)<=2 and abs(r.short_multiplier-row.short_multiplier)<=0.15]
        valid=[r for r in neigh if r.robust_score>-1e8]
        neighbourhood.append({
            "rank":rank,"strategy_type":row.strategy_type,"timeframe_min":row.interval,"mode":row.mode,"neighbour_count":len(valid),
            "selection_positive_ratio":sum(r.selection_score>0 for r in valid)/len(valid) if valid else 0,
            "validation_positive_ratio":sum(r.validation_score>0 for r in valid)/len(valid) if valid else 0,
            "development_holdout_positive_ratio":sum(r.test.total_return_pct>0 for r in valid)/len(valid) if valid else 0,
            "median_validation_score":float(np.median([r.validation_score for r in valid])) if valid else 0,
            "median_development_holdout_pct":float(np.median([r.test.total_return_pct for r in valid])) if valid else 0,
        })

    # Overfit / transfer flags for the most interesting research results.
    flags=[]
    if not rdf.empty:
        tmp=rdf[rdf["selection_score"]>-1e8].sort_values("selection_score",ascending=False).head(2000)
        for _,r in tmp.iterrows():
            reasons=[]
            if r["selection_score"]>0 and r["validation_score"]<=0: reasons.append("selection_positive_validation_nonpositive")
            if r["selection_median_return_pct"]>0 and r["validation_median_return_pct"]<0: reasons.append("median_sign_flip")
            if r["validation_positive_ratio"]<0.5: reasons.append("less_than_half_validation_windows_positive")
            if r["test_total_return_pct"]<0: reasons.append("development_holdout_negative")
            if reasons:
                flags.append({"strategy_type":r["strategy_type"],"timeframe_min":r["timeframe_min"],"mode":r["mode"],"stage":r["stage"],
                              "selection_score":r["selection_score"],"validation_score":r["validation_score"],"robust_score":r["robust_score"],
                              "development_holdout_pct":r["test_total_return_pct"],"flags":";".join(reasons)})

    # Adaptive V2 regime diagnostics for top 20 adaptive rows.
    regime_diag=[]; regime_samples=[]
    adaptive_top=sorted([r for r in all_rows if r.strategy_type=="ADAPTIVE_V2" and r.robust_score>-1e8], key=lambda r:r.robust_score, reverse=True)[:20]
    for rank,row in enumerate(adaptive_top,1):
        df=data_by_tf.get(row.interval)
        if df is None or df.empty: continue
        h=df["high"].to_numpy(np.float64); l=df["low"].to_numpy(np.float64); c=df["close"].to_numpy(np.float64)
        reg=online_regime_v2(h,l,c,row.ema_fast,row.ema_slow,row.adx_period,row.adx_threshold,row.regime_separation_atr,row.regime_confirm_bars,row.regime_min_hold_bars)
        plan=build_walk_forward_plan(df); vs=int(plan["validation_start"]); hs=int(plan["holdout_start"]); rs=int(plan["research_start"])
        parts=(("SELECTION",rs,max(rs,vs-1)),("VALIDATION",vs,max(vs,hs-1)),("DEVELOPMENT_HOLDOUT",hs,len(df)-1))
        for part,a,b in parts:
            arr=reg[a:b+1]; nn=max(1,len(arr))
            regime_diag.append({"rank":rank,"timeframe_min":row.interval,"part":part,"ema_fast":row.ema_fast,"ema_slow":row.ema_slow,
                "adx_period":row.adx_period,"adx_threshold":row.adx_threshold,"separation_atr":row.regime_separation_atr,"confirm_bars":row.regime_confirm_bars,"min_hold_bars":row.regime_min_hold_bars,
                "up_ratio":float(np.sum(arr==1)/nn),"down_ratio":float(np.sum(arr==-1)/nn),"flat_ratio":float(np.sum(arr==0)/nn),"switches":int(np.sum(arr[1:]!=arr[:-1])) if len(arr)>1 else 0})
        step=max(1,len(df)//400)
        for i in range(0,len(df),step):
            regime_samples.append({"rank":rank,"timeframe_min":row.interval,"datetime_utc":str(df["datetime"].iloc[i]),"close":float(c[i]),"state":"UP" if reg[i]==1 else ("DOWN" if reg[i]==-1 else "FLAT")})

    # Trade audit: overall leaders + dedicated Adaptive V2 leaders. This keeps
    # adaptive evidence in the report even when BASE occupies the entire TOP-100.
    trade_records=[]; fee=float((cfg or {}).get("fee",0) or 0); slip=float((cfg or {}).get("slippage",0) or 0)
    audit_rows=[]; seen=set()
    for rr in list(top_rows[:15]) + list(adaptive_top[:10]):
        kk=_result_key(rr)
        if kk not in seen:
            seen.add(kk); audit_rows.append(rr)
    for rank,row in enumerate(audit_rows,1):
        df=data_by_tf.get(row.interval)
        if df is None or df.empty: continue
        plan=build_walk_forward_plan(df); desired,_=build_selected_signal(df,row)
        rs=int(plan["research_start"]); vs=int(plan["validation_start"]); hs=int(plan["holdout_start"])
        parts=(("SELECTION",rs,max(rs,vs-1)),("VALIDATION",vs,max(vs,hs-1)),("DEVELOPMENT_HOLDOUT",hs,len(df)-1))
        for part,a,b in parts:
            _,trades,_=backtest_desired_with_trades(df,desired,a,b,fee,slip)
            for tr in trades:
                trade_records.append({"rank":rank,"strategy_type":row.strategy_type,"timeframe_min":row.interval,"mode":row.mode,"stage":row.stage,"part":part,
                    "side":tr.side,"entry_time_utc":str(tr.entry_time),"exit_time_utc":str(tr.exit_time),"entry_price":tr.entry_price,"exit_price":tr.exit_price,
                    "gross_return_pct":tr.gross_return_pct,"net_return_pct":tr.net_return_pct})

    valid_all=[r for r in all_rows if r.robust_score>-1e8]
    useful_total=int((research_summary or {}).get("committed_useful_checks",0) or 0)
    if useful_total <= 0:
        useful_total=sum(int(x.get("useful_backtest_evaluations",0) or 0) for x in search_stats)
    full_result_count=int((research_summary or {}).get("committed_result_rows",0) or len(all_rows))
    consistency={
        "full_disk_results":full_result_count,"retained_report_sample":len(all_rows),"top_results":len(top_rows),"nan_robust_scores_in_sample":sum(not math.isfinite(float(r.robust_score)) for r in all_rows),
        "candidate_expansion_uses_validation":False,"candidate_expansion_uses_development_holdout":False,"final_rank_uses_development_holdout":False,
        "adaptive_v2_uses_future":False,"signal_execution_rule":"signal after closed candle; execution at next open",
        "fresh_forward_test_available":False,
    }
    summary={
        "created_utc":datetime.now(timezone.utc).isoformat(),"application":APP_NAME,"version":VERSION,"config":_json_safe(cfg or {}),"search_seconds":round(float(search_seconds or 0),3),
        "all_unique_results":full_result_count,"retained_report_sample":len(all_rows),"valid_results_in_sample":len(valid_all),"top_results":len(top_rows),"useful_backtest_evaluations":useful_total,
        "strategy_type_counts":dict((research_summary or {}).get("strategy_type_counts",{})) or {k:sum(r.strategy_type==k for r in all_rows) for k in sorted(set(r.strategy_type for r in all_rows))},
        "research_storage":_json_safe(research_summary or {}),
        "development_holdout_positive_count":sum(r.test.total_return_pct>0 for r in valid_all),
        "strict_development_survivor_count":sum(r.train.total_return_pct>0 and r.test.total_return_pct>0 and r.test.profit_factor>1 and r.test.max_drawdown_pct<=20 and r.test.trades>=5 for r in valid_all),
        "search_stats":_json_safe(search_stats),"data_diagnostics":data_diag,"consistency_checks":consistency,
        "methodology":{
            "selection":"early chronological research windows; only phase allowed to expand search",
            "validation":"later chronological research windows; ranks completed candidates but cannot expand search",
            "development_holdout":"historical final block, excluded from search/ranking but already exposed by prior reports and therefore not a fresh test",
            "fresh_forward_test":"requires future candles collected after this version was designed",
        },
        "multiple_testing_warning":f"{full_result_count:,} stored strategies / {useful_total:,} evaluations: isolated winners are expected by chance; prefer broad neighbour stability and transfer from SELECTION to VALIDATION. Detailed CSV analyses use a retained representative sample because the full raw stream is disk-backed.",
        "system":{"platform":platform.platform(),"python":sys.version,"python_executable":sys.executable,"cpu_logical":os.cpu_count(),"numpy":np.__version__,"pandas":pd.__version__,"requests":getattr(requests,"__version__","unknown"),"numba_available":NUMBA_AVAILABLE},
        "known_model_limits":[
            "Funding is not included.","Closed-trade drawdown is used; intrabar floating drawdown is not modelled.","Liquidation and margin tiers are not modelled.",
            "The historical development holdout has been inspected in previous development cycles and is not a fresh forward test.",
            "Diagnostic 21-day UP/DOWN/FLAT labels use completed windows only for analysis; live ADAPTIVE_V2 uses past-only EMA/ADX/hysteresis.",
        ],
    }

    readme=(
        f"BYBIT SUPERTREND LAB v{VERSION} - FULL AUDIT REPORT\n\n"
        "Start with REPORT/diagnostic_summary.json, REPORT/strategy_type_summary.csv, REPORT/parameter_coverage.csv, "
        "REPORT/base_cluster_landscape.csv, REPORT/neighbourhood_stability_top50.csv and REPORT/development_holdout_survivors.csv.\n\n"
        "IMPORTANT: SELECTION expands the search; VALIDATION does not. DEVELOPMENT_HOLDOUT is excluded from ranking but has been observed in prior reports, so a truly fresh test requires future candles.\n"
    )
    report_index={
        "diagnostic_summary.json":"main methodology/search/data/system summary",
        "strategy_type_summary.csv":"BASE vs ADAPTIVE_V2 transfer by timeframe",
        "parameter_coverage.csv":"how far the search went in width/depth and by stage",
        "base_cluster_landscape.csv":"binned BASE parameter regions with selection/validation/development statistics",
        "neighbourhood_stability_top50.csv":"local neighbour stability around research leaders",
        "development_holdout_survivors.csv":"development-only survivors; NOT a fresh forward test",
        "overfit_flags_top_selection.csv":"high-selection candidates that deteriorate later",
        "selection_validation_holdout_plan.csv":"exact chronology and which windows may expand search",
        "adaptive_v2_regime_diagnostics_top20.csv":"EMA/ADX/hysteresis state mix and switch counts",
        "trades_top20.csv":"trade-level audit split into selection/validation/development holdout",
        "fresh_forward_test_status.json":"explicit statement that truly unseen future data is still needed",
        "research_storage_summary.json":"checkpoint/chunk counts and location of the full disk-backed result store",
        "research_stage_counts.csv":"exact durable counts by timeframe/stage from the research store",
    }

    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr("REPORT/README_REPORT.txt",readme)
        z.writestr("REPORT/report_index.json",json.dumps(report_index,ensure_ascii=False,indent=2))
        z.writestr("REPORT/diagnostic_summary.json",json.dumps(summary,ensure_ascii=False,indent=2))
        z.writestr("REPORT/top100_snapshot.json",json.dumps(top_snapshot,ensure_ascii=False,indent=2))
        z.writestr("REPORT/search_stats.json",json.dumps(_json_safe(search_stats),ensure_ascii=False,indent=2))
        z.writestr("REPORT/data_diagnostics.json",json.dumps(data_diag,ensure_ascii=False,indent=2))
        z.writestr("REPORT/consistency_checks.json",json.dumps(consistency,ensure_ascii=False,indent=2))
        z.writestr("REPORT/fresh_forward_test_status.json",json.dumps({"fresh_forward_test_available":False,"reason":"historical final block was inspected in earlier development reports","next_valid_test":"new candles collected after v0.7.x design"},ensure_ascii=False,indent=2))
        if research_summary:
            z.writestr("REPORT/research_storage_summary.json",json.dumps(_json_safe(research_summary),ensure_ascii=False,indent=2))
            stage_counts=(research_summary or {}).get("stage_counts",{}) or {}
            if stage_counts:
                z.writestr("REPORT/research_stage_counts.csv",pd.DataFrame([{"timeframe_stage":k,"stored_rows":v} for k,v in sorted(stage_counts.items())]).to_csv(index=False))
            for src_key,arc_name in [("checkpoint_path","RESEARCH/checkpoint.json"),("index_path","RESEARCH/research_index.json")]:
                fp=Path(str(research_summary.get(src_key,""))) if research_summary.get(src_key) else None
                if fp and fp.exists() and fp.is_file(): z.write(fp,arc_name)
            session_dir=Path(str(research_summary.get("session_dir",""))) if research_summary.get("session_dir") else None
            if session_dir:
                rf=session_dir/"README_RESEARCH.txt"
                if rf.exists(): z.write(rf,"RESEARCH/README_RESEARCH.txt")
        if strategy_summary: z.writestr("REPORT/strategy_type_summary.csv",pd.DataFrame(strategy_summary).to_csv(index=False))
        if coverage: z.writestr("REPORT/parameter_coverage.csv",pd.DataFrame(coverage).to_csv(index=False))
        if plan_rows: z.writestr("REPORT/selection_validation_holdout_plan.csv",pd.DataFrame(plan_rows).to_csv(index=False))
        z.writestr("REPORT/development_holdout_survivors.csv",survivors.to_csv(index=False))
        if not clusters_df.empty: z.writestr("REPORT/base_cluster_landscape.csv",clusters_df.to_csv(index=False))
        if neighbourhood: z.writestr("REPORT/neighbourhood_stability_top50.csv",pd.DataFrame(neighbourhood).to_csv(index=False))
        if flags: z.writestr("REPORT/overfit_flags_top_selection.csv",pd.DataFrame(flags).to_csv(index=False))
        if regime_diag: z.writestr("REPORT/adaptive_v2_regime_diagnostics_top20.csv",pd.DataFrame(regime_diag).to_csv(index=False))
        if regime_samples: z.writestr("REPORT/adaptive_v2_regime_samples_top20.csv",pd.DataFrame(regime_samples).to_csv(index=False))
        if trade_records: z.writestr("REPORT/trades_top20.csv",pd.DataFrame(trade_records).to_csv(index=False))
        if not rdf.empty:
            base_ranked=rdf[(rdf["strategy_type"]=="BASE")&(rdf["robust_score"]>-1e8)].sort_values("robust_score",ascending=False).head(200)
            ad_ranked=rdf[(rdf["strategy_type"]=="ADAPTIVE_V2")&(rdf["robust_score"]>-1e8)].sort_values("robust_score",ascending=False).head(200)
            val_ranked=rdf[rdf["validation_score"]>-1e8].sort_values("validation_score",ascending=False).head(200)
            z.writestr("REPORT/top200_base_research.csv",base_ranked.to_csv(index=False)); z.writestr("REPORT/top200_adaptive_v2_research.csv",ad_ranked.to_csv(index=False)); z.writestr("REPORT/top200_validation.csv",val_ranked.to_csv(index=False))
        if not topdf.empty: z.writestr("REPORT/top100_full.csv",topdf.to_csv(index=False))
        for name in ["BYBIT_SUPERTREND_LAB.py","START.bat","SELF_TEST.py","SELF_TEST.bat","DIAGNOSE.py","DIAGNOSE.bat","requirements.txt","README.txt"]:
            fp=BASE_DIR/name
            if fp.exists() and fp.is_file(): z.write(fp,f"PROGRAM/{name}")
        for folder,arc in [(LOGS_DIR,"LOGS"),(RESULTS_DIR,"RESULTS")]:
            for fp in sorted(folder.glob("*")):
                if fp.is_file(): z.write(fp,f"{arc}/{fp.name}")
        for fp in sorted(DATA_DIR.glob("*.csv")):
            if fp.is_file():
                z.write(fp,f"DATA/{fp.name}")
                try: z.writestr(f"DATA_HASHES/{fp.name}.sha256.txt",_sha256_file(fp)+"  "+fp.name+"\n")
                except Exception as exc: z.writestr(f"DATA_HASHES/{fp.name}.error.txt",str(exc))
    file_log(f"CHATGPT report created: {out}")
    return out


class PlotWindow(tk.Toplevel):
    def __init__(self, master, title: str, fig: Figure):
        super().__init__(master)
        self.title(title)
        self.geometry("1050x720")
        self.minsize(780, 540)
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas


class ScrollableFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, style="Panel.TFrame")
        self.canvas = tk.Canvas(self, bg="#151c25", highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style="Panel.TFrame")
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.canvas.bind_all("<MouseWheel>", self._on_wheel, add="+")

    def _on_inner(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_wheel(self, event):
        try:
            x, y = self.winfo_pointerxy()
            widget = self.winfo_containing(x, y)
            if widget is not None and str(widget).startswith(str(self)):
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} v{VERSION}")
        self.geometry("1420x860")
        self.minsize(900, 620)
        self.configure(bg="#0b0f14")

        self.msg_queue: queue.Queue = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.data_by_tf: dict[str, pd.DataFrame] = {}
        self.loaded_key = None
        self.all_results: list[AutoResult] = []
        self.top_results: list[AutoResult] = []
        self.selected_row: Optional[AutoResult] = None
        self.last_config = {}
        self.cpu_count = max(1, os.cpu_count() or 1)
        self.is_busy = False
        self.busy_started_at = 0.0
        self.last_ui_activity = time.monotonic()
        self.search_started_at = 0.0
        self.last_search_seconds = 0.0
        self.search_stats: list[dict] = []
        self.last_report_path: Optional[Path] = None
        self.last_research_summary: dict = {}
        self.status_phase = "ГОТОВО"
        self.status_checked = 0
        self.status_total = 0
        self.status_speed = 0.0
        self.status_eta = "—"
        self.status_workers = 0

        file_log(f"START {APP_NAME} v{VERSION} | base={BASE_DIR} | python={os.sys.executable}")
        self._setup_style()
        self._build_ui()
        self.after(100, self._drain_queue)
        self.after(500, self._heartbeat)
        self.after(250, lambda: self._log(f"Запущена версия {VERSION} · журнал: {RUNTIME_LOG}"))

    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background="#0b0f14", foreground="#e7edf5", font=("Segoe UI", 10))
        style.configure("TFrame", background="#0b0f14")
        style.configure("Panel.TFrame", background="#151c25")
        style.configure("TLabel", background="#0b0f14", foreground="#dbe5ef")
        style.configure("Panel.TLabel", background="#151c25", foreground="#dbe5ef")
        style.configure("Muted.TLabel", background="#0b0f14", foreground="#91a0b2")
        style.configure("BigStatus.TLabel", background="#151c25", foreground="#ffffff", font=("Segoe UI Semibold", 12))
        style.configure("Title.TLabel", background="#0b0f14", foreground="#ffffff", font=("Segoe UI Semibold", 20))
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 11), padding=(12, 10))
        style.configure("Danger.TButton", font=("Segoe UI", 10), padding=(10, 8))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 7))
        style.configure("TEntry", fieldbackground="#0e141b", foreground="#ffffff", insertcolor="#ffffff", padding=6)
        style.configure("TCombobox", fieldbackground="#0e141b", foreground="#ffffff", padding=5)
        style.configure("TCheckbutton", background="#151c25", foreground="#dbe5ef")
        style.configure("Treeview", background="#111821", fieldbackground="#111821", foreground="#e7edf5", rowheight=27, borderwidth=0)
        style.configure("Treeview.Heading", background="#263242", foreground="#ffffff", font=("Segoe UI Semibold", 9), relief="flat")
        style.map("Treeview", background=[("selected", "#314c70")], foreground=[("selected", "#ffffff")])
        style.configure("Horizontal.TProgressbar", troughcolor="#202a36", background="#4b8edb")
        style.configure("TLabelframe", background="#151c25", foreground="#ffffff")
        style.configure("TLabelframe.Label", background="#151c25", foreground="#ffffff", font=("Segoe UI Semibold", 10))

    def _build_ui(self):
        # Grid keeps the status panel visible at every window size.
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 6))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="BYBIT SUPERTREND LAB", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="selection → validation · поиск в ширину и глубину · Adaptive V2", style="Muted.TLabel").grid(row=0, column=1, sticky="w", padx=(10, 0), pady=(7, 0))
        ttk.Label(header, text="БЭКТЕСТ · ОРДЕРА НЕ ОТКРЫВАЕТ", style="Muted.TLabel").grid(row=0, column=2, sticky="e", pady=(7, 0))

        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 6))
        left_host = ttk.Frame(main, style="Panel.TFrame", width=370)
        right = ttk.Frame(main, style="Panel.TFrame")
        main.add(left_host, weight=0)
        main.add(right, weight=1)

        left_scroll = ScrollableFrame(left_host)
        left_scroll.pack(fill=tk.BOTH, expand=True)
        self._build_left(left_scroll.inner)
        self._build_right(right)
        self._build_bottom()

    def _build_left(self, parent):
        box = ttk.LabelFrame(parent, text="Автоматический поиск", padding=10)
        box.pack(fill=tk.X, padx=8, pady=(8, 5))

        self.symbol_var = tk.StringVar(value="BTCUSDT")
        self.months_var = tk.StringVar(value="18")
        self.fee_var = tk.StringVar(value="0.055")
        self.slippage_var = tk.StringVar(value="0.02")
        self.depth_var = tk.StringVar(value="Глубокий")
        self._labeled_entry(box, "Инструмент", self.symbol_var, 0)
        self._labeled_combo(box, "История, месяцев", self.months_var, ["6", "12", "18", "24", "36"], 1)

        ttk.Label(box, text="РЕЖИМ ПОИСКА", style="Panel.TLabel").grid(row=2, column=0, sticky="nw", padx=12, pady=(7, 4))
        depth_frame = ttk.Frame(box, style="Panel.TFrame")
        depth_frame.grid(row=2, column=1, sticky="ew", padx=12, pady=(4, 2))
        for i, (value, caption) in enumerate((
            ("Стандартный", "СТАНДАРТНЫЙ"),
            ("Глубокий", "ГЛУБОКИЙ"),
            ("Максимальный", "МАКСИМАЛЬНЫЙ"),
        )):
            ttk.Radiobutton(depth_frame, text=caption, variable=self.depth_var, value=value).grid(row=i, column=0, sticky="w", pady=2)
        ttk.Label(
            box, text="Глубокий — рекомендуемый. LONG / SHORT / BOTH программа проверяет сама.",
            style="Muted.TLabel", wraplength=210, justify=tk.LEFT,
        ).grid(row=3, column=1, sticky="w", padx=12, pady=(0, 6))

        ttk.Label(box, text="Таймфреймы", style="Panel.TLabel").grid(row=4, column=0, sticky="nw", padx=12, pady=6)
        tf_frame = ttk.Frame(box, style="Panel.TFrame")
        tf_frame.grid(row=4, column=1, sticky="ew", padx=12, pady=4)
        self.tf_vars = {}
        defaults = {"5", "15", "30", "60"}
        for i, tf in enumerate(AUTO_INTERVALS):
            var = tk.BooleanVar(value=tf in defaults)
            self.tf_vars[tf] = var
            ttk.Checkbutton(tf_frame, text=f"{tf}м", variable=var).grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 12), pady=2)

        self._labeled_entry(box, "Комиссия / сторона, %", self.fee_var, 5)
        self._labeled_entry(box, "Проскальзывание / сторона, %", self.slippage_var, 6)
        box.columnconfigure(1, weight=1)

        research = ttk.LabelFrame(parent, text="Новая логика проверки", padding=9)
        research.pack(fill=tk.X, padx=8, pady=5)
        ttk.Label(
            research,
            text=(
                "• Последний участок истории — финальный HOLDOUT. Он не влияет на выбор.\n"
                "• До него идут последовательные walk-forward окна по 21 дню.\n"
                "• Обычный Supertrend и адаптивная версия сравниваются одновременно.\n"
                "• Адаптивная версия: свой Supertrend для роста и падения; во флэте — пауза.\n"
                "• Режим рынка определяется только по прошлым закрытым свечам.\n"
                "• Отчёт отдельно показывает выживших на HOLDOUT и устойчивость соседних параметров."
            ),
            style="Panel.TLabel", justify=tk.LEFT, wraplength=315,
        ).pack(anchor="w")

        cpu_box = ttk.LabelFrame(parent, text="Процессор", padding=9)
        cpu_box.pack(fill=tk.X, padx=8, pady=5)
        auto_workers = max(1, min(32, self.cpu_count - 1 if self.cpu_count > 2 else self.cpu_count))
        self.workers_var = tk.StringVar(value=str(auto_workers))
        ttk.Label(cpu_box, text=f"Windows видит: {self.cpu_count} логических потоков", style="Panel.TLabel").pack(anchor="w")
        ttk.Label(cpu_box, text=f"Автоматически будет использовано: {auto_workers} процессов", style="Panel.TLabel").pack(anchor="w", pady=(2, 0))
        ttk.Label(cpu_box, text=f"Ускорение Numba: {'ВКЛЮЧЕНО' if NUMBA_AVAILABLE else 'НЕТ (будет заметно медленнее)'}", style="Panel.TLabel").pack(anchor="w", pady=(2, 0))

        action = ttk.Frame(parent, style="Panel.TFrame")
        action.pack(fill=tk.X, padx=8, pady=(7, 5))
        self.download_btn = ttk.Button(action, text="1. СКАЧАТЬ ИСТОРИЮ", command=self._on_download_click, style="Accent.TButton")
        self.download_btn.pack(fill=tk.X)
        self.auto_btn = ttk.Button(action, text="2. ЗАПУСТИТЬ ИССЛЕДОВАНИЕ", command=self.start_auto_search, style="Accent.TButton")
        self.auto_btn.pack(fill=tk.X, pady=(7, 0))
        self.report_btn = ttk.Button(action, text="3. СОЗДАТЬ ОТЧЁТ ДЛЯ CHATGPT", command=self.start_report_export)
        self.report_btn.pack(fill=tk.X, pady=(7, 0))
        self.stop_btn = ttk.Button(action, text="ОСТАНОВИТЬ", command=self.request_stop, state=tk.DISABLED, style="Danger.TButton")
        self.stop_btn.pack(fill=tk.X, pady=(7, 0))

        hint = ttk.LabelFrame(parent, text="Какую глубину выбирать", padding=8)
        hint.pack(fill=tk.X, padx=8, pady=(5, 10))
        ttk.Label(
            hint,
            text=(
                "Стандартный — быстрая проверка.\n"
                "Глубокий — основной режим, рассчитан на большой поиск.\n"
                "Максимальный — значительно тяжелее; имеет смысл после того, как Глубокий покажет устойчивые области."
            ), style="Panel.TLabel", wraplength=315, justify=tk.LEFT,
        ).pack(anchor="w")

    def _build_right(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)
        top = ttk.Frame(parent, style="Panel.TFrame")
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 5))
        ttk.Label(top, text="ТОП-100: research-рейтинг BASE + ADAPTIVE V2", style="Panel.TLabel").pack(side=tk.LEFT)
        ttk.Button(top, text="Тепловая карта", command=self.show_heatmap).pack(side=tk.RIGHT)
        ttk.Button(top, text="Кривая капитала", command=self.show_equity).pack(side=tk.RIGHT, padx=(0, 5))
        ttk.Button(top, text="Сделки", command=self.show_trades).pack(side=tk.RIGHT, padx=(0, 5))

        table_host = ttk.Frame(parent, style="Panel.TFrame")
        table_host.grid(row=1, column=0, sticky="nsew", padx=8)
        table_host.rowconfigure(0, weight=1)
        table_host.columnconfigure(0, weight=1)
        cols = ("rank", "tf", "mode", "atr", "mult", "sel", "val", "score", "wf", "median", "research", "holdout", "pf", "dd", "trades")
        self.tree = ttk.Treeview(table_host, columns=cols, show="headings", selectmode="browse")
        labels = {
            "rank":"#", "tf":"TF", "mode":"Режим", "atr":"ATR", "mult":"Множ.", "sel":"Selection", "val":"Validation", "score":"Итог",
            "wf":"+ окон", "median":"Медиана окна %", "research":"Research %", "holdout":"Dev holdout %",
            "pf":"PF", "dd":"Просадка %", "trades":"Сделок"
        }
        widths = {"rank":42,"tf":50,"mode":80,"atr":60,"mult":70,"sel":78,"val":78,"score":78,"wf":74,"median":105,"research":82,"holdout":98,"pf":58,"dd":82,"trades":70}
        for c in cols:
            self.tree.heading(c, text=labels[c])
            self.tree.column(c, width=widths[c], anchor=tk.CENTER, stretch=False)
        yscroll = ttk.Scrollbar(table_host, orient=tk.VERTICAL, command=self.tree.yview)
        xscroll = ttk.Scrollbar(table_host, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        detail = ttk.LabelFrame(parent, text="Подробности выбранного варианта", padding=7)
        detail.grid(row=2, column=0, sticky="ew", padx=8, pady=(5, 8))
        self.detail_text = tk.Text(detail, height=7, bg="#0e141b", fg="#dce7f3", insertbackground="#ffffff", relief=tk.FLAT, font=("Consolas", 9), wrap=tk.WORD)
        self.detail_text.pack(fill=tk.X)
        self.detail_text.insert("1.0", "После исследования здесь появятся SELECTION, VALIDATION, development holdout и результаты по режимам рынка.")
        self.detail_text.configure(state=tk.DISABLED)

    def _build_bottom(self):
        # Fixed bottom row: it never disappears behind the main content.
        panel = ttk.Frame(self, style="Panel.TFrame")
        panel.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        panel.columnconfigure(0, weight=1)
        self.big_status_var = tk.StringVar(value="ГОТОВО · выберите параметры и запустите исследование")
        ttk.Label(panel, textvariable=self.big_status_var, style="BigStatus.TLabel", anchor="w").grid(row=0, column=0, sticky="ew", padx=9, pady=(7, 4))
        self.progress = ttk.Progressbar(panel, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, sticky="ew", padx=9)

        stat = ttk.Frame(panel, style="Panel.TFrame")
        stat.grid(row=2, column=0, sticky="ew", padx=9, pady=(4, 2))
        for i in range(6):
            stat.columnconfigure(i, weight=1)
        self.stage_var = tk.StringVar(value="Этап: —")
        self.checked_var = tk.StringVar(value="Проверок: 0")
        self.speed_var = tk.StringVar(value="Скорость: —")
        self.elapsed_var = tk.StringVar(value="Время: 00:00")
        self.eta_var = tk.StringVar(value="Осталось: —")
        self.cpu_var = tk.StringVar(value="CPU: —")
        for i, var in enumerate([self.stage_var, self.checked_var, self.speed_var, self.elapsed_var, self.eta_var, self.cpu_var]):
            ttk.Label(stat, textvariable=var, style="Panel.TLabel", anchor="w").grid(row=0, column=i, sticky="ew", padx=(0, 8))

        info = ttk.Frame(panel, style="Panel.TFrame")
        info.grid(row=3, column=0, sticky="ew", padx=9, pady=(1, 2))
        info.columnconfigure(0, weight=1)
        info.columnconfigure(1, weight=1)
        self.data_var = tk.StringVar(value="История ещё не загружена")
        self.counter_var = tk.StringVar(value="")
        ttk.Label(info, textvariable=self.data_var, style="Panel.TLabel", anchor="w").grid(row=0, column=0, sticky="ew")
        ttk.Label(info, textvariable=self.counter_var, style="Panel.TLabel", anchor="e").grid(row=0, column=1, sticky="ew")

        self.log_text = tk.Text(panel, height=4, bg="#0b1016", fg="#aebdcd", relief=tk.FLAT, font=("Consolas", 8), wrap=tk.WORD)
        self.log_text.grid(row=4, column=0, sticky="ew", padx=9, pady=(2, 8))
        self.log_text.configure(state=tk.DISABLED)

    def _labeled_entry(self, parent, label, variable, row):
        ttk.Label(parent, text=label, style="Panel.TLabel").grid(row=row, column=0, sticky="w", padx=12, pady=5)
        ttk.Entry(parent, textvariable=variable, width=16).grid(row=row, column=1, sticky="ew", padx=12, pady=5)

    def _labeled_combo(self, parent, label, variable, values, row):
        ttk.Label(parent, text=label, style="Panel.TLabel").grid(row=row, column=0, sticky="w", padx=12, pady=5)
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=14).grid(row=row, column=1, sticky="ew", padx=12, pady=5)

    def _selected_tfs(self):
        return [tf for tf in AUTO_INTERVALS if self.tf_vars[tf].get()]

    def _read_config(self):
        symbol = self.symbol_var.get().upper().strip()
        if not symbol:
            raise ValueError("Укажите инструмент, например BTCUSDT")
        months = int(self.months_var.get())
        tfs = self._selected_tfs()
        if not tfs:
            raise ValueError("Выберите хотя бы один таймфрейм")
        fee = float(self.fee_var.get().replace(",", ".")) / 100.0
        slippage = float(self.slippage_var.get().replace(",", ".")) / 100.0
        if not (0 <= fee <= 0.02):
            raise ValueError("Комиссия выглядит ошибочной. Например: 0.055")
        if not (0 <= slippage <= 0.02):
            raise ValueError("Проскальзывание выглядит ошибочным. Например: 0.02")
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=round(months * 30.4375))
        workers = max(1, min(32, int(self.workers_var.get()), self.cpu_count))
        depth = self.depth_var.get().strip() if hasattr(self, "depth_var") else "Глубокий"
        return {
            "symbol": symbol, "months": months, "tfs": tfs, "fee": fee, "slippage": slippage,
            "start_dt": start_dt, "end_dt": end_dt, "workers": workers, "depth": depth,
        }

    def _config_key(self, cfg):
        return (cfg["symbol"], cfg["months"], tuple(cfg["tfs"]))

    def _set_busy(self, busy: bool):
        self.is_busy = busy
        if busy:
            self.busy_started_at = time.monotonic()
            self.last_ui_activity = time.monotonic()
            self.status_checked = 0
            self.status_total = 0
            self.status_speed = 0.0
            self.status_eta = "—"
            self.elapsed_var.set("Время: 00:00") if hasattr(self, "elapsed_var") else None
            try:
                self.progress.stop()
                self.progress.configure(mode="indeterminate")
                self.progress.start(12)
            except Exception:
                pass
        else:
            try:
                self.progress.stop()
                self.progress.configure(mode="determinate")
            except Exception:
                pass
        self.download_btn.configure(state=tk.DISABLED if busy else tk.NORMAL)
        self.auto_btn.configure(state=tk.DISABLED if busy else tk.NORMAL)
        self.report_btn.configure(state=tk.DISABLED if busy else tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL if busy else tk.DISABLED)
        if not busy:
            self.download_btn.configure(text="1. СКАЧАТЬ ИСТОРИЮ")

    def _post(self, kind, *payload):
        try:
            file_log(f"QUEUE {kind}: " + " | ".join(str(x)[:500] for x in payload))
        except Exception:
            pass
        self.msg_queue.put((kind, payload))

    def _log(self, text: str):
        file_log(text)
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{stamp}] {text}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _drain_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                self.last_ui_activity = time.monotonic()
                try:
                    if kind == "status":
                        self.big_status_var.set(payload[0])
                        if hasattr(self, "stage_var"):
                            short_phase = str(payload[0]).split("·")[0].strip()
                            self.stage_var.set(f"Этап: {short_phase}")
                        if len(payload) > 1 and payload[1]:
                            self._log(payload[1])
                    elif kind == "progress":
                        pct = float(payload[0])
                        if pct > 0:
                            try:
                                self.progress.stop()
                                self.progress.configure(mode="determinate")
                            except Exception:
                                pass
                            self.progress["value"] = pct
                        if len(payload) > 1:
                            self.counter_var.set(payload[1])
                    elif kind == "search_status":
                        info = payload[0] if payload else {}
                        phase = str(info.get("phase", "ПОИСК"))
                        self.stage_var.set(f"Этап: {phase}")
                        checked = int(info.get("checked", 0) or 0)
                        total = int(info.get("total", 0) or 0)
                        self.status_checked = checked
                        self.status_total = total
                        if total > 0:
                            self.checked_var.set(f"Проверок: {checked:,} / {total:,}")
                        else:
                            self.checked_var.set(f"Проверок: {checked:,}")
                        speed = float(info.get("speed", 0.0) or 0.0)
                        self.status_speed = speed
                        self.speed_var.set(f"Скорость: {speed:,.0f}/с" if speed > 0 else "Скорость: —")
                        eta = info.get("eta")
                        self.status_eta = fmt_seconds(float(eta)) if eta is not None else "—"
                        self.eta_var.set(f"Осталось: {self.status_eta}")
                        workers = int(info.get("workers", 0) or 0)
                        self.status_workers = workers
                        self.cpu_var.set(f"CPU: {workers} проц.") if workers else self.cpu_var.set("CPU: —")
                        detail = str(info.get("detail", ""))
                        if detail:
                            self.counter_var.set(detail)
                    elif kind == "data_done":
                        self.data_by_tf, self.loaded_key = payload
                        self._after_data_loaded()
                    elif kind == "search_done":
                        self.all_results, self.top_results, paths = payload[:3]
                        if len(payload) > 3:
                            self.last_report_path = Path(payload[3])
                        if len(payload) > 4 and isinstance(payload[4], dict):
                            self.last_research_summary = payload[4]
                        self._after_search(paths)
                    elif kind == "report_done":
                        self.last_report_path = Path(payload[0])
                        self._set_busy(False)
                        self.progress["value"] = 100
                        self.big_status_var.set("ОТЧЁТ ДЛЯ CHATGPT ГОТОВ")
                        self.counter_var.set(self.last_report_path.name)
                        self._log(f"Отчёт для ChatGPT: {self.last_report_path}")
                        messagebox.showinfo("Отчёт готов", f"Создан один ZIP-файл:\n\n{self.last_report_path}\n\nПрикрепите его к сообщению в ChatGPT.")
                    elif kind == "error":
                        self._set_busy(False)
                        self.big_status_var.set("ОШИБКА · подробности записаны в logs")
                        self._log(payload[0])
                        err_path = LOGS_DIR / "last_error.txt"
                        try:
                            err_path.write_text(payload[0], encoding="utf-8")
                        except Exception:
                            pass
                        messagebox.showerror("Ошибка", payload[0])
                    elif kind == "stopped":
                        self._set_busy(False)
                        self.big_status_var.set("ОСТАНОВЛЕНО ПОЛЬЗОВАТЕЛЕМ")
                        self._log("Операция остановлена пользователем")
                except Exception as exc:
                    file_log(f"QUEUE HANDLER ERROR kind={kind}: {exc}\n{traceback.format_exc(limit=5)}")
        except queue.Empty:
            pass
        except Exception as exc:
            file_log(f"QUEUE DRAIN ERROR: {exc}\n{traceback.format_exc(limit=5)}")
        finally:
            try:
                self.after(100, self._drain_queue)
            except tk.TclError:
                pass

    def _heartbeat(self):
        try:
            if self.is_busy:
                elapsed = max(0.0, time.monotonic() - self.busy_started_at)
                self.elapsed_var.set(f"Время: {fmt_seconds(elapsed)}")
                worker_state = "работает" if (self.worker and self.worker.is_alive()) else "завершается"
                if self.status_speed > 0:
                    self.speed_var.set(f"Скорость: {self.status_speed:,.0f}/с")
                    self.eta_var.set(f"Осталось: {self.status_eta}")
                if time.monotonic() - self.last_ui_activity > 3.0:
                    self.counter_var.set(f"Операция {worker_state} · подробности: logs/runtime.log")
            self.after(500, self._heartbeat)
        except tk.TclError:
            pass
        except Exception as exc:
            file_log(f"HEARTBEAT ERROR: {exc}")
            try:
                self.after(1000, self._heartbeat)
            except Exception:
                pass

    def request_stop(self):
        self.stop_event.set()
        self.big_status_var.set("ОСТАНАВЛИВАЮ…")
        self._log("Запрошена остановка. Текущие процессы будут завершены после ближайшей контрольной точки.")

    def _download_all(self, cfg):
        data = {}
        tfs = cfg["tfs"]
        for idx, tf in enumerate(tfs):
            if self.stop_event.is_set():
                raise StopRequested()
            self._post("status", f"СКАЧИВАЮ {tf} МИН · {idx+1}/{len(tfs)}", f"Начинаю загрузку {cfg['symbol']} {tf}м")

            def cb(local_pct, msg, idx=idx, tf=tf):
                overall = ((idx + local_pct / 100.0) / len(tfs)) * 100.0
                self._post("progress", overall, f"Загрузка {tf}м · {local_pct:.1f}%")
                self._post("status", f"ЗАГРУЗКА BYBIT · {msg}")
                self._post("search_status", {
                    "phase": f"Загрузка {tf}м ({idx+1}/{len(tfs)})",
                    "checked": 0, "total": 0, "speed": 0, "eta": None, "workers": 0,
                    "detail": f"{local_pct:.1f}% · {msg}",
                })

            reused_df, reused_path = find_reusable_candles(cfg["symbol"], tf, cfg["start_dt"], cfg["end_dt"])
            if reused_df is not None:
                data[tf] = reused_df
                self._post("status", f"{tf} МИН · ИСПОЛЬЗУЮ СОХРАНЁННУЮ ИСТОРИЮ", f"{reused_path.name}: {len(reused_df):,} свечей · повторно не скачиваю")
                continue

            partial = DATA_DIR / f"{cfg['symbol']}_{tf}m_DOWNLOADING.csv"
            self._post("status", f"СКАЧИВАЮ {tf} МИН · создаю временный файл", f"Промежуточные данные: {partial.name}")
            df = download_bybit_klines(
                cfg["symbol"], tf, cfg["start_dt"], cfg["end_dt"], cb, self.stop_event, partial_path=partial
            )
            path = save_candles(df, cfg["symbol"], tf)
            try:
                partial.unlink(missing_ok=True)
            except Exception as exc:
                file_log(f"PARTIAL cleanup error {partial}: {exc}")
            data[tf] = df
            self._post("status", f"{tf} МИН ГОТОВО · {len(df):,} свечей", f"Сохранено: {path.name}")
        return data

    def _on_download_click(self):
        """GUI wrapper: a click must never fail silently."""
        file_log("CLICK download button")
        try:
            self.download_btn.configure(text="НАЖАТО · ЗАПУСКАЮ…")
            self.big_status_var.set("КНОПКА НАЖАТА · ПРОВЕРЯЮ ПАРАМЕТРЫ…")
            self.counter_var.set("Подготовка загрузки…")
            self.progress["value"] = 0
            self.update_idletasks()
            self.start_download_only()
        except Exception as exc:
            details = f"Ошибка обработчика кнопки: {exc}\n\n{traceback.format_exc(limit=8)}"
            file_log(details)
            try:
                self._set_busy(False)
                self.big_status_var.set("ОШИБКА ПРИ НАЖАТИИ КНОПКИ")
                self._log(details)
                messagebox.showerror("Ошибка запуска загрузки", details)
            except Exception:
                pass

    def start_download_only(self):
        file_log("ENTER start_download_only")
        if self.worker and self.worker.is_alive():
            self._log("Загрузка не запущена: уже выполняется другая операция")
            self.big_status_var.set("УЖЕ ВЫПОЛНЯЕТСЯ ДРУГАЯ ОПЕРАЦИЯ")
            return
        try:
            cfg = self._read_config()
        except Exception as exc:
            messagebox.showerror("Параметры", str(exc))
            return
        self.stop_event.clear()
        self._set_busy(True)
        self.progress["value"] = 0
        self.counter_var.set("")
        self.big_status_var.set("ПОДКЛЮЧАЮСЬ К BYBIT…")
        self.counter_var.set("Создаю рабочий поток загрузки…")
        self.update_idletasks()
        self._log(f"Загрузка истории {cfg['symbol']}: {', '.join(x+'м' for x in cfg['tfs'])}, {cfg['months']} мес.")

        def run():
            file_log("DOWNLOAD WORKER enter")
            try:
                data = self._download_all(cfg)
                self._post("data_done", data, self._config_key(cfg))
                file_log("DOWNLOAD WORKER success")
            except StopRequested:
                file_log("DOWNLOAD WORKER stopped")
                self._post("stopped")
            except Exception as exc:
                details = f"{exc}\n\n{traceback.format_exc(limit=8)}"
                file_log(f"DOWNLOAD WORKER error: {details}")
                self._post("error", details)
            finally:
                file_log("DOWNLOAD WORKER exit")

        self.worker = threading.Thread(target=run, daemon=True, name="BybitDownloadWorker")
        self.worker.start()
        file_log(f"THREAD started name={self.worker.name} alive={self.worker.is_alive()}")
        self.counter_var.set("Рабочий поток запущен · соединяюсь с api.bybit.com…")


    def _after_data_loaded(self):
        self._set_busy(False)
        self.progress.configure(mode="determinate")
        self.progress["value"] = 100
        total = sum(len(df) for df in self.data_by_tf.values())
        detail = " · ".join(f"{tf}м: {len(df):,}" for tf, df in sorted(self.data_by_tf.items(), key=lambda x: int(x[0])))
        self.data_var.set(f"Загружено {total:,} свечей · {detail}")
        self.counter_var.set("")
        self.big_status_var.set("ИСТОРИЯ ЗАГРУЖЕНА · можно запускать исследование")
        self.stage_var.set("Этап: история готова")
        self.checked_var.set(f"Свечей: {total:,}")
        self.speed_var.set("Скорость: —")
        self.eta_var.set("Осталось: —")
        self.cpu_var.set("CPU: —")
        self._log("Загрузка истории полностью завершена")

    def start_auto_search(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            cfg = self._read_config()
        except Exception as exc:
            messagebox.showerror("Параметры", str(exc))
            return

        self.stop_event.clear()
        self._set_busy(True)
        self.progress["value"] = 0
        self.tree.delete(*self.tree.get_children())
        self.all_results = []
        self.top_results = []
        self.selected_row = None
        self.last_config = cfg.copy()
        self.search_started_at = time.monotonic()
        self.last_search_seconds = 0.0
        self.search_stats = []
        need_download = self.loaded_key != self._config_key(cfg) or any(tf not in self.data_by_tf for tf in cfg["tfs"])
        self._log(f"{cfg['depth']} автопоиск: {cfg['symbol']} · TF {', '.join(cfg['tfs'])} · CPU процессов {cfg['workers']}")

        def run():
            persistence = None
            try:
                data = self.data_by_tf
                if need_download:
                    self._post("status", "СНАЧАЛА СКАЧИВАЮ НУЖНУЮ ИСТОРИЮ…", "Текущая история не совпадает с выбранными настройками")
                    data = self._download_all(cfg)
                    self.data_by_tf = data
                    self.loaded_key = self._config_key(cfg)
                if self.stop_event.is_set():
                    raise StopRequested()

                # Preflight: fail early with a useful explanation instead of a generic
                # "no results" after silently skipping every timeframe.
                preflight = []
                for tf in cfg["tfs"]:
                    if tf not in data:
                        preflight.append(f"{tf}м: данные отсутствуют")
                        continue
                    try:
                        plan = build_walk_forward_plan(data[tf])
                        preflight.append(
                            f"{tf}м: {len(data[tf]):,} свечей, {plan['total_days']:.1f} дней, "
                            f"WF {len(plan['windows'])}: SELECTION {len(plan['selection_windows'])} + VALIDATION {len(plan['validation_windows'])}; DEV HOLDOUT {plan['holdout_days']} дн."
                        )
                    except Exception as exc:
                        preflight.append(f"{tf}м: ошибка плана — {exc}")
                self._post("status", f"ПРЕДПРОВЕРКА · режим {cfg['depth']}", " | ".join(preflight))
                viable = []
                for tf in cfg["tfs"]:
                    try:
                        if tf in data and len(build_walk_forward_plan(data[tf])["windows"]) >= 4:
                            viable.append(tf)
                    except Exception:
                        pass
                if not viable:
                    raise RuntimeError(
                        "Ни один выбранный таймфрейм не имеет 4 walk-forward окон. "
                        "Проверьте длину истории и timestamp. Предпроверка: " + " | ".join(preflight)
                    )
                persistence = ResearchPersistence(cfg, data)
                self._post(
                    "status",
                    "ДИСКОВОЕ ХРАНИЛИЩЕ ГОТОВО" if not persistence.resumed else "ВОССТАНАВЛИВАЮ ПО CHECKPOINT",
                    f"{persistence.root.name} · сохранение каждые ~{CHECKPOINT_USEFUL_EVERY:,} полезных проверок",
                )
                all_rows = self._run_full_search(cfg, data, persistence)
                top = diversified_top(all_rows, 100)
                self.last_search_seconds = max(0.0, time.monotonic() - self.search_started_at)
                persistence.finalize({
                    "search_seconds": self.last_search_seconds,
                    "retained_report_rows": len(all_rows),
                    "top_rows": len(top),
                    "search_stats": self.search_stats,
                })
                research_summary = persistence.summary()
                paths = save_results(all_rows, top, cfg["symbol"], research_summary=research_summary)
                self.last_research_summary = research_summary
                self._post("status", "ПОИСК ГОТОВ · СОБИРАЮ ОТЧЁТ ДЛЯ CHATGPT…", "Полные сырые результаты остаются в research/; в ZIP кладу индекс, диагностику и лучшие выборки")
                report_path = create_chatgpt_report(cfg, self.search_stats, all_rows, top, data, self.last_search_seconds, research_summary=research_summary)
                self._post("search_done", all_rows, top, paths, str(report_path), research_summary)
            except StopRequested:
                if persistence is not None:
                    try:
                        persistence.safe_stop("user_stop")
                        self.last_research_summary = persistence.summary()
                    except Exception as stop_exc:
                        file_log(f"CHECKPOINT stop error: {stop_exc}")
                self._post("stopped")
            except Exception as exc:
                if persistence is not None:
                    try:
                        persistence.safe_stop(f"error:{type(exc).__name__}")
                        self.last_research_summary = persistence.summary()
                    except Exception as stop_exc:
                        file_log(f"CHECKPOINT error-save failed: {stop_exc}")
                self._post("error", f"{exc}\n\n{traceback.format_exc(limit=8)}")

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def _run_full_search(self, cfg, data, persistence: ResearchPersistence):
        depth = cfg.get("depth", "Глубокий")
        # v0.7 keeps v0.6 WIDTH+DEPTH logic, but the full result stream is now
        # disk-backed. Search expansion still uses SELECTION only; VALIDATION can
        # rank completed candidates but cannot tell the optimiser where to look.
        if depth == "Стандартный":
            coarse_periods = list(range(2, 181, 4))
            coarse_mults = [round(x, 3) for x in np.arange(0.4, 9.0001, 0.20)]
            fine_seed_count, fine_p_radius, fine_m_radius, fine_m_step = 28, 4, 0.40, 0.05
            cluster_seed_count, cluster_p_radius, cluster_m_radius, cluster_m_step = 12, 10, 0.70, 0.05
            micro_seed_count, micro_p_radius, micro_m_radius, micro_m_step = 8, 3, 0.12, 0.01
            component_limit = 5
        elif depth == "Максимальный":
            coarse_periods = list(range(2, 321, 2))
            coarse_mults = [round(x, 3) for x in np.arange(0.25, 14.0001, 0.05)]
            fine_seed_count, fine_p_radius, fine_m_radius, fine_m_step = 180, 8, 0.70, 0.015
            cluster_seed_count, cluster_p_radius, cluster_m_radius, cluster_m_step = 70, 24, 1.30, 0.015
            micro_seed_count, micro_p_radius, micro_m_radius, micro_m_step = 45, 7, 0.28, 0.0025
            component_limit = 12
        else:  # Глубокий
            coarse_periods = list(range(2, 242, 6))
            coarse_mults = [round(x, 3) for x in np.arange(0.30, 12.0001, 0.10)]
            fine_seed_count, fine_p_radius, fine_m_radius, fine_m_step = 30, 3, 0.30, 0.025
            cluster_seed_count, cluster_p_radius, cluster_m_radius, cluster_m_step = 12, 6, 0.45, 0.025
            micro_seed_count, micro_p_radius, micro_m_radius, micro_m_step = 8, 2, 0.08, 0.005
            component_limit = 4

        modes = ("BOTH", "LONG", "SHORT")
        total_tfs = len(cfg["tfs"])
        self.search_stats = []
        resume_base_useful = persistence.committed_useful
        estimated_useful_total = max(1, persistence.committed_useful)
        report_pool: list[AutoResult] = []
        if persistence.resumed:
            self._post(
                "status",
                "НАЙДЕН CHECKPOINT · ПРОДОЛЖАЮ ИССЛЕДОВАНИЕ",
                f"Уже сохранено {persistence.committed_useful:,} полезных проверок · {persistence.committed_rows:,} результатов · {persistence.cp.get('chunk_count',0)} пакетов",
            )

        # Coarse estimate improves the ETA before later stages are known.
        for tf in cfg["tfs"]:
            try:
                p0 = build_walk_forward_plan(data[tf])
                checks = len(p0["windows"]) + 4
                base_configs = sum(1 for p in coarse_periods if p < p0["holdout_start"]) * len(coarse_mults) * len(modes)
                estimated_useful_total += base_configs * checks
            except Exception:
                pass

        for tf_index, tf in enumerate(cfg["tfs"]):
            tf_started = time.monotonic()
            if self.stop_event.is_set():
                raise StopRequested()
            df = data[tf]
            n = len(df)
            plan = build_walk_forward_plan(df)
            wf_windows = plan["windows"]
            if len(wf_windows) < 4:
                self._post("status", f"ПРОПУСК {tf}м · мало walk-forward окон", f"{tf}м: найдено только {len(wf_windows)} окон")
                continue

            months = (df["datetime"].dt.year * 100 + df["datetime"].dt.month).to_numpy(np.int32)
            payload = {
                "open": df["open"].to_numpy(np.float64),
                "high": df["high"].to_numpy(np.float64),
                "low": df["low"].to_numpy(np.float64),
                "close": df["close"].to_numpy(np.float64),
                "months": months,
            }
            min_trades = max(24, int(cfg["months"] * 2.2))
            checks_per_config = len(wf_windows) + 4
            regime_counts = {
                "UP": sum(1 for w in wf_windows if w[2] == "UP"),
                "DOWN": sum(1 for w in wf_windows if w[2] == "DOWN"),
                "FLAT": sum(1 for w in wf_windows if w[2] == "FLAT"),
            }

            coarse_candidates = {
                (p, m, mode)
                for p in coarse_periods if p < plan["holdout_start"]
                for m in coarse_mults
                for mode in modes
            }

            # Neutral breadth probes, unchanged in principle from v0.6.
            breadth_candidates = set()
            if depth == "Стандартный":
                p_long = range(30, 141, 5); m_long = np.arange(1.5, 4.0001, 0.10)
                p_fast = range(2, 13); m_fast = np.arange(6.0, 10.0001, 0.10)
                p_edge = range(150, 241, 10); m_edge = np.arange(0.5, 10.0001, 0.50)
            elif depth == "Максимальный":
                p_long = range(20, 241); m_long = np.arange(1.0, 5.0001, 0.015)
                p_fast = range(2, 21); m_fast = np.arange(5.0, 12.5001, 0.015)
                p_edge = range(220, 401, 3); m_edge = np.arange(0.3, 14.0001, 0.10)
            else:
                p_long = range(30, 181, 2); m_long = np.arange(1.2, 4.5001, 0.05)
                p_fast = range(2, 19); m_fast = np.arange(5.5, 11.0001, 0.05)
                p_edge = range(180, 301, 5); m_edge = np.arange(0.5, 12.0001, 0.25)
            for p0 in p_long:
                if p0 < plan["holdout_start"]:
                    for m in m_long: breadth_candidates.add((p0, round(float(m), 4), "BOTH"))
            for p0 in p_fast:
                if p0 < plan["holdout_start"]:
                    for m in m_fast: breadth_candidates.add((p0, round(float(m), 4), "BOTH"))
            for p0 in p_edge:
                if p0 < plan["holdout_start"]:
                    for m in m_edge:
                        for mode in modes: breadth_candidates.add((p0, round(float(m), 4), mode))
            breadth_candidates -= coarse_candidates

            workers = min(cfg["workers"], max(1, len(set(p for p,_,_ in coarse_candidates | breadth_candidates))))
            self._post(
                "status",
                f"{tf}м · ДИСКОВЫЙ ПОИСК · {workers} процессов",
                f"Checkpoint каждые ~{CHECKPOINT_USEFUL_EVERY:,} проверок · в RAM только рабочий TOP · сохранено {persistence.committed_useful:,}",
            )

            stage_rows: dict[str, list[AutoResult]] = {}
            stage_counts: dict[str, int] = {}
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(payload,)) as executor:
                def run_stage(candidates, stage_name, stage_label, progress_from, progress_to):
                    nonlocal estimated_useful_total
                    candidates = set(candidates)
                    stage_configs = len(candidates)
                    stage_counts[stage_name] = stage_configs
                    estimated_useful_total += stage_configs * checks_per_config
                    if not candidates:
                        return []
                    if persistence.is_stage_complete(tf, stage_name):
                        retained = persistence.load_stage_seed(tf, stage_name)
                        meta = persistence.stage_meta(tf, stage_name)
                        self._post("status", f"{tf}м · {stage_label} · УЖЕ ГОТОВО", f"Checkpoint: {meta.get('candidate_count',stage_configs):,} вариантов · загружаю рабочий TOP")
                        return retained

                    grouped = group_specs(candidates)
                    retained = persistence.load_stage_seed(tf, stage_name)

                    # IMPORTANT: never submit the whole stage at once. A completed
                    # Future keeps its full result in Future._result until the
                    # Future itself is released. v0.7.0 kept every Future in one
                    # dict until stage end, so checkpointing cleared buffer_rows
                    # but did NOT free those result batches. On large cluster stages
                    # this could consume many GB and eventually kill a worker.
                    todo_items = []
                    already_configs = 0
                    for period, specs in grouped.items():
                        unit_id = str(period)
                        unit_configs = sum(len(modes0) for _mult, modes0 in specs)
                        if persistence.unit_done(tf, stage_name, unit_id):
                            already_configs += unit_configs
                        else:
                            todo_items.append((period, specs, unit_id, unit_configs))

                    stage_done_configs = already_configs
                    stage_t0 = time.monotonic()
                    max_inflight = max(2, workers * 2)
                    inflight = {}
                    item_iter = iter(todo_items)
                    exhausted = False
                    batches_since_gc = 0

                    def submit_more():
                        nonlocal exhausted
                        while not exhausted and len(inflight) < max_inflight:
                            try:
                                period, specs, unit_id, unit_configs = next(item_iter)
                            except StopIteration:
                                exhausted = True
                                break
                            task = (period, specs, plan, cfg["fee"], cfg["slippage"], min_trades, tf, stage_name)
                            inflight[executor.submit(_worker_eval_period, task)] = (unit_id, unit_configs)

                    submit_more()
                    while inflight:
                        if self.stop_event.is_set():
                            for f in list(inflight):
                                f.cancel()
                            raise StopRequested()
                        done, _pending = wait(tuple(inflight.keys()), return_when=FIRST_COMPLETED)
                        for fut in done:
                            unit_id, unit_configs = inflight.pop(fut)
                            try:
                                batch = fut.result()
                            except BrokenProcessPool as exc:
                                persistence.flush(reason=f"broken_pool_{tf}_{stage_name}", force=True)
                                persistence.save_partial_seed(tf, stage_name, retained)
                                file_log(f"BrokenProcessPool {tf}m/{stage_name}: {exc}; checkpoint={persistence.committed_useful:,}")
                                raise RuntimeError(
                                    "Один из расчётных процессов аварийно завершился. "
                                    f"Checkpoint безопасно сохранён на {persistence.committed_useful:,} полезных проверках. "
                                    "Перезапустите тот же поиск — программа продолжит с checkpoint. "
                                    "v0.7.1 ограничивает число одновременно находящихся в памяти пакетов, "
                                    "поэтому повторная ошибка значительно менее вероятна."
                                ) from exc

                            retained = _retain_search_rows(retained, batch, RETAIN_ROWS_PER_STAGE)
                            flushed = persistence.queue_unit(tf, stage_name, unit_id, batch, unit_configs * checks_per_config)
                            stage_done_configs += unit_configs
                            # Drop the batch reference immediately. Because this Future
                            # was popped from `inflight`, the Future and its _result can
                            # now be reclaimed after this loop iteration.
                            del batch
                            batches_since_gc += 1
                            if flushed:
                                persistence.save_partial_seed(tf, stage_name, retained)
                                self._post("status", f"CHECKPOINT СОХРАНЁН · {tf}м {stage_label}", f"На диск: {persistence.committed_useful:,} проверок · {persistence.committed_rows:,} результатов · RAM {_memory_percent():.1f}%")
                            ram = _memory_percent()
                            if ram >= 94.0:
                                persistence.flush(reason=f"memory_guard_stop_{ram:.1f}", force=True)
                                persistence.save_partial_seed(tf, stage_name, retained)
                                raise RuntimeError(
                                    f"Защитная остановка: занято {ram:.1f}% оперативной памяти. "
                                    "Checkpoint сохранён; повторный запуск продолжит исследование."
                                )
                            checked_now = persistence.committed_useful + persistence.buffer_useful
                            elapsed = max(0.001, time.monotonic() - self.search_started_at)
                            session_checked = max(0, checked_now - resume_base_useful)
                            speed = session_checked / elapsed if session_checked else 0.0
                            frac = stage_done_configs / max(1, stage_configs)
                            local = progress_from + frac * (progress_to - progress_from)
                            overall = (tf_index + local / 100.0) / total_tfs * 100.0
                            remain = max(0, estimated_useful_total - checked_now)
                            eta = remain / speed if speed > 0 else None
                            self._post("progress", overall, f"{tf}м · {stage_label}: {stage_done_configs:,}/{stage_configs:,}")
                            self._post("search_status", {
                                "phase": f"{tf}м · {stage_label}", "checked": checked_now,
                                "total": max(estimated_useful_total, checked_now), "speed": speed, "eta": eta, "workers": workers,
                                "detail": (
                                    f"Диск: {persistence.committed_useful:,} · буфер: {persistence.buffer_useful:,}/{CHECKPOINT_USEFUL_EVERY:,} · "
                                    f"RAM {ram:.1f}% · в полёте {len(inflight):,}/{max_inflight} · retained {len(retained):,} · "
                                    f"SELECTION {len(plan['selection_windows'])} / VALIDATION {len(plan['validation_windows'])}"
                                ),
                            })
                            if flushed or batches_since_gc >= 8:
                                gc.collect(); batches_since_gc = 0
                            fut = None
                        done.clear()
                        submit_more()

                    meta = {
                        "candidate_count": stage_configs,
                        "useful_checks": stage_configs * checks_per_config,
                        "retained_rows": len(retained),
                        "seconds_last_segment": round(time.monotonic()-stage_t0,3),
                    }
                    persistence.save_stage_seed(tf, stage_name, retained, meta)
                    return retained

                broad_rows = run_stage(coarse_candidates, "broad", "широкий базовый", 0.0, 18.0)
                breadth_rows = run_stage(breadth_candidates, "breadth", "поиск в ширину", 18.0, 30.0)
                stage_rows["broad"] = broad_rows; stage_rows["breadth"] = breadth_rows

                seed_source = broad_rows + breadth_rows
                seeds = choose_seed_rows(seed_source, fine_seed_count)
                fine_candidates = set()
                for seed in seeds:
                    for p0 in range(max(2, seed.atr_period - fine_p_radius), seed.atr_period + fine_p_radius + 1):
                        lo = max(0.15, seed.multiplier - fine_m_radius); hi = min(16.0, seed.multiplier + fine_m_radius)
                        m = lo
                        while m <= hi + 1e-12:
                            fine_candidates.add((p0, round(m, 4), seed.mode)); m += fine_m_step
                fine_candidates -= coarse_candidates; fine_candidates -= breadth_candidates
                fine_rows = run_stage(fine_candidates, "fine", "точный вокруг SELECTION-лидеров", 30.0, 48.0)
                stage_rows["fine"] = fine_rows

                source2 = seed_source + fine_rows
                cluster_seeds = choose_seed_rows(source2, cluster_seed_count)
                both_pool = sorted(
                    [r for r in source2 if r.strategy_type == "BASE" and r.mode == "BOTH" and r.selection_score > -1e8],
                    key=lambda r: r.selection_score, reverse=True,
                )
                for r in both_pool:
                    if len(cluster_seeds) >= cluster_seed_count + max(6, cluster_seed_count // 3): break
                    if not any(r.interval == x.interval and r.mode == x.mode and abs(r.atr_period-x.atr_period)<=6 and abs(r.multiplier-x.multiplier)<=0.35 for x in cluster_seeds):
                        cluster_seeds.append(r)
                cluster_candidates = set()
                for seed in cluster_seeds:
                    for p0 in range(max(2, seed.atr_period-cluster_p_radius), seed.atr_period+cluster_p_radius+1):
                        lo=max(0.12, seed.multiplier-cluster_m_radius); hi=min(18.0, seed.multiplier+cluster_m_radius); m=lo
                        while m <= hi + 1e-12:
                            cluster_candidates.add((p0, round(m,4), seed.mode)); m += cluster_m_step
                cluster_candidates -= coarse_candidates | breadth_candidates | fine_candidates
                cluster_rows = run_stage(cluster_candidates, "cluster_wide", "кластеры в ширину", 48.0, 68.0)
                stage_rows["cluster_wide"] = cluster_rows

                source3 = source2 + cluster_rows
                micro_seeds = choose_seed_rows(source3, micro_seed_count)
                micro_candidates = set()
                for seed in micro_seeds:
                    for p0 in range(max(2, seed.atr_period-micro_p_radius), seed.atr_period+micro_p_radius+1):
                        lo=max(0.10, seed.multiplier-micro_m_radius); hi=min(20.0, seed.multiplier+micro_m_radius); m=lo
                        while m <= hi + 1e-12:
                            micro_candidates.add((p0, round(m,5), seed.mode)); m += micro_m_step
                micro_candidates -= coarse_candidates | breadth_candidates | fine_candidates | cluster_candidates
                micro_rows = run_stage(micro_candidates, "cluster_deep", "кластеры в глубину", 68.0, 82.0)
                stage_rows["cluster_deep"] = micro_rows

            base_rows_tf = []
            for arr in stage_rows.values():
                base_rows_tf = _retain_search_rows(base_rows_tf, arr, RETAIN_ROWS_PER_STAGE * 2)

            # ----- Adaptive V2: EMA direction + ADX + ATR separation + hysteresis -----
            long_components = _adaptive_component_rows(base_rows_tf, "LONG", component_limit)
            short_components = _adaptive_component_rows(base_rows_tf, "SHORT", component_limit)
            if depth == "Стандартный":
                ema_pairs=[(8,21),(12,26),(20,50)]; adx_periods=[14]; adx_thresholds=[18,25]
                separations=[0.10,0.25]; confirms=[2]; holds=[3,6]
            elif depth == "Максимальный":
                ema_pairs=[(5,13),(6,18),(8,21),(12,26),(20,50),(30,80)]
                adx_periods=[8,10,14,20]; adx_thresholds=[12,15,18,20,25,30]
                separations=[0.0,0.08,0.15,0.25,0.40]; confirms=[2,3,4]; holds=[3,6,12]
            else:
                ema_pairs=[(6,18),(8,21),(12,26),(20,50)]
                adx_periods=[10,14]; adx_thresholds=[15,20,25]
                separations=[0.10,0.25]; confirms=[2,3]; holds=[4,8]
            regime_configs = [
                {"ema_fast":ef,"ema_slow":es,"adx_period":ap,"adx_threshold":at,"separation_atr":sep,"confirm_bars":cb,"min_hold_bars":hb}
                for ef,es in ema_pairs for ap in adx_periods for at in adx_thresholds for sep in separations for cb in confirms for hb in holds
                if ef < es
            ]
            adaptive_total = len(long_components) * len(short_components) * len(regime_configs)
            estimated_useful_total += adaptive_total * checks_per_config
            adaptive_stage = "adaptive_v2"
            adaptive_rows: list[AutoResult] = []
            if persistence.is_stage_complete(tf, adaptive_stage):
                adaptive_rows = persistence.load_stage_seed(tf, adaptive_stage)
                self._post("status", f"{tf}м · ADAPTIVE V2 · УЖЕ ГОТОВО", f"Загружаю retained {len(adaptive_rows):,} из checkpoint")
            elif long_components and short_components and adaptive_total:
                adaptive_rows = persistence.load_stage_seed(tf, adaptive_stage)
                regime_units = [regime_configs[i:i+ADAPTIVE_REGIME_CHUNK] for i in range(0,len(regime_configs),ADAPTIVE_REGIME_CHUNK)]
                adaptive_workers = min(cfg["workers"], max(1, len(regime_units)))
                self._post(
                    "status", f"{tf}м · ADAPTIVE V2 · {adaptive_workers} ПРОЦЕССОВ",
                    f"{len(long_components)} LONG × {len(short_components)} SHORT × {len(regime_configs)} режимов = {adaptive_total:,} · задачи дробятся по {ADAPTIVE_REGIME_CHUNK} режимов",
                )
                already_ad = 0
                adaptive_todo = []
                for idx, chunk in enumerate(regime_units):
                    unit_id = f"regime_chunk_{idx:05d}"
                    unit_configs = len(chunk) * len(long_components) * len(short_components)
                    if persistence.unit_done(tf, adaptive_stage, unit_id):
                        already_ad += unit_configs
                    else:
                        adaptive_todo.append((chunk, unit_id, unit_configs))

                with ProcessPoolExecutor(max_workers=adaptive_workers, initializer=_worker_init, initargs=(payload,)) as ad_executor:
                    done_ad = already_ad
                    max_inflight_ad = max(2, adaptive_workers * 2)
                    inflight = {}
                    item_iter = iter(adaptive_todo)
                    exhausted = False
                    batches_since_gc = 0

                    def submit_more_adaptive():
                        nonlocal exhausted
                        while not exhausted and len(inflight) < max_inflight_ad:
                            try:
                                chunk, unit_id, unit_configs = next(item_iter)
                            except StopIteration:
                                exhausted = True
                                break
                            task=(chunk, long_components, short_components, ema_pairs, adx_periods, plan, cfg["fee"], cfg["slippage"], min_trades, tf)
                            inflight[ad_executor.submit(_worker_eval_adaptive_v2_chunk, task)] = (unit_id, unit_configs)

                    submit_more_adaptive()
                    while inflight:
                        if self.stop_event.is_set():
                            for f in list(inflight): f.cancel()
                            raise StopRequested()
                        done, _pending = wait(tuple(inflight.keys()), return_when=FIRST_COMPLETED)
                        for fut in done:
                            unit_id, unit_configs = inflight.pop(fut)
                            try:
                                batch=fut.result()
                            except BrokenProcessPool as exc:
                                persistence.flush(reason=f"broken_pool_{tf}_{adaptive_stage}", force=True)
                                persistence.save_partial_seed(tf, adaptive_stage, adaptive_rows)
                                file_log(f"BrokenProcessPool {tf}m/{adaptive_stage}: {exc}; checkpoint={persistence.committed_useful:,}")
                                raise RuntimeError(
                                    "Один из процессов Adaptive V2 аварийно завершился. "
                                    f"Checkpoint сохранён на {persistence.committed_useful:,} полезных проверках. "
                                    "Перезапустите тот же поиск — незавершённые пакеты будут пересчитаны, сохранённые пропущены."
                                ) from exc
                            adaptive_rows = _retain_search_rows(adaptive_rows, batch, RETAIN_ROWS_PER_STAGE)
                            flushed=persistence.queue_unit(tf, adaptive_stage, unit_id, batch, unit_configs*checks_per_config)
                            done_ad += unit_configs
                            del batch
                            batches_since_gc += 1
                            if flushed:
                                persistence.save_partial_seed(tf, adaptive_stage, adaptive_rows)
                                self._post("status", f"CHECKPOINT СОХРАНЁН · {tf}м Adaptive V2", f"На диск: {persistence.committed_useful:,} проверок · RAM {_memory_percent():.1f}%")
                            ram=_memory_percent()
                            if ram>=94.0:
                                persistence.flush(reason=f"memory_guard_stop_{ram:.1f}",force=True); persistence.save_partial_seed(tf,adaptive_stage,adaptive_rows)
                                raise RuntimeError(f"Защитная остановка: занято {ram:.1f}% RAM. Checkpoint сохранён; перезапуск продолжит расчёт.")
                            checked_now=persistence.committed_useful+persistence.buffer_useful
                            elapsed=max(0.001,time.monotonic()-self.search_started_at); session_checked=max(0,checked_now-resume_base_useful); speed=session_checked/elapsed if session_checked else 0.0
                            frac=done_ad/max(1,adaptive_total); overall=(tf_index+(82+frac*18)/100.0)/total_tfs*100.0
                            remain=max(0,estimated_useful_total-checked_now); eta=remain/speed if speed>0 else None
                            self._post("progress",overall,f"{tf}м · Adaptive V2: {done_ad:,}/{adaptive_total:,}")
                            self._post("search_status",{
                                "phase":f"{tf}м · Adaptive V2","checked":checked_now,"total":max(estimated_useful_total,checked_now),
                                "speed":speed,"eta":eta,"workers":adaptive_workers,
                                "detail":f"Диск {persistence.committed_useful:,} · буфер {persistence.buffer_useful:,}/{CHECKPOINT_USEFUL_EVERY:,} · RAM {ram:.1f}% · в полёте {len(inflight):,}/{max_inflight_ad} · retained {len(adaptive_rows):,}",
                            })
                            if flushed or batches_since_gc >= 8:
                                gc.collect(); batches_since_gc = 0
                            fut = None
                        done.clear()
                        submit_more_adaptive()
                persistence.save_stage_seed(tf,adaptive_stage,adaptive_rows,{
                    "candidate_count":adaptive_total,"useful_checks":adaptive_total*checks_per_config,"retained_rows":len(adaptive_rows),
                    "regime_configs":len(regime_configs),"workers":adaptive_workers,
                })
            elif not persistence.is_stage_complete(tf, adaptive_stage):
                persistence.save_stage_seed(tf,adaptive_stage,adaptive_rows,{"candidate_count":0,"useful_checks":0,"retained_rows":0})

            stage_counts[adaptive_stage]=adaptive_total
            tf_rows=[]
            for arr in stage_rows.values():
                tf_rows=_retain_search_rows(tf_rows,arr,RETAIN_ROWS_PER_STAGE*3)
            tf_rows=_retain_search_rows(tf_rows,adaptive_rows,RETAIN_ROWS_PER_STAGE*3)
            report_pool=_retain_search_rows(report_pool,tf_rows,25000)

            # Use durable stage metadata for exact counts even after a resumed run.
            def meta_count(stage):
                m=persistence.stage_meta(tf,stage); return int(m.get("candidate_count",stage_counts.get(stage,0)) or 0)
            tf_candidate_count=sum(meta_count(st) for st in ["broad","breadth","fine","cluster_wide","cluster_deep","adaptive_v2"])
            tf_useful=tf_candidate_count*checks_per_config
            self.search_stats.append({
                "timeframe_min":tf,"candles":n,"workers":workers,"depth":depth,"numba":NUMBA_AVAILABLE,
                "walk_forward_windows":len(wf_windows),"selection_windows":len(plan["selection_windows"]),
                "validation_windows":len(plan["validation_windows"]),"walk_forward_test_days":plan["test_days"],
                "warmup_days":plan["warmup_days"],"development_holdout_days":plan["holdout_days"],"regime_windows":regime_counts,
                "broad_candidates":meta_count("broad"),"breadth_candidates":meta_count("breadth"),
                "fine_candidates":meta_count("fine"),"cluster_wide_candidates":meta_count("cluster_wide"),
                "cluster_deep_candidates":meta_count("cluster_deep"),
                "adaptive_v2_long_components":len(long_components),"adaptive_v2_short_components":len(short_components),
                "adaptive_v2_regime_configs":len(regime_configs),"adaptive_v2_candidates":meta_count("adaptive_v2"),
                "adaptive_v2_workers":min(cfg["workers"],max(1,math.ceil(len(regime_configs)/ADAPTIVE_REGIME_CHUNK))) if adaptive_total else 0,
                "evaluated_strategy_configs":tf_candidate_count,"useful_backtest_evaluations":tf_useful,
                "retained_in_memory":len(tf_rows),"seconds":round(time.monotonic()-tf_started,3),
                "disk_committed_useful":persistence.committed_useful,"disk_committed_rows":persistence.committed_rows,
            })
            self._post("status",f"{tf}м ГОТОВО · ВСЁ ЦЕННОЕ НА ДИСКЕ",f"{tf_candidate_count:,} стратегий · {tf_useful:,} полезных проверок · retained RAM {len(tf_rows):,} · chunks {persistence.cp.get('chunk_count',0)}")
            gc.collect()

        persistence.flush(reason="search_finished",force=True)
        if not report_pool:
            # Last-resort: gather retained seeds from completed stages.
            for tf in cfg["tfs"]:
                for st in ["broad","breadth","fine","cluster_wide","cluster_deep","adaptive_v2"]:
                    report_pool=_retain_search_rows(report_pool,persistence.load_stage_seed(tf,st),25000)
        if not report_pool:
            raise RuntimeError("Не удалось получить ни одного результата")
        report_pool.sort(key=lambda r:(r.robust_score,r.validation_score,r.selection_score),reverse=True)
        self._post("progress",100.0,"Поиск завершён · полные результаты сохранены в research/")
        return report_pool


    def _after_search(self, paths):
        self._set_busy(False)
        self.progress["value"] = 100
        all_path, top_path = paths
        self.big_status_var.set("ГОТОВО · ТОП-100 ПО SELECTION → VALIDATION СФОРМИРОВАН")
        total_useful = int((self.last_research_summary or {}).get("committed_useful_checks",0) or 0)
        if total_useful <= 0:
            total_useful = sum(int(x.get("useful_backtest_evaluations", 0) or 0) for x in self.search_stats)
        total_rows = int((self.last_research_summary or {}).get("committed_result_rows",0) or len(self.all_results))
        self.counter_var.set(f"На диске: {total_rows:,} стратегий · полезных проверок: {total_useful:,}")
        self._log(f"Рабочая выборка: {all_path.name}")
        self._log(f"ТОП-100: {top_path.name}")
        if self.last_research_summary:
            self._log(f"Полное хранилище: {self.last_research_summary.get('session_dir','research/')} · chunks {self.last_research_summary.get('chunk_count',0)}")
        if self.last_report_path:
            self._log(f"Полный отчёт для ChatGPT: {self.last_report_path.name}")
        total_candles = sum(len(df) for df in self.data_by_tf.values())
        self.data_var.set(f"Данные: {total_candles:,} свечей · SELECTION расширяет поиск · VALIDATION только проверяет · Dev holdout вне рейтинга")

        for iid in self.tree.get_children():
            self.tree.delete(iid)
        for rank, row in enumerate(self.top_results, start=1):
            t = row.test
            wf = f"{row.wf_positive_windows}/{row.wf_windows}"
            if row.strategy_type != "BASE":
                mode_text = "ADAPT.V2" if row.strategy_type == "ADAPTIVE_V2" else "АДАПТ."
                atr_text = f"L{row.long_atr_period}/S{row.short_atr_period}"
                mult_text = f"{row.long_multiplier:g}/{row.short_multiplier:g}"
            else:
                mode_text = row.mode
                atr_text = str(row.atr_period)
                mult_text = f"{row.multiplier:g}"
            self.tree.insert("", tk.END, iid=str(rank - 1), values=(
                rank, f"{row.interval}м", mode_text, atr_text, mult_text,
                f"{row.selection_score:.3f}", f"{row.validation_score:.3f}", f"{row.robust_score:.3f}",
                wf, f"{row.wf_median_return_pct:+.2f}",
                f"{row.train.total_return_pct:+.2f}", f"{t.total_return_pct:+.2f}",
                f"{t.profit_factor:.2f}", f"{t.max_drawdown_pct:.2f}", t.trades,
            ))
        if self.top_results:
            self.tree.selection_set("0")
            self.tree.focus("0")
            self.tree.see("0")
            self.on_tree_select()

    def on_tree_select(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx >= len(self.top_results):
            return
        row = self.top_results[idx]
        self.selected_row = row
        research, holdout = row.train, row.test
        regimes = (
            f"РОСТ: {row.up_windows} окон, прибыльных {row.up_positive_ratio*100:.0f}%, ср. {row.up_avg_return_pct:+.2f}% · "
            f"ПАДЕНИЕ: {row.down_windows} окон, прибыльных {row.down_positive_ratio*100:.0f}%, ср. {row.down_avg_return_pct:+.2f}% · "
            f"БОКОВИК: {row.flat_windows} окон, прибыльных {row.flat_positive_ratio*100:.0f}%, ср. {row.flat_avg_return_pct:+.2f}%"
        )
        if row.strategy_type != "BASE":
            if row.strategy_type == "ADAPTIVE_V2":
                regime_desc = (
                    f"EMA {row.ema_fast}/{row.ema_slow} · ADX{row.adx_period}>={row.adx_threshold:g} · "
                    f"sep={row.regime_separation_atr:g}ATR · confirm={row.regime_confirm_bars} · min-hold={row.regime_min_hold_bars}"
                )
            else:
                regime_desc = f"{row.regime_hours}ч / ±{row.regime_threshold_pct:g}%"
            strategy_line = (
                f"TF {row.interval} минут · {row.strategy_type} · "
                f"РОСТ: ATR={row.long_atr_period}, ×{row.long_multiplier:g} · "
                f"ПАДЕНИЕ: ATR={row.short_atr_period}, ×{row.short_multiplier:g} · {regime_desc} · "
                f"research score={row.robust_score:.3f}"
            )
            online_line = (
                f"Past-only классификатор: РОСТ {row.online_up_ratio*100:.1f}% · "
                f"ПАДЕНИЕ {row.online_down_ratio*100:.1f}% · БОКОВИК/ПАУЗА {row.online_flat_ratio*100:.1f}% · "
                f"переключений {row.regime_switches}\n"
            )
        else:
            strategy_line = f"TF {row.interval} минут · {row.mode} · ATR={row.atr_period} · множитель={row.multiplier:g} · research score={row.robust_score:.3f}"
            online_line = ""
        text = (
            strategy_line + "\n\n"
            f"SELECTION: score {row.selection_score:.3f} · окон {row.selection_windows} · + {row.selection_positive_ratio*100:.1f}% · медиана {row.selection_median_return_pct:+.2f}% · худшее {row.selection_worst_return_pct:+.2f}%\n"
            f"VALIDATION: score {row.validation_score:.3f} · окон {row.validation_windows} · + {row.validation_positive_ratio*100:.1f}% · медиана {row.validation_median_return_pct:+.2f}% · худшее {row.validation_worst_return_pct:+.2f}%\n"
            f"ВСЕ RESEARCH WF: прибыльных окон {row.wf_positive_windows}/{row.wf_windows} ({row.wf_positive_ratio*100:.1f}%) · медиана {row.wf_median_return_pct:+.2f}%\n"
            f"Research aggregate: сделок {research.trades} · результат {research.total_return_pct:+.2f}% · "
            f"PF {research.profit_factor:.2f} · просадка {research.max_drawdown_pct:.2f}%\n"
            f"{regimes}\n"
            f"{online_line}\n"
            f"DEVELOPMENT HOLDOUT — НЕ УЧАСТВУЕТ В РЕЙТИНГЕ, НО УЖЕ НЕ ЯВЛЯЕТСЯ НОВЫМ ЭКЗАМЕНОМ:\n"
            f"сделок {holdout.trades} · результат {holdout.total_return_pct:+.2f}% · PF {holdout.profit_factor:.2f} · "
            f"просадка {holdout.max_drawdown_pct:.2f}% · Win rate {holdout.win_rate:.1f}%\n\n"
            "Комиссия и проскальзывание учтены. Funding пока не учитывается. Новый честный forward-test потребует будущих свечей. Сигнал формируется после закрытия свечи, исполнение — на следующем open."
        )
        self.detail_text.configure(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert("1.0", text)
        self.detail_text.configure(state=tk.DISABLED)

    def _selected_calc(self):
        if self.selected_row is None:
            raise ValueError("Сначала выполните автопоиск и выберите строку")
        row = self.selected_row
        if row.interval not in self.data_by_tf:
            raise ValueError("Нет данных выбранного таймфрейма")
        df = self.data_by_tf[row.interval]
        desired, label = build_selected_signal(df, row)
        plan = build_walk_forward_plan(df)
        return df, row, desired, label, plan

    def start_report_export(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Отчёт", "Сначала дождитесь завершения текущей операции или остановите её.")
            return
        cfg = self.last_config.copy() if self.last_config else {}
        if not cfg:
            try:
                cfg = self._read_config()
            except Exception:
                cfg = {"symbol": self.symbol_var.get().upper().strip() or "UNKNOWN", "tfs": self._selected_tfs()}
        self._set_busy(True)
        self.big_status_var.set("СОБИРАЮ ПОЛНЫЙ ОТЧЁТ ДЛЯ CHATGPT…")
        self.stage_var.set("Этап: отчёт")
        self.checked_var.set("Упаковка файлов")
        self.counter_var.set("Упаковываю код, логи, результаты и данные…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)

        def run():
            try:
                path = create_chatgpt_report(
                    cfg,
                    self.search_stats,
                    self.all_results,
                    self.top_results,
                    self.data_by_tf,
                    self.last_search_seconds,
                    research_summary=self.last_research_summary or None,
                )
                self._post("report_done", str(path))
            except Exception as exc:
                self._post("error", f"Ошибка создания отчёта: {exc}\n\n{traceback.format_exc(limit=8)}")

        self.worker = threading.Thread(target=run, daemon=True, name="ReportExportWorker")
        self.worker.start()

    def show_heatmap(self):
        if not self.all_results or self.selected_row is None:
            messagebox.showinfo("Тепловая карта", "Сначала выполните автопоиск и выберите результат.")
            return
        row = self.selected_row
        if row.strategy_type == "ADAPTIVE_V2":
            subset = [r for r in self.all_results if r.strategy_type == "ADAPTIVE_V2" and r.interval == row.interval
                      and r.long_atr_period == row.long_atr_period and abs(r.long_multiplier-row.long_multiplier) < 1e-9
                      and r.short_atr_period == row.short_atr_period and abs(r.short_multiplier-row.short_multiplier) < 1e-9
                      and r.regime_separation_atr == row.regime_separation_atr
                      and r.regime_confirm_bars == row.regime_confirm_bars and r.regime_min_hold_bars == row.regime_min_hold_bars]
            pairs = sorted({(r.ema_fast,r.ema_slow) for r in subset})
            thrs = sorted({round(r.adx_threshold,3) for r in subset})
            if not pairs or not thrs:
                messagebox.showinfo("Тепловая карта", "Недостаточно соседних Adaptive V2 вариантов для карты."); return
            matrix=np.full((len(pairs),len(thrs)),np.nan); pi={v:i for i,v in enumerate(pairs)}; ti={v:i for i,v in enumerate(thrs)}
            for r in subset: matrix[pi[(r.ema_fast,r.ema_slow)],ti[round(r.adx_threshold,3)]]=r.robust_score
            fig=Figure(figsize=(10,6),dpi=100); ax=fig.add_subplot(111); im=ax.imshow(matrix,aspect="auto",origin="lower")
            ax.set_title(f"{row.interval}м · Adaptive V2 · research score"); ax.set_xlabel("ADX threshold"); ax.set_ylabel("EMA fast/slow")
            ax.set_xticks(range(len(thrs))); ax.set_xticklabels([f"{x:g}" for x in thrs])
            ax.set_yticks(range(len(pairs))); ax.set_yticklabels([f"{a}/{b}" for a,b in pairs])
            fig.colorbar(im,ax=ax,label="Research score"); fig.tight_layout(); PlotWindow(self,"Тепловая карта",fig); return
        if row.strategy_type == "ADAPTIVE":
            subset = [r for r in self.all_results if r.strategy_type == "ADAPTIVE" and r.interval == row.interval
                      and r.long_atr_period == row.long_atr_period and abs(r.long_multiplier-row.long_multiplier) < 1e-9
                      and r.short_atr_period == row.short_atr_period and abs(r.short_multiplier-row.short_multiplier) < 1e-9]
            hours = sorted({r.regime_hours for r in subset}); thrs = sorted({round(r.regime_threshold_pct, 4) for r in subset})
            if not hours or not thrs:
                messagebox.showinfo("Тепловая карта", "Недостаточно соседних адаптивных вариантов для карты."); return
            matrix=np.full((len(hours),len(thrs)),np.nan); hi={v:i for i,v in enumerate(hours)}; ti={v:i for i,v in enumerate(thrs)}
            for r in subset: matrix[hi[r.regime_hours],ti[round(r.regime_threshold_pct,4)]]=r.robust_score
            fig=Figure(figsize=(10,6),dpi=100); ax=fig.add_subplot(111); im=ax.imshow(matrix,aspect="auto",origin="lower")
            ax.set_title(f"{row.interval}м · adaptive v1"); ax.set_xlabel("Порог, %"); ax.set_ylabel("Часы")
            ax.set_xticks(range(len(thrs))); ax.set_xticklabels([f"{x:g}" for x in thrs],rotation=45,ha="right"); ax.set_yticks(range(len(hours))); ax.set_yticklabels([str(x) for x in hours])
            fig.colorbar(im,ax=ax,label="Research score"); fig.tight_layout(); PlotWindow(self,"Тепловая карта",fig); return
        subset = [r for r in self.all_results if r.strategy_type == "BASE" and r.interval == row.interval and r.mode == row.mode]
        periods = sorted({r.atr_period for r in subset}); mults = sorted({round(r.multiplier, 4) for r in subset})
        matrix = np.full((len(periods), len(mults)), np.nan); pi={p:i for i,p in enumerate(periods)}; mi={m:i for i,m in enumerate(mults)}
        for r in subset: matrix[pi[r.atr_period], mi[round(r.multiplier,4)]] = r.robust_score
        fig=Figure(figsize=(10,6),dpi=100); ax=fig.add_subplot(111); im=ax.imshow(matrix, aspect="auto", origin="lower")
        ax.set_title(f"{row.interval}м · {row.mode} · устойчивость walk-forward"); ax.set_xlabel("Множитель Supertrend"); ax.set_ylabel("ATR период")
        xs=max(1,len(mults)//12); ys=max(1,len(periods)//15); ax.set_xticks(range(0,len(mults),xs)); ax.set_xticklabels([f"{mults[i]:g}" for i in range(0,len(mults),xs)], rotation=45,ha="right")
        ax.set_yticks(range(0,len(periods),ys)); ax.set_yticklabels([str(periods[i]) for i in range(0,len(periods),ys)])
        fig.colorbar(im,ax=ax,label="Robust score"); fig.tight_layout(); PlotWindow(self,"Тепловая карта",fig)

    def show_equity(self):
        try:
            df, row, desired, label, plan = self._selected_calc()
        except Exception as exc:
            messagebox.showinfo("Кривая капитала", str(exc)); return
        fee=self.last_config["fee"]; slip=self.last_config["slippage"]; research_start=int(plan["research_start"]); holdout_start=int(plan["holdout_start"])
        _,_,eq1=backtest_desired_with_trades(df, desired, research_start, holdout_start-1, fee, slip)
        _,_,eq2=backtest_desired_with_trades(df, desired, holdout_start, len(df)-1, fee, slip)
        fig=Figure(figsize=(10,6),dpi=100); ax=fig.add_subplot(111); ax.plot(np.arange(len(eq1)),eq1,label="Walk-forward research")
        off=len(eq1)-1; ax.plot(off+np.arange(len(eq2)),eq2*eq1[-1],label="Development holdout"); ax.axvline(off,linestyle="--",linewidth=1)
        ax.set_title(f"Кривая капитала · {row.interval}м · {label}"); ax.set_xlabel("Закрытые сделки"); ax.set_ylabel("Капитал, старт = 1.0"); ax.grid(True,alpha=0.25); ax.legend(); fig.tight_layout(); PlotWindow(self,"Кривая капитала",fig)

    def show_trades(self):
        try:
            df, row, desired, label, plan = self._selected_calc()
        except Exception as exc:
            messagebox.showinfo("Сделки", str(exc)); return
        fee=self.last_config["fee"]; slip=self.last_config["slippage"]; research_start=int(plan["research_start"]); holdout_start=int(plan["holdout_start"])
        _,t1,_=backtest_desired_with_trades(df, desired, research_start, holdout_start-1, fee, slip)
        _,t2,_=backtest_desired_with_trades(df, desired, holdout_start, len(df)-1, fee, slip)
        trades=[("RESEARCH",t) for t in t1]+[("DEV HOLDOUT",t) for t in t2]
        if not trades:
            messagebox.showinfo("Сделки","Сделок нет"); return
        win=tk.Toplevel(self); win.title(f"Сделки · {row.interval}м · {label}"); win.geometry("1200x650")
        cols=("part","side","entry_t","exit_t","entry","exit","gross","net"); tree=ttk.Treeview(win,columns=cols,show="headings")
        labels={"part":"Участок","side":"Сторона","entry_t":"Вход UTC","exit_t":"Выход UTC","entry":"Цена входа","exit":"Цена выхода","gross":"До расходов %","net":"Чистыми %"}
        for c in cols: tree.heading(c,text=labels[c]); tree.column(c,width=140,anchor=tk.CENTER)
        tree.pack(fill=tk.BOTH,expand=True,padx=10,pady=10)
        for part,t in trades:
            tree.insert("",tk.END,values=(part,t.side,t.entry_time.strftime("%Y-%m-%d %H:%M"),t.exit_time.strftime("%Y-%m-%d %H:%M"),f"{t.entry_price:.2f}",f"{t.exit_price:.2f}",f"{t.gross_return_pct:+.4f}",f"{t.net_return_pct:+.4f}"))


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    mp.freeze_support()
    main()

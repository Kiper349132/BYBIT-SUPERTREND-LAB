"""Candle download / CSV storage. Parent process only (pandas, requests)."""
from __future__ import annotations

import csv
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests

import lab_core as core
from lab_log import file_log

APP_NAME = "Bybit Supertrend Lab"
API_URL = "https://api.bybit.com/v5/market/kline"
SUPPORTED_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720"]
AUTO_INTERVALS = ["1", "3", "5", "15", "30", "60"]
CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]


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


def _frame_from_array(arr: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(arr, columns=CANDLE_COLUMNS)
    frame = frame.dropna()
    frame["timestamp"] = frame["timestamp"].astype("int64")
    return frame


def download_bybit_klines(symbol: str, interval: str, start_dt: datetime, end_dt: datetime,
                          progress: Optional[Callable[[float, str], None]] = None,
                          stop_event: Optional[threading.Event] = None,
                          partial_path: Optional[Path] = None,
                          session: Optional[requests.Session] = None) -> pd.DataFrame:
    symbol = symbol.upper().strip()
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Таймфрейм {interval} не поддерживается")
    if end_dt <= start_dt:
        raise ValueError("Конечная дата должна быть позже начальной")

    start_ms = dt_to_ms(start_dt)
    end_ms = min(dt_to_ms(end_dt), now_utc_ms())
    cursor_end = end_ms
    # v0.8: every 1000-row batch is parsed into a float64 array immediately
    # (56 bytes per candle). v0.7.1 kept all candles as lists of strings
    # (~0.5 KB per candle, ~400 MB for 1m / 18 months).
    parts: list[np.ndarray] = []
    received = 0
    session = session or requests.Session()
    session.headers.update({"User-Agent": f"{APP_NAME}/{core.VERSION}"})
    expected = max(1, math.ceil((end_ms - start_ms) / interval_ms(interval)))
    request_no = 0

    if partial_path is not None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        with partial_path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(CANDLE_COLUMNS)
    if progress:
        progress(0.0, f"{interval}м: подключение к Bybit…")
    file_log(f"HTTP begin symbol={symbol} tf={interval} start={start_dt.isoformat()} end={end_dt.isoformat()}")

    while cursor_end >= start_ms:
        if stop_event and stop_event.is_set():
            raise StopRequested()
        params = {"category": "linear", "symbol": symbol, "interval": interval,
                  "start": start_ms, "end": cursor_end, "limit": 1000}
        payload = None
        for attempt in range(1, 7):
            try:
                response = session.get(API_URL, params=params, timeout=12)
                if response.status_code == 429:
                    raise RuntimeError("HTTP 429: лимит запросов Bybit")
                response.raise_for_status()
                payload = response.json()
                if payload.get("retCode") != 0:
                    raise RuntimeError(f"Bybit retCode={payload.get('retCode')}: {payload.get('retMsg', 'неизвестная ошибка')}")
                break
            except Exception as exc:
                file_log(f"HTTP ERROR tf={interval} attempt={attempt}: {exc}")
                if progress:
                    progress(min(99.0, received / expected * 100.0), f"{interval}м: ошибка связи, повтор {attempt}/6 — {exc}")
                if attempt == 6:
                    raise RuntimeError(f"Не удалось получить свечи Bybit: {exc}") from exc
                time.sleep(min(20.0, 0.8 * 2 ** (attempt - 1)))
        batch = (payload or {}).get("result", {}).get("list", [])
        request_no += 1
        if not batch:
            break
        arr = np.array([[float(x) for x in row[:7]] for row in batch], dtype=np.float64)
        parts.append(arr)
        received += len(arr)
        if partial_path is not None:
            try:
                with partial_path.open("a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerows(batch)
            except Exception as exc:
                raise RuntimeError(f"Не удалось записать временный файл {partial_path.name}: {exc}") from exc
        oldest = int(arr[:, 0].min())
        pct = min(100.0, received / expected * 100.0)
        if progress:
            oldest_dt = datetime.fromtimestamp(oldest / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            progress(pct, f"{interval}м: запрос №{request_no} · получено {received:,}/{expected:,} свечей · {pct:.1f}% · дошли до {oldest_dt}")
        if oldest <= start_ms:
            break
        if oldest >= cursor_end:
            raise RuntimeError("Bybit вернул повторяющийся диапазон. Загрузка остановлена.")
        cursor_end = oldest - 1
        time.sleep(0.035)

    if not parts:
        raise RuntimeError("Bybit не вернул свечи для выбранного периода")
    file_log(f"HTTP complete tf={interval} raw_rows={received} requests={request_no}")
    frame = _frame_from_array(np.concatenate(parts))
    frame = frame.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    frame = frame[(frame["timestamp"] >= start_ms) & (frame["timestamp"] <= end_ms)]
    if not frame.empty and int(frame.iloc[-1]["timestamp"]) + interval_ms(interval) > now_utc_ms():
        frame = frame.iloc[:-1].copy()   # the last, still open candle is never used
    frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.reset_index(drop=True)
    if len(frame) < 300:
        raise RuntimeError(f"Слишком мало данных: {len(frame)} свечей")
    if progress:
        progress(100.0, f"{interval}м: загрузка завершена · {len(frame):,} закрытых свечей")
    return frame


def save_candles(df: pd.DataFrame, symbol: str, interval: str, data_dir: Path) -> Path:
    first = df.iloc[0]["datetime"].strftime("%Y%m%d")
    last = df.iloc[-1]["datetime"].strftime("%Y%m%d")
    path = data_dir / f"{symbol}_{interval}m_{first}_{last}.csv"
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def load_candles_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"timestamp", "open", "high", "low", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"В CSV нет колонок: {', '.join(sorted(missing))}")
    for col in ["timestamp", "open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col in df.columns else 0.0
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def _dates_from_name(path: Path, interval: str) -> Optional[tuple[datetime, datetime]]:
    """SYMBOL_15m_YYYYMMDD_YYYYMMDD.csv -> (first day, last day) without reading the file."""
    try:
        parts = path.stem.split("_")
        a = datetime.strptime(parts[-2], "%Y%m%d").replace(tzinfo=timezone.utc)
        b = datetime.strptime(parts[-1], "%Y%m%d").replace(tzinfo=timezone.utc)
        return a, b
    except Exception:
        return None


def find_reusable_candles(symbol: str, interval: str, start_dt: datetime, end_dt: datetime, data_dir: Path):
    """Reuse a previously downloaded CSV that covers the requested period.

    v0.8 fixes:
    * the result is TRIMMED to exactly the requested [start, end] range
      (v0.7.1 happily returned 36 months for an 18-month request);
    * the end may lag by at most max(3 candles, 6 hours), not 1% of the period;
    * candidate files are pre-filtered by the dates in their file names, so
      unsuitable 1-minute CSVs are not parsed at all.
    """
    pattern = f"{symbol.upper()}_{interval}m_*.csv"
    candidates = [p for p in data_dir.glob(pattern) if "DOWNLOADING" not in p.name.upper()]
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    start_ms, end_ms = dt_to_ms(start_dt), dt_to_ms(end_dt)
    step = interval_ms(interval)
    start_tol = 3 * step
    end_tol = max(3 * step, 6 * 3600 * 1000)
    for path in candidates:
        rng = _dates_from_name(path, interval)
        if rng is not None:
            # names carry days only; allow one day of slack each side
            if dt_to_ms(rng[0]) - 86_400_000 > start_ms + start_tol or dt_to_ms(rng[1]) + 2 * 86_400_000 < end_ms - end_tol:
                continue
        try:
            df = load_candles_csv(path)
            if df.empty:
                continue
            first, last = int(df["timestamp"].iloc[0]), int(df["timestamp"].iloc[-1])
            if first - start_ms > start_tol or end_ms - last > end_tol + step:
                continue
            trimmed = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)].reset_index(drop=True)
            if len(trimmed) < 300:
                continue
            trimmed.attrs["source_path"] = str(path)
            file_log(f"REUSE data {path.name}: rows={len(df)} -> trimmed {len(trimmed)}")
            return trimmed, path
        except Exception as exc:
            file_log(f"REUSE skip {path.name}: {exc}")
    return None, None


def df_to_market(df: pd.DataFrame) -> dict:
    ts = df["timestamp"].to_numpy(np.int64)
    return {
        "timestamp": ts,
        "open": df["open"].to_numpy(np.float64), "high": df["high"].to_numpy(np.float64),
        "low": df["low"].to_numpy(np.float64), "close": df["close"].to_numpy(np.float64),
        "months": core.month_codes_from_ms(ts),
    }


def market_to_df(m: dict) -> pd.DataFrame:
    df = pd.DataFrame({k: np.asarray(m[k]) for k in ("timestamp", "open", "high", "low", "close")})
    df["volume"] = 0.0
    df["turnover"] = 0.0
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

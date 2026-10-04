"""Shared helpers for the logic audit (read-only use of the program modules).

Audit data lives OUTSIDE the repository (AUDIT_DATA env var); nothing here
modifies the main program.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
DATA = Path(os.environ.get("AUDIT_DATA", "/tmp/audit_data"))

import lab_core as core  # noqa: E402

TF_NAME = {"5": "5m", "15": "15m", "30": "30m", "60": "1h"}


def load_binance(tf: str) -> dict:
    """Binance USDT-M BTCUSDT perpetual klines (public archive), 2025-03..2026-08.
    Proxy for Bybit BTCUSDT linear (Bybit API is geo-blocked here)."""
    cache = DATA / f"binance_{tf}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    rows = []
    for zp in sorted((DATA / "zips").glob(f"BTCUSDT-{TF_NAME[tf]}-*.zip")):
        with zipfile.ZipFile(zp) as zf:
            txt = zf.read(zf.namelist()[0]).decode()
        for line in txt.splitlines():
            if not line or not line[0].isdigit():
                continue
            p = line.split(",")
            rows.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    a = np.array(rows, dtype=np.float64)
    a = a[np.argsort(a[:, 0], kind="stable")]
    _, keep = np.unique(a[:, 0], return_index=True)
    a = a[keep]
    ts = a[:, 0].astype(np.int64)
    m = {"timestamp": ts, "open": a[:, 1].copy(), "high": a[:, 2].copy(), "low": a[:, 3].copy(), "close": a[:, 4].copy(),
         "months": core.month_codes_from_ms(ts)}
    np.savez(cache, **m)
    return m


def load_funding() -> tuple[np.ndarray, np.ndarray]:
    """(funding_time_ms, rate) - Binance 8h funding, positive = longs pay shorts."""
    t, r = [], []
    for zp in sorted((DATA / "zips").glob("BTCUSDT-fundingRate-*.zip")):
        with zipfile.ZipFile(zp) as zf:
            txt = zf.read(zf.namelist()[0]).decode()
        for line in txt.splitlines():
            p = line.split(",")
            if p and p[0].isdigit():
                t.append(int(p[0])); r.append(float(p[2]))
    o = np.argsort(t)
    return np.asarray(t, dtype=np.int64)[o], np.asarray(r)[o]

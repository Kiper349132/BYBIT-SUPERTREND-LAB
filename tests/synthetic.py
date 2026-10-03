"""Deterministic synthetic market data for tests (no network)."""
from __future__ import annotations

import numpy as np

import lab_core as core


def synthetic_market(n: int = 9000, tf: str = "60", seed: int = 20261002, end_ms: int = 1_780_000_000_000) -> dict:
    rng = np.random.default_rng(seed)
    third = n // 3
    returns = np.r_[
        0.00006 + 0.0018 * rng.normal(size=third),
        -0.00005 + 0.0022 * rng.normal(size=third),
        0.00001 + 0.0015 * rng.normal(size=n - 2 * third),
    ]
    close = 60000 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]] * (1 + 0.00012 * rng.normal(size=n))
    high = np.maximum(open_, close) * (1 + np.abs(0.0009 * rng.normal(size=n)))
    low = np.minimum(open_, close) * (1 - np.abs(0.0009 * rng.normal(size=n)))
    step = int(tf) * 60_000
    ts = (end_ms // step) * step - np.arange(n, dtype=np.int64)[::-1] * step
    return {"timestamp": ts.astype(np.int64), "open": open_, "high": high, "low": low, "close": close,
            "months": core.month_codes_from_ms(ts)}

"""v0.8 must not change any backtest / score arithmetic of v0.7.1.

Every metric is compared for EXACT equality against a frozen copy of the
v0.7.1 code (tests/reference_v071.py).
"""
import numpy as np
import pandas as pd
import pytest

import lab_core as core
import reference_v071 as ref
from synthetic import synthetic_market

TF = "60"


@pytest.fixture(scope="module")
def market():
    return synthetic_market(9000, TF)


@pytest.fixture(scope="module")
def frame(market):
    df = pd.DataFrame({k: market[k] for k in ("timestamp", "open", "high", "low", "close")})
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def _metrics_equal(a, b):
    for f in ("trades", "wins", "losses", "win_rate", "total_return_pct", "profit_factor", "max_drawdown_pct",
              "avg_trade_pct", "median_trade_pct", "best_trade_pct", "worst_trade_pct", "long_return_pct",
              "short_return_pct", "long_trades", "short_trades", "profitable_months", "months",
              "profitable_month_ratio", "score"):
        assert getattr(a, f) == getattr(b, f), f


COMMON = ("selection_score", "validation_score", "robust_score", "selection_windows", "selection_positive_ratio",
          "selection_median_return_pct", "selection_worst_return_pct", "validation_windows", "validation_positive_ratio",
          "validation_median_return_pct", "validation_worst_return_pct", "wf_windows", "wf_positive_windows",
          "wf_positive_ratio", "wf_median_return_pct", "wf_worst_return_pct", "wf_best_return_pct", "up_windows",
          "up_positive_ratio", "up_avg_return_pct", "down_windows", "down_positive_ratio", "down_avg_return_pct",
          "flat_windows", "flat_positive_ratio", "flat_avg_return_pct")


def test_month_codes_match_pandas(market, frame):
    expected = (frame["datetime"].dt.year * 100 + frame["datetime"].dt.month).to_numpy(np.int32)
    assert np.array_equal(core.month_codes_from_ms(market["timestamp"]), expected)


def test_walk_forward_plan_identical(market, frame):
    old = ref.build_walk_forward_plan(frame)
    new = core.build_walk_forward_plan_arrays(market["timestamp"], market["open"], market["close"])
    for k in old:
        assert old[k] == new[k], k


def test_indicators_identical(market):
    h, l, c = market["high"], market["low"], market["close"]
    for p in (2, 7, 14, 50):
        assert np.array_equal(ref._fast_atr(h, l, c, p), core._fast_atr(h, l, c, p), equal_nan=True)
        assert np.array_equal(ref._adx_array(h, l, c, p), core._adx_array(h, l, c, p))
        assert np.array_equal(ref._ema_array(c, p), core._ema_array(c, p))
    reg_old = ref.online_regime_v2(h, l, c, 8, 21, 14, 20.0, 0.15, 3, 6)
    reg_new = core.online_regime_v2(h, l, c, 8, 21, 14, 20.0, 0.15, 3, 6)
    assert np.array_equal(reg_old, reg_new)


def test_base_evaluation_bit_exact(market):
    plan = core.build_walk_forward_plan_arrays(market["timestamp"], market["open"], market["close"])
    ref._worker_init({k: market[k] for k in ("open", "high", "low", "close", "months")})
    core.worker_set_inline(TF, market, plan)
    specs = [(1.3, ("BOTH", "LONG", "SHORT")), (2.5, ("BOTH", "LONG", "SHORT")), (4.75, ("BOTH", "SHORT"))]
    for period in (5, 14, 33):
        old = ref._worker_eval_period((period, specs, plan, 0.00055, 0.0002, 26, TF, "broad"))
        new = core._worker_eval_period((TF, period, specs, 0.00055, 0.0002, 26, "broad"))
        assert len(old) == len(new) == 8
        for o, n in zip(old, new):
            assert (o.interval, o.mode, o.atr_period, o.multiplier, o.stage) == (n.interval, n.mode, n.atr_period, n.multiplier, n.stage)
            _metrics_equal(o.train, n.train)
            _metrics_equal(o.test, n.test)
            for f in COMMON:
                assert getattr(o, f) == getattr(n, f), f


def test_adaptive_v2_bit_exact(market):
    plan = core.build_walk_forward_plan_arrays(market["timestamp"], market["open"], market["close"])
    arrays = {k: market[k] for k in ("open", "high", "low", "close", "months")}
    ref._worker_init(arrays)
    old_rows = ref._worker_eval_period((14, [(2.0, ("LONG", "SHORT")), (3.1, ("LONG", "SHORT"))], plan, 0.00055, 0.0002, 26, TF, "broad"))
    longs = [r for r in old_rows if r.mode == "LONG"]
    shorts = [r for r in old_rows if r.mode == "SHORT"]
    ema_pairs = [(8, 21), (12, 26)]
    adx_periods = [10, 14]
    regime_chunk = [
        {"ema_fast": ef, "ema_slow": es, "adx_period": ap, "adx_threshold": 20, "separation_atr": sep, "confirm_bars": 2, "min_hold_bars": 4}
        for ef, es in ema_pairs for ap in adx_periods for sep in (0.1, 0.25)
    ]
    old = ref._worker_eval_adaptive_v2_chunk((regime_chunk, longs, shorts, ema_pairs, adx_periods, plan, 0.00055, 0.0002, 26, TF))
    h, l, c = market["high"], market["low"], market["close"]
    feats = {name: core.compute_feature(name, h, l, c) for name in core.feature_names([x for p in ema_pairs for x in p], adx_periods)}
    core.worker_set_inline(TF, market, plan, feats)
    new = core._worker_eval_adaptive_v2_chunk((TF, regime_chunk, longs, shorts, 0.00055, 0.0002, 26))
    assert len(old) == len(new) == len(regime_chunk) * 4
    for o, n in zip(old, new):
        _metrics_equal(o.train, n.train)
        _metrics_equal(o.test, n.test)
        for f in COMMON + ("online_up_ratio", "online_down_ratio", "online_flat_ratio", "regime_switches"):
            assert getattr(o, f) == getattr(n, f), f


def test_selection_only_regime_stats(market):
    """sel_* statistics are computed from SELECTION windows only."""
    plan = core.build_walk_forward_plan_arrays(market["timestamp"], market["open"], market["close"])
    core.worker_set_inline(TF, market, plan)
    row = core._worker_eval_period((TF, 14, [(2.5, ("BOTH",))], 0.00055, 0.0002, 26, "broad"))[0]
    sel_regimes = [w[2] for w in plan["selection_windows"]]
    assert row.sel_up_windows == sel_regimes.count("UP")
    assert row.sel_down_windows == sel_regimes.count("DOWN")
    assert row.sel_flat_windows == sel_regimes.count("FLAT")
    assert row.sel_up_windows + row.sel_down_windows + row.sel_flat_windows == row.selection_windows
    assert row.up_windows + row.down_windows + row.flat_windows == row.wf_windows > row.selection_windows

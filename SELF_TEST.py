"""Локальная проверка BYBIT SUPERTREND LAB v0.7.1 без интернета."""
import importlib.util
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = HERE / "BYBIT_SUPERTREND_LAB.py"
spec = importlib.util.spec_from_file_location("bybit_lab", MAIN)
lab = importlib.util.module_from_spec(spec)
sys.modules["bybit_lab"] = lab
spec.loader.exec_module(lab)

rng = np.random.default_rng(20261002)
n = 14000
returns = np.r_[
    0.00005 + 0.0018 * rng.normal(size=n//3),
    -0.00004 + 0.0022 * rng.normal(size=n//3),
    0.00001 + 0.0015 * rng.normal(size=n - 2*(n//3)),
]
close = 60000 * np.exp(np.cumsum(returns))
open_ = np.r_[close[0], close[:-1]] * (1 + 0.00012 * rng.normal(size=n))
high = np.maximum(open_, close) * (1 + np.abs(0.0009 * rng.normal(size=n)))
low = np.minimum(open_, close) * (1 - np.abs(0.0009 * rng.normal(size=n)))
ts = 1700000000000 + np.arange(n, dtype=np.int64) * 60 * 60_000

df = pd.DataFrame({"timestamp":ts,"open":open_,"high":high,"low":low,"close":close,"volume":1.0,"turnover":1.0})
df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
months = (df["datetime"].dt.year * 100 + df["datetime"].dt.month).to_numpy(np.int32)

atr = lab.wilder_atr(high, low, close, 14)
atr_fast = lab._fast_atr(high, low, close, 14)
assert np.allclose(atr[np.isfinite(atr)], atr_fast[np.isfinite(atr_fast)], rtol=1e-12, atol=1e-12)
trend, st = lab.supertrend_from_atr(high, low, close, atr, 2.5)
trend_fast = lab._fast_trend(high, low, close, atr_fast, 2.5)
assert np.array_equal(trend, trend_fast)

fast = lab.backtest_fast_arrays(open_, close, months, trend, 14, n-1, 0.00055, 0.0002, "BOTH", 1)
full, trades, equity = lab.backtest_with_trades(df, trend, 14, n-1, 0.00055, 0.0002, "BOTH")
assert fast.trades == full.trades == len(trades)
assert abs(fast.total_return_pct - full.total_return_pct) < 1e-9
assert len(equity) == len(trades) + 1

plan = lab.build_walk_forward_plan(df)
assert len(plan["windows"]) >= 4
assert plan["research_start"] < plan["validation_start"] <= plan["holdout_start"] < len(df)
assert len(plan["selection_windows"]) >= 2
assert len(plan["validation_windows"]) >= 2

# Regression test for v0.4.0: walk-forward planning must never depend on
# pandas datetime internal resolution (ns/us/ms). Raw Bybit timestamp is authoritative.
df_alt_datetime = df.copy()
df_alt_datetime["datetime"] = pd.to_datetime(np.arange(n, dtype=np.int64), unit="ms", utc=True)
plan_alt = lab.build_walk_forward_plan(df_alt_datetime)
assert len(plan_alt["windows"]) == len(plan["windows"])
assert plan_alt["holdout_start"] == plan["holdout_start"]
assert abs(plan_alt["total_days"] - plan["total_days"]) < 1e-12

# Worker-level walk-forward evaluation.
payload = {
    "open": open_.astype(np.float64), "high": high.astype(np.float64),
    "low": low.astype(np.float64), "close": close.astype(np.float64), "months": months,
}
lab._worker_init(payload)
rows = lab._worker_eval_period((14, [(2.5, ("BOTH","LONG","SHORT"))], plan, 0.00055, 0.0002, 5, "60", "selftest"))
assert len(rows) == 3
for r in rows:
    assert r.wf_windows == len(plan["windows"])
    assert r.selection_windows == len(plan["selection_windows"])
    assert r.validation_windows == len(plan["validation_windows"])
    assert np.isfinite(r.selection_score)
    assert np.isfinite(r.validation_score)
    assert np.isfinite(r.robust_score)

top = lab.diversified_top(rows, 100)
assert len(top) >= 1

# Adaptive regime must be past-only: changing future prices cannot change old labels.
reg1 = lab.online_regime_from_past(close, "60", 24, 1.0)
close_future_changed = close.copy()
cut = n // 2
close_future_changed[cut+1:] *= 1.8
reg2 = lab.online_regime_from_past(close_future_changed, "60", 24, 1.0)
assert np.array_equal(reg1[:cut+1], reg2[:cut+1])

# Fast adaptive segment must match the transparent desired-vector backtest.
latr = lab._fast_atr(high, low, close, 12)
satr = lab._fast_atr(high, low, close, 20)
ltrend = lab._fast_trend(high, low, close, latr, 2.2)
strend = lab._fast_trend(high, low, close, satr, 3.1)
reg = lab.online_regime_from_past(close, "60", 24, 0.75)
desired = np.zeros(n, dtype=np.int8)
desired[(reg == 1) & (ltrend == 1)] = 1
desired[(reg == -1) & (strend == -1)] = -1
slow_ad = lab.backtest_desired_fast_arrays(open_, close, months, desired, 30, n-1, 0.00055, 0.0002, 1)
fast_tuple = lab._adaptive_segment_nb(open_, close, ltrend, strend, reg, 30, n-1, 0.00055, 0.0002)
fast_ad = lab._metrics_from_adaptive_tuple(fast_tuple, 1)
assert slow_ad.trades == fast_ad.trades
assert abs(slow_ad.total_return_pct - fast_ad.total_return_pct) < 1e-8
assert abs(slow_ad.profit_factor - fast_ad.profit_factor) < 1e-8
assert abs(slow_ad.max_drawdown_pct - fast_ad.max_drawdown_pct) < 1e-8

# Build one adaptive result to make sure the new report/result fields are valid.
lr = next(r for r in rows if r.mode == "LONG")
sr = next(r for r in rows if r.mode == "SHORT")
adaptive = lab.evaluate_adaptive_candidate(
    "60", lr, sr, 24, 0.75, ltrend, strend, reg,
    open_, close, months, plan, 0.00055, 0.0002, 5,
)
assert adaptive.strategy_type == "ADAPTIVE"
assert adaptive.regime_hours == 24
assert adaptive.wf_windows == len(plan["windows"])
assert np.isfinite(adaptive.robust_score)

# Adaptive V2 must also be past-only and hysteresis must reduce flip-flopping.
regv1 = lab.online_regime_v2(high, low, close, 8, 21, 14, 20.0, 0.15, 3, 6)
high2, low2, close2 = high.copy(), low.copy(), close.copy()
high2[cut+1:] *= 1.7; low2[cut+1:] *= 1.7; close2[cut+1:] *= 1.7
regv2 = lab.online_regime_v2(high2, low2, close2, 8, 21, 14, 20.0, 0.15, 3, 6)
assert np.array_equal(regv1[:cut+1], regv2[:cut+1])
rp = {"ema_fast":8,"ema_slow":21,"adx_period":14,"adx_threshold":20.0,"separation_atr":0.15,"confirm_bars":3,"min_hold_bars":6}
adaptive2 = lab.evaluate_adaptive_v2_candidate(
    "60", lr, sr, rp, ltrend, strend, regv1, open_, close, months, plan, 0.00055, 0.0002, 5
)
assert adaptive2.strategy_type == "ADAPTIVE_V2"
assert adaptive2.validation_windows == len(plan["validation_windows"])
assert np.isfinite(adaptive2.robust_score)

diag = lab.diagnose_dataframe(df, "60")
assert diag["duplicate_timestamps"] == 0
assert diag["gap_events"] == 0


# v0.7 crash-safe disk persistence: a one-million-check batch must become
# durable before its unit is marked complete, and stage seeds must reload.
cfg_store={"symbol":"SELFTEST","months":18,"tfs":["60"],"fee":0.00055,"slippage":0.0002,"depth":"Стандартный","workers":2}
store=lab.ResearchPersistence(cfg_store,{"60":df})
try:
    sample_rows=rows[:2]
    flushed=store.queue_unit("60","selftest_stage","14",sample_rows,lab.CHECKPOINT_USEFUL_EVERY)
    assert flushed is True
    assert store.unit_done("60","selftest_stage","14")
    assert store.committed_useful >= lab.CHECKPOINT_USEFUL_EVERY
    assert store.committed_rows >= len(sample_rows)
    # Simulate a process restart before the stage is finalized.
    store2=lab.ResearchPersistence(cfg_store,{"60":df})
    assert store2.resumed is True
    assert store2.unit_done("60","selftest_stage","14")
    assert store2.committed_useful >= lab.CHECKPOINT_USEFUL_EVERY
    store.save_stage_seed("60","selftest_stage",sample_rows,{"candidate_count":len(sample_rows),"useful_checks":lab.CHECKPOINT_USEFUL_EVERY})
    assert store.is_stage_complete("60","selftest_stage")
    loaded=store.load_stage_seed("60","selftest_stage")
    assert len(loaded)==len(sample_rows)
    assert loaded[0].interval==sample_rows[0].interval
    store.finalize({"self_test":True})
    assert store.summary()["status"]=="complete"
finally:
    shutil.rmtree(store.root,ignore_errors=True)

# v0.7.1 regression: a persistently high *system* RAM percentage must not
# cause a checkpoint after every tiny completed unit. It may force an early
# emergency flush only after MEMORY_FLUSH_MIN_USEFUL has accumulated.
orig_mem = lab._memory_percent
lab._memory_percent = lambda: 90.0
cfg_guard={"symbol":"SELFTEST_GUARD","months":18,"tfs":["60"],"fee":0.00055,"slippage":0.0002,"depth":"Стандартный","workers":2}
guard=lab.ResearchPersistence(cfg_guard,{"60":df})
try:
    step=max(1, lab.MEMORY_FLUSH_MIN_USEFUL // 20)
    flushed_count=0
    for i in range(19):
        assert guard.queue_unit("60","guard",str(i),rows[:1],step) is False
    flushed_count += int(guard.queue_unit("60","guard","19",rows[:1],step))
    assert flushed_count == 1
    assert guard.cp.get("chunk_count",0) == 1
finally:
    lab._memory_percent = orig_mem
    shutil.rmtree(guard.root,ignore_errors=True)

print("SELF TEST: PASS")
print(f"Numba: {lab.NUMBA_AVAILABLE}")
print(f"Walk-forward окон: {len(plan['windows'])}")
print(f"DEVELOPMENT HOLDOUT: {plan['holdout_days']} дней")
print(f"Сделок базового теста: {full.trades}")
print(f"Adaptive V1/V2 past-only test: PASS · сделок V1 {fast_ad.trades}")
print(f"SELECTION/VALIDATION: {len(plan['selection_windows'])}/{len(plan['validation_windows'])}")
print("Disk checkpoint 1,000,000 useful checks: PASS")
print("High-RAM flush hysteresis: PASS")

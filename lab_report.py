"""Results export, trade listings and the ChatGPT audit report (parent only)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

import lab_core as core
from lab_core import (
    AutoResult, BacktestMetrics, NUMBA_AVAILABLE, VERSION, metrics_from_returns, online_regime_from_past,
    online_regime_v2, supertrend_from_atr, wilder_atr, _effective_entry_array, _effective_exit_array,
)
from lab_core import result_key as _result_key
from lab_log import LOGS_DIR, RUNTIME_LOG, file_log

APP_NAME = "Bybit Supertrend Lab"
BASE_DIR = Path(__file__).resolve().parent
from lab_log import HOME_DIR  # noqa: E402
RESULTS_DIR = HOME_DIR / "results"
REPORTS_DIR = HOME_DIR / "reports"
DATA_DIR = HOME_DIR / "data"
LOG_TAIL_BYTES = 5 * 1024 * 1024
PROGRAM_FILES = ["BYBIT_SUPERTREND_LAB.py", "lab_core.py", "lab_storage.py", "lab_search.py", "lab_data.py",
                 "lab_report.py", "lab_gui.py", "lab_log.py", "lab_cli.py", "START.bat", "SELF_TEST.py", "SELF_TEST.bat",
                 "DIAGNOSE.py", "DIAGNOSE.bat", "requirements.txt", "README.txt"]


def build_walk_forward_plan(df: pd.DataFrame) -> dict:
    return core.build_walk_forward_plan_arrays(
        pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(np.float64),
        df["open"].to_numpy(np.float64), df["close"].to_numpy(np.float64))


@dataclass
class Trade:
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    gross_return_pct: float
    net_return_pct: float


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
            "sel_up_windows": row.sel_up_windows, "sel_up_positive_ratio": row.sel_up_positive_ratio,
            "sel_up_avg_return_pct": row.sel_up_avg_return_pct,
            "sel_down_windows": row.sel_down_windows, "sel_down_positive_ratio": row.sel_down_positive_ratio,
            "sel_down_avg_return_pct": row.sel_down_avg_return_pct,
            "sel_flat_windows": row.sel_flat_windows, "sel_flat_positive_ratio": row.sel_flat_positive_ratio,
            "sel_flat_avg_return_pct": row.sel_flat_avg_return_pct,
        }
        for prefix, metrics in (("train", row.train), ("test", row.test)):
            for k, v in asdict(metrics).items():
                rec[f"{prefix}_{k}"] = v
        records.append(rec)
    return pd.DataFrame(records)


def save_results(all_rows: list[AutoResult], top_rows: list[AutoResult], symbol: str, research_summary: Optional[dict] = None) -> tuple[Path, ...]:
    """Save human-sized extracts. The full result stream lives in research/."""
    from lab_storage import CHECKPOINT_USEFUL_EVERY
    RESULTS_DIR.mkdir(exist_ok=True)
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
        return retained_path, top_path, info
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
    result_files: Optional[list] = None,
) -> Path:
    """Create a reproducible v0.8 audit package.

    v0.8 keeps the ZIP bounded: only the result files of THIS run, the candle
    files actually used, the tail of the runtime log, and the explainability
    files (regions/*.json) of the research store. v0.7.1 packed every CSV in
    data/ and results/ (including unfinished downloads) and the whole log.

    The report intentionally separates SELECTION, VALIDATION and the historical
    DEVELOPMENT_HOLDOUT. Since earlier reports have already exposed the last
    historical block, v0.7 never calls it a fresh/untouched forward test.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol = (cfg or {}).get("symbol", "UNKNOWN")
    REPORTS_DIR.mkdir(exist_ok=True)
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
        "candidate_expansion_uses_validation":False,
        "adaptive_component_selection_uses_validation":False,
        "validation_leakage_fix":"v0.8: Adaptive V2 LONG/SHORT components are ranked with selection-window regime stats (sel_*); v0.7.1 used stats of all research windows including VALIDATION","candidate_expansion_uses_development_holdout":False,"final_rank_uses_development_holdout":False,
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
        z.writestr("REPORT/fresh_forward_test_status.json",json.dumps({"fresh_forward_test_available":False,"reason":"historical final block was inspected in earlier development reports","next_valid_test":"new candles collected after v0.8.x design"},ensure_ascii=False,indent=2))
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
        for name in PROGRAM_FILES:
            fp=BASE_DIR/name
            if fp.exists() and fp.is_file(): z.write(fp,f"PROGRAM/{name}")
        # Logs: only the tail of the runtime log + last error (v0.7.1: everything).
        if RUNTIME_LOG.exists():
            with RUNTIME_LOG.open("rb") as fh:
                fh.seek(max(0, RUNTIME_LOG.stat().st_size - LOG_TAIL_BYTES))
                z.writestr("LOGS/runtime_tail.log", fh.read())
        err = LOGS_DIR / "last_error.txt"
        if err.exists(): z.write(err, "LOGS/last_error.txt")
        for fp in (result_files or []):
            fp = Path(fp)
            if fp.exists() and fp.is_file(): z.write(fp, f"RESULTS/{fp.name}")
        if research_summary and research_summary.get("session_dir"):
            regions_dir = Path(str(research_summary["session_dir"])) / "regions"
            for fp in sorted(regions_dir.glob("*.json")):
                z.write(fp, f"RESEARCH/regions/{fp.name}")
        # Candles actually used by this run (never *_DOWNLOADING partial files).
        for tf, df in sorted(data_by_tf.items(), key=lambda x: int(x[0])):
            src = (getattr(df, "attrs", {}) or {}).get("source_path")
            name = f"{symbol}_{tf}m_used.csv"
            payload = df[["timestamp","open","high","low","close"]].to_csv(index=False).encode("utf-8")
            z.writestr(f"DATA/{name}", payload)
            z.writestr(f"DATA_HASHES/{name}.sha256.txt", hashlib.sha256(payload).hexdigest()+"  "+name+(f"  (source: {Path(src).name})" if src else "")+"\n")
    file_log(f"CHATGPT report created: {out}")
    return out

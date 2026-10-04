"""Multiple-testing audit: run the FULL program search (v0.8 pipeline, same
depth / timeframe / period / costs) on real candles and on many synthetic
"null" histories, and compare the real TOP with the distribution of what the
optimiser finds by chance.

Null models (both keep the exact real timestamps, so the walk-forward plan is
identical):

  sign   STRICT NULL. Every real candle is kept either as-is or mirrored
         (price reflected), chosen by a fair coin. Bar-by-bar |moves|, fat tails,
         volatility clustering, volatility regimes, gaps (open vs previous
         close) and wick shapes are preserved EXACTLY; direction and any
         directional persistence are destroyed, so no strategy can have a real
         edge. Whatever the optimiser "finds" here is pure selection luck.

  block  STATIONARY BLOCK BOOTSTRAP (Politis-Romano, mean block 3 days) of real
         candles. Keeps local autocorrelation / momentum, gaps, wick shapes and
         volatility clusters inside blocks, and the overall drift on average.
         Tests whether the specific real history matters beyond its generic
         statistical properties. It is NOT a no-edge null for trend following
         (momentum inside blocks survives).

    python audit/null_tests.py --tf 60 --model sign --seeds 0-59 --out results.jsonl
    python audit/null_tests.py --tf 60 --model real --out results.jsonl
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from common import core, load_binance  # noqa: F401  (sets sys.path)
from lab_search import ResearchRunner, Landscape  # noqa: F401
from lab_storage import ResearchStore


def decompose(m):
    o, h, l, c = (np.log(m[k]) for k in ("open", "high", "low", "close"))
    gap = np.r_[0.0, o[1:] - c[:-1]]
    body = c - o
    uw = h - np.maximum(o, c)
    lw = np.minimum(o, c) - l
    return gap, body, uw, lw, o[0]


def rebuild(m, gap, body, uw, lw, o0):
    n = len(gap)
    o = np.empty(n); c = np.empty(n)
    o[0] = o0; c[0] = o0 + body[0]
    # vectorised: c_t = o0 + cumsum(body) + cumsum(gap[1:])
    c = o0 + np.cumsum(body) + np.cumsum(gap)
    o = c - body
    h = np.maximum(o, c) + uw
    l = np.minimum(o, c) - lw
    out = dict(m)
    out.update(open=np.exp(o), high=np.exp(h), low=np.exp(l), close=np.exp(c))
    return out


def null_sign(m, seed):
    gap, body, uw, lw, o0 = decompose(m)
    s = np.random.default_rng(seed).choice([-1.0, 1.0], size=len(gap))
    flip = s < 0
    uw2 = np.where(flip, lw, uw); lw2 = np.where(flip, uw, lw)   # mirrored candle
    return rebuild(m, gap * s, body * s, uw2, lw2, o0)


def null_block(m, seed, mean_block_bars):
    gap, body, uw, lw, o0 = decompose(m)
    n = len(gap)
    rng = np.random.default_rng(seed)
    idx = np.empty(n, dtype=np.int64)
    p = 1.0 / mean_block_bars
    i = rng.integers(n)
    for t in range(n):
        if t > 0 and rng.random() < p:
            i = rng.integers(n)
        idx[t] = i
        i = (i + 1) % n
    return rebuild(m, gap[idx], body[idx], uw[idx], lw[idx], o0)


def neighbour_stats(rows, top):
    if top.strategy_type != "BASE":
        return None
    nb = [r for r in rows if r.strategy_type == "BASE" and r.mode == top.mode and abs(r.atr_period - top.atr_period) <= 3
          and abs(r.multiplier - top.multiplier) <= 0.25 and r is not top]
    if not nb:
        return {"n": 0}
    rob = np.array([r.robust_score for r in nb])
    return {"n": len(nb), "pos_ratio": float(np.mean(rob > 0)), "median_robust": float(np.median(rob))}


def summarize(res, store) -> dict:
    rows = res.report_pool
    valid = [r for r in rows if r.robust_score > -1e8]
    by_rob = sorted(valid, key=lambda r: r.robust_score, reverse=True)
    top = by_rob[0] if by_rob else None
    by_sel = max(rows, key=lambda r: r.selection_score)
    regions = {}
    for st in ("fine", "cluster_wide", "cluster_deep"):
        for p in (store.regions_dir).glob(f"*_{st}.json"):
            reg = json.loads(p.read_text())["regions"]
            regions[st] = max((x["q25_selection_score"] for x in reg), default=None)

    def row_info(r):
        return None if r is None else {
            "type": r.strategy_type, "mode": r.mode, "atr": r.atr_period, "mult": r.multiplier,
            "robust": r.robust_score, "selection": r.selection_score, "validation": r.validation_score,
            "research_ret": r.train.total_return_pct, "research_dd": r.train.max_drawdown_pct, "trades": r.train.trades,
            "research_pf": r.train.profit_factor, "wf_pos": r.wf_positive_ratio, "val_pos": r.validation_positive_ratio,
            "holdout_ret": r.test.total_return_pct, "holdout_dd": r.test.max_drawdown_pct, "holdout_trades": r.test.trades}
    rob = np.array([r.robust_score for r in by_rob]) if by_rob else np.array([])
    return {
        "top1": row_info(top), "best_selection": row_info(by_sel),
        "top10_mean_robust": float(rob[:10].mean()) if rob.size else None,
        "top100_mean_robust": float(rob[:100].mean()) if rob.size else None,
        "n_robust_positive": int((rob > 0).sum()),
        "top1_neighbours": neighbour_stats(rows, top) if top else None,
        "best_region_q25": regions,
        "top100_holdout_mean": float(np.mean([r.test.total_return_pct for r in by_rob[:100]])) if by_rob else None,
        "top100_holdout_pos": float(np.mean([r.test.total_return_pct > 0 for r in by_rob[:100]])) if by_rob else None,
    }


def run_one(market_tf: dict, tf: str, depth: str, workers: int, label: str) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="audit_null_"))
    try:
        cfg = {"symbol": f"AUD{label}", "months": 18, "tfs": [tf], "fee": 0.00055, "slippage": 0.0002, "depth": depth}
        store = ResearchStore.create(tmp, cfg, {tf: market_tf})
        t0 = time.monotonic()
        res = ResearchRunner(cfg, store, workers=workers, max_tasks_per_child=None).run()
        out = summarize(res, store)
        out["seconds"] = round(time.monotonic() - t0, 1)
        out["committed_rows"] = store.committed_rows
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="60")
    ap.add_argument("--model", choices=["real", "sign", "block"], required=True)
    ap.add_argument("--seeds", default="0-0")
    ap.add_argument("--depth", default="Глубокий")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = load_binance(a.tf)
    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds = [None] if a.model == "real" else range(lo, hi + 1)
    bars_per_day = 1440 // int(a.tf)
    for seed in seeds:
        if a.model == "real":
            mk = m
        elif a.model == "sign":
            mk = null_sign(m, 1000 + seed)
        else:
            mk = null_block(m, 2000 + seed, 3 * bars_per_day)
        mk["months"] = core.month_codes_from_ms(mk["timestamp"])
        summary = run_one(mk, a.tf, a.depth, a.workers, f"{a.model}{seed}")
        rec = {"tf": a.tf, "model": a.model, "seed": seed, "depth": a.depth, **summary}
        with open(a.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({"tf": a.tf, "model": a.model, "seed": seed, "sec": summary["seconds"],
                          "top1_robust": (summary["top1"] or {}).get("robust")}), flush=True)


if __name__ == "__main__":
    main()

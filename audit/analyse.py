"""Analysis of dense grids (search logic, score incentives, transfer) and of
the null-test series (multiple testing). Prints a JSON summary and writes
charts next to the output file.

    python audit/analyse.py --out-dir DIR --grid60 grid_60.npz --grid15 ... --grid5 ... --nulls null_60.jsonl null_15.jsonl --realism realism_60.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import core, load_binance

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_SIGN, C_BLOCK, C_REAL = "#2a78d6", "#1baf7a", "#eb6834"
MODES = ("BOTH", "LONG", "SHORT")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def neighbourhood(g, idx, dp=3, dm=0.25):
    sel = (g["mode"] == g["mode"][idx]) & (np.abs(g["period"] - g["period"][idx]) <= dp) & (np.abs(g["mult"] - g["mult"][idx]) <= dm + 1e-9)
    sel[idx] = False
    return sel


def plateau_scores(g, valid, rp, rm, key="selection_score", min_members=9):
    """q25 of `key` in each point's neighbourhood (the program's region rule)."""
    s = g[key]
    out = np.full(len(s), -np.inf)
    order = np.argsort(-np.where(valid, s, -np.inf))[:6000]
    for i in order:
        if not valid[i]:
            continue
        nb = (g["mode"] == g["mode"][i]) & (np.abs(g["period"] - g["period"][i]) <= rp) & (np.abs(g["mult"] - g["mult"][i]) <= rm + 1e-9)
        v = s[nb]
        if v.size >= min_members:
            out[i] = np.quantile(v, 0.25)
    return out


def analyse_grid(path, rp, rm):
    z = np.load(path, allow_pickle=True)
    g = {k: z[k] for k in z.files if k != "meta"}
    valid = (g["robust_score"] > -1e8)
    vs = (g["selection_score"] > -1e8) & (g["validation_score"] > -1e8)
    res = {"configs": int(len(g["period"])), "valid_robust": int(valid.sum())}
    # transfer: does an in-sample score predict the next phase?
    res["spearman_selection_vs_validation"] = spearman(g["selection_score"][vs], g["validation_score"][vs])
    res["spearman_robust_vs_holdout_return"] = spearman(g["robust_score"][valid], g["holdout_ret"][valid])
    res["spearman_research_return_vs_holdout_return"] = spearman(g["research_ret"][valid], g["holdout_ret"][valid])
    order_sel = np.argsort(-np.where(vs, g["selection_score"], -np.inf))
    top1pct = order_sel[: max(1, int(vs.sum() * 0.01))]
    res["top1pct_by_selection_validation_median"] = float(np.median(g["validation_score"][top1pct]))
    res["all_validation_median"] = float(np.median(g["validation_score"][vs]))
    order_rob = np.argsort(-np.where(valid, g["robust_score"], -np.inf))
    t100 = order_rob[:100]
    res["top100_by_robust"] = {
        "holdout_mean": float(g["holdout_ret"][t100].mean()), "holdout_pos_share": float((g["holdout_ret"][t100] > 0).mean()),
        "research_ret_median": float(np.median(g["research_ret"][t100])), "trades_median": float(np.median(g["trades"][t100])),
        "dd_median": float(np.median(g["research_dd"][t100])),
        "modes": {MODES[k]: int((g["mode"][t100] == k).sum()) for k in range(3)},
    }
    res["all_valid"] = {"holdout_mean": float(g["holdout_ret"][valid].mean()), "holdout_pos_share": float((g["holdout_ret"][valid] > 0).mean()),
                        "trades_median": float(np.median(g["trades"][valid])), "dd_median": float(np.median(g["research_dd"][valid]))}
    # score incentives
    res["spearman_robust_vs_trades"] = spearman(g["robust_score"][valid], g["trades"][valid])
    res["spearman_robust_vs_research_dd"] = spearman(g["robust_score"][valid], g["research_dd"][valid])
    res["spearman_robust_vs_research_return"] = spearman(g["robust_score"][valid], g["research_ret"][valid])
    top_ret = set(np.argsort(-np.where(valid, g["research_ret"], -np.inf))[:100].tolist())
    res["overlap_top100_return_vs_top100_robust"] = len(top_ret & set(t100.tolist()))
    # peak vs plateau of the TOP
    ratios = []
    for i in t100[:30]:
        nb = neighbourhood(g, i) & valid
        if nb.sum() >= 5:
            med = np.median(g["robust_score"][nb])
            ratios.append({"own": float(g["robust_score"][i]), "neigh_median": float(med), "neigh_pos": float((g["robust_score"][nb] > 0).mean())})
    res["top30_own_vs_neighbourhood"] = {
        "median_own": float(np.median([r["own"] for r in ratios])), "median_neigh_median": float(np.median([r["neigh_median"] for r in ratios])),
        "median_neigh_positive_share": float(np.median([r["neigh_pos"] for r in ratios]))}
    # best point vs best plateau (selection-only, as the search sees it)
    pl = plateau_scores(g, vs, rp, rm)
    ib, ip = int(order_sel[0]), int(np.argmax(pl))
    res["best_selection_point"] = {"mode": MODES[g["mode"][ib]], "atr": int(g["period"][ib]), "mult": float(g["mult"][ib]),
                                   "selection": float(g["selection_score"][ib]), "validation": float(g["validation_score"][ib]),
                                   "holdout": float(g["holdout_ret"][ib])}
    res["best_selection_plateau"] = {"mode": MODES[g["mode"][ip]], "atr": int(g["period"][ip]), "mult": float(g["mult"][ip]),
                                     "q25": float(pl[ip]), "selection": float(g["selection_score"][ip]),
                                     "validation": float(g["validation_score"][ip]), "holdout": float(g["holdout_ret"][ip])}
    # does plateau-picking transfer better than point-picking? (top-50 each, validation median)
    top_pl = np.argsort(-pl)[:50]
    res["validation_median_top50_points"] = float(np.median(g["validation_score"][order_sel[:50]]))
    res["validation_median_top50_plateaus"] = float(np.median(g["validation_score"][top_pl]))
    res["holdout_mean_top50_points"] = float(np.mean(g["holdout_ret"][order_sel[:50]]))
    res["holdout_mean_top50_plateaus"] = float(np.mean(g["holdout_ret"][top_pl]))
    res["best_robust_in_grid"] = float(g["robust_score"][order_rob[0]])
    res["_grid"] = g
    return res


def hierarchy_coverage(grid_res, realism_path, tf="60"):
    """Did the hierarchical search find what the exhaustive grid finds?"""
    r = json.loads(Path(realism_path).read_text())
    top = r["top"]
    best_h = max(x["robust_score"] for x in top if x["strategy_type"] == "BASE")
    g = grid_res["_grid"]
    valid = g["robust_score"] > -1e8
    better = int((g["robust_score"][valid] > best_h).sum())
    regions = r["trace"][tf]["regions"]
    bp = grid_res["best_selection_plateau"]
    dist = []
    for st, regs in regions.items():
        for x in regs:
            if x["mode"] == bp["mode"]:
                dist.append((abs(x["atr_period"] - bp["atr"]), abs(x["multiplier"] - bp["mult"]), st))
    near = min(dist, key=lambda t: (t[0] / 6 + t[1] / 0.1)) if dist else None
    return {"best_base_robust_found_by_hierarchy": best_h, "best_base_robust_in_dense_grid": grid_res["best_robust_in_grid"],
            "dense_configs_with_higher_robust": better, "nearest_region_to_best_dense_plateau": near,
            "regions_per_stage": {k: len(v) for k, v in regions.items()}}


def null_analysis(paths):
    recs = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    out = {}
    metrics = {
        "top1_robust": lambda r: r["top1"]["robust"] if r["top1"] else np.nan,
        "top10_mean_robust": lambda r: r["top10_mean_robust"],
        "top100_mean_robust": lambda r: r["top100_mean_robust"],
        "best_selection_score": lambda r: r["best_selection"]["selection"],
        "top1_research_return_pct": lambda r: r["top1"]["research_ret"] if r["top1"] else np.nan,
        "top1_research_dd_pct": lambda r: r["top1"]["research_dd"] if r["top1"] else np.nan,
        "top1_trades": lambda r: r["top1"]["trades"] if r["top1"] else np.nan,
        "top1_neighbour_positive_share": lambda r: (r["top1_neighbours"] or {}).get("pos_ratio", np.nan),
        "best_region_q25_fine": lambda r: r["best_region_q25"].get("fine"),
        "n_robust_positive": lambda r: r["n_robust_positive"],
        "top100_holdout_mean_pct": lambda r: r["top100_holdout_mean"],
    }
    for tf in sorted({r["tf"] for r in recs}, key=int):
        real = [r for r in recs if r["tf"] == tf and r["model"] == "real"]
        for model in ("sign", "block"):
            nulls = [r for r in recs if r["tf"] == tf and r["model"] == model]
            if not nulls or not real:
                continue
            res = {"n_runs": len(nulls)}
            for name, f in metrics.items():
                vals = np.array([f(r) if f(r) is not None else np.nan for r in nulls], dtype=float)
                vals = vals[np.isfinite(vals)]
                rv = f(real[0])
                if rv is None or vals.size == 0:
                    continue
                ge = int((vals >= rv).sum())
                res[name] = {"real": rv, "null_median": float(np.median(vals)), "null_p05": float(np.quantile(vals, 0.05)),
                             "null_p95": float(np.quantile(vals, 0.95)), "null_max": float(vals.max()),
                             "real_percentile": float((vals < rv).mean() * 100),
                             "p_value_one_sided": (ge + 1) / (vals.size + 1)}
            out[f"{tf}_{model}"] = res
    return out, recs


def chart_nulls(recs, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tfs = sorted({r["tf"] for r in recs}, key=int)
    panels = [("top1_robust", "Лучший итоговый score (TOP-1, robust)", lambda r: r["top1"]["robust"] if r["top1"] else None),
              ("top10", "Средний score TOP-10", lambda r: r["top10_mean_robust"]),
              ("ret", "Доходность TOP-1 в research-зоне, %", lambda r: r["top1"]["research_ret"] if r["top1"] else None)]
    fig, axes = plt.subplots(len(tfs), len(panels), figsize=(13, 3.6 * len(tfs)), facecolor=SURFACE, squeeze=False)
    for i, tf in enumerate(tfs):
        for j, (_k, title, f) in enumerate(panels):
            ax = axes[i][j]; ax.set_facecolor(SURFACE)
            data = []
            for model, color, name in (("sign", C_SIGN, "строгий null (без направления)"), ("block", C_BLOCK, "block-bootstrap")):
                v = [f(r) for r in recs if r["tf"] == tf and r["model"] == model and f(r) is not None]
                if v:
                    data.append((v, color, f"{name}, n={len(v)}"))
            lo = min(min(v) for v, *_ in data) if data else 0; hi = max(max(v) for v, *_ in data) if data else 1
            real = [f(r) for r in recs if r["tf"] == tf and r["model"] == "real"]
            if real:
                lo, hi = min(lo, real[0]), max(hi, real[0])
            bins = np.linspace(lo, hi, 25)
            for v, color, name in data:
                ax.hist(v, bins=bins, color=color, alpha=0.55, label=name, edgecolor=SURFACE, linewidth=1)
            if real:
                ax.axvline(real[0], color=C_REAL, linewidth=2.5, label="реальные данные")
                ax.annotate("реальные", xy=(real[0], ax.get_ylim()[1] * 0.92), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=9)
            ax.set_title(f"{tf}м · {title}", loc="left", fontsize=10, color=INK)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors=INK2, labelsize=8); ax.grid(color=GRID, linewidth=0.6, axis="y"); ax.set_axisbelow(True)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Что программа находит на случайных историях (гистограммы) и на реальной (линия)", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out_png, dpi=110, facecolor=SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--grid60"); ap.add_argument("--grid15"); ap.add_argument("--grid5")
    ap.add_argument("--nulls", nargs="*", default=[])
    ap.add_argument("--realism")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    grids = {}
    # Deep profile region radii: 2 x coarse step (6 periods, 0.10 multiplier)
    for tf, p in (("60", a.grid60), ("15", a.grid15), ("5", a.grid5)):
        if p and Path(p).exists():
            grids[tf] = analyse_grid(p, 12, 0.2)
            summary[f"grid_{tf}"] = {k: v for k, v in grids[tf].items() if k != "_grid"}
    if "60" in grids and a.realism:
        summary["hierarchy_vs_dense_60"] = hierarchy_coverage(grids["60"], a.realism)
    if a.nulls:
        summary["nulls"], recs = null_analysis(a.nulls)
        chart_nulls(recs, out / "null_tests.png")
    (out / "analysis.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()

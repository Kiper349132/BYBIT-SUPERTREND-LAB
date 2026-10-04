"""Density of SELECTION score vs VALIDATION score over all dense-grid configs."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID, REAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#eb6834"


def main(out, *grids):
    fig, axes = plt.subplots(1, len(grids), figsize=(5.2 * len(grids), 4.8), facecolor=SURFACE)
    for ax, spec in zip(np.atleast_1d(axes), grids):
        tf, path = spec.split("=")
        z = np.load(path, allow_pickle=True)
        s, v, rob = z["selection_score"], z["validation_score"], z["robust_score"]
        ok = (s > -1e8) & (v > -1e8)
        s, v = np.clip(s[ok], -1, 1), np.clip(v[ok], -1, 1)
        ax.set_facecolor(SURFACE)
        hb = ax.hexbin(s, v, gridsize=45, bins="log", cmap="Blues", mincnt=1, linewidths=0)
        top = np.argsort(-np.where(ok, rob, -np.inf))[:100]
        ax.scatter(np.clip(z["selection_score"][top], -1, 1), np.clip(z["validation_score"][top], -1, 1), s=14,
                   color=REAL, edgecolor=SURFACE, linewidth=0.8, label="TOP-100 программы", zorder=3)
        rho = np.corrcoef(np.argsort(np.argsort(s)), np.argsort(np.argsort(v)))[0, 1]
        ax.set_title(f"{tf}м · {ok.sum():,} конфигураций · ранговая корреляция {rho:+.2f}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("SELECTION score (то, по чему ищется)", color=INK2); ax.set_ylabel("VALIDATION score (следующий период)", color=INK2)
        ax.axhline(0, color=GRID, linewidth=1); ax.axvline(0, color=GRID, linewidth=1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.colorbar(hb, ax=ax, shrink=0.8).set_label("число конфигураций (лог.)", color=INK2)
    fig.tight_layout(); fig.savefig(out, dpi=110, facecolor=SURFACE)


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])

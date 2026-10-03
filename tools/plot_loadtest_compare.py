"""Before/after chart from two tools/loadtest.py CSV files (same scenario).

    python tools/plot_loadtest_compare.py OLD.csv NEW.csv OUT.png "title"
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OLD, NEW = "#eb6834", "#2a78d6"     # categorical slots 2 and 1 of the reference palette


def load(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    f = lambda k: [float(r[k]) if r[k] not in ("", None) else float("nan") for r in rows]
    return {"min": [x / 60 for x in f("t_sec")], "parent": f("parent_mb"), "worker": f("workers_max_mb"),
            "total": f("total_mb"), "rows": [x / 1000 for x in f("rows")]}


def main(old_csv, new_csv, out_png, title):
    old, new = load(old_csv), load(new_csv)
    panels = [("parent", "Родительский процесс (GUI/оркестратор), МБ"),
              ("worker", "Самый большой расчётный процесс, МБ"),
              ("total", "Всё вместе: родитель + все расчётные процессы, МБ")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10.5, 9.5), sharex=True, facecolor=SURFACE)
    for ax, (key, label) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for d, color, name in ((old, OLD, "v0.7.1"), (new, NEW, "v0.8")):
            xs = [x for x, y in zip(d["min"], d[key]) if y == y and not (key == "worker" and y == 0)]
            ys = [y for y in d[key] if y == y and not (key == "worker" and y == 0)]
            if not xs:
                continue
            ax.plot(xs, ys, color=color, linewidth=2, label=name, solid_capstyle="round")
            ax.annotate(f"{name}: {ys[-1]:,.0f}" if key == "rows" else f"{name}: макс {max(ys):,.0f}",
                        xy=(xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
        ax.set_title(label, loc="left", fontsize=10.5, color=INK)
        ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    axes[-1].set_xlabel("минуты от старта", color=INK2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 0.93, 0.98))
    fig.savefig(out_png, dpi=110, facecolor=SURFACE)


if __name__ == "__main__":
    main(*sys.argv[1:5])

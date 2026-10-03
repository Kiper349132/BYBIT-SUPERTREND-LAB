"""Long-run memory load test.

Runs a heavy research on synthetic candles for a fixed time and samples the
resident memory (RSS) of the parent process and of every worker process every
`--interval` seconds. Produces a CSV time series and a PNG chart.

    python tools/loadtest.py --minutes 20 --workers 3 --spec 3:262800,15:52560 --depth Глубокий --out reports/loadtest_new
    python tools/loadtest.py ... --old path/to/v0.7.1/BYBIT_SUPERTREND_LAB.py --out reports/loadtest_old

`--old` drives the original v0.7.1 search code (for before/after comparison)
with the spawn start method that Windows uses. Requires psutil
(pip install psutil). Works on Windows and Linux.

NOTE: top-level imports are deliberately tiny - worker processes re-import
this file as __mp_main__.
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sampler(stop, out_rows, interval, extra):
    import psutil
    me = psutil.Process(os.getpid())
    t0 = time.monotonic()
    while not stop.is_set():
        t = time.monotonic() - t0
        try:
            parent = me.memory_info().rss / 1048576
            workers = {}
            uss = {}
            for ch in me.children(recursive=True):
                try:
                    cmd = " ".join(ch.cmdline())
                    if "spawn_main" in cmd or "multiprocessing-fork" in cmd:
                        if "resource_tracker" in cmd:
                            continue
                        workers[ch.pid] = ch.memory_info().rss / 1048576
                        # USS = private memory (what counts against the Windows commit limit);
                        # RSS also includes shared libraries and memory-mapped candle files.
                        uss[ch.pid] = ch.memory_full_info().uss / 1048576
                except Exception:
                    pass
            vm = psutil.virtual_memory()
            out_rows.append({"t": round(t, 1), "parent_mb": round(parent, 1), "workers": workers,
                             "workers_uss_max_mb": round(max(uss.values()), 1) if uss else 0.0,
                             "parent_uss_mb": round(me.memory_full_info().uss / 1048576, 1),
                             "workers_sum_mb": round(sum(workers.values()), 1),
                             "system_used_pct": vm.percent, **extra()})
        except Exception as exc:
            out_rows.append({"t": round(t, 1), "error": str(exc)})
        stop.wait(interval)


def _synthetic(spec, seed=20261002):
    sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
    from synthetic import synthetic_market
    return {tf: synthetic_market(int(n), tf, seed=seed + i) for i, (tf, n) in enumerate(p.split(":") for p in spec.split(","))}


def run_new(args, market, cfg, stop_evt, extra_holder):
    sys.path.insert(0, str(ROOT))
    from lab_search import ResearchRunner
    from lab_storage import ResearchStore
    store = ResearchStore.create(Path(args.workdir), cfg, market)
    extra_holder["fn"] = lambda: {"committed_rows": store.committed_rows, "committed_useful": store.committed_useful + store.pending_useful}
    res = ResearchRunner(cfg, store, workers=args.workers, stop_event=stop_evt,
                         max_tasks_per_child=args.max_tasks_per_child or None).run()
    return res.status


def run_old(args, market, cfg, stop_evt, extra_holder):
    import importlib.util
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    import pandas as pd
    old_path = Path(args.old).resolve()
    sys.path.insert(0, str(old_path.parent))
    spec = importlib.util.spec_from_file_location(old_path.stem, old_path)
    old = importlib.util.module_from_spec(spec); sys.modules[old_path.stem] = old; spec.loader.exec_module(old)
    old.ProcessPoolExecutor = partial(ProcessPoolExecutor, mp_context=mp.get_context("spawn"))   # Windows behaviour
    data = {}
    for tf, m in market.items():
        df = pd.DataFrame({k: m[k] for k in ("timestamp", "open", "high", "low", "close")})
        df["volume"] = 0.0; df["turnover"] = 0.0
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        data[tf] = df
    pers = old.ResearchPersistence(cfg, data)
    extra_holder["fn"] = lambda: {"committed_rows": pers.committed_rows + len(pers.buffer_rows),
                                  "committed_useful": pers.committed_useful + pers.buffer_useful}

    class FakeApp:
        stop_event = stop_evt
        search_started_at = time.monotonic()
        search_stats = []

        def _post(self, *a):
            pass
    try:
        old.App._run_full_search(FakeApp(), cfg, data, pers)
        return "complete"
    except old.StopRequested:
        pers.safe_stop("loadtest_stop")
        return "stopped"


def plot(rows, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in rows if "error" not in r]
    t = [r["t"] / 60 for r in rows]
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(t, [r["parent_mb"] for r in rows], label="родительский процесс", color="#2a6fdb", linewidth=2)
    pids = sorted({pid for r in rows for pid in r["workers"]})
    for i, pid in enumerate(pids):
        xs = [r["t"] / 60 for r in rows if pid in r["workers"]]
        ys = [r["workers"][pid] for r in rows if pid in r["workers"]]
        ax[0].plot(xs, ys, color="#e07b39", alpha=0.6, linewidth=1, label="workers (каждый)" if i == 0 else None)
    ax[0].set_ylabel("RSS, МБ"); ax[0].legend(loc="upper left"); ax[0].grid(alpha=0.3); ax[0].set_title(title)
    ax[1].plot(t, [r["parent_mb"] + r["workers_sum_mb"] for r in rows], color="#444", label="родитель + все workers")
    ax[1].set_ylabel("суммарно, МБ"); ax[1].set_xlabel("минуты"); ax[1].grid(alpha=0.3); ax[1].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--spec", default="3:262800,15:52560")
    ap.add_argument("--depth", default="Глубокий")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--max-tasks-per-child", type=int, default=400)
    ap.add_argument("--old", default="")
    ap.add_argument("--workdir", default="")
    ap.add_argument("--out", default=str(ROOT / "reports" / "loadtest"))
    args = ap.parse_args()
    import tempfile
    args.workdir = args.workdir or tempfile.mkdtemp(prefix="lab_loadtest_")
    if args.old:
        os.environ.setdefault("LAB_HOME", args.workdir)
    market = _synthetic(args.spec)
    tfs = sorted(market, key=int)
    cfg = {"symbol": "LOAD", "months": 18, "tfs": tfs, "fee": 0.00055, "slippage": 0.0002, "depth": args.depth, "workers": args.workers}
    stop_run = threading.Event(); stop_sampling = threading.Event()
    rows = []
    extra = {"fn": lambda: {}}
    sampler = threading.Thread(target=_sampler, args=(stop_sampling, rows, args.interval, lambda: extra["fn"]()), daemon=True)
    sampler.start()
    timer = threading.Timer(args.minutes * 60, stop_run.set); timer.start()
    t0 = time.monotonic()
    status = (run_old if args.old else run_new)(args, market, cfg, stop_run, extra)
    stop_at = time.monotonic()
    timer.cancel(); time.sleep(args.interval * 2); stop_sampling.set(); sampler.join()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_sec", "parent_mb", "workers_n", "workers_max_mb", "workers_sum_mb", "total_mb", "system_used_pct", "rows", "useful",
                    "parent_uss_mb", "workers_uss_max_mb"])
        for r in rows:
            if "error" in r:
                continue
            ws = list(r["workers"].values())
            w.writerow([r["t"], r["parent_mb"], len(ws), round(max(ws), 1) if ws else 0, r["workers_sum_mb"],
                        round(r["parent_mb"] + r["workers_sum_mb"], 1), r["system_used_pct"], r.get("committed_rows", ""), r.get("committed_useful", ""),
                        r.get("parent_uss_mb", ""), r.get("workers_uss_max_mb", "")])
    good = [r for r in rows if "error" not in r and r["workers"]]
    summary = {
        "mode": "v0.7.1" if args.old else "v0.8", "status": status, "spec": args.spec, "depth": args.depth, "workers": args.workers,
        "run_seconds": round(stop_at - t0, 1), "samples": len(rows),
        "parent_peak_mb": max((r["parent_mb"] for r in good), default=0),
        "worker_peak_mb": max((max(r["workers"].values()) for r in good), default=0),
        "worker_private_uss_peak_mb": max((r.get("workers_uss_max_mb", 0) for r in good), default=0),
        "parent_private_uss_peak_mb": max((r.get("parent_uss_mb", 0) for r in good), default=0),
        "total_peak_mb": max((r["parent_mb"] + r["workers_sum_mb"] for r in good), default=0),
        "rows_at_end": good[-1].get("committed_rows") if good else 0, "useful_at_end": good[-1].get("committed_useful") if good else 0,
    }
    Path(str(out) + ".json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    plot(rows, str(out) + ".png", f"{summary['mode']} · {args.depth} · {args.spec} · {args.workers} workers")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

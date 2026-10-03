"""Headless research runner (no GUI). Used by tests and load tests; also handy
for long runs on a server.

    python lab_cli.py run --research-dir DIR --synthetic 60:9000,30:16000 --depth Тест --workers 3
    python lab_cli.py run --research-dir DIR --resume RUN_DIR

Ctrl+C (SIGINT) performs the same safe stop as the GUI STOP button: workers are
stopped, finished units are committed, exit code 2.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import lab_core as core
from lab_search import ResearchRunner, recommended_workers
from lab_storage import ResearchStore


def _synthetic(spec: str, seed: int) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
    from synthetic import synthetic_market
    out = {}
    for i, part in enumerate(spec.split(",")):
        tf, n = part.split(":")
        out[tf] = synthetic_market(int(n), tf, seed=seed + i)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--research-dir", required=True)
    r.add_argument("--resume", default="")
    r.add_argument("--synthetic", default="60:9000")
    r.add_argument("--seed", type=int, default=20261002)
    r.add_argument("--depth", default="Тест")
    r.add_argument("--months", type=int, default=12)
    r.add_argument("--workers", type=int, default=0)
    r.add_argument("--max-tasks-per-child", type=int, default=400)
    r.add_argument("--checkpoint-every", type=int, default=0)
    r.add_argument("--result-json", default="")
    a = ap.parse_args(argv)

    # Crash injection for tests: simulate a hard process death (power loss,
    # Windows killing the process) at precise points.
    crash_at = os.environ.get("LAB_TEST_CRASH_AT", "")
    if crash_at:
        import lab_storage
        point, nth = crash_at.split(":")
        counter = {"n": 0}

        def _hook(p):
            if p == point:
                counter["n"] += 1
                if counter["n"] >= int(nth):
                    os._exit(77)
        lab_storage.CRASH_HOOK = _hook
    crash_units = int(os.environ.get("LAB_TEST_CRASH_AFTER_UNITS", "0") or 0)
    unit_counter = {"n": 0}

    def unit_hook(runner, tf, stage, uid):
        unit_counter["n"] += 1
        if crash_units and unit_counter["n"] >= crash_units:
            os._exit(77)

    if a.checkpoint_every:
        import lab_storage
        lab_storage.CHECKPOINT_USEFUL_EVERY = int(a.checkpoint_every)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    if a.resume:
        store = ResearchStore.open(Path(a.resume))
        cfg = dict(store.run_meta["config"])
    else:
        market = _synthetic(a.synthetic, a.seed)
        cfg = {"symbol": "SYNTH", "months": a.months, "tfs": sorted(market, key=int), "fee": 0.00055,
               "slippage": 0.0002, "depth": a.depth}
        store = ResearchStore.create(Path(a.research_dir), cfg, market)
    print(json.dumps({"event": "store", "root": str(store.root), "pid": os.getpid()}), flush=True)
    max_candles = max(int(n) for n in store.run_meta.get("rows", {"x": 0}).values())
    workers = a.workers or recommended_workers(max_candles)[0]
    t0 = time.monotonic()
    res = ResearchRunner(cfg, store, workers=workers, stop_event=stop,
                         max_tasks_per_child=a.max_tasks_per_child or None, unit_hook=unit_hook,
                         on_status=lambda d: d.get("log") and print(json.dumps({"event": "log", "text": d["log"]}, ensure_ascii=False), flush=True)).run()
    out = {"event": "done", "status": res.status, "seconds": round(time.monotonic() - t0, 3), "root": str(store.root),
           "committed_useful": store.committed_useful, "message": res.message}
    if res.status == "complete":
        store.finalize({"cli": True})
        if a.result_json:
            Path(a.result_json).write_text(json.dumps({
                "top": [core.autoresult_to_dict(x) for x in res.top],
                "report_pool": [core.autoresult_to_dict(x) for x in res.report_pool],
                "trace": res.trace,
            }, ensure_ascii=False, default=list), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return 0 if res.status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

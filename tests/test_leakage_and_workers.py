"""VALIDATION must not steer the search; workers must stay lean and not leak."""
import time

import numpy as np

import lab_core as core
from lab_search import ResearchRunner, WorkerPool
from lab_storage import ResearchStore
from resume_helpers import make_cfg, synthetic_set


def _perturb_after_validation_start(market: dict) -> dict:
    """Rewrite every candle from the first VALIDATION window on (validation +
    development holdout). Everything before it - the SELECTION phase - is
    untouched, and all indicators are causal."""
    out = {}
    for tf, m in market.items():
        plan = core.build_walk_forward_plan_arrays(m["timestamp"], m["open"], m["close"])
        vs = int(plan["validation_start"])
        rng = np.random.default_rng(99)
        k = len(m["close"]) - vs
        factor = np.exp(np.cumsum(0.004 * rng.normal(size=k)))   # a completely different path
        new = {key: np.array(v, copy=True) for key, v in m.items()}
        for key in ("open", "high", "low", "close"):
            new[key][vs:] = m[key][vs:] * factor
        new["high"][vs:] = np.maximum.reduce([new["high"][vs:], new["open"][vs:], new["close"][vs:]])
        new["low"][vs:] = np.minimum.reduce([new["low"][vs:], new["open"][vs:], new["close"][vs:]])
        out[tf] = new
    return out


def test_validation_data_cannot_change_search_decisions(tmp_path):
    base = synthetic_set()
    pert = _perturb_after_validation_start(base)
    traces, tops = [], []
    for i, market in enumerate((base, pert)):
        store = ResearchStore.create(tmp_path / str(i), make_cfg(), market)
        res = ResearchRunner(make_cfg(), store, workers=2).run()
        assert res.status == "complete"
        traces.append(res.trace); tops.append(res.top)
    for tf in traces[0]:
        a, b = traces[0][tf], traces[1][tf]
        # every SEARCH decision is identical ...
        assert a["regions"] == b["regions"]
        assert a["candidates"] == b["candidates"]          # fine / cluster_wide / cluster_deep
        assert a["components"] == b["components"]          # Adaptive V2 LONG/SHORT parts
    # ... while validation-based ranking did see the different data
    assert [r.validation_score for r in tops[0]] != [r.validation_score for r in tops[1]]


def test_workers_do_not_import_gui_or_pandas(tmp_path):
    market = synthetic_set("60:9000")
    store = ResearchStore.create(tmp_path, {**make_cfg(), "tfs": ["60"]}, market)
    plans = {"60": core.build_walk_forward_plan_arrays(market["60"]["timestamp"], market["60"]["open"], market["60"]["close"])}
    pool = WorkerPool(2, store.market_dir, plans, max_tasks_per_child=None)
    try:
        list(pool.run(core._worker_eval_period, [(i, ("60", 10 + i, [(2.0, ("BOTH",))], 0.00055, 0.0002, 26, "broad")) for i in range(4)],
                      lambda: None))
        probes = [pool.executor.submit(core._worker_probe).result() for _ in range(4)]
    finally:
        pool.close()
    for p in probes:
        assert p["heavy_modules"] == [], p
        assert p["rss_mb"] < 200, p


def test_single_pool_no_memory_growth(tmp_path):
    """Hundreds of tasks through ONE pool: worker RSS must plateau."""
    market = synthetic_set("60:9000")
    store = ResearchStore.create(tmp_path, {**make_cfg(), "tfs": ["60"]}, market)
    plans = {"60": core.build_walk_forward_plan_arrays(market["60"]["timestamp"], market["60"]["open"], market["60"]["close"])}
    pool = WorkerPool(2, store.market_dir, plans, max_tasks_per_child=None)
    specs = [(round(1.0 + 0.25 * j, 3), ("BOTH", "LONG", "SHORT")) for j in range(12)]
    samples = []
    try:
        for rnd in range(8):
            items = [(i, ("60", 5 + (i % 60), specs, 0.00055, 0.0002, 26, "broad")) for i in range(60)]
            for _item, res in pool.run(core._worker_eval_period, items, lambda: None):
                del res
            rss = [pool.executor.submit(core._worker_probe).result()["rss_mb"] for _ in range(6)]
            samples.append(max(rss))
    finally:
        pool.close()
    # first round includes warm-up; afterwards growth must be negligible
    growth = samples[-1] - samples[1]
    assert growth < 8.0, samples


def test_max_tasks_per_child_recycles_without_hanging(tmp_path):
    market = synthetic_set("60:9000")
    store = ResearchStore.create(tmp_path, {**make_cfg(), "tfs": ["60"]}, market)
    plans = {"60": core.build_walk_forward_plan_arrays(market["60"]["timestamp"], market["60"]["open"], market["60"]["close"])}
    pool = WorkerPool(2, store.market_dir, plans, max_tasks_per_child=3)
    t0 = time.monotonic()
    try:
        items = [(i, ("60", 5 + i, [(2.0, ("BOTH",))], 0.00055, 0.0002, 26, "broad")) for i in range(40)]
        done = list(pool.run(core._worker_eval_period, items, lambda: None))
    finally:
        pool.close()
    assert len(done) == 40
    assert time.monotonic() - t0 < 120

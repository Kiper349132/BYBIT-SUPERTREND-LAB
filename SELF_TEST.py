"""Локальная проверка BYBIT SUPERTREND LAB v0.8 без интернета (1-2 минуты).

Полный набор автоматических тестов: python -m pytest tests
"""
import json
import shutil
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "tests")]


def main() -> int:
    import numpy as np

    import lab_core as core
    from lab_search import ResearchRunner
    from lab_storage import ResearchStore
    from resume_helpers import canonical, make_cfg, synthetic_set

    m = synthetic_set("60:9000")["60"]
    h, l, c = m["high"], m["low"], m["close"]
    atr = core.wilder_atr(h, l, c, 14)
    assert np.array_equal(atr, core._fast_atr(h, l, c, 14), equal_nan=True)
    trend, _ = core.supertrend_from_atr(h, l, c, atr, 2.5)
    assert np.array_equal(trend, core._fast_trend(h, l, c, atr, 2.5))
    print("Supertrend numba == numpy: PASS")

    # Adaptive regime must be past-only: changing the future cannot change the past.
    cut = len(c) // 2
    h2, l2, c2 = h.copy(), l.copy(), c.copy()
    h2[cut + 1:] *= 1.7; l2[cut + 1:] *= 1.7; c2[cut + 1:] *= 1.7
    r1 = core.online_regime_v2(h, l, c, 8, 21, 14, 20.0, 0.15, 3, 6)
    r2 = core.online_regime_v2(h2, l2, c2, 8, 21, 14, 20.0, 0.15, 3, 6)
    assert np.array_equal(r1[:cut + 1], r2[:cut + 1])
    print("Adaptive V2 past-only: PASS")

    tmp = Path(tempfile.mkdtemp(prefix="lab_selftest_"))
    try:
        cfg = make_cfg()
        market = synthetic_set()
        s1 = ResearchStore.create(tmp / "a", cfg, market)
        full = ResearchRunner(cfg, s1, workers=2).run()
        assert full.status == "complete" and len(full.top) == 100

        # stop in the middle, damage the index, restart, resume
        s2 = ResearchStore.create(tmp / "b", cfg, market)
        stop = threading.Event()
        n = {"k": 0}

        def hook(runner, *_):
            n["k"] += 1
            if n["k"] == 40:
                stop.set()
        part = ResearchRunner(cfg, s2, workers=2, stop_event=stop, unit_hook=hook).run()
        assert part.status == "stopped"
        s2.index_path.write_text("damaged")
        resumed = ResearchRunner(cfg, ResearchStore.open(s2.root), workers=3).run()
        assert resumed.status == "complete"
        a = canonical(full.top, full.report_pool, full.trace)
        b = canonical(resumed.top, resumed.report_pool, resumed.trace)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        print("Стоп + повреждённый checkpoint + продолжение == непрерывный прогон: PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("SELF TEST: PASS")
    print(f"Numba: {core.NUMBA_AVAILABLE}")
    return 0


if __name__ == "__main__":   # required: worker processes re-import this file
    raise SystemExit(main())

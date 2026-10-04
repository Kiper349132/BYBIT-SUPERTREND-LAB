"""Power check for the null test: plant a small, known directional persistence
into strict-null histories (bar signs follow a 2-state Markov chain that keeps
the previous sign with probability q) and see whether the full program search
then scores above the strict-null 95th percentile."""
import json
import sys

import numpy as np

from common import core, load_binance
from null_tests import decompose, rebuild, run_one


def planted(m, seed, q):
    gap, body, uw, lw, o0 = decompose(m)
    rng = np.random.default_rng(seed)
    s = np.empty(len(gap)); s[0] = 1.0
    keep = rng.random(len(gap)) < q
    for t in range(1, len(gap)):
        s[t] = s[t - 1] if keep[t] else -s[t - 1]
    # the sign applies to the bar's direction: align with the real bar's own sign
    base = np.sign(body + gap); base[base == 0] = 1.0
    flip = s * base
    uw2 = np.where(flip < 0, lw, uw); lw2 = np.where(flip < 0, uw, lw)
    return rebuild(m, gap * flip, body * flip, uw2, lw2, o0)


if __name__ == "__main__":
    tf, out = "60", sys.argv[1]
    m = load_binance(tf)
    for q in (0.52, 0.53):
        for seed in range(5):
            mk = planted(m, 5000 + seed, q)
            r = np.diff(np.log(mk["close"])); ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
            mk["months"] = core.month_codes_from_ms(mk["timestamp"])
            s = run_one(mk, tf, "Глубокий", 3, f"pow{q}_{seed}")
            rec = {"q": q, "seed": seed, "ac1": ac1, "top1_robust": s["top1"]["robust"] if s["top1"] else None,
                   "top1_ret": s["top1"]["research_ret"] if s["top1"] else None, "holdout_top100": s["top100_holdout_mean"]}
            print(json.dumps(rec), flush=True)
            with open(out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

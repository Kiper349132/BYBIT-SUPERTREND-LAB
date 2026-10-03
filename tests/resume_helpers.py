import json

import lab_core as core
from synthetic import synthetic_market

SEED = 20261002
SYNTH_SPEC = "30:16000,60:9000"


def synthetic_set(spec: str = SYNTH_SPEC, seed: int = SEED) -> dict:
    # identical to lab_cli._synthetic
    out = {}
    for i, part in enumerate(spec.split(",")):
        tf, n = part.split(":")
        out[tf] = synthetic_market(int(n), tf, seed=seed + i)
    return out


def make_cfg(depth="Тест"):
    return {"symbol": "SYNTH", "months": 12, "tfs": ["30", "60"], "fee": 0.00055, "slippage": 0.0002, "depth": depth}


def canonical(top, report_pool, trace) -> dict:
    """Everything the run decided, as plain JSON (floats compared exactly;
    json keeps NaN/inf as tokens so they compare equal too)."""
    payload = {
        "top": [core.autoresult_to_dict(x) if not isinstance(x, dict) else x for x in top],
        "report_pool": [core.autoresult_to_dict(x) if not isinstance(x, dict) else x for x in report_pool],
        "trace": trace,
    }
    return json.loads(json.dumps(payload, sort_keys=True, default=list))


def diff_summary(a: dict, b: dict) -> str:
    out = []
    for k in ("top", "report_pool"):
        if a[k] != b[k]:
            out.append(f"{k}: {len(a[k])} vs {len(b[k])} rows differ")
    for tf in sorted(set(a["trace"]) | set(b["trace"])):
        ta, tb = a["trace"].get(tf, {}), b["trace"].get(tf, {})
        for key in sorted(set(ta) | set(tb)):
            if ta.get(key) != tb.get(key):
                out.append(f"trace[{tf}][{key}] differs")
    return "; ".join(out) or "identical"

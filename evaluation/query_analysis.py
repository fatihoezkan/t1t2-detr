"""Per-query usage on the test split for one finished run.

Reports per output slot, over the test voxels: active_rate (share of voxels with existence
probability >= theta), max_prob, mean_prob, median predicted weight when active, and the 10th
to 90th percentile of predicted T1 and T2 when active. theta is the run's own fitted threshold
from summary.json. Writes results/<run>/query_analysis.json and prints a table.

Usage: PYTHONPATH=.:datagen python3 evaluation/query_analysis.py <run> [<run> ...]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

from t1t2.runs import load_run


def analyse(run, device="cpu"):
    """Measure which query slots are active and what they predict."""
    loaded = load_run(Path("results") / run, device)
    cfg, rd, theta = loaded.cfg, loaded.dir, loaded.fitted_threshold
    q, _ = loaded.predict("test")
    params = np.asarray(q["params"])          # (N, Q, 3) T1 ms / T2 ms / w
    probs = np.asarray(q["exist_prob"])       # (N, Q)
    n_vox, n_q = probs.shape

    slots = []
    for j in range(n_q):
        act = probs[:, j] >= theta
        n_act = int(act.sum())
        rec = {
            "slot": j + 1,                    # 1-indexed, as the thesis writes them
            "active_count": n_act,
            "active_rate": n_act / n_vox,
            "max_prob": float(probs[:, j].max()),
            "mean_prob": float(probs[:, j].mean()),
        }
        if n_act:
            t1, t2, w = params[act, j, 0], params[act, j, 1], params[act, j, 2]
            rec.update(
                median_w_active=float(np.median(w)),
                median_t1_active_ms=float(np.median(t1)),
                median_t2_active_ms=float(np.median(t2)),
                t1_p10_ms=float(np.percentile(t1, 10)), t1_p90_ms=float(np.percentile(t1, 90)),
                t2_p10_ms=float(np.percentile(t2, 10)), t2_p90_ms=float(np.percentile(t2, 90)),
            )
        else:
            rec.update({k: None for k in
                        ("median_w_active", "median_t1_active_ms", "median_t2_active_ms",
                         "t1_p10_ms", "t1_p90_ms", "t2_p10_ms", "t2_p90_ms")})
        slots.append(rec)

    active = [s for s in slots if s["active_count"] > 0]
    never = [s for s in slots if s["active_count"] == 0]
    out = {
        "run": run, "theta": theta, "theta_source": "run's own fitted threshold",
        "n_voxels": n_vox, "n_queries": n_q,
        "t1_range_ms": [cfg.data.t1_min, cfg.data.t1_max],
        "t2_range_ms": [cfg.data.t2_min, cfg.data.t2_max],
        "n_slots_ever_active": len(active),
        "n_slots_never_active": len(never),
        "never_active_slots": [s["slot"] for s in never],
        "never_active_max_prob_range": (
            [min(s["max_prob"] for s in never), max(s["max_prob"] for s in never)]
            if never else None),
        "slots": slots,
    }
    (rd / "query_analysis.json").write_text(json.dumps(out, indent=1))
    return out


def report(o):
    """Print query activity, prediction ranges, and duplicate counts."""
    print(f"\n=== {o['run']}   theta={o['theta']:.2f}   {o['n_voxels']} test voxels ===")
    print(f"{'slot':>4} {'active%':>8} {'max prob':>9} {'med w':>7} "
          f"{'T1 p10-p90 (ms)':>19} {'T2 p10-p90 (ms)':>19}")
    for s in o["slots"]:
        if s["active_count"]:
            print(f"{s['slot']:>4} {100*s['active_rate']:8.2f} {s['max_prob']:9.4f} "
                  f"{s['median_w_active']:7.3f} "
                  f"{s['t1_p10_ms']:8.0f}-{s['t1_p90_ms']:<10.0f} "
                  f"{s['t2_p10_ms']:8.0f}-{s['t2_p90_ms']:<10.0f}")
        else:
            print(f"{s['slot']:>4} {0.0:8.2f} {s['max_prob']:9.4f} {'--':>7} "
                  f"{'--':>19} {'--':>19}")
    print(f"  ever active: {o['n_slots_ever_active']}   never active: "
          f"{o['n_slots_never_active']} (slots {o['never_active_slots']}, "
          f"max prob {o['never_active_max_prob_range'][0]:.3f}-"
          f"{o['never_active_max_prob_range'][1]:.3f})")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        report(analyse(r))

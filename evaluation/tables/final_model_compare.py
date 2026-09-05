#!/usr/bin/env python3
"""Score the final model against the reference, four seeds against four seeds, and print it.

The criteria are the ones written into configs/seeds/final_uniform_q6_seed*.yaml before any of
those runs existed; this only reads the stored output and applies them. Reads
results/threshold_sweep/<run>.json, results/nd_evaluation/<run>.json and
results/nd_evaluation/tables_2d_3d.json. Prints only, writes nothing.
Usage: python3 evaluation/tables/final_model_compare.py [--dry]
       --dry checks the mechanics on the reference group against itself
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

REF = ["baseline_v2_reproduction", "baseline_seed20260725",
       "baseline_seed20260726", "baseline_seed20260727"]
FINAL = [f"final_uniform_q6_seed{s}" for s in (20260724, 20260725, 20260726, 20260727)]
LOSS_UNIFORM = "loss_uniform"

# rulers: range over the four reference seeds (Section 5.1 of the thesis)
RULER = {"strict@.50": 1.95, "strict@.75": 0.98, "mAP@7 2D": 0.0168,
         "mAP@7 3D": 0.0199, "K=3 strict": 2.85, "count@.75": 0.80}


def _sweep(r, dim, th, key):
    d = json.load(open(RES / "threshold_sweep" / f"{r}.json"))[dim]
    return next(x for x in d if abs(x["threshold"] - th) < 1e-9)[key]


def _perk(r, k):
    d = json.load(open(RES / "threshold_sweep" / f"{r}.json"))["2d_by_k"]["0.75"]
    if "strict" in d:
        return d["strict"][k]
    return d[k] if isinstance(d[k], float) else d[k]["strict"]


def _map(r):
    m2 = json.load(open(RES / "nd_evaluation" / f"{r}.json"))["map"]["map@7"]
    t = json.load(open(RES / "nd_evaluation" / "tables_2d_3d.json"))
    return m2, (t[r]["3d"]["map"]["map@7"] if r in t else None)


def metrics(r):
    m2, m3 = _map(r)
    return {"strict@.50": _sweep(r, "2d", 0.50, "voxel_acc"),
            "strict@.75": _sweep(r, "2d", 0.75, "voxel_acc"),
            "count@.75": _sweep(r, "2d", 0.75, "count_acc"),
            "mAP@7 2D": m2, "mAP@7 3D": m3, "K=3 strict": _perk(r, "3")}


ORDER = ["strict@.50", "strict@.75", "K=3 strict", "count@.75", "mAP@7 2D", "mAP@7 3D"]


def group(runs, label):
    vals = {}
    for k in ORDER:
        xs = [metrics(r)[k] for r in runs]
        xs = [x for x in xs if x is not None]
        vals[k] = (statistics.mean(xs), max(xs) - min(xs), len(xs))
    print(f"\n{label}  (n={len(runs)})")
    for k in ORDER:
        m, rg, n = vals[k]
        d = 4 if m < 2 else 2
        print(f"   {k:12s} mean {m:8.{d}f}   range {rg:.{d}f}")
    return vals


def main(dry=False):
    a = group(REF, "reference")
    bruns = REF if dry else FINAL
    missing = [r for r in bruns if not (RES / r).exists()]
    if missing:
        print("\nnot finished yet: " + ", ".join(missing))
        return
    b = group(bruns, "final model (uniform weighting + 6 queries)"
              if not dry else "DRY RUN: reference against itself")
    lu = metrics(LOSS_UNIFORM)

    print("\ndifference against the reference, judged by the four-seed ruler")
    for k in ORDER:
        diff = b[k][0] - a[k][0]
        d = 4 if abs(b[k][0]) < 2 else 2
        rul = RULER[k]
        verdict = "clears" if abs(diff) > rul else "inside ruler"
        print(f"   {k:12s} {diff:+8.{d}f}   ruler {rul:<7} {verdict}")

    print("\ncriteria, fixed before the runs")
    ca = (b["strict@.50"][0] - a["strict@.50"][0] > RULER["strict@.50"] and
          b["strict@.75"][0] - a["strict@.75"][0] > RULER["strict@.75"])
    cb = b["mAP@7 2D"][0] >= lu["mAP@7 2D"] - RULER["mAP@7 2D"]
    cc = b["K=3 strict"][0] >= lu["K=3 strict"] - RULER["K=3 strict"]
    for tag, ok, txt in (("a", ca, "beats the reference on strict accuracy at BOTH thresholds"),
                         ("b", cb, f"mAP@7 not below loss_uniform ({lu['mAP@7 2D']:.4f}) by > ruler"),
                         ("c", cc, f"K=3 not below loss_uniform ({lu['K=3 strict']:.2f}) by > ruler")):
        print(f"   ({tag}) {'PASS' if ok else 'FAIL'}  {txt}")

    print("\nreading, also fixed in advance")
    d50 = b["strict@.50"][0] - lu["strict@.50"]
    d75 = b["strict@.75"][0] - lu["strict@.75"]
    print(f"   against loss_uniform alone: strict@.50 {d50:+.2f}, strict@.75 {d75:+.2f}")
    if d50 > RULER["strict@.50"] and d75 > RULER["strict@.75"]:
        print("   -> ADDITIVE. This is the final model of the thesis.")
    elif d50 < -RULER["strict@.50"] or d75 < -RULER["strict@.75"]:
        print("   -> NEGATIVE INTERACTION. Report loss_uniform as final; this is the evidence.")
    else:
        print("   -> REDUNDANT within the ruler: the query cut adds nothing once the loss is")
        print("      fixed, i.e. both were suppressing the same spurious proposals.")
        print("      Report loss_uniform as final, because it is the simpler recipe.")


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)

#!/usr/bin/env python3
"""Print the run-to-run spread of a group of runs for every quantity the chapter quotes.

Chapter 5 judges every arm against a ruler; this reports the real spread (min, max, range, std)
over the reference and its seed replicates. Reads results/<run>/{metrics_detr,
parameter_recovery_detr}.json, results/threshold_sweep/<run>.json and
results/nd_evaluation/{<run>,tables_2d_3d,tables_2d_3d_extra}.json. Prints only, writes nothing.
Usage: python3 evaluation/tables/seed_spread.py baseline_v2_reproduction baseline_seed20260725 ...
       --check reproduces the frozen-vs-repeat pair that set the original rulers
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

CHECK = ["t1_3500_t2_500_weighted_long", "baseline_v2_reproduction"]


def _j(*p):
    """Read a JSON file under the results folder."""
    return json.load(open(RES.joinpath(*p)))


def _sweep_at(run, dim, theta, key):
    """Get a saved metric at the requested threshold and dimension."""
    return next(r for r in _j("threshold_sweep", run + ".json")[dim]
                if abs(r["threshold"] - theta) < 1e-9)[key]


def metrics(run):
    """Collect one run's scores for measuring variation across seeds."""
    m = _j(run, "metrics_detr.json")
    bin0 = _j(run, "parameter_recovery_detr.json")["bins"][0]      # w in [0.05, 0.10)
    try:
        t23 = _j("nd_evaluation", "tables_2d_3d.json")
    except FileNotFoundError:
        t23 = {}
    try:                       # older runs live in the _extra file, with a flatter schema
        t23x = _j("nd_evaluation", "tables_2d_3d_extra.json")
    except FileNotFoundError:
        t23x = {}
    out = {
        "count acc @own theta (%)":      m["count_accuracy"] * 100,
        "count acc @0.75 (%)":           _sweep_at(run, "2d", 0.75, "count_acc"),
        "strict 2D @0.50 (%)":           _sweep_at(run, "2d", 0.50, "voxel_acc"),
        "strict 2D @0.75 (%)":           _sweep_at(run, "2d", 0.75, "voxel_acc"),
        "strict 3D @0.50 (%)":           _sweep_at(run, "3d", 0.50, "voxel_acc"),
        "strict 3D @0.75 (%)":           _sweep_at(run, "3d", 0.75, "voxel_acc"),
        "mAP@7 2D":                      _j("nd_evaluation", run + ".json")["map"]["map@7"],
        "small-comp T1 err (%)":         bin0["t1_relative_error_median"] * 100,
        "small-comp detection (%)":      bin0["match_rate"] * 100,
        "existence F1":                  m["existence_f1"],
        "T1 abs err median (ms)":        m["t1_abs_median_ms"],
    }
    if run in t23:
        out["mAP@7 3D"] = t23[run]["3d"]["map"]["map@7"]
    elif run in t23x:
        out["mAP@7 3D"] = t23x[run]["3d"]["map7"]
    return out


ORDER = ["count acc @own theta (%)", "count acc @0.75 (%)",
         "strict 2D @0.50 (%)", "strict 2D @0.75 (%)",
         "strict 3D @0.50 (%)", "strict 3D @0.75 (%)",
         "mAP@7 2D", "mAP@7 3D",
         "small-comp T1 err (%)", "small-comp detection (%)",
         "existence F1", "T1 abs err median (ms)"]


def report(runs):
    """Print metric ranges and standard deviations across runs."""
    vals = {r: metrics(r) for r in runs}
    n = len(runs)
    print(f"\nrun-to-run spread over {n} run(s): {', '.join(runs)}\n")
    print(f"{'quantity':28s} {'min':>10s} {'max':>10s} {'range':>9s} {'std':>9s}")
    print("-" * 70)
    for k in ORDER:
        xs = [vals[r][k] for r in runs if k in vals[r]]
        if len(xs) < 2:
            continue
        d = 4 if max(abs(x) for x in xs) < 2 else 2
        sd = statistics.stdev(xs) if len(xs) > 1 else float("nan")
        print(f"{k:28s} {min(xs):10.{d}f} {max(xs):10.{d}f} "
              f"{max(xs)-min(xs):9.{d}f} {sd:9.{d}f}")
    print(f"\nn = {n}. Report as range or std with n stated, never as a confidence interval.")
    if n == 2:
        print("Two runs give a difference, not a spread: this is the lower-bound ruler.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--check"]:
        runs = [r for r in CHECK if (RES / r / "metrics_detr.json").exists()]
        for r in CHECK:
            if r not in runs:
                print(f"{r} is not under results/, skipped")
        if len(runs) >= 2:
            report(runs)
        else:
            print("--check needs both runs; nothing to compare")
    elif args:
        report(args)
    else:
        raise SystemExit(__doc__)

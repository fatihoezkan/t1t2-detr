#!/usr/bin/env python3
"""Calibrate the existence threshold per run on validation, then read test accuracy at it.

Runs peak at different thresholds (0.64 to 0.95 measured), so one shared theta is not neutral.
Per run, theta is swept over threshold_sweep.GRID on validation, the argmax of strict voxel
accuracy is taken, and that theta is applied unchanged to test; no test data enters the choice.
Strict accuracy has an interior optimum, whereas the F1 in run_nd_evaluation.py saturates at
the edge of the grid. Writes results/threshold_val/<run>.json with the validation curve. Usage:
    PYTHONPATH=.:datagen python3 evaluation/calibrate_threshold.py <run> [<run> ...]
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import threshold_sweep as ts               # shares GRID and score() with the sweep

from t1t2.nd_metrics import TAU_BASE, dataset_records
from t1t2.runs import load_run

OUT = Path("results/threshold_val")


def _curve(run, split):
    """Score one split of the run at each confidence threshold."""
    # unfiltered ND records at tau = 7 %, then one score row per threshold
    q, trues = run.predict(split)
    recs, ngt = dataset_records(q, trues, run.spans, TAU_BASE, exist_thresh=0.0,
                               include_weight=False)
    return [ts.score(recs, ngt, T) for T in ts.GRID]


def main(runs):
    """Choose thresholds on validation data and report test accuracy."""
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'run':34s} {'val theta':>10s} {'val acc':>8s} {'test acc':>9s} {'test@.75':>9s}")
    for run in runs:
        # best threshold on validation by strict voxel accuracy
        loaded = load_run(Path("results") / run)
        val = _curve(loaded, "val")
        best = max(val, key=lambda r: r["voxel_acc"])
        theta = best["threshold"]
        # test accuracy at that threshold and at the declared 0.75
        test = _curve(loaded, "test")
        at = lambda t: next(r for r in test if abs(r["threshold"] - t) < 1e-9)["voxel_acc"]
        rec = {"run": run, "val_theta": theta, "val_voxel_acc": best["voxel_acc"],
               "test_voxel_acc_at_val_theta": at(theta), "test_voxel_acc_at_075": at(0.75),
               "val_curve": val}
        # write and print
        (OUT / f"{run}.json").write_text(json.dumps(rec, indent=1))
        print(f"{run:34s} {theta:10.2f} {best['voxel_acc']:8.2f} "
              f"{rec['test_voxel_acc_at_val_theta']:9.2f} {rec['test_voxel_acc_at_075']:9.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])

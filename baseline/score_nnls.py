"""Score an NNLS run with this repository's own ND/mAP code, so it lands beside the DETR runs.

Reads baseline/nnls_baseline.py output (results/nnls/<run>/<split>_predictions.npz) and applies
exactly the functions the DETR tables use: nd_metrics.dataset_records / map_101_from_records /
stratified_map, evaluation.threshold_sweep.score / score_by_k, and eval.parameter_recovery_analysis.

Operating point: the confidence threshold is calibrated on the validation split, once for best
strict voxel accuracy and once for best F1, then applied unchanged to test. The grid runs from
0.01 because NNLS confidence is a signal fraction, which can be much smaller than a DETR
existence probability. Fractions need not be equal (e.g. 0.8, 0.1, 0.1 for K=3). mAP needs no
operating threshold, but depends on each method's confidence ranking.

Usage: PYTHONPATH=.:datagen python baseline/score_nnls.py nnls_chi2_g50 [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from t1t2.nd_metrics import (TAUS_DEFAULT, TAU_BASE, dataset_records, log_spans,
                             map_101_from_records, stratified_map)
from evaluation.threshold_sweep import score, score_by_k

try:    # t1t2.eval imports torch, which the ND metrics themselves do not need
    from t1t2.eval import parameter_recovery_analysis, predictions_from_query_outputs
    HAVE_EVAL = True
except ModuleNotFoundError:
    HAVE_EVAL = False

EPS = 1e-12                      # drops empty slots: a slot with zero fraction is not a prediction
GRID = [round(0.01 * i, 2) for i in range(1, 51)]        # 0.01 .. 0.50
SPANS = log_spans(50.0, 3500.0, 5.0, 500.0)
WEIGHT_BINS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]


def load(run, split):
    d = np.load(Path("results/nnls") / run / f"{split}_predictions.npz")
    q = {"params": d["params"], "exist_prob": d["exist_prob"]}
    T, n_comp = d["trues"], d["n_comp"]
    trues = [[tuple(T[i, k]) for k in range(int(n_comp[i]))] for i in range(len(n_comp))]
    return q, trues, n_comp


def calibrate(run, spans=SPANS):
    """Best-strict-accuracy and best-F1 thresholds on validation."""
    q, trues, _ = load(run, "val")
    recs, ngt = dataset_records(q, trues, spans, TAU_BASE, exist_thresh=EPS)
    rows = [score(recs, ngt, T) for T in GRID]
    best_strict = max(rows, key=lambda r: r["voxel_acc"])
    best_f1 = max(rows, key=lambda r: r["f1"])
    return {"grid": rows,
            "theta_strict": best_strict["threshold"], "val_voxel_acc": best_strict["voxel_acc"],
            "theta_f1": best_f1["threshold"], "val_f1": best_f1["f1"]}


def main(runs):
    for run in runs:
        info = json.loads((Path("results/nnls") / run / "info.json").read_text())
        spans = log_spans(*info["t1_range"], *info["t2_range"])
        cal = calibrate(run, spans)
        q, trues, n_comp = load(run, "test")
        recs, ngt = dataset_records(q, trues, spans, TAU_BASE, exist_thresh=EPS)
        out = {"run": run, "n_voxels": len(trues), "tau": TAU_BASE, "spans": spans,
               "calibration": cal,
               # map_101_from_records returns (mAP, precisions, recalls); keep the scalar
               "mAP": {f"mAP@{int(t * 100)}": map_101_from_records(
                   *dataset_records(q, trues, spans, t, exist_thresh=EPS))[0]
                   for t in TAUS_DEFAULT},
               "sweep": [score(recs, ngt, T) for T in GRID],
               "at_theta_strict": score(recs, ngt, cal["theta_strict"]),
               "at_theta_f1": score(recs, ngt, cal["theta_f1"]),
               "by_k": {"theta_strict": score_by_k(recs, ngt, cal["theta_strict"]),
                        "theta_f1": score_by_k(recs, ngt, cal["theta_f1"])},
               "stratified_map": stratified_map(q, trues, spans, TAU_BASE)}
        if HAVE_EVAL:
            preds = predictions_from_query_outputs(q, exist_thresh=cal["theta_strict"])
            rec = parameter_recovery_analysis(preds, trues, weight_bins=WEIGHT_BINS)
            out["parameter_recovery"] = {"summary": rec["summary"], "bins": rec["bins"]}
        path = Path("results/nnls") / run / "nd_scores.json"
        path.write_text(json.dumps(out, indent=2,
                                   default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
        m = out["mAP"]
        s = out["at_theta_strict"]
        print(f"{run}: mAP@7 {m['mAP@7']:.4f} | theta {cal['theta_strict']:.2f} | "
              f"strict {s['voxel_acc']:.2f}% | count {s['count_acc']:.2f}% | "
              f"P {s['precision']:.3f} R {s['recall']:.3f} | {path}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["nnls_chi2_g50"])

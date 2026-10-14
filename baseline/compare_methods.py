"""Build the classical-versus-DETR comparison table from stored scores.

Reads the NNLS scores written by baseline/score_nnls.py and the DETR scores the thesis tables
already use (results/nd_evaluation/<run>.json for mAP and the stratified views,
results/threshold_val/<run>.json for the validation-calibrated threshold and the strict accuracy
it gives on test, results/threshold_sweep/<run>.json for count accuracy, precision and recall at
that threshold). DETR families are averaged over their four seeds and the min-to-max range is
reported for the two headline columns; the NNLS rows are single deterministic runs, so their
range columns stay empty.

Every method is scored on the same 9,999 test voxels with the same ND rule at tau = 7 %, and
every method sits at its own validation-calibrated operating point, which is the same protocol
the thesis uses between DETR arms.

Usage: PYTHONPATH=.:datagen python baseline/compare_methods.py [out.csv]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

FAMILIES = {
    "DETR reference (4 seeds)": ["baseline_v2_reproduction", "baseline_seed20260725",
                                 "baseline_seed20260726", "baseline_seed20260727"],
    "DETR loss_uniform (4 seeds)": ["loss_uniform", "loss_uniform_seed20260725",
                                    "loss_uniform_seed20260726", "loss_uniform_seed20260727"],
}
NNLS_RUNS = [("nnls_chi2_g30", "NNLS chi2 (grid 30, 771 atoms)"),
             ("nnls_chi2_g50", "NNLS chi2 (grid 50, 2149 atoms)")]
FIELDS = ["method", "mAP5", "mAP7", "mAP7_range", "mAP10", "theta", "strict", "strict_range",
          "count", "precision", "recall", "map7_w_small", "map7_w_mid", "map7_w_large",
          "map7_k1", "map7_k2", "map7_k3"]


def _at(sweep, theta):
    """The sweep row at one threshold; the grid is in steps of 0.05, so this is an exact lookup."""
    for row in sweep["2d"]:
        if abs(row["threshold"] - theta) < 1e-9:
            return row
    raise KeyError(f"threshold {theta} not in sweep grid")


def detr_row(label, runs):
    nd = [json.loads(Path(f"results/nd_evaluation/{r}.json").read_text()) for r in runs]
    tv = [json.loads(Path(f"results/threshold_val/{r}.json").read_text()) for r in runs]
    sw = [json.loads(Path(f"results/threshold_sweep/{r}.json").read_text()) for r in runs]
    pt = [_at(s, v["val_theta"]) for s, v in zip(sw, tv)]
    m7 = [d["map"]["map@7"] for d in nd]
    strict = [v["test_voxel_acc_at_val_theta"] for v in tv]
    w = lambda band: np.mean([d["stratified_map7"]["by_weight"][band]["map"] for d in nd])  # noqa: E731
    k = lambda n: np.mean([d["stratified_map7"]["by_n_comp"][f"n_comp={n}"]["map"] for d in nd])  # noqa: E731
    return dict(method=label,
                mAP5=np.mean([d["map"]["map@5"] for d in nd]), mAP7=np.mean(m7),
                mAP7_range=np.ptp(m7), mAP10=np.mean([d["map"]["map@10"] for d in nd]),
                theta=np.mean([v["val_theta"] for v in tv]),
                strict=np.mean(strict), strict_range=np.ptp(strict),
                count=np.mean([p["count_acc"] for p in pt]),
                precision=np.mean([p["precision"] for p in pt]),
                recall=np.mean([p["recall"] for p in pt]),
                map7_w_small=w("w<0.3"), map7_w_mid=w("0.3<=w<0.6"), map7_w_large=w("w>=0.6"),
                map7_k1=k(1), map7_k2=k(2), map7_k3=k(3))


def nnls_row(run, label):
    n = json.loads((Path("results/nnls") / run / "nd_scores.json").read_text())
    s, st = n["at_theta_strict"], n["stratified_map"]
    return dict(method=label, mAP5=n["mAP"]["mAP@5"], mAP7=n["mAP"]["mAP@7"],
                mAP7_range=float("nan"), mAP10=n["mAP"]["mAP@10"],
                theta=n["calibration"]["theta_strict"], strict=s["voxel_acc"],
                strict_range=float("nan"), count=s["count_acc"], precision=s["precision"],
                recall=s["recall"],
                map7_w_small=st["by_weight"]["w<0.3"]["map"],
                map7_w_mid=st["by_weight"]["0.3<=w<0.6"]["map"],
                map7_w_large=st["by_weight"]["w>=0.6"]["map"],
                map7_k1=st["by_n_comp"]["n_comp=1"]["map"],
                map7_k2=st["by_n_comp"]["n_comp=2"]["map"],
                map7_k3=st["by_n_comp"]["n_comp=3"]["map"])


def main(out_path):
    rows = [detr_row(label, runs) for label, runs in FAMILIES.items()]
    rows += [nnls_row(run, label) for run, label in NNLS_RUNS
             if (Path("results/nnls") / run / "nd_scores.json").exists()]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    head = (f"{'method':32}{'mAP@5':>8}{'mAP@7':>8}{'mAP@10':>8}{'theta':>7}"
            f"{'strict%':>9}{'count%':>8}{'prec':>7}{'rec':>7}")
    print(head)
    for r in rows:
        print(f"{r['method']:32}{r['mAP5']:8.4f}{r['mAP7']:8.4f}{r['mAP10']:8.4f}"
              f"{r['theta']:7.2f}{r['strict']:9.2f}{r['count']:8.2f}"
              f"{r['precision']:7.3f}{r['recall']:7.3f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/nnls/nnls_vs_detr_comparison.csv")

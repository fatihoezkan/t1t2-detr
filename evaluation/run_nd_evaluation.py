"""ND / mAP evaluation for one finished run directory.

Loads config.yaml and checkpoints/best.pt, runs inference on validation and test, calibrates
the existence threshold on validation (grid 0.25 to 0.75 step 0.05, best F1 under ND matching
at tau = 7 %), then scores test: threshold-free mAP at tau = 5, 7, 10 % and their mean, exact
metrics at the calibrated threshold, and mAP@7 stratified by true weight and by compartment
count. Writes <out_dir>/<run_name>.json (default results/nd_evaluation/) with a per-voxel record
dump for paired_tests.py. Usage: python evaluation/run_nd_evaluation.py results/<run> [out_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from t1t2.nd_metrics import (
    TAUS_DEFAULT, TAU_BASE, dataset_records, map_101_from_records,
    exact_metrics_from_records, calibrate_threshold_nd, stratified_map,
    bootstrap_map_ci,
)
from t1t2.runs import load_run


def evaluate_run(run_dir, out_dir, device="cpu", limit=None, n_boot=300, log=print):
    """Evaluate a saved model with normalized-distance matching and save results."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = load_run(run_dir, device)
    cfg, spans, name = run.cfg, run.spans, run.cfg.name

    log(f"[{name}] inference on val ...")
    q_val, t_val = run.predict("val", limit=limit)
    log(f"[{name}] inference on test ...")
    q_test, t_test = run.predict("test", limit=limit)

    # -- threshold calibrated on VAL only ------------------------------------------
    best_t, calib_table = calibrate_threshold_nd(q_val, t_val, spans, tau=TAU_BASE)
    log(f"[{name}] calibrated existence threshold (val, F1@ND7): {best_t:.2f}")

    # -- threshold-free mAP on TEST -------------------------------------------------
    maps, pr_curves = {}, {}
    recs7 = n_gt7 = None
    for tau in TAUS_DEFAULT:
        recs, n_gt = dataset_records(q_test, t_test, spans, tau, exist_thresh=0.0)
        m, prec, rec = map_101_from_records(recs, n_gt)
        maps[f"map@{round(tau * 100)}"] = m
        # decimate the PR curve for storage
        step = max(1, len(prec) // 2000)
        pr_curves[f"tau_{round(tau * 100)}"] = {
            "precision": prec[::step].tolist(), "recall": rec[::step].tolist()}
        if abs(tau - TAU_BASE) < 1e-9:
            recs7, n_gt7 = recs, n_gt
    maps["map_avg"] = float(np.mean([maps[f"map@{round(t * 100)}"] for t in TAUS_DEFAULT]))
    ci = bootstrap_map_ci(recs7, n_gt7, n_boot=n_boot)
    log(f"[{name}] mAP@7 = {maps['map@7']:.4f}  (95% CI {ci['lo']:.4f}-{ci['hi']:.4f})")

    # -- exact metrics on TEST at the val-calibrated threshold ----------------------
    recs_op, n_gt_op = dataset_records(q_test, t_test, spans, TAU_BASE,
                                       exist_thresh=best_t)
    exact = exact_metrics_from_records(recs_op, n_gt_op)
    log(f"[{name}] P {exact['precision']:.4f}  R {exact['recall']:.4f}  "
        f"F1 {exact['f1']:.4f}  mean dT1 {exact['mean_dt1_ms']:.2f} ms  "
        f"mean dT2 {exact['mean_dt2_ms']:.2f} ms")

    strat = stratified_map(q_test, t_test, spans, tau=TAU_BASE)

    result = {
        "name": name,
        "run_dir": str(run_dir),
        "epoch": run.epoch,
        "spans_log": {"t1": spans[0], "t2": spans[1]},
        "ranges_ms": {"t1": [cfg.data.t1_min, cfg.data.t1_max],
                      "t2": [cfg.data.t2_min, cfg.data.t2_max]},
        "test_paths": list(cfg.data.test_path),
        "n_test_voxels": len(t_test),
        "n_val_voxels": len(t_val),
        "existence_threshold": best_t,
        "calibration_split": "val",
        "calibration_table": {str(k): v for k, v in calib_table.items()},
        "map": maps,
        "map7_ci95": ci,
        "exact_at_threshold": exact,
        "stratified_map7": strat,
        "pr_curves": pr_curves,
        # raw per-voxel record dump for paired bootstraps across models
        "_records_tau7": [[{k: (None if isinstance(v, float) and not np.isfinite(v)
                                else v) for k, v in r.items()} for r in recs]
                          for recs in recs7],
        "_n_gt": n_gt7.tolist(),
    }
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(result))
    log(f"[{name}] written {out_path}")
    return result


if __name__ == "__main__":
    run_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "results/nd_evaluation"
    evaluate_run(run_dir, out_dir)

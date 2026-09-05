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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import threshold_sweep as ts               # shares GRID and score() with the sweep

from t1t2.config import load_config
from t1t2.data import TargetNormalizer, VoxelDataset
from t1t2.eval import detr_query_outputs, true_compartments
from t1t2.model import build_model
from t1t2.nd_metrics import TAU_BASE, log_spans, dataset_records

OUT = Path("results/threshold_val")


def _load(run, device="cpu"):
    rd = Path("results") / run
    cfg = load_config(rd / "config.yaml")
    model = build_model(cfg.model)
    ck = torch.load(rd / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"] if "model" in ck else ck["state_dict"])
    model.to(device).eval()
    norm = TargetNormalizer.from_config(cfg.data)
    spans = log_spans(cfg.data.t1_min, cfg.data.t1_max, cfg.data.t2_min, cfg.data.t2_max)
    return cfg, model, norm, spans


def _curve(paths, cfg, model, norm, spans, device="cpu"):
    ds = VoxelDataset(paths, cfg.data, norm)
    q = detr_query_outputs(model, ds, torch.device(device), norm)
    trues = true_compartments(ds)
    recs, ngt = dataset_records(q, trues, spans, TAU_BASE, exist_thresh=0.0,
                               include_weight=False)
    return [ts.score(recs, ngt, T) for T in ts.GRID]


def main(runs):
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'run':34s} {'val theta':>10s} {'val acc':>8s} {'test acc':>9s} {'test@.75':>9s}")
    for run in runs:
        cfg, model, norm, spans = _load(run)
        val = _curve(cfg.data.val_path, cfg, model, norm, spans)
        best = max(val, key=lambda r: r["voxel_acc"])
        theta = best["threshold"]
        test = _curve(cfg.data.test_path, cfg, model, norm, spans)
        at = lambda t: next(r for r in test if abs(r["threshold"] - t) < 1e-9)["voxel_acc"]
        rec = {"run": run, "val_theta": theta, "val_voxel_acc": best["voxel_acc"],
               "test_voxel_acc_at_val_theta": at(theta), "test_voxel_acc_at_075": at(0.75),
               "val_curve": val}
        (OUT / f"{run}.json").write_text(json.dumps(rec, indent=1))
        print(f"{run:34s} {theta:10.2f} {best['voxel_acc']:8.2f} "
              f"{rec['test_voxel_acc_at_val_theta']:9.2f} {rec['test_voxel_acc_at_075']:9.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])

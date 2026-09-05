#!/usr/bin/env python3
"""Write results/nd_evaluation/tables_2d_3d.json: mAP@5/7/10 and exact metrics, 2D and 3D.

Reads results/<run>/config.yaml, results/<run>/checkpoints/best.pt and the test split named in
the config. The 3D form calls t1t2.nd_metrics with include_weight=True: a prediction is accepted
only if T1, T2 and the signal fraction are all within tau, and the fraction enters the ranking
sum. The table builders read this file for the mAP@7 3D column.
Usage: PYTHONPATH=.:datagen python3 evaluation/tables/build_2d_3d_tables.py [--verify] <run> ...
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "datagen"))
os.chdir(ROOT)                      # data paths in config.yaml are relative to the repo root

from t1t2.nd_metrics import (                                    # noqa: E402
    TAUS_DEFAULT, TAU_BASE, dataset_records, map_101_from_records,
    exact_metrics_from_records,
)
from t1t2.runs import load_run                                   # noqa: E402

RES = ROOT / "results"
OUT = RES / "nd_evaluation" / "tables_2d_3d.json"
THRESHOLD = 0.75          # the declared cut-off used by the exact F1 columns


def scores_for(run, device="cpu"):
    """Evaluate a model with and without signal fraction in the distance."""
    loaded = load_run(RES / run, device)
    q, t = loaded.predict("test")
    spans = loaded.spans

    out = {"threshold": THRESHOLD}
    for dim, inc_w in (("2d", False), ("3d", True)):
        maps = {}
        for tau in TAUS_DEFAULT:
            recs, n_gt = dataset_records(q, t, spans, tau, exist_thresh=0.0,
                                         include_weight=inc_w)
            maps[f"map@{round(tau * 100)}"] = map_101_from_records(recs, n_gt)[0]
        maps["map_avg"] = float(np.mean(
            [maps[f"map@{round(x * 100)}"] for x in TAUS_DEFAULT]))
        recs_op, n_gt_op = dataset_records(q, t, spans, TAU_BASE,
                                           exist_thresh=THRESHOLD, include_weight=inc_w)
        out[dim] = {"map": {k: round(v, 4) for k, v in maps.items()},
                    "exact": exact_metrics_from_records(recs_op, n_gt_op)}
    return out


def main():
    """Build or verify the saved 2D and 3D comparison scores."""
    args = [a for a in sys.argv[1:] if a != "--verify"]
    verify = "--verify" in sys.argv
    if not args:
        raise SystemExit(__doc__)
    table = json.loads(OUT.read_text()) if OUT.exists() else {}
    for run in args:
        got = scores_for(run)
        if verify and run in table:
            old = table[run]
            print(f"\n{run}: recomputed vs stored")
            for dim in ("2d", "3d"):
                for k in ("map@5", "map@7", "map@10", "map_avg"):
                    a, b = got[dim]["map"][k], old[dim]["map"][k]
                    flag = "OK " if abs(a - b) < 5e-4 else "DIFF"
                    print(f"  {flag} {dim} {k:8s} recomputed {a:.4f}  stored {b:.4f}")
                for k in ("TP", "FP", "FN"):
                    a, b = got[dim]["exact"][k], old[dim]["exact"][k]
                    print(f"  {'OK ' if a == b else 'DIFF'} {dim} {k:8s} "
                          f"recomputed {a}  stored {b}")
        else:
            table[run] = got
            print(f"{run}: 2D mAP@7 {got['2d']['map']['map@7']:.4f}  "
                  f"3D mAP@7 {got['3d']['map']['map@7']:.4f}")
    if not verify:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(table, indent=1))
        print(f"\nwrote {OUT}  ({len(table)} runs)")


if __name__ == "__main__":
    main()

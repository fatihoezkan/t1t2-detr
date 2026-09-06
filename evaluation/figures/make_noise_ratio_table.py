"""Per-compartment table of signal amplitude against noise on the main test set.

Writes results/compartment_noise_ratio_test.parquet, one row per true compartment of the
9,999 test voxels: its signal fraction w, the peak amplitude A of its own signal, the
voxel's noise sigma, the ratio r = A / sigma, and whether the reference and the final
model found it (ND rule, tau = 7 %, at each run's fitted threshold). Read by
make_noise_effect_figure.py. Needs results/nd_evaluation/<run>.json for both runs.
Usage: PYTHONPATH=.:datagen python3 evaluation/figures/make_noise_ratio_table.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

from t1t2.config import load_config
from t1t2.physics import forward_numpy, load_protocol

RUNS = {"found_reference": "baseline_v2_reproduction", "found_final": "loss_uniform"}
OUT = ROOT / "results" / "compartment_noise_ratio_test.parquet"

# test set and protocol
cfg = load_config(f"configs/{RUNS['found_reference']}.yaml")
test = pd.concat([pd.read_parquet(p) for p in cfg.data.test_path], ignore_index=True)
proto = load_protocol()

# The ND records list every query of every voxel with the ground-truth index it was
# assigned to; a compartment counts as found when a query above the run's fitted
# threshold was assigned to it.
found = {}
for col, run in RUNS.items():
    theta = json.load(open(ROOT / "results" / run / "threshold_calibration.json"))["selected_threshold"]
    records = json.load(open(ROOT / "results" / "nd_evaluation" / f"{run}.json"))["_records_tau7"]
    found[col] = [{r["gt"] for r in recs if r["gt"] is not None and r["prob"] >= theta}
                  for recs in records]

# one row per true compartment with the peak amplitude of its own signal over sigma
rows = []
for v, row in test.iterrows():
    for c in range(int(row["n_comp"])):
        w, t1, t2 = row[f"w_{c + 1}"], row[f"T1_{c + 1}"], row[f"T2_{c + 1}"]
        amp = w * np.abs(forward_numpy(proto, [t1], [t2], [1.0])).max()
        rows.append({"voxel": v, "K": int(row["n_comp"]), "c": c, "w": w, "A": amp,
                     "sigma": row["sigma"], "r": amp / row["sigma"], "snr": row["snr"],
                     **{col: c in found[col][v] for col in RUNS}})
D = pd.DataFrame(rows)
# coarse bins for reading the table by hand; the figure script bins on its own
D["rbin"] = pd.cut(D.r, [-np.inf, 2, 3, 5, 10, 20, 50, np.inf],
                   labels=["<2", "2-3", "3-5", "5-10", "10-20", "20-50", ">50"])
D["wbin"] = pd.cut(D.w, [-np.inf, 0.1, 0.3, 0.6, np.inf],
                   labels=["w<0.1", "0.1-0.3", "0.3-0.6", ">0.6"])
D["snrbin"] = pd.cut(D.snr, [-np.inf, 60, 100, np.inf], labels=["SNR 30-60", "60-100", "100-150"])
D.to_parquet(OUT, index=False)
print(f"wrote {OUT.relative_to(ROOT)}: {len(D)} compartments, found {D.found_reference.mean():.3f} (reference) "
      f"/ {D.found_final.mean():.3f} (final)")

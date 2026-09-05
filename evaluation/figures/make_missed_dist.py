"""What the missed compartments look like: their distribution over true T2 and over the
signal fraction, against the distribution of all compartments. If long-T2 compartments were
harder to detect, the missed density would lean right of the overall density in the T2
panel. Missed means not accepted by the ND rule at the run's fitted theta. Reads
results/<run>/ (config, summary.json, best.pt) and the test parquets.
Writes figures/17_missed_dist.png.
Usage: PYTHONPATH=.:datagen python3 evaluation/figures/make_missed_dist.py
"""
import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch

mpl.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from t1t2 import nd_metrics as ndm  # noqa: E402
from t1t2.config import load_config  # noqa: E402
from t1t2.data import TargetNormalizer, VoxelDataset  # noqa: E402
from t1t2.eval import detr_query_outputs  # noqa: E402
from t1t2.model import build_model  # noqa: E402

OUT = ROOT / "figures" / "17_missed_dist.png"
BASE, SMALL, TINY = 9, 8, 7
mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": BASE, "axes.titlesize": BASE, "axes.labelsize": BASE,
    "xtick.labelsize": TINY, "ytick.labelsize": TINY, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white"})
TAU, TE_MAX = 0.07, 150.0
RUNS = [("baseline_v2_reproduction", "reference", "#7f7f7f"),
        ("loss_uniform", "final model", "#1f4e79")]


def collect(run):
    rd = ROOT / "results" / run
    cfg = load_config(rd / "config.yaml")
    thr = float(json.load(open(rd / "summary.json"))["threshold_calibration"]["selected_threshold"])
    model = build_model(cfg.model)
    ck = torch.load(rd / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"] if "model" in ck else ck["state_dict"]); model.eval()
    norm = TargetNormalizer.from_config(cfg.data)
    ds = VoxelDataset(cfg.data.test_path, cfg.data, norm)
    q = detr_query_outputs(model, ds, torch.device("cpu"), norm)
    P = np.asarray(q["params"]); E = np.asarray(q["exist_prob"])
    df = pd.read_parquet(cfg.data.test_path)
    kmax = max(int(c.split("_")[1]) for c in df.columns if c.startswith("T1_"))
    spans = ndm.log_spans(cfg.data.t1_min, cfg.data.t1_max, cfg.data.t2_min, cfg.data.t2_max)
    all_t2, all_w, mis_t2, mis_w = [], [], [], []
    for i, row in enumerate(df.itertuples(index=False)):
        trues = [(getattr(row, f"T1_{k}"), getattr(row, f"T2_{k}"), getattr(row, f"w_{k}"))
                 for k in range(1, kmax + 1) if np.isfinite(getattr(row, f"T1_{k}"))]
        recs = ndm.voxel_records(P[i], E[i], trues, spans, TAU, exist_thresh=thr)
        hit = {r["gt"] for r in recs if r["gt"] is not None}
        for g, t in enumerate(trues):
            all_t2.append(t[1]); all_w.append(t[2])
            if g not in hit:
                mis_t2.append(t[1]); mis_w.append(t[2])
    print(f"{run}: {len(mis_t2)} missed of {len(all_t2)}")
    return map(np.asarray, (all_t2, all_w, mis_t2, mis_w)), cfg


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.9))
first = True
for run, label, col in RUNS:
    (all_t2, all_w, mis_t2, mis_w), cfg = collect(run)
    e2 = np.logspace(np.log10(cfg.data.t2_min), np.log10(cfg.data.t2_max), 25)
    ew = np.linspace(0.05, 1.0, 25)
    if first:
        ax1.hist(all_t2, bins=e2, density=True, color="0.88", label="all compartments")
        ax2.hist(all_w, bins=ew, density=True, color="0.88", label="all compartments")
        first = False
    ax1.hist(mis_t2, bins=e2, density=True, histtype="step", lw=1.7, color=col,
             label=f"missed, {label}")
    ax2.hist(mis_w, bins=ew, density=True, histtype="step", lw=1.7, color=col,
             label=f"missed, {label}")
ax1.set_xscale("log")
ax1.axvline(TE_MAX, color="k", lw=1.0, ls="--")
ax1.text(TE_MAX * 1.1, ax1.get_ylim()[1] * 0.93, "longest TE", fontsize=TINY, ha="left",
         path_effects=[pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])
ax1.set_xlabel("true $T_2$ (ms)"); ax1.set_ylabel("density")
ax1.set_title("where the missed compartments sit in $T_2$", fontsize=SMALL)
ax1.legend(fontsize=TINY)
ax2.set_xlabel("true signal fraction"); ax2.set_title(
    "and what signal fraction they carry", fontsize=SMALL)
ax2.legend(fontsize=TINY)
fig.suptitle("The missed compartments against the whole population "
             "(densities, ND rule at each run's fitted θ)", x=0.01, ha="left", fontsize=BASE)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT); print("wrote", OUT)

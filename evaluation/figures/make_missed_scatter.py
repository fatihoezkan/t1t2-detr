"""Every true compartment in the (T2, signal fraction) plane, matched against missed: the
scatter form of the missed-distribution question. If long-T2 compartments were harder to
detect, red would cluster right of the longest-TE line; instead it clusters in the
low-fraction band. Matched means accepted by the ND rule at the run's fitted theta. Reads
results/<run>/ (config, summary.json, best.pt) and the test parquets.
Writes figures/17_missed_scatter.png.
Usage: PYTHONPATH=.:datagen python3 evaluation/figures/make_missed_scatter.py
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

OUT = ROOT / "figures" / "17_missed_scatter.png"
BASE, SMALL, TINY = 9, 8, 7
mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": BASE, "axes.titlesize": BASE, "axes.labelsize": BASE,
    "xtick.labelsize": TINY, "ytick.labelsize": TINY, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white"})
TAU, TE_MAX = 0.07, 150.0
RUNS = [("baseline_v2_reproduction", "reference (baseline)"),
        ("loss_uniform", "final model (loss_uniform)")]


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
    F, M = [], []
    for i, row in enumerate(df.itertuples(index=False)):
        trues = [(getattr(row, f"T1_{k}"), getattr(row, f"T2_{k}"), getattr(row, f"w_{k}"))
                 for k in range(1, kmax + 1) if np.isfinite(getattr(row, f"T1_{k}"))]
        recs = ndm.voxel_records(P[i], E[i], trues, spans, TAU, exist_thresh=thr)
        hit = {r["gt"] for r in recs if r["gt"] is not None}
        for g, t in enumerate(trues):
            (F if g in hit else M).append((t[1], t[2]))
    print(f"{run}: matched {len(F)}, missed {len(M)}")
    return np.asarray(F), np.asarray(M), cfg, thr


fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), sharey=True)
for ax, (run, label) in zip(axes, RUNS):
    F, M, cfg, thr = collect(run)
    A = np.vstack([F, M])
    ax.scatter(A[:, 0], A[:, 1], s=6, alpha=0.18, color="#bdbdbd", edgecolors="none",
               rasterized=True, label="all compartments")
    ax.scatter(F[:, 0], F[:, 1], s=4, alpha=0.30, color="#2e6db4", edgecolors="none",
               rasterized=True, label="matched")
    ax.scatter(M[:, 0], M[:, 1], s=5, alpha=0.40, color="#c62828", edgecolors="none",
               rasterized=True, label="unmatched")
    ax.set_xscale("log")
    ax.axvline(TE_MAX, color="k", lw=1.0, ls="--")
    ax.text(TE_MAX * 1.1, 0.965, "longest TE", fontsize=TINY, ha="left",
            path_effects=[pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])
    ax.set_xlim(cfg.data.t2_min * 0.9, cfg.data.t2_max * 1.1)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"{label}, θ = {thr:.2f}", fontsize=SMALL)
    ax.set_xlabel("true $T_2$ (ms)")
    ax.text(0.985, 0.03, f"matched {len(F):,} / unmatched {len(M):,}",
            transform=ax.transAxes, fontsize=TINY, ha="right", va="bottom",
            path_effects=[pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()])
axes[0].set_ylabel("true signal fraction")
fig.suptitle("Every true compartment in the ($T_2$, signal fraction) plane: "
             "matched vs missed (τ = 7% box and the existence threshold)",
             x=0.01, ha="left", fontsize=BASE)
fig.tight_layout()
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=SMALL, markerscale=4,
           bbox_to_anchor=(0.5, 0.0))
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT); print("wrote", OUT)

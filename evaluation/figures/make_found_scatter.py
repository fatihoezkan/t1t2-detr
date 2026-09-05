"""Every true compartment of the test set in the (T1, T2) plane, found against missed.

One panel per model (reference, loss_uniform). Without a flag a compartment counts as found
when the ND rule accepts it at the run's fitted theta (tau = 7% box and the existence
threshold): figures/14_found_missed.png. With --map7 the existence threshold is dropped
(theta = 0), the view the mAP@7 protocol takes: figures/15_found_missed_map7.png. Reads
results/<run>/ (config, summary.json, best.pt) and the test parquets.
Usage: PYTHONPATH=.:datagen python3 evaluation/figures/make_found_scatter.py [--map7]
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

BASE, SMALL, TINY = 9, 8, 7
mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": BASE, "axes.titlesize": BASE, "axes.labelsize": BASE,
    "xtick.labelsize": TINY, "ytick.labelsize": TINY, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white"})
TAU, TE_MAX = 0.07, 150.0
MAP7 = "--map7" in sys.argv   # no existence threshold: the mAP@7 matching protocol
RUNS = [("baseline_v2_reproduction", "reference (baseline)"), ("loss_uniform", "final model (loss_uniform)")]
OUT = ROOT / "figures" / ("15_found_missed_map7.png" if MAP7 else "14_found_missed.png")


def found_missed(run):
    rd = ROOT / "results" / run
    cfg = load_config(rd / "config.yaml")
    thr = 0.0 if MAP7 else float(json.load(open(rd / "summary.json"))["threshold_calibration"]["selected_threshold"])
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
            (F if g in hit else M).append((t[0], t[1]))
    print(f"{run}: found {len(F)}, missed {len(M)} ({100*len(M)/(len(F)+len(M)):.1f}% missed)")
    return np.asarray(F), np.asarray(M), cfg, thr


fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), sharey=True)
for ax, (run, label) in zip(axes, RUNS):
    F, M, cfg, thr = found_missed(run)
    A = np.vstack([F, M])
    ax.scatter(A[:, 0], A[:, 1], s=6, alpha=0.18, color="#bdbdbd", edgecolors="none",
               rasterized=True, label="all compartments")
    ax.scatter(F[:, 0], F[:, 1], s=4, alpha=0.30, color="#2e6db4", edgecolors="none",
               rasterized=True, label="matched")
    ax.scatter(M[:, 0], M[:, 1], s=5, alpha=0.40, color="#c62828", edgecolors="none",
               rasterized=True, label="unmatched")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot([cfg.data.t2_min, cfg.data.t1_max], [cfg.data.t2_min, cfg.data.t1_max],
            color="0.45", lw=0.8, ls=":")
    ax.axhline(TE_MAX, color="k", lw=1.0, ls="--")
    ax.text(cfg.data.t1_max * 0.9, TE_MAX * 1.15, "longest TE", fontsize=TINY,
            ha="right", va="bottom",
            path_effects=[pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])
    ax.set_xlim(cfg.data.t1_min * 0.9, cfg.data.t1_max * 1.1)
    ax.set_ylim(cfg.data.t2_min * 0.9, cfg.data.t2_max * 1.1)
    ax.set_title(f"{label}, no existence threshold" if MAP7 else f"{label}, θ = {thr:.2f}", fontsize=SMALL)
    ax.set_xlabel("true $T_1$ (ms)")
    ax.text(0.985, 0.02, f"matched {len(F):,} / unmatched {len(M):,} of {len(F)+len(M):,}",
            transform=ax.transAxes, fontsize=TINY, ha="right", va="bottom",
            path_effects=[pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()])
axes[0].set_ylabel("true $T_2$ (ms)")
fig.suptitle("Every true compartment of the test set: found vs missed "
             + ("(τ = 7% acceptance box, NO existence threshold — the view mAP@7 takes)" if MAP7 else "(τ = 7% acceptance box AND the existence threshold, at each run's fitted θ)"), x=0.01, ha="left", fontsize=BASE, y=1.02)
fig.tight_layout()
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=SMALL, markerscale=4,
           bbox_to_anchor=(0.5, 0.0))
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT); print("wrote", OUT)

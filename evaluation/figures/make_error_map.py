"""Where in the (T1, T2) plane the errors of the final model (loss_uniform) sit.

Three binned maps over the true compartment positions: the share of compartments the ND
rule finds, and the median relative T1 and T2 error of the evaluation's matched prediction.
The longest echo time (150 ms) is drawn on the T2 axis; the long-T2 argument of the
discussion predicts the T2-error map should degrade above it. Reads results/<run>/ (config,
summary.json, best.pt) and the test parquets. Writes figures/13_error_map.png.
Usage: PYTHONPATH=.:datagen python3 evaluation/figures/make_error_map.py
"""
import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from scipy.stats import binned_statistic_2d

mpl.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from t1t2 import nd_metrics as ndm  # noqa: E402
from t1t2.config import load_config  # noqa: E402
from t1t2.data import TargetNormalizer, VoxelDataset  # noqa: E402
from t1t2.eval import _match, detr_query_outputs  # noqa: E402
from t1t2.model import build_model  # noqa: E402

OUT = ROOT / "figures" / "13_error_map.png"
BASE, SMALL, TINY = 9, 8, 7
mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": BASE, "axes.titlesize": BASE, "axes.labelsize": BASE,
    "xtick.labelsize": TINY, "ytick.labelsize": TINY, "axes.titlelocation": "left",
    "figure.facecolor": "white", "axes.facecolor": "white"})
TAU, RUN, TE_MAX = 0.07, "loss_uniform", 150.0
MIN_N = 25

rd = ROOT / "results" / RUN
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

det_x, det_y, det_f = [], [], []          # every true compartment: found or not (ND rule)
err_x, err_y, err1, err2 = [], [], [], [] # every matched pair: relative errors
for i, row in enumerate(df.itertuples(index=False)):
    trues = [(getattr(row, f"T1_{k}"), getattr(row, f"T2_{k}"), getattr(row, f"w_{k}"))
             for k in range(1, kmax + 1) if np.isfinite(getattr(row, f"T1_{k}"))]
    recs = ndm.voxel_records(P[i], E[i], trues, spans, TAU, exist_thresh=thr)
    hit = {r["gt"] for r in recs if r["gt"] is not None}
    for g, t in enumerate(trues):
        det_x.append(t[0]); det_y.append(t[1]); det_f.append(1.0 if g in hit else 0.0)
    pred = [(float(P[i, k, 0]), float(P[i, k, 1]), float(P[i, k, 2]))
            for k in np.where(E[i] >= thr)[0]]
    for pp, tt in _match(pred, trues):
        err_x.append(tt[0]); err_y.append(tt[1])
        err1.append(abs(pp[0] - tt[0]) / tt[0] * 100)
        err2.append(abs(pp[1] - tt[1]) / tt[1] * 100)
print(f"{RUN}: {len(det_x)} true compartments, {len(err_x)} matched pairs, theta {thr}")

xe = np.logspace(np.log10(cfg.data.t1_min), np.log10(cfg.data.t1_max), 23)
ye = np.logspace(np.log10(cfg.data.t2_min), np.log10(cfg.data.t2_max), 23)
lx, ly = np.log(xe), np.log(ye)


def binned(x, y, v, stat):
    s, _, _, _ = binned_statistic_2d(np.log(x), np.log(y), v, statistic=stat, bins=[lx, ly])
    n, _, _, _ = binned_statistic_2d(np.log(x), np.log(y), v, statistic="count", bins=[lx, ly])
    return np.where(n >= MIN_N, s, np.nan)


panels = [
    ("share of compartments found (%)", binned(det_x, det_y, det_f, "mean") * 100, "viridis", (0, 100)),
    ("median rel. $T_1$ error (%)", binned(err_x, err_y, err1, "median"), "magma_r", (0, 12)),
    ("median rel. $T_2$ error (%)", binned(err_x, err_y, err2, "median"), "magma_r", (0, 12)),
]
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
for ax, (title, M, cmap, clim) in zip(axes, panels):
    pc = ax.pcolormesh(xe, ye, M.T, cmap=cmap, vmin=clim[0], vmax=clim[1], shading="flat")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot([cfg.data.t2_min, cfg.data.t1_max], [cfg.data.t2_min, cfg.data.t1_max],
            color="0.45", lw=0.8, ls=":")
    ax.axhline(TE_MAX, color="k", lw=1.0, ls="--")
    ax.text(cfg.data.t1_max * 0.9, TE_MAX * 1.18, "longest TE", fontsize=TINY,
            ha="right", va="bottom",
            path_effects=[pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])
    ax.set_xlim(cfg.data.t1_min, cfg.data.t1_max)
    ax.set_ylim(cfg.data.t2_min, cfg.data.t2_max)
    ax.set_title(title, fontsize=SMALL)
    ax.set_xlabel("true $T_1$ (ms)")
    fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
axes[0].set_ylabel("true $T_2$ (ms)")
fig.suptitle(f"Where the errors sit in the plane ({RUN}, its own fitted θ; bins with"
             f" ≥ {MIN_N} compartments)", x=0.01, ha="left", fontsize=BASE, y=1.04)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT); print("wrote", OUT)

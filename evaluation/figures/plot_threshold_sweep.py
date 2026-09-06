"""ND voxel accuracy against the existence threshold, for the reference and the eleven
single-change arms.

Shows why the threshold is calibrated on validation instead of searched on test: the curves
are broad and keep rising almost to the top of the range, so a search on test would mostly
return the largest value it is allowed to try. Reads results/threshold_sweep/<run>.json.
Writes figures/fig_threshold_sweep.png.
Usage: python3 evaluation/figures/plot_threshold_sweep.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "fig_threshold_sweep.png"
SWEEP = ROOT / "results" / "threshold_sweep"

HIGHLIGHT = {
    "baseline_v2_reproduction": ("reference", "#000000", 2.4, "-"),
    "loss_uniform":             ("loss_uniform", "#1f77b4", 2.0, "-"),
    "exist_weight_03":          ("exist_weight_03", "#d62728", 2.0, "-"),
    "queries_4":                ("queries_4", "#ff7f0e", 2.0, "--"),
}
OTHERS = ["data_loguniform", "queries_6", "aux_loss", "decoder_2", "decoder_6",
          "exist_head_shared", "physics_clean", "physics_noisy"]

# left: 2-D rule, right: 3-D rule; grey for the other runs, coloured for the highlighted ones
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
for ax, dim, title in zip(axes, ("2d", "3d"),
                          ("2D: $T_1$ and $T_2$", "3D: $T_1$, $T_2$ and weight")):
    for run in OTHERS:
        d = json.load(open(SWEEP / f"{run}.json"))[dim]
        ax.plot([r["threshold"] for r in d], [r["voxel_acc"] for r in d],
                color="#bbbbbb", lw=1.0, zorder=1)
    for run, (label, color, lw, ls) in HIGHLIGHT.items():
        d = json.load(open(SWEEP / f"{run}.json"))[dim]
        ax.plot([r["threshold"] for r in d], [r["voxel_acc"] for r in d],
                color=color, lw=lw, ls=ls, label=label, zorder=3)
    # the two declared thresholds
    for t in (0.5, 0.75):
        ax.axvline(t, color="#555555", ls=":", lw=1.1, zorder=2)
        ax.text(t, 0.985, f"{t:.2f}", fontsize=8, color="#555555",
                transform=ax.get_xaxis_transform(), va="top", ha="right",
                rotation=90)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("existence threshold")
    ax.set_xlim(0.05, 0.95)
    ax.grid(alpha=0.25, lw=0.6)
axes[0].set_ylabel("voxels fully solved (%)")
axes[0].plot([], [], color="#bbbbbb", lw=1.0, label="other runs")
axes[0].legend(fontsize=8.5, loc="upper left", framealpha=0.9)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=200)
print("wrote", OUT.relative_to(ROOT))

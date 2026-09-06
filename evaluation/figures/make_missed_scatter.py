"""Every true compartment in the (T2, signal fraction) plane, matched against missed: the
scatter form of the missed-distribution question. If long-T2 compartments were harder to
detect, red would cluster right of the longest-TE line; instead it clusters in the
low-fraction band. Matched means accepted by the ND rule at the run's fitted theta. Reads
results/<run>/ (config, summary.json, best.pt) and the test parquets.
Writes figures/17_missed_scatter.png.
Usage: PYTHONPATH=.:datagen python3 evaluation/figures/make_missed_scatter.py
"""
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

from t1t2 import nd_metrics as ndm  # noqa: E402
from t1t2.runs import load_run  # noqa: E402

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
    """Separate found and missed compartments by T2 and signal fraction."""
    # inference, then the ND rule per voxel
    loaded = load_run(ROOT / "results" / run)
    cfg, spans, thr = loaded.cfg, loaded.spans, loaded.fitted_threshold
    q, all_trues = loaded.predict("test")
    P = np.asarray(q["params"]); E = np.asarray(q["exist_prob"])
    F, M = [], []
    # (T2, w) of every compartment, split by found or missed
    for i, trues in enumerate(all_trues):
        recs = ndm.voxel_records(P[i], E[i], trues, spans, TAU, exist_thresh=thr)
        hit = {r["gt"] for r in recs if r["gt"] is not None}
        for g, t in enumerate(trues):
            (F if g in hit else M).append((t[1], t[2]))
    print(f"{run}: matched {len(F)}, missed {len(M)}")
    return np.asarray(F), np.asarray(M), cfg, thr


# one panel per run in the (T2, w) plane
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

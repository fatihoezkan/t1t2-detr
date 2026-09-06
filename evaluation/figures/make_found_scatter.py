"""Every true compartment of the test set in the (T1, T2) plane, found against missed.

One panel per model (reference, loss_uniform). Without a flag a compartment counts as found
when the ND rule accepts it at the run's fitted theta (tau = 7% box and the existence
threshold): figures/14_found_missed.png. With --map7 the existence threshold is dropped
(theta = 0), the view the mAP@7 protocol takes: figures/15_found_missed_map7.png. Reads
results/<run>/ (config, summary.json, best.pt) and the test parquets.
Usage: PYTHONPATH=.:datagen python3 evaluation/figures/make_found_scatter.py [--map7]
"""
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

from t1t2 import nd_metrics as ndm  # noqa: E402
from t1t2.runs import load_run  # noqa: E402

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
    """(found, missed) lists of (T1, T2) over every true compartment, plus the run's config
    and threshold."""
    # inference, then the ND rule per voxel
    loaded = load_run(ROOT / "results" / run)
    cfg, spans = loaded.cfg, loaded.spans
    thr = 0.0 if MAP7 else loaded.fitted_threshold
    q, all_trues = loaded.predict("test")
    P = np.asarray(q["params"]); E = np.asarray(q["exist_prob"])
    F, M = [], []
    # a true compartment is found if some record was assigned to it
    for i, trues in enumerate(all_trues):
        recs = ndm.voxel_records(P[i], E[i], trues, spans, TAU, exist_thresh=thr)
        hit = {r["gt"] for r in recs if r["gt"] is not None}
        for g, t in enumerate(trues):
            (F if g in hit else M).append((t[0], t[1]))
    print(f"{run}: found {len(F)}, missed {len(M)} ({100*len(M)/(len(F)+len(M)):.1f}% missed)")
    return np.asarray(F), np.asarray(M), cfg, thr


# one panel per run: all compartments in grey, matched in blue, unmatched in red
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
             + ("(τ = 7% acceptance box, NO existence threshold, the view mAP@7 takes)" if MAP7 else "(τ = 7% acceptance box AND the existence threshold, at each run's fitted θ)"), x=0.01, ha="left", fontsize=BASE, y=1.02)
fig.tight_layout()
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=SMALL, markerscale=4,
           bbox_to_anchor=(0.5, 0.0))
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT); print("wrote", OUT.relative_to(ROOT))

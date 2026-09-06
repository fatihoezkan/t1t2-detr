"""Predicted against true over every matched compartment of the test set.

Two rows (reference, loss_uniform), three columns (T1, T2, signal fraction). Each dot is
one matched compartment: x is the ground truth, y the prediction the ND rule assigned to
it at that run's own fitted threshold, the protocol under which the thesis reports
parameter errors. The tau = 7% acceptance corridor is drawn around the diagonal for T1
and T2. Reads results/<run>/ (config, summary.json, best.pt) and the test parquets.
Writes figures/12_pred_true_scatter.png. Usage: PYTHONPATH=.:datagen python3 evaluation/figures/make_scatter_figure.py
"""
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

from t1t2.eval import _match  # noqa: E402
from t1t2.runs import load_run  # noqa: E402

OUT = ROOT / "figures" / "12_pred_true_scatter.png"
BASE, SMALL, TINY = 9, 8, 7
mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": BASE, "axes.titlesize": BASE, "axes.labelsize": BASE,
    "xtick.labelsize": TINY, "ytick.labelsize": TINY, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white"})
TAU = 0.07
RUNS = [("baseline_v2_reproduction", "reference"), ("loss_uniform", "loss_uniform")]


def pairs_for(run):
    """Load a model and collect matched predictions and true values."""
    # inference, then the queries above the fitted threshold
    loaded = load_run(ROOT / "results" / run)
    cfg, spans, thr = loaded.cfg, loaded.spans, loaded.fitted_threshold
    q, all_trues = loaded.predict("test")
    P = np.asarray(q["params"]); E = np.asarray(q["exist_prob"])
    out = []
    for i, trues in enumerate(all_trues):
        pred = [(float(P[i, k, 0]), float(P[i, k, 1]), float(P[i, k, 2]))
                for k in np.where(E[i] >= thr)[0]]
        # The evaluation's own assignment (Hungarian in log T1-T2), distance-unbounded,
        # so the scatter shows the full spread; the tau corridor marks what the ND rule
        # would additionally accept.
        for pp, tt in _match(pred, trues):
            out.append((tt[0], tt[1], tt[2], pp[0], pp[1], pp[2]))
    a = np.asarray(out)
    print(run, "matched pairs:", len(a), "theta", thr)
    return a, spans, cfg


# rows: runs; columns: T1, T2, weight
fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0))
for r, (run, label) in enumerate(RUNS):
    a, spans, cfg = pairs_for(run)
    for c, (name, ti, pi, span, lo, hi, log) in enumerate([
            ("$T_1$ (ms)", 0, 3, spans[0], cfg.data.t1_min, cfg.data.t1_max, True),
            ("$T_2$ (ms)", 1, 4, spans[1], cfg.data.t2_min, cfg.data.t2_max, True),
            ("signal fraction", 2, 5, None, 0.0, 1.0, False)]):
        ax = axes[r][c]
        x, y = a[:, ti], a[:, pi]
        ax.scatter(x, y, s=3, alpha=0.10, color="#1f4e79", edgecolors="none", rasterized=True)
        # log axes with the identity line and the tau corridor; linear for the weight
        if log:
            ax.set_xscale("log"); ax.set_yscale("log")
            lim = (lo * 0.85, hi * 1.18)
            g = np.array(lim)
            ax.plot(g, g, color="0.3", lw=0.9)
            f = float(np.exp(TAU * span))
            ax.plot(g, g * f, color="0.55", lw=0.8, ls="--")
            ax.plot(g, g / f, color="0.55", lw=0.8, ls="--")
        else:
            lim = (0.0, 1.02)
            ax.plot(lim, lim, color="0.3", lw=0.9)
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        # median error printed in the corner
        med = np.median(np.abs(y - x) / np.maximum(x, 1e-9)) * 100 if log else np.median(np.abs(y - x))
        note = f"median rel. err. {med:.1f}%" if log else f"median abs. err. {med:.3f}"
        ax.text(0.03, 0.97, note, transform=ax.transAxes, fontsize=TINY, va="top")
        if r == 1: ax.set_xlabel("true " + name)
        if c == 0: ax.set_ylabel(label + "\npredicted", fontsize=SMALL)
        if r == 0: ax.set_title(name, fontsize=SMALL)
fig.suptitle("Predicted against true, every matched compartment of the test set "
             f"(τ = {TAU:.0%} corridor dashed)", x=0.01, ha="left", fontsize=BASE, y=1.0)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT); print("wrote", OUT.relative_to(ROOT))

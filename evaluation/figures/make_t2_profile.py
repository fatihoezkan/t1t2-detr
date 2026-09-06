"""Detection and error as a function of true T2 alone.

The 1D companion to the error map: found share (top) and median relative T1/T2 error
(bottom) against true T2, all T1 pooled, for the reference and the final model. The longest
echo time is the vertical line; the long-T2 argument predicts the T2-error curve rises to
its right while T1 stays flat. Reads results/<run>/ (config, summary.json, best.pt) and the
test parquets. Writes figures/16_t2_profile.png.
Usage: PYTHONPATH=.:datagen python3 evaluation/figures/make_t2_profile.py
"""
import os
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from t1t2 import nd_metrics as ndm  # noqa: E402
from t1t2.eval import _match  # noqa: E402
from t1t2.runs import load_run  # noqa: E402

OUT = ROOT / "figures" / "16_t2_profile.png"
BASE, SMALL, TINY = 9, 8, 7
mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": BASE, "axes.titlesize": BASE, "axes.labelsize": BASE,
    "xtick.labelsize": TINY, "ytick.labelsize": TINY, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white"})
TAU, TE_MAX = 0.07, 150.0
RUNS = [("baseline_v2_reproduction", "reference", "#7f7f7f"),
        ("loss_uniform", "final model", "#1f4e79")]


def collect(run):
    """Gather detection results and parameter errors across true T2 values."""
    # inference, then the ND rule and the Hungarian match per voxel
    loaded = load_run(ROOT / "results" / run)
    cfg, spans, thr = loaded.cfg, loaded.spans, loaded.fitted_threshold
    q, all_trues = loaded.predict("test")
    P = np.asarray(q["params"]); E = np.asarray(q["exist_prob"])
    det_t2, det_f, m_t2, e1, e2 = [], [], [], [], []
    for i, trues in enumerate(all_trues):
        recs = ndm.voxel_records(P[i], E[i], trues, spans, TAU, exist_thresh=thr)
        hit = {r["gt"] for r in recs if r["gt"] is not None}
        for g, t in enumerate(trues):
            det_t2.append(t[1]); det_f.append(1.0 if g in hit else 0.0)
        pred = [(float(P[i, k, 0]), float(P[i, k, 1]), float(P[i, k, 2]))
                for k in np.where(E[i] >= thr)[0]]
        for pp, tt in _match(pred, trues):
            m_t2.append(tt[1])
            e1.append(abs(pp[0] - tt[0]) / tt[0] * 100)
            e2.append(abs(pp[1] - tt[1]) / tt[1] * 100)
    # summary above and below the longest TE
    det_t2, det_f = np.asarray(det_t2), np.asarray(det_f)
    m_t2, e1, e2 = np.asarray(m_t2), np.asarray(e1), np.asarray(e2)
    hi = m_t2 > TE_MAX
    print(f"{run}: found {100*det_f[det_t2>TE_MAX].mean():.1f}% above TE vs "
          f"{100*det_f[det_t2<=TE_MAX].mean():.1f}% below | "
          f"median T2 err {np.median(e2[hi]):.1f}% above vs {np.median(e2[~hi]):.1f}% below | "
          f"T1 err {np.median(e1[hi]):.1f}% vs {np.median(e1[~hi]):.1f}%")
    return (det_t2, det_f, m_t2, e1, e2, cfg)


def prof(x, v, edges, stat):
    """Summarize each log-spaced bin that has at least 100 samples."""
    idx = np.digitize(np.log(x), np.log(edges)) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= 100:
            out[b] = stat(v[sel])
    return out


# top: found share; bottom: median T2 and T1 error, both against true T2
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
for run, label, col in RUNS:
    det_t2, det_f, m_t2, e1, e2, cfg = collect(run)
    edges = np.logspace(np.log10(cfg.data.t2_min), np.log10(cfg.data.t2_max), 19)
    mid = np.sqrt(edges[:-1] * edges[1:])
    ax1.plot(mid, 100 * prof(det_t2, det_f, edges, np.mean), color=col, lw=1.6, label=label)
    ax2.plot(mid, prof(m_t2, e2, edges, np.median), color=col, lw=1.6,
             label=f"{label}, $T_2$ error")
    ax2.plot(mid, prof(m_t2, e1, edges, np.median), color=col, lw=1.2, ls=":",
             label=f"{label}, $T_1$ error")
for ax in (ax1, ax2):
    ax.set_xscale("log")
    ax.axvline(TE_MAX, color="k", lw=1.0, ls="--")
ax1.text(TE_MAX * 1.08, 20, "longest TE", fontsize=TINY, ha="left",
         path_effects=[pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])
ax1.set_ylabel("share of compartments found (%)"); ax1.set_ylim(0, 100)
ax1.legend(fontsize=TINY, loc="lower left")
ax2.set_ylabel("median relative error (%)"); ax2.set_ylim(bottom=0)
ax2.set_xlabel("true $T_2$ (ms)")
ax2.legend(fontsize=TINY, loc="upper left")
fig.suptitle("Detection and error against true $T_2$ (all $T_1$ pooled)",
             x=0.01, ha="left", fontsize=BASE)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT); print("wrote", OUT)

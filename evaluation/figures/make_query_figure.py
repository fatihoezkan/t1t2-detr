"""Query usage figure for the reference run (baseline_v2_reproduction).

Panels: how often each of the ten queries fires at the run's fitted threshold, the
compartment fraction each working query predicts, and where each looks in the (T1, T2)
plane. Reads results/<run>/{config.yaml, summary.json, checkpoints/best.pt} and the test
parquets named in the config; inference runs on the CPU. Writes figures/11_queries.png.
Usage: PYTHONPATH=.:datagen python3 evaluation/figures/make_query_figure.py
"""
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

from t1t2.runs import load_run  # noqa: E402

RUN = "baseline_v2_reproduction"
OUT = ROOT / "figures" / "11_queries.png"
BASE, SMALL, TINY = 9, 8, 7
mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
 "font.size": BASE, "axes.titlesize": BASE, "axes.labelsize": BASE, "legend.fontsize": SMALL,
 "xtick.labelsize": TINY, "ytick.labelsize": TINY, "axes.titlelocation": "left", "axes.titlepad": 6,
 "axes.spines.top": False, "axes.spines.right": False, "xtick.direction": "out", "ytick.direction": "out",
 "legend.frameon": False, "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white"})
C_GT, C_NOISE = "#1f4e79", "#b0b0b0"
QCOL = ["#1f4e79", "#c1670c", "#7b3294", "#2e7d32"]   # one per working query; the reference has 4
HALO = [pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()]


def loglabels(ax, which="both"):
    """Add readable tick labels to logarithmic axes."""
    def fmt(v, _):
        return (f"{v/1000:.1f}".rstrip("0").rstrip(".") + "k") if v >= 1000 else f"{v:g}"
    for axis, (lo, hi) in ([(ax.xaxis, ax.get_xlim())] if which in ("both", "x") else []) + \
                          ([(ax.yaxis, ax.get_ylim())] if which in ("both", "y") else []):
        def locs(subs, n):
            L = mticker.LogLocator(subs=subs, numticks=n); L.set_axis(axis)
            return [t for t in L.tick_values(lo, hi) if lo <= t <= hi]
        maj = locs((1., 2., 5.), 10)
        if len(maj) < 2: maj = locs("all", 10)
        axis.set_major_locator(mticker.FixedLocator(maj))
        axis.set_minor_locator(mticker.NullLocator())
        axis.set_major_formatter(mticker.FuncFormatter(fmt))


# inference on the test split; a query 'works' if it fires in more than 1 % of the voxels
loaded = load_run(ROOT / "results" / RUN)
cfg, theta = loaded.cfg, loaded.fitted_threshold
q, _ = loaded.predict("test")
P = np.asarray(q["params"]); E = np.asarray(q["exist_prob"])
N, Q, _ = P.shape
rate = np.array([(E[:, j] >= theta).mean() for j in range(Q)])
work = [j for j in range(Q) if rate[j] > 0.01]
print("theta", theta, "| working queries (1-indexed):", [j + 1 for j in work],
      "| firing rates (%)", [f"{100*rate[j]:.1f}" for j in work])

fig = plt.figure(figsize=(11.6, 6.2))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.50, wspace=0.28)

# (a) how often each of the ten queries fires
ax = fig.add_subplot(gs[0, :2])
# grey bars, coloured for the working queries
cols = ["0.80"] * Q
for k, j in enumerate(work): cols[j] = QCOL[k]
ax.bar(np.arange(1, Q + 1), 100 * rate, color=cols, width=0.72)
for j in range(Q):
    if rate[j] < 0.01:
        ax.annotate("never" if rate[j] == 0 else f"{100*rate[j]:.2f}%", (j + 1, 1.5), ha="center",
                    va="bottom", fontsize=TINY - 0.5, color="0.45", rotation=90)
    else:
        ax.annotate(f"{100*rate[j]:.0f}%", (j + 1, 100 * rate[j] + 1.5), ha="center", va="bottom",
                    fontsize=TINY, color=cols[j])
ax.set_xticks(np.arange(1, Q + 1)); ax.set_xlabel("query"); ax.set_ylabel("voxels where it fires (%)")
ax.set_ylim(0, 78); ax.set_title("(a) Four of the ten queries do all the work")

# (b) the predicted compartment fraction of each working query when it fires
ax = fig.add_subplot(gs[0, 2:])
data = [P[:, j, 2][E[:, j] >= theta] for j in work]
bp = ax.boxplot(data, vert=False, widths=0.62, showfliers=False, patch_artist=True,
                medianprops=dict(color="white", lw=1.6))
for k, b in enumerate(bp["boxes"]):
    b.set_facecolor(QCOL[k]); b.set_edgecolor(QCOL[k]); b.set_alpha(0.85)
for part in ("whiskers", "caps"):
    for k, ln in enumerate(bp[part]): ln.set_color(QCOL[k // 2])
for k, d in enumerate(data):
    md = np.median(d); side = "right" if md > 0.9 else "center"
    ax.annotate(f"median {md:.2f}", (md - 0.02 if md > 0.9 else md, k + 1.42), ha=side, va="bottom",
                fontsize=TINY, color=QCOL[k], path_effects=HALO)
ax.set_yticks(range(1, len(work) + 1)); ax.set_yticklabels([f"query {j+1}" for j in work])
ax.set_xlabel("predicted compartment fraction"); ax.set_xlim(0, 1.04); ax.set_ylim(0.4, len(work) + 0.95)
ax.invert_yaxis()
ax.set_title("(b) What they do divide up is compartment size")

# (c) 2-D histogram of the (T1, T2) predictions of each working query, one panel each
t1e = np.logspace(np.log10(50), np.log10(4000), 33)
t2e = np.logspace(np.log10(5), np.log10(600), 33)
H = []
for j in work:
    m = E[:, j] >= theta
    h, _, _ = np.histogram2d(P[m, j, 0], P[m, j, 1], bins=[t1e, t2e])
    H.append(h / max(h.max(), 1))
for k, (j, h) in enumerate(zip(work, H)):
    ax = fig.add_subplot(gs[1, k])
    ax.pcolormesh(t1e, t2e, h.T, cmap="YlGnBu", vmin=0, vmax=1, shading="auto")
    ax.plot([5, 4000], [5, 4000], color="0.5", lw=0.7, ls=":")
    mm = E[:, j] >= theta
    mt1, mt2 = np.median(P[mm, j, 0]), np.median(P[mm, j, 1])
    for kk, jj in enumerate(work):     # every panel carries all four medians, for comparison
        m2 = E[:, jj] >= theta
        ax.plot(np.median(P[m2, jj, 0]), np.median(P[m2, jj, 1]), marker="o", ms=3.4, mfc="none",
                mec="0.45", mew=0.8, zorder=6)
    ax.plot(mt1, mt2, marker="o", ms=6.0, mfc=QCOL[k], mec="white", mew=1.1, zorder=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(50, 4000); ax.set_ylim(5, 600)
    ax.set_xlabel("$T_1$ (ms)")
    if k == 0: ax.set_ylabel("$T_2$ (ms)")
    else: ax.tick_params(labelleft=False)
    ax.set_title(f"(c{k+1}) query {j+1}   median {np.median(P[E[:, j] >= theta, j, 0]):.0f} / {np.median(P[E[:, j] >= theta, j, 1]):.0f} ms", color=QCOL[k], fontsize=TINY + 0.5)
    loglabels(ax)
fig.text(0.008, 0.475, "(c) Where each working query looks. The filled dot is that query's median; "
         "the open circles repeat all four medians in every panel.", ha="left", fontsize=BASE)
fig.suptitle(f"Query usage in the reference run, at its threshold $\\theta$ = {theta:.2f}",
             fontsize=BASE, x=0.008, y=0.985, ha="left")
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, bbox_inches="tight"); print("wrote", OUT.relative_to(ROOT))

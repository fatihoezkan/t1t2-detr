"""What the noise does to small compartments, in three panels (final model, test set).

(a) Share of true compartments found at low SNR (30 to 60) and high SNR (100 to 150), by
the compartment's share of the signal. (b) One two-compartment test voxel: the small
compartment's own signal and what is left of it after the best single-compartment fit, in
units of sigma. (c) Share of the smaller compartment found in two-compartment voxels, by
that residual in units of sigma. Reads results/compartment_noise_ratio_test.parquet and the n2
test parquet; writes figures/20_noise_small_compartments.png and results/separability_k2_test.parquet. Usage: PYTHONPATH=.:datagen python3 evaluation/figures/make_noise_effect_figure.py
"""
import os
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from t1t2.physics import forward_numpy, load_protocol  # noqa: E402

RESULTS = ROOT / "results"
BASE, SMALL, TINY = 9, 8, 7
mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": BASE, "axes.titlesize": BASE, "axes.labelsize": BASE,
    "xtick.labelsize": TINY, "ytick.labelsize": TINY, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white"})
OUT = ROOT / "figures" / "20_noise_small_compartments.png"
GREY, BLUE, ORANGE, LIGHT = "#7f7f7f", "#1f4e79", "#c65911", "#9dbbd8"
# Per-compartment table of the test set: voxel, K, c, w, A (compartment amplitude), sigma,
# r = A / sigma, snr, and the found flags of the reference and the final model at their
# fitted thresholds.
D = pd.read_parquet(RESULTS / "compartment_noise_ratio_test.parquet")

# residual of the best one-compartment fit, K = 2 voxels
proto = load_protocol()
ti = np.asarray(getattr(proto, "ti", getattr(proto, "TI", None)), float); te = np.asarray(getattr(proto, "te", getattr(proto, "TE", None)), float)
# The n2 test voxels start at row 3333 of the concatenated (n1, n2, n3) test set.
df2 = pd.read_parquet(ROOT / "data" / "t1_3500_t2_500_100k" / "n2" / "test.parquet"); off = 3333
T1g = np.logspace(np.log10(50), np.log10(3500), 60); T2g = np.logspace(np.log10(5), np.log10(500), 60)
G = np.array([forward_numpy(proto, np.array([a]), np.array([b]), np.array([1.0])).ravel() for a in T1g for b in T2g])
Gn2 = (G ** 2).sum(1)
rows, store = [], {}
for i, r in enumerate(df2.itertuples(index=False)):
    S = forward_numpy(proto, np.array([r.T1_1, r.T1_2]), np.array([r.T2_1, r.T2_2]), np.array([r.w_1, r.w_2])).ravel()
    proj = G @ S; res2 = S @ S - proj ** 2 / Gn2; b = int(np.argmin(res2))
    resid = S - (proj[b] / Gn2[b]) * G[b]
    rows.append((off + i, np.sqrt(max(res2[b], 0) / 64) / r.sigma)); store[off + i] = (S, resid)
R = pd.DataFrame(rows, columns=["voxel", "res_over_sigma"])
k2 = D[D.K == 2]; small = k2.loc[k2.groupby("voxel").w.idxmin()].merge(R, on="voxel")
small.to_parquet(RESULTS / "separability_k2_test.parquet")
print("K=2 smallest: amplitude/sigma median %.1f, residual/sigma median %.2f" % (small.r.median(), small.res_over_sigma.median()))

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(12.5, 3.7), gridspec_kw={"width_ratios": [1, 1.35, 1]})

# (a) who the noise hurts: the small compartments
wb = [(0.05, 0.1, "$w<0.1$"), (0.1, 0.3, "$0.1$–$0.3$"), (0.3, 0.6, "$0.3$–$0.6$"), (0.6, 1.01, "$w\\geq0.6$")]
lo = D[(D.snr >= 30) & (D.snr < 60)]; hi = D[(D.snr >= 100) & (D.snr <= 150)]
ylo = [100 * lo[(lo.w >= a) & (lo.w < b)].found_final.mean() for a, b, _ in wb]
yhi = [100 * hi[(hi.w >= a) & (hi.w < b)].found_final.mean() for a, b, _ in wb]
x = np.arange(4); bw = 0.38
axA.bar(x - bw / 2, ylo, bw, color=LIGHT, label="more noise: SNR 30–60")
axA.bar(x + bw / 2, yhi, bw, color=BLUE, label="less noise: SNR 100–150")
for xi, (a, b) in enumerate(zip(ylo, yhi)):
    axA.text(xi - bw / 2, a + 1.5, f"{a:.0f}", ha="center", fontsize=TINY, color=BLUE)
    axA.text(xi + bw / 2, b + 1.5, f"{b:.0f}", ha="center", fontsize=TINY, color=BLUE)
axA.set_xticks(x); axA.set_xticklabels([l for _, _, l in wb]); axA.set_ylim(0, 108)
axA.set_ylabel("share of true compartments found (%)"); axA.set_xlabel("compartment's share of the signal $w$")
axA.set_title("(a) noise hurts the small compartments only"); axA.legend(fontsize=TINY, loc="upper left")

# (b) one voxel: the small compartment's own signal (grey) against what is left after the
# one-compartment refit (blue), both in units of the noise level sigma, same scale, with
# the +-sigma band.
cand = small[(small.r > 15) & (small.r < 30) & (small.res_over_sigma > 0.5) & (small.res_over_sigma < 0.8) & (~small.found_final.astype(bool))]
ex = cand.sort_values("w").iloc[len(cand) // 2]; v = int(ex.voxel); row = df2.iloc[v - off]
sigma = float(row.sigma); S, resid = store[v]
smallidx = 0 if row.w_1 < row.w_2 else 1
w_s = (row.w_1, row.w_2)[smallidx]; t1_s = (row.T1_1, row.T1_2)[smallidx]; t2_s = (row.T2_1, row.T2_2)[smallidx]
w_l = (row.w_1, row.w_2)[1 - smallidx]; t1_l = (row.T1_1, row.T1_2)[1 - smallidx]; t2_l = (row.T2_1, row.T2_2)[1 - smallidx]
g_small = w_s * forward_numpy(proto, np.array([t1_s]), np.array([t2_s]), np.array([1.0])).ravel()
order = np.lexsort((te, ti)); xs = np.arange(64)
axB.axhspan(-1, 1, color=ORANGE, alpha=0.18, lw=0); axB.text(63.3, -2.1, "noise band $\\pm\\sigma$", ha="right", va="top", fontsize=TINY, color=ORANGE)
axB.plot(xs, g_small[order] / sigma, "-", color=GREY, lw=1.4, label="the small compartment's own signal")
axB.plot(xs, resid[order] / sigma, "-", color=BLUE, lw=1.6, label="mismatch after fitting one compartment")
axB.axhline(0, color="k", lw=0.5)
for k in range(1, 8): axB.axvline(8 * k - 0.5, color="0.85", lw=0.6)
axB.set_xticks(np.arange(4, 64, 8)); axB.set_xticklabels([f"{int(t)}" for t in np.sort(np.unique(ti))]); axB.set_xlabel("inversion time TI (ms); the 8 echo times inside each block")
axB.set_ylabel("signal in units of the noise level $\\sigma$"); axB.set_xlim(-0.5, 63.5)
axB.set_title(f"(b) test voxel {v}: small compartment $w={w_s:.2f}$ next to a large one $w={w_l:.2f}$", fontsize=TINY); axB.legend(fontsize=TINY, loc="upper left")
axB.text(0.99, 0.04, f"$T_1$ {t1_s:.0f} vs {t1_l:.0f} ms, $T_2$ {t2_s:.0f} vs {t2_l:.0f} ms\nSNR of this voxel: {row.snr:.0f}\nsmall compartment not detected by the final model",
         transform=axB.transAxes, ha="right", va="bottom", fontsize=TINY)
print(f"example voxel {v}: w_small {w_s:.2f}, amp/sigma {ex.r:.1f}, residual/sigma {ex.res_over_sigma:.2f}, T1 {t1_s:.0f}/{t1_l:.0f}, T2 {t2_s:.0f}/{t2_l:.0f}, SNR {row.snr:.0f}")

# (c) the residual decides
edges = [0, 1, 2, 4, 1e9]; labels = ["below $\\sigma$", "$\\sigma$ to $2\\sigma$", "$2\\sigma$ to $4\\sigma$", "above $4\\sigma$"]
vals, ns = [], []
for a, b in zip(edges[:-1], edges[1:]):
    m = (small.res_over_sigma >= a) & (small.res_over_sigma < b); vals.append(100 * small.found_final[m].mean()); ns.append(int(m.sum()))
axC.bar(np.arange(4), vals, 0.6, color=[LIGHT, LIGHT, BLUE, BLUE])
for i, (vv, n) in enumerate(zip(vals, ns)): axC.text(i, vv + 1.5, f"{vv:.0f} %\n$n$={n}", ha="center", fontsize=TINY, color=BLUE)
axC.set_xticks(np.arange(4)); axC.set_xticklabels(labels); axC.set_ylim(0, 112)
axC.set_xlabel("mismatch after fitting one compartment"); axC.set_ylabel("share of small compartments found (%)")
axC.set_title("(c) the mismatch decides, across all two-compartment voxels")
fig.suptitle("What the noise does to small compartments (final model, strict rule at its fitted $\\theta$, test set)", x=0.01, ha="left", fontsize=BASE + 1)
OUT.parent.mkdir(exist_ok=True)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT); print("wrote", OUT)

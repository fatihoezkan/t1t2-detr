"""Cumulative distributions of the parameter errors over the whole test set.

ECDFs of the relative T1 error, the relative T2 error and the absolute signal-fraction
error over every matched compartment of the 9 999 test voxels, for the reference baseline
and the final model, split by the true compartment count. Runs inference for both models
on the test split (about a minute each on a CPU) and reads results/<run>/{summary.json,
metrics_detr.json} and the test parquets. Writes figures/19_error_distribution.png and
results/error_distribution_summary.json. Usage: PYTHONPATH=.:datagen python3 evaluation/figures/make_error_distribution.py
"""
import json
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

from t1t2.config import load_config  # noqa: E402
from t1t2.eval import _match  # noqa: E402
from t1t2.runs import load_run  # noqa: E402

RESULTS = ROOT / "results"
BASE, SMALL, TINY = 9, 8, 7
mpl.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": BASE, "axes.titlesize": BASE, "axes.labelsize": BASE,
    "xtick.labelsize": TINY, "ytick.labelsize": TINY, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white"})
RUNS = [("baseline_v2_reproduction", "reference", "#7f7f7f"), ("loss_uniform", "final model", "#1f4e79")]
LS = {1: "-", 2: "--", 3: ":"}
OUT_FIG = ROOT / "figures" / "19_error_distribution.png"
OUT_JSON = RESULTS / "error_distribution_summary.json"
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

# ground truth of the test set from the parquets
cfg = load_config(RESULTS / "loss_uniform" / "config.yaml")
df = pd.concat([pd.read_parquet(p) for p in cfg.data.test_path], ignore_index=True)
kmax = max(int(c.split("_")[1]) for c in df.columns if c.startswith("T1_"))
trues = [[(getattr(r, f"T1_{k}"), getattr(r, f"T2_{k}"), getattr(r, f"w_{k}"))
          for k in range(1, kmax + 1) if np.isfinite(getattr(r, f"T1_{k}"))]
         for r in df.itertuples(index=False)]
K = np.array([len(t) for t in trues])  # compartment count per voxel
# The shaded band marks where the tau = 7 % acceptance edge of the strict rule falls in
# relative terms. It is a share of the log range, hence asymmetric: -26/+35 % for T1,
# -28/+38 % for T2.
span1 = np.log(cfg.data.t1_max / cfg.data.t1_min); span2 = np.log(cfg.data.t2_max / cfg.data.t2_min)
band1 = (100 * (1 - np.exp(-0.07 * span1)), 100 * (np.exp(0.07 * span1) - 1))
band2 = (100 * (1 - np.exp(-0.07 * span2)), 100 * (np.exp(0.07 * span2) - 1))

# inference for both runs, matched-pair errors and their percentiles
data, summary = {}, {}
for run, label, col in RUNS:
    loaded = load_run(RESULTS / run)
    q, _ = loaded.predict("test")
    P, E, thr = q["params"], q["exist_prob"], loaded.fitted_threshold
    # The count accuracy recomputed here is checked against the stored metrics to make sure
    # the predictions line up with the test set.
    stored = json.load(open(RESULTS / run / "metrics_detr.json"))["count_accuracy"]
    n_pred = (E >= thr).sum(1); cacc = float((n_pred == K).mean())
    assert abs(cacc - stored) < 2e-3, f"{run}: order check failed {cacc:.4f} vs stored {stored:.4f}"
    # Matching is the evaluation's own Hungarian assignment in log(T1, T2) at each run's
    # fitted theta, the protocol under which the thesis reports parameter errors.
    rows = []
    for i, tr in enumerate(trues):
        pred = [(float(P[i, k, 0]), float(P[i, k, 1]), float(P[i, k, 2])) for k in np.where(E[i] >= thr)[0]]
        for pp, tt in _match(pred, tr):
            rows.append((K[i], tt[2], abs(pp[0] - tt[0]) / tt[0] * 100, abs(pp[1] - tt[1]) / tt[1] * 100, abs(pp[2] - tt[2])))
    # columns: K, true w, rel T1 err (%), rel T2 err (%), abs w err
    A = np.array(rows); data[label] = A
    # percentiles overall and per K
    s = {"theta_fit": thr, "n_matched": int(len(A))}
    for name, j in (("t1_rel", 2), ("t2_rel", 3), ("w_abs", 4)):
        s[name] = {"median": float(np.median(A[:, j])), "p75": float(np.percentile(A[:, j], 75)),
                   "p90": float(np.percentile(A[:, j], 90)), "p95": float(np.percentile(A[:, j], 95))}
        for k in (1, 2, 3):
            m = A[:, 0] == k
            s[f"{name}_K{k}"] = {"median": float(np.median(A[m, j])), "p90": float(np.percentile(A[m, j], 90)),
                                 "share_within_band": None}
        s[f"{name}_share_le_band"] = float(np.mean(A[:, j] <= (band1[0] if j == 2 else band2[0] if j == 3 else 0.07)))
    summary[label] = s
    print(label, json.dumps({k: v for k, v in s.items() if not k.endswith(("K1", "K2", "K3"))}, indent=None)[:600])
OUT_JSON.write_text(json.dumps(summary, indent=1))


def ecdf(ax, x, **kw):
    """Plot the percentage of values at or below each value."""
    x = np.sort(x); ax.plot(x, np.arange(1, len(x) + 1) / len(x) * 100, **kw)


# three ECDF panels: T1, T2, weight; one line per (run, K)
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
titles = ["relative $T_1$ error (%)", "relative $T_2$ error (%)", "absolute signal-fraction error"]
for j, ax in enumerate(axes):
    col_idx = j + 2
    for run, label, col in RUNS:
        A = data[label]
        for k in (1, 2, 3):
            m = A[:, 0] == k
            ecdf(ax, A[m, col_idx], color=col, ls=LS[k], lw=1.3,
                 label=f"{label}, $K={k}$")
    # acceptance band on the T1/T2 panels, a line at tau on the weight panel
    if j < 2:
        band = band1 if j == 0 else band2
        ax.axvspan(band[0], band[1], color="#c65911", alpha=0.12, lw=0)
        ax.text(band[0] * 1.02, 3, "$\\tau = 7\\,\\%$\nacceptance edge", fontsize=TINY, color="#c65911", va="bottom")
        ax.set_xscale("log"); ax.set_xlim(0.1, 300)
        ax.set_xticks([0.1, 1, 10, 100]); ax.set_xticklabels(["0.1", "1", "10", "100"])
    else:
        ax.axvline(0.07, color="#c65911", alpha=0.5, lw=1, ls="-")
        ax.text(0.075, 3, "$\\tau = 0.07$", fontsize=TINY, color="#c65911", va="bottom")
        ax.set_xlim(0, 0.5)
    ax.set_title(titles[j]); ax.set_ylim(0, 100); ax.set_ylabel("share of matched compartments (%)" if j == 0 else "")
    ax.grid(alpha=0.25, lw=0.5)
axes[0].legend(fontsize=TINY, loc="upper left", ncol=1)
fig.suptitle("The whole distribution of the parameter errors over the test set: cumulative share of matched compartments below a given error",
             x=0.01, ha="left", fontsize=BASE + 1)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT_FIG); print("wrote", OUT_FIG)

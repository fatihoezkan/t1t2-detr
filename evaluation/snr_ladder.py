"""Score finished runs on the fixed-SNR test sets and plot them against SNR. Run from the repo root.

Rungs share voxels and noise draw, only sigma differs. Per run and rung: strict and count accuracy
at the validation-calibrated threshold (results/threshold_val/<run>.json) and median relative
T1/T2 error over matched compartments at the fitted threshold (summary.json), per family (mean,
min-to-max range over its runs). SNR 20 lies below the training range (30 to 150): written to
the JSON, left out of the figure and table. Writes summary.json, snr_ladder.png, snr_ladder.tex
to --out-dir (default results/snr_ladder/). Usage: snr_ladder.py [--family NAME=RUN,RUN] [--replot]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from t1t2 import nd_metrics as ndm  # noqa: E402
from t1t2.eval import _match  # noqa: E402
from t1t2.runs import load_run  # noqa: E402
from threshold_sweep import score, score_by_k  # noqa: E402

TAU = 0.07
SNRS = (20, 40, 60, 100, 150)          # every rung the generator writes
SNRS_IN_RANGE = (40, 60, 100, 150)     # the rungs inside the training range, drawn and tabulated
TRAIN_RANGE = (30, 150)

DEFAULT_FAMILIES = {
    "reference": ["baseline_v2_reproduction", "baseline_seed20260725",
                  "baseline_seed20260726", "baseline_seed20260727"],
    "final": ["loss_uniform", "loss_uniform_seed20260725",
              "loss_uniform_seed20260726", "loss_uniform_seed20260727"],
}
COLOURS = {"reference": "#7f7f7f", "final": "#1f4e79"}

# The four quantities drawn in the top row, with their axis titles.
QUANTITIES = [
    ("strict_acc", "strict voxel accuracy (%)"),
    ("count_acc", "count accuracy (%)"),
    ("t1_rel_median", "median relative $T_1$ error (%)"),
    ("t2_rel_median", "median relative $T_2$ error (%)"),
]


def ladder_paths(cfg, snr: int) -> list[str]:
    """The fixed-SNR files sitting next to each test split of the run's config."""
    paths = cfg.data.test_path if isinstance(cfg.data.test_path, list) else [cfg.data.test_path]
    return [str(p).replace("test.parquet", f"test_snr{snr}.parquet") for p in paths]


def relative_errors(query_outputs, trues, theta):
    """Relative T1 and T2 errors in percent, over matched compartments at threshold theta."""
    P = np.asarray(query_outputs["params"])
    E = np.asarray(query_outputs["exist_prob"])
    e1, e2 = [], []
    for i, true in enumerate(trues):
        pred = [(float(P[i, k, 0]), float(P[i, k, 1]), float(P[i, k, 2]))
                for k in np.where(E[i] >= theta)[0]]
        for p, t in _match(pred, true):
            e1.append(abs(p[0] - t[0]) / t[0] * 100.0)
            e2.append(abs(p[1] - t[1]) / t[1] * 100.0)
    return e1, e2


def evaluate_run(results_dir: Path, run: str, device: str) -> dict:
    """Score one trained model at each test noise level."""
    loaded = load_run(results_dir / run, device)
    # Two thresholds: the strict-accuracy one swept on validation by calibrate_threshold.py,
    # and the run's own fitted one from summary.json.
    theta_cal = float(json.loads((results_dir / "threshold_val" / f"{run}.json").read_text())["val_theta"])
    theta_fit = loaded.fitted_threshold
    out = {"theta_calibrated": theta_cal, "theta_fitted": theta_fit, "rungs": {}}
    for snr in SNRS:
        q, trues = loaded.predict(ladder_paths(loaded.cfg, snr))
        recs, ngt = ndm.dataset_records(q, trues, loaded.spans, TAU, exist_thresh=0.0)
        s = score(recs, ngt, theta_cal)
        by_k = score_by_k(recs, ngt, theta_cal)["strict"]
        e1, e2 = relative_errors(q, trues, theta_fit)
        out["rungs"][str(snr)] = {
            "n_voxels": int(len(ngt)),
            "strict_acc": s["voxel_acc"],
            "count_acc": s["count_acc"],
            "strict_k1": by_k.get("1"),
            "strict_k2": by_k.get("2"),
            "strict_k3": by_k.get("3"),
            "t1_rel_median": float(np.median(e1)),
            "t2_rel_median": float(np.median(e2)),
            "n_matched": len(e1),
            "extrapolation": not (TRAIN_RANGE[0] <= snr <= TRAIN_RANGE[1]),
        }
        print(f"{run:32s} SNR {snr:3d}  strict {s['voxel_acc']:6.2f}  count {s['count_acc']:6.2f}  "
              f"K=3 {by_k.get('3', float('nan')):6.2f}  T1 {np.median(e1):5.2f} %  "
              f"T2 {np.median(e2):5.2f} %")
    return out


def stack(summary: dict, family: str, key: str, snrs=SNRS_IN_RANGE) -> np.ndarray:
    """One row per run of the family, one column per rung."""
    runs = summary[family]
    return np.array([[runs[r]["rungs"][str(s)][key] for s in snrs] for r in runs], dtype=float)


def draw(summary: dict, out_png: Path) -> None:
    """Plot model performance across the test noise levels."""
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.titlelocation": "left", "legend.frameon": False,
    })
    families = list(summary)
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.6))

    def curve(ax, key, title):
        """Plot each model family's mean and range across noise levels."""
        for fam in families:
            A = stack(summary, fam, key)
            colour = COLOURS.get(fam)
            ax.plot(SNRS_IN_RANGE, A.mean(0), "-o", color=colour, ms=3.5, lw=1.4, label=fam)
            ax.fill_between(SNRS_IN_RANGE, A.min(0), A.max(0), color=colour, alpha=0.18, lw=0)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xticks(SNRS_IN_RANGE)
        ax.set_xticklabels([str(s) for s in SNRS_IN_RANGE])
        ax.set_xlim(35, 170)

    for ax, (key, title) in zip(axes[0], QUANTITIES):
        curve(ax, key, title)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    for ax, k in zip(axes[1], (1, 2, 3)):
        curve(ax, f"strict_k{k}", f"strict voxel accuracy, $K={k}$ (%)")
        ax.set_xlabel("SNR of the test set")
    axes[1, 3].axis("off")
    axes[1, 3].text(0.0, 0.85,
                    "Lines: mean over the runs of a family\nBand: min to max over those runs\n\n"
                    "Accuracies at each run's validation-\ncalibrated threshold; parameter errors\n"
                    "over matched compartments at the\nrun's own fitted threshold.",
                    transform=axes[1, 3].transAxes, va="top", fontsize=8)
    fig.suptitle("Behaviour against the noise level on the fixed-SNR test sets inside the "
                 "training range (paired voxels)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png)
    print("wrote", out_png)


def latex_table(summary: dict, out_tex: Path) -> None:
    """Mean [range] per rung and family, as tabular rows the thesis can \\input."""
    families = list(summary)

    def cell(fam, key, i):
        """Format a family's mean and range as a LaTeX table entry."""
        col = stack(summary, fam, key)[:, i]
        return f"{col.mean():.2f} {{\\scriptsize[{col.max() - col.min():.2f}]}}"

    n = len(families)
    head = " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{title}}}" for _, title in QUANTITIES)
    lines = [rf"\begin{{tabular}}{{l{'r' * (4 * n)}}}", r"\toprule", f"& {head} \\\\",
             "SNR & " + " & ".join(fam for _ in QUANTITIES for fam in families) + r" \\", r"\midrule"]
    for i, snr in enumerate(SNRS_IN_RANGE):
        cells = [cell(fam, key, i) for key, _ in QUANTITIES for fam in families]
        lines.append(" & ".join([str(snr)] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_tex.write_text("\n".join(lines) + "\n")
    print("wrote", out_tex)


def parse_families(items: list[str] | None) -> dict[str, list[str]]:
    """Read named groups of runs from command-line options."""
    if not items:
        return DEFAULT_FAMILIES
    families = {}
    for item in items:
        name, _, runs = item.partition("=")
        if not runs:
            raise SystemExit(f"--family expects name=run1,run2,...; got {item!r}")
        families[name] = [r for r in runs.split(",") if r]
    return families


def main() -> None:
    """Evaluate noise robustness and save the tables and plots."""
    ap = argparse.ArgumentParser(description="Score finished runs on the fixed-SNR test sets.")
    ap.add_argument("--results-dir", default="results", help="Where the runs live.")
    ap.add_argument("--out-dir", default="results/snr_ladder", help="Where to write the outputs.")
    ap.add_argument("--family", action="append",
                    help="name=run1,run2,... Repeatable. Defaults to the thesis's two families.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--replot", action="store_true",
                    help="Skip inference and redraw from the JSON already in --out-dir.")
    a = ap.parse_args()

    results_dir, out_dir = Path(a.results_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "summary.json"

    if a.replot and out_json.exists():
        summary = json.loads(out_json.read_text())
    else:
        families = parse_families(a.family)
        summary = {fam: {run: evaluate_run(results_dir, run, a.device) for run in runs}
                   for fam, runs in families.items()}
        out_json.write_text(json.dumps(summary, indent=1))
        print("wrote", out_json)

    draw(summary, out_dir / "snr_ladder.png")
    latex_table(summary, out_dir / "snr_ladder.tex")


if __name__ == "__main__":
    main()

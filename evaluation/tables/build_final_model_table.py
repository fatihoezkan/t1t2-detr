#!/usr/bin/env python3
"""Write tables/tab_final_model.tex: the three model families, each over four seeds.

Reference is signal-fraction weighting with 10 queries, loss_uniform is uniform weighting with
10 queries (the single change that helped), final_uniform_q6 is uniform weighting with 6 queries
(the combination). Every cell is the mean over the seeds that exist with the range in brackets;
runs that have not finished are left out and the n column says so. Reads
results/threshold_sweep/<run>.json and results/nd_evaluation/<run>.json.
Usage: python3 evaluation/tables/build_final_model_table.py
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "tables" / "tab_final_model.tex"
SEEDS = (20260724, 20260725, 20260726, 20260727)

FAMILIES = [
    ("Reference", r"signal fraction, 10 q",
     ["baseline_v2_reproduction"] + [f"baseline_seed{s}" for s in SEEDS[1:]]),
    ("\\texttt{loss\\_uniform}", r"uniform, 10 q",
     ["loss_uniform"] + [f"loss_uniform_seed{s}" for s in SEEDS[1:]]),
    ("\\texttt{final\\_uniform\\_q6}", r"uniform, 6 q",
     [f"final_uniform_q6_seed{s}" for s in SEEDS]),
]


def _sweep(r, dim, th, key):
    """Get a saved metric at the requested threshold and dimension."""
    d = json.load(open(RES / "threshold_sweep" / f"{r}.json"))[dim]
    return next(x for x in d if abs(x["threshold"] - th) < 1e-9)[key]


def _perk(r, k):
    """Get strict accuracy for one compartment count at threshold 0.75."""
    d = json.load(open(RES / "threshold_sweep" / f"{r}.json"))["2d_by_k"]["0.75"]
    if "strict" in d:
        return d["strict"][k]
    return d[k] if isinstance(d[k], float) else d[k]["strict"]


def metrics(r):
    """Collect the scores used to compare the final model families."""
    m2 = json.load(open(RES / "nd_evaluation" / f"{r}.json"))["map"]["map@7"]
    return [_sweep(r, "2d", 0.50, "voxel_acc"), _sweep(r, "2d", 0.75, "voxel_acc"),
            _perk(r, "3"), _sweep(r, "2d", 0.75, "count_acc"), m2]


DIGITS = [2, 2, 2, 2, 4]


def main():
    """Write the final-model table with means and ranges across seeds."""
    rows = []
    for label, recipe, runs in FAMILIES:
        have = [r for r in runs if (RES / "threshold_sweep" / f"{r}.json").exists()]
        if not have:
            continue
        cols = list(zip(*[metrics(r) for r in have]))
        cells = []
        for c, d in zip(cols, DIGITS):
            m = statistics.mean(c)
            rg = max(c) - min(c)
            cells.append(f"{m:.{d}f} {{\\scriptsize[{rg:.{d}f}]}}" if len(c) > 1 else f"{m:.{d}f} {{\\scriptsize[n=1]}}")
        rows.append(f"{label} & {recipe} & {len(have)} & " + " & ".join(cells) + r" \\")
    out = [r"\begin{tabular}{llcrrrrr}", r"\toprule",
           r"Model & Recipe & $n$ & strict 2D & strict 2D & $K{=}3$ & count acc. & mAP@7 \\",
           r" & & & $\theta{=}.50$ & $\theta{=}.75$ & $\theta{=}.75$ & $\theta{=}.75$ & 2D \\",
           r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n")
    print("wrote", OUT)
    print("\n".join(rows))


if __name__ == "__main__":
    main()

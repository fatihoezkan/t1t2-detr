#!/usr/bin/env python3
"""Write tables/tab_progression.tex: every model of the thesis in one table.

The order is the story: baseline, then the single changes, then the combinations, then the
final model. The primary column is test strict voxel accuracy at the threshold each run chose
on its own validation split (evaluation/calibrate_threshold.py), the same procedure for every
run, never touching test data. Reads results/threshold_val/<run>.json and
results/nd_evaluation/<run>.json. Runs that have not finished are skipped.
Usage: python3 evaluation/tables/build_progression_table.py
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
VAL = RES / "threshold_val"
OUT = ROOT / "tables" / "tab_progression.tex"
S = (20260724, 20260725, 20260726, 20260727)

GROUPS = [
    ("\\emph{baseline}", [
        ("Reference", "unchanged",
         ["baseline_v2_reproduction"] + [f"baseline_seed{s}" for s in S[1:]]),
    ]),
    ("\\emph{one change}", [
        (r"\texttt{loss\_uniform}", "uniform loss weighting",
         ["loss_uniform"] + [f"loss_uniform_seed{s}" for s in S[1:]]),
        (r"\texttt{aux\_loss}", "loss on every decoder layer", ["aux_loss"]),
        (r"\texttt{physics\_noisy}", "signal-consistency term", ["physics_noisy"]),
        (r"\texttt{physics\_clean}", "consistency, clean target", ["physics_clean"]),
        (r"\texttt{exist\_head\_shared}", "shared existence head", ["exist_head_shared"]),
        (r"\texttt{queries\_6}", "6 queries", ["queries_6"]),
        (r"\texttt{queries\_4}", "4 queries", ["queries_4"]),
        (r"\texttt{decoder\_6}", "6 decoder layers", ["decoder_6"]),
        (r"\texttt{decoder\_2}", "2 decoder layers", ["decoder_2"]),
        (r"\texttt{data\_loguniform}$^{*}$", "log-uniform $T_1$", ["data_loguniform"]),
        (r"\texttt{exist\_weight\_03}", "existence weight 0.3", ["exist_weight_03"]),
    ]),
    ("\\emph{several changes}", [
        ("v3", "2 decoder layers, 6 queries, shared existence head, "
               "$\\sqrt{w}$ weighting, signal-consistency term", ["baseline_v3"]),
        ("v3 $-$ sqrt", "v3, fraction weighting", ["baseline_v3_no_sqrt"]),
        ("v3 $-$ consistency", "v3, no physics term", ["baseline_v3_no_physics"]),
        ("v4", "6 queries, shared existence head, signal-consistency term",
         ["baseline_v4"]),
        (r"\texttt{final\_uniform\_q6}", "uniform weighting + 6 queries",
         [f"final_uniform_q6_seed{s}" for s in S]),
    ]),
]


def acc(r):
    return json.load(open(VAL / f"{r}.json"))["test_voxel_acc_at_val_theta"]


def theta(r):
    return json.load(open(VAL / f"{r}.json"))["val_theta"]


def map7(r):
    return json.load(open(RES / "nd_evaluation" / f"{r}.json"))["map"]["map@7"]


def have(runs):
    return [r for r in runs if (VAL / f"{r}.json").exists()]


def main():
    ref = have(GROUPS[0][1][0][2])
    ra = [acc(r) for r in ref]
    base, ruler = statistics.mean(ra), max(ra) - min(ra)

    lines = [r"\begin{tabular}{l p{4.3cm} cccrl}", r"\toprule",
             r"Model & What changed & $n$ & $\theta$ & strict acc.\ (\%) & $\Delta$ & \\",
             r"\midrule"]
    for gname, entries in GROUPS:
        lines.append(f"\\multicolumn{{7}}{{l}}{{{gname}}} \\\\")
        for label, what, runs in entries:
            rs = have(runs)
            if not rs:
                continue
            xs = [acc(r) for r in rs]
            m = statistics.mean(xs)
            d = m - base
            th = f"{statistics.mean(theta(r) for r in rs):.2f}"
            v = "better" if d > ruler else ("worse" if d < -ruler else "---")
            dtxt = f"{d:+.2f}" if rs != ref else "---"
            if v != "---":
                dtxt = r"\textbf{" + dtxt + "}"
            rng = f" {{\\scriptsize[{max(xs)-min(xs):.2f}]}}" if len(xs) > 1 else ""
            lines.append(f"\\quad {label} & {what} & {len(rs)} & {th} & "
                         f"{m:.2f}{rng} & {dtxt} & {v} \\\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT)
    print(f"baseline {base:.2f}, ruler {ruler:.2f} pp")
    for l in lines:
        if l.startswith("\\quad"):
            print("  ", l[:96])


if __name__ == "__main__":
    main()

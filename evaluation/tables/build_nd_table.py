"""tables/tab_nd.tex: mAP and exact F1 of every main-matrix run, in the 2D and 3D forms.

2D accepts a prediction on T1 and T2; 3D also requires the signal fraction within tau.
Reads results/nd_evaluation/tables_2d_3d.json (written by build_2d_3d_tables.py). The
frozen v1 baseline row is written only if that run is under results/.
Usage: python3 evaluation/tables/build_nd_table.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES, OUT = ROOT / "results", ROOT / "tables" / "tab_nd.tex"

T = json.load(open(RES / "nd_evaluation" / "tables_2d_3d.json"))
ROWS = [
    ("baseline_v2_reproduction", "reproduction (reference)"),
    ("t1_3500_t2_500_weighted_long", "frozen baseline"),
    ("loss_uniform", r"\texttt{loss\_uniform}"),
    ("aux_loss", r"\texttt{aux\_loss}"),
    ("queries_6", r"\texttt{queries\_6}"),
    ("queries_4", r"\texttt{queries\_4}"),
    ("exist_weight_03", r"\texttt{exist\_weight\_03}"),
    ("decoder_2", r"\texttt{decoder\_2}"),
    ("decoder_6", r"\texttt{decoder\_6}"),
    ("exist_head_shared", r"\texttt{exist\_head\_shared}"),
    ("physics_clean", r"\texttt{physics\_clean}"),
    ("physics_noisy", r"\texttt{physics\_noisy}"),
    ("baseline_v3", "combined v3"),
    ("baseline_v3_no_sqrt", "v3 $-$ sqrt"),
    ("baseline_v3_no_physics", "v3 $-$ consistency"),
    ("baseline_v4", "combined v4"),
]

lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{mAP (avg)} & \multicolumn{2}{c}{mAP@7} & \multicolumn{2}{c}{exact F$_1$} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Run & 2D & 3D & 2D & 3D & 2D & 3D \\", r"\midrule"]
for run, label in ROWS:
    if run not in T:
        print(f"skip {run}: not in tables_2d_3d.json")
        continue
    d2, d3 = T[run]["2d"], T[run]["3d"]
    lines.append(f"{label} & {d2['map']['map_avg']:.4f} & {d3['map']['map_avg']:.4f} & "
                 f"{d2['map']['map@7']:.4f} & {d3['map']['map@7']:.4f} & "
                 f"{d2['exact']['f1']:.4f} & {d3['exact']['f1']:.4f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}")

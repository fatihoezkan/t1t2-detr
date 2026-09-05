"""tables/tab_criteria.tex: the pre-specified criteria of the combined models v3 and v4.

The criteria are the ones written into configs/combined/baseline_v3.yaml and
baseline_v4.yaml before those runs were submitted. The measured values come from
results/<run>/metrics_detr.json and parameter_recovery_detr.json.
Usage: python3 evaluation/tables/build_criteria_table.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES, OUT = ROOT / "results", ROOT / "tables" / "tab_criteria.tex"


def metrics(run):
    return json.load(open(RES / run / "metrics_detr.json"))


def faintest_band_t1(run):
    return json.load(open(RES / run / "parameter_recovery_detr.json"))["bins"][0]["t1_relative_error_median"] * 100


v3, v4 = metrics("baseline_v3"), metrics("baseline_v4")
# (model, criterion as printed, measured value, comparison, limit, decimals)
CRITERIA = [
    ("v3", r"(a) count accuracy $\geq 78.1\,\%$", v3["count_accuracy"] * 100, ">=", 78.1, 2),
    ("v3", r"(b) smallest-band median rel.\ T$_1$ error $\leq 25\,\%$", faintest_band_t1("baseline_v3"), "<=", 25, 2),
    ("v3", r"(c) pooled median abs.\ T$_1$ error $\leq 27$\,ms", v3["t1_abs_median_ms"], "<=", 27, 2),
    ("v4", r"(a) count accuracy $\geq 78.1\,\%$", v4["count_accuracy"] * 100, ">=", 78.1, 2),
    ("v4", r"(b) pooled median abs.\ T$_1$ error $\leq 27$\,ms", v4["t1_abs_median_ms"], "<=", 27, 2),
    ("v4", r"(c) existence F$_1 \geq 0.944$", v4["existence_f1"], ">=", 0.944, 4),
]

lines = [r"\begin{tabular}{llrl}", r"\toprule",
         r"Model & Pre-specified criterion & Measured & Verdict \\", r"\midrule"]
for model, text, value, op, limit, dec in CRITERIA:
    ok = value >= limit if op == ">=" else value <= limit
    verdict = "pass" if ok else r"\textbf{fail}"
    lines.append(f"{model} & {text} & {value:.{dec}f} & {verdict} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}")

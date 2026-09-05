#!/usr/bin/env python3
"""Compare finished experiment arms against the reference run, one config change per arm.

Reads results/<run>/summary.json and config.yaml. Prints metrics as rows and runs as columns,
reference first, with a signed delta per arm, then a per-arm verdict from the config diff
(single change / NOT INTERPRETABLE / DIFFERENT DATASET / CONTROL). Writes comparison.md,
comparison_metrics.csv and comparison_arms.csv to results/_comparison/. One run per arm, so no
error bars; see docs/experiments.md.
    python evaluation/compare_experiments.py --all | <run> [<run> ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

# The reference run. Hard-coded so it cannot be moved by a flag. It is baseline_v2_reproduction
# rather than the original baseline because its config ships in configs/ and can be re-run.
BASELINE_RUN = "baseline_v2_reproduction"

REPO_ROOT = Path(__file__).resolve().parent.parent


# Metric rows: the reported metrics plus training and threshold provenance, in reading order.
#
# Each entry is (label, source, key, formatting, direction) where
#   source     'detr'  -> summary["detr"][key]                      (test-split metrics)
#              'recov' -> summary["detr"]["parameter_recovery"]["summary"][key]
#              'run'   -> summary[key]                              (training provenance)
#              'thr'   -> summary["threshold_calibration"][key]
#   formatting 'pct'   -> percentage (value is a fraction in [0, 1])
#              'ms'    -> milliseconds, 2 decimals
#              'num'   -> plain number, 4 decimals
#              'loss'  -> 6 decimals; loss differences between arms sit in the 5th decimal
#              'sec'   -> seconds, 1 decimal
#              'int'   -> integer
#   direction  +1 higher is better, -1 lower is better, 0 neither (provenance / diagnostics)
#
# direction only labels a delta as better or worse. It does not pick a winner: several metrics
# trade against each other (exist_weight_03 is the clearest case).
METRICS: list[tuple[str, str, str, str, int]] = [
    # --- the headline set ---
    ("Compartment-count accuracy",             "detr",  "count_accuracy",                        "pct", +1),
    ("Count MAE (compartments)",               "detr",  "count_mae",                             "num", -1),
    ("Existence precision",                    "detr",  "existence_precision",                   "num", +1),
    ("Existence recall",                       "detr",  "existence_recall",                      "num", +1),
    ("Existence F1",                           "detr",  "existence_f1",                          "num", +1),
    ("T1 relative error, median",              "detr",  "t1_rel_median",                         "pct", -1),
    ("T2 relative error, median",              "detr",  "t2_rel_median",                         "pct", -1),
    # Medians. Read from the unambiguous aliases; the JSON's t1_mae_ms etc. are also medians.
    ("T1 absolute error, median (ms)",         "detr",  "t1_abs_median_ms",                      "ms",  -1),
    ("T2 absolute error, median (ms)",         "detr",  "t2_abs_median_ms",                      "ms",  -1),
    ("Weight absolute error, median",          "detr",  "w_abs_median",                          "num", -1),
    # The corresponding means.
    ("T1 absolute error, mean (ms)",           "detr",  "t1_abs_mean_ms",                        "ms",  -1),
    ("T2 absolute error, mean (ms)",           "detr",  "t2_abs_mean_ms",                        "ms",  -1),
    ("Weight absolute error, mean",            "detr",  "w_abs_mean",                            "num", -1),
    ("Recovered signal fraction",              "recov", "recovered_signal_fraction",             "pct", +1),
    ("False positives per voxel",              "detr",  "false_positive_compartments_per_voxel", "num", -1),
    ("Missed compartments per voxel",          "detr",  "missed_compartments_per_voxel",         "num", -1),
    # --- resolved by compartment count ---
    ("Count accuracy, K = 1",                  "detr",  "count_accuracy_n1",                     "pct", +1),
    ("Count accuracy, K = 2",                  "detr",  "count_accuracy_n2",                     "pct", +1),
    ("Count accuracy, K = 3",                  "detr",  "count_accuracy_n3",                     "pct", +1),
    ("T1 relative error, median, K = 3",       "detr",  "n3_t1_rel_median",                      "pct", -1),
    ("T2 relative error, median, K = 3",       "detr",  "n3_t2_rel_median",                      "pct", -1),
    ("Existence recall, K = 3",                "detr",  "existence_recall_n3",                   "num", +1),
    ("Parameter-set error",                    "recov", "parameter_set_error",                   "num", -1),
    # --- training provenance, to tell "better model" from "trained longer" ---
    ("Best epoch",                             "run",   "best_epoch",                            "int",  0),
    ("Epochs run",                             "run",   "epochs_run",                            "int",  0),
    ("Best validation parameter loss",         "run",   "best_parameter_val_loss",               "loss", -1),
    ("Best validation total loss",             "run",   "best_total_val_loss",                   "loss",  0),
    ("Training wall time (s)",                 "run",   "wall_seconds",                          "sec",   0),
    ("Training steps",                         "run",   "total_steps",                           "int",  0),
    ("Existence threshold (validation)",       "thr",   "selected_threshold",                    "num",  0),
    ("Test voxels",                            "detr",  "n_voxels",                              "int",  0),
]

# Legacy names. eval._regression_block computes t1_mae_ms, t2_mae_ms and w_mae with _median(),
# so despite the names they are medians. Older summary.json files only have these; the fallback
# is announced in the report notes.
MEDIAN_ALIASES = {
    "t1_abs_median_ms": "t1_mae_ms",
    "t2_abs_median_ms": "t2_mae_ms",
    "w_abs_median": "w_mae",
}

# Per-weight-bin recovery block; the loss_uniform arm is judged on it.
WEIGHT_BIN_FIELDS = [
    ("detection rate", "match_rate", "pct", +1),
    ("median T1 rel. error", "t1_relative_error_median", "pct", -1),
    ("median T2 rel. error", "t2_relative_error_median", "pct", -1),
]

# Config fields allowed to differ without counting as a change: a new run needs a new name, and
# notes is free text.
IGNORED_CONFIG_FIELDS = {"name", "notes"}


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------
def flatten(d: dict, prefix: str = "") -> dict:
    """Flatten a nested config dict to `section.field` keys so two configs can be diffed.

    Lists are kept whole, not expanded per index: a data path list is one setting, and expanding
    it would report three differences for data_loguniform where there is one.
    """
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = tuple(v) if isinstance(v, list) else v
    return out


def load_run(run_dir: Path) -> dict:
    """Read one finished run (summary.json plus config.yaml), or raise with a readable message.

    A run counts as finished once summary.json exists; it is written last, after evaluation.
    The config goes through t1t2.config.load_config rather than raw YAML so that fields absent
    from an older YAML come back as dataclass defaults. Otherwise model.exist_head, which the
    baseline YAML omits, would diff against every arm that writes it explicitly.
    """
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{run_dir}: no summary.json. Either the run has not finished, or evaluation "
            f"failed after training. Check the SLURM log before treating this as a result."
        )
    with open(summary_path) as f:
        summary = json.load(f)

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"{run_dir}: no config.yaml, so its config cannot be diffed.")

    sys.path.insert(0, str(REPO_ROOT))
    from t1t2.config import load_config

    cfg = load_config(config_path)
    return {
        "name": run_dir.name,
        "dir": run_dir,
        "summary": summary,
        "config": flatten(asdict(cfg)),
    }


def metric_value(run: dict, source: str, key: str) -> tuple[float | None, str | None]:
    """Pull one metric. Returns (value, note); note is set when a fallback or a gap was hit."""
    s = run["summary"]
    if source == "detr":
        block = s.get("detr", {})
    elif source == "recov":
        block = s.get("detr", {}).get("parameter_recovery", {}).get("summary", {})
    elif source == "thr":
        block = s.get("threshold_calibration", {})
    else:
        block = s

    if key in block:
        v = block[key]
        return (None if v is None else float(v)), None

    alias = MEDIAN_ALIASES.get(key)
    if alias and alias in block:
        return float(block[alias]), (
            f"{run['name']}: no `{key}`; fell back to `{alias}`, which despite its name holds "
            f"a MEDIAN (eval.py::_regression_block uses _median())."
        )
    return None, f"{run['name']}: `{key}` missing from summary.json."


# --------------------------------------------------------------------------------------------
# Interpretability verdict: the one-change rule, checked mechanically
# --------------------------------------------------------------------------------------------
def config_diff(baseline: dict, arm: dict) -> list[tuple[str, object, object]]:
    """Every config field where the arm differs from the baseline, ignoring name and notes."""
    keys = (set(baseline["config"]) | set(arm["config"])) - IGNORED_CONFIG_FIELDS
    diffs = []
    for k in sorted(keys):
        b = baseline["config"].get(k, "<absent>")
        a = arm["config"].get(k, "<absent>")
        if b != a:
            diffs.append((k, b, a))
    return diffs


def dataset_differs(baseline: dict, arm: dict) -> bool:
    """True when the arm is scored on a different test set than the baseline.

    Two runs on different test sets are measured on different voxel populations; their delta
    mixes the change under test with the difference between the two draws.
    """
    return baseline["config"].get("data.test_path") != arm["config"].get("data.test_path")


def verdict(baseline: dict, arm: dict) -> dict:
    """Classify an arm by whether its delta is attributable to one named config change."""
    diffs = config_diff(baseline, arm)
    data_fields = [d for d in diffs if d[0].startswith("data.")]
    other = [d for d in diffs if not d[0].startswith("data.")]
    cross_dataset = dataset_differs(baseline, arm)

    # The four loss.signal_consistency* fields are one conceptual change (switch the physics
    # term on with these settings): the weight/target/warmup fields mean nothing without the
    # switch. They collapse to one diff only when the switch itself flipped; two physics arms
    # differing in, say, target only keep that as a single change of its own.
    _SC_PREFIX = "loss.signal_consistency"
    sc_fields = [d for d in diffs if d[0].startswith(_SC_PREFIX)]
    sc_switch_flipped = any(d[0] == "loss.signal_consistency" for d in sc_fields)
    effective_n = len(diffs) - (len(sc_fields) - 1 if sc_switch_flipped and sc_fields else 0)

    if arm["name"] == baseline["name"]:
        tag, note = "REFERENCE", "the reference run itself"
    elif not diffs:
        tag = "CONTROL"
        note = ("identical config under a new name; this is the reproduction control, and its "
                "delta against the baseline IS the run-to-run noise floor")
    elif cross_dataset:
        # All data.* path edits are one conceptual change (use the other dataset family), so a
        # three-path swap is not three changes. Anything on top of it is.
        extra = len(other)
        tag = "DIFFERENT DATASET" if extra == 0 else "DIFFERENT DATASET + NOT INTERPRETABLE"
        note = ("scored on a different test set, so this is not a controlled comparison: the "
                "delta mixes the change under test with the different voxel draw")
        if extra:
            note += f"; and {extra} further config field(s) also differ"
    elif len(diffs) == 1:
        k, b, a = diffs[0]
        tag = "single change"
        note = f"`{k}`: {b!r} -> {a!r}"
    elif effective_n == 1 and sc_switch_flipped:
        tag = "single change"
        note = ("physics term switched on as one conceptual change: "
                + "; ".join(f"`{k}`: {b!r} -> {a!r}" for k, b, a in sc_fields))
    else:
        tag = "NOT INTERPRETABLE"
        note = (f"{len(diffs)} config fields differ, so no observed difference can be "
                f"attributed to any one of them")

    return {
        "run": arm["name"],
        "verdict": tag,
        "n_config_diffs": len(diffs),
        "differing_fields": "; ".join(k for k, _, _ in diffs) or "-",
        "detail": note,
        "cross_dataset": cross_dataset,
        "diffs": diffs,
        "data_fields": len(data_fields),
    }


# --------------------------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------------------------
def fmt(value: float | None, kind: str) -> str:
    """Format a metric with the right units and precision."""
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if kind == "pct":
        return f"{100.0 * value:.2f} %"
    if kind == "ms":
        return f"{value:.2f}"
    if kind == "loss":
        return f"{value:.6f}"
    if kind == "sec":
        return f"{value:.1f}"
    if kind == "int":
        return f"{int(round(value)):,}"
    return f"{value:.4f}"


def fmt_delta(base: float | None, value: float | None, kind: str, direction: int) -> str:
    """Signed delta in the metric's own units, with a better/worse marker.

    Percentage metrics get percentage points, not a relative change: 78.54 % -> 79.54 % is
    "+1.00 pp", not "+1.3 %".
    """
    if base is None or value is None:
        return "-"
    if isinstance(base, float) and math.isnan(base):
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    d = value - base
    if kind == "pct":
        body = f"{100.0 * d:+.2f} pp"
    elif kind == "ms":
        body = f"{d:+.2f}"
    elif kind == "loss":
        body = f"{d:+.6f}"
    elif kind == "sec":
        body = f"{d:+.1f}"
    elif kind == "int":
        body = f"{int(round(d)):+,}"
    else:
        body = f"{d:+.4f}"
    if direction == 0 or d == 0.0:
        return body
    return f"{body} ({'better' if d * direction > 0 else 'worse'})"


# --------------------------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------------------------
def build_metric_rows(runs: list[dict], baseline: dict) -> tuple[list[list[str]], list[str], list[str]]:
    """Build the metric table as a list of string rows, plus the header and any notes."""
    notes: list[str] = []
    header = ["Metric", f"{baseline['name']}  (reference)"]
    for r in runs[1:]:
        header += [r["name"], "Δ"]

    rows: list[list[str]] = []
    for label, source, key, kind, direction in METRICS:
        base_v, note = metric_value(baseline, source, key)
        if note:
            notes.append(note)
        row = [label, fmt(base_v, kind)]
        for r in runs[1:]:
            v, note = metric_value(r, source, key)
            if note:
                notes.append(note)
            row += [fmt(v, kind), fmt_delta(base_v, v, kind, direction)]
        rows.append(row)
    return rows, header, notes


def weight_bin_block(runs: list[dict], baseline: dict) -> tuple[list[list[str]], list[str]]:
    """Recovery resolved by true compartment weight, the block the loss_uniform arm is judged on."""
    def bins_of(run):
        """Look up recovery results by signal-fraction group."""
        b = run["summary"].get("detr", {}).get("parameter_recovery", {}).get("bins", [])
        return {x["label"]: x for x in b}

    base_bins = bins_of(baseline)
    if not base_bins:
        return [], []
    header = ["Weight bin", "quantity", f"{baseline['name']}"]
    for r in runs[1:]:
        header += [r["name"], "Δ"]

    rows = []
    for label in base_bins:
        for qname, field, kind, direction in WEIGHT_BIN_FIELDS:
            bv = base_bins[label].get(field)
            row = [label, qname, fmt(bv, kind)]
            for r in runs[1:]:
                rb = bins_of(r).get(label, {})
                v = rb.get(field)
                row += [fmt(v, kind), fmt_delta(bv, v, kind, direction)]
            rows.append(row)
    return rows, header


def markdown_table(header: list[str], rows: list[list[str]], align_first_left=True) -> str:
    """Arrange headers and rows into a readable Markdown table."""
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]
    def line(cells):
        """Format one table row with evenly padded columns."""
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    sep = "|" + "|".join(
        ("-" * (w + 2)) if (i == 0 and align_first_left) else ("-" * (w + 1) + ":")
        for i, w in enumerate(widths)
    ) + "|"
    return "\n".join([line(header), sep] + [line(r) for r in rows])


FOOTER = """
---

## How to read this table

**One change per arm.** Every arm in `docs/experiments.md` makes exactly one change relative to
the reference run. Most are literally one config field; a few touch a small group of fields that
together switch one feature on, and those are collapsed here into a single diff. The arm table
above states, per arm, whether that holds. An arm marked `NOT INTERPRETABLE` has more than one
genuine difference. Its numbers are still printed, because leaving them out would be its own
distortion, but no difference it shows can be attributed to a single cause.

**Percentage points, not percent.** Deltas on percentage metrics are in percentage points. A
move from 78.54 % to 79.54 % is `+1.00 pp`. Writing it as "+1.3 %" is a different quantity and
reads as a larger effect than it is.

**Medians are labelled as medians.** `metrics_detr.json` has fields named `t1_mae_ms`,
`t2_mae_ms` and `w_mae` that `eval._regression_block` computes with `_median()`. This table
reads the unambiguous aliases instead and labels every such row "absolute error, median". The
genuine means are reported as separate rows: on the reference run the median T1 absolute error
is 29.60 ms against a mean of 122.50 ms, a factor of 4.1. That gap is a result in its own
right, since it says the error is concentrated in a minority of hard voxels.

**One seed per arm, so there are no error bars here.** Every run in this table is seed
20260724, run once, which gives a point estimate and no interval. The measured spread across
repeated training of the same config is reported in `docs/experiments.md`; a delta smaller than
that spread should be reported as within run-to-run variation rather than as an effect.

**A different test set is a different question.** An arm marked `DIFFERENT DATASET` is scored on
another voxel family, so its delta mixes the change under test with the difference between two
independent draws. `data_loguniform` is in that position by design: its base seed is 3500501
against the reference's 3500500, because reusing the seed would have given the two families
bit-identical noise realisations and coupled the comparison.

**Three arms carry confounders that no config could remove.** `queries_4` and `queries_6` change
the parameter count and the existence class balance along with the query budget. At
`n_queries = 4` with three compartments the existence `pos_weight` reaches its clamp floor of
0.50, where the term stops up-weighting positives and begins down-weighting them.
`exist_head_shared` changes the parameter count by 2.60 %. Each of these belongs in the same
sentence as its result. The numbers are in `docs/experiments.md`.
"""


def write_report(runs: list[dict], out_dir: Path, notes_extra: list[str]) -> dict:
    """Save the experiment comparison as Markdown, CSV, and JSON."""
    baseline = runs[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, header, notes = build_metric_rows(runs, baseline)
    verdicts = [verdict(baseline, r) for r in runs]
    wb_rows, wb_header = weight_bin_block(runs, baseline)

    # --- markdown ---
    parts = [
        "# Experiment comparison against the reference run",
        "",
        f"**Reference:** `{baseline['name']}`, see `docs/experiments.md`. "
        f"Never retrained, never re-evaluated.",
        f"**Runs compared:** {len(runs)} "
        f"({len(runs) - 1} arm{'s' if len(runs) != 2 else ''} besides the reference)",
        "",
        "## 1. Metrics",
        "",
        "Metrics are rows; runs are columns. The first column is the reference run. "
        "Δ is the arm minus the baseline, in the metric's own units (percentage points for "
        "percentages).",
        "",
        markdown_table(header, rows),
        "",
        "## 2. Is each arm interpretable?",
        "",
        "One row per run. `single change` is the only verdict under which a delta can be "
        "attributed to a named cause.",
        "",
        markdown_table(
            ["Run", "Verdict", "# config diffs", "Differing fields", "Detail"],
            [[v["run"], v["verdict"], str(v["n_config_diffs"]),
              v["differing_fields"], v["detail"]] for v in verdicts],
        ),
    ]

    if wb_rows:
        parts += [
            "",
            "## 3. Recovery by true compartment weight",
            "",
            "The main finding sits in this block rather than in the aggregates: recovery "
            "degrades steadily as the true compartment weight falls. Whether that gradient is "
            "the information limit of a 64-measurement protocol or an artefact of "
            "`signal_fraction` loss weighting is what the `loss_uniform` arm was run to "
            "separate; see `docs/experiments.md`.",
            "",
            markdown_table(wb_header, wb_rows),
        ]

    all_notes = notes + notes_extra
    if all_notes:
        seen, uniq = set(), []
        for n in all_notes:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        parts += ["", "## Data notes", ""] + [f"- {n}" for n in uniq]

    parts += [FOOTER]
    md_path = out_dir / "comparison.md"
    md_path.write_text("\n".join(parts) + "\n")

    # --- CSVs ---
    metrics_csv = out_dir / "comparison_metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
        if wb_rows:
            w.writerow([])
            w.writerow(wb_header)
            w.writerows(wb_rows)

    arms_csv = out_dir / "comparison_arms.csv"
    with open(arms_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "verdict", "n_config_diffs", "differing_fields",
                    "cross_dataset", "detail"])
        for v in verdicts:
            w.writerow([v["run"], v["verdict"], v["n_config_diffs"],
                        v["differing_fields"], int(v["cross_dataset"]), v["detail"]])

    return {"markdown": md_path, "metrics_csv": metrics_csv, "arms_csv": arms_csv,
            "verdicts": verdicts, "n_metric_rows": len(rows), "header": header}


# --------------------------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------------------------
def discover_runs(results_dir: Path) -> list[str]:
    """Every immediate subdirectory of results/ that holds a summary.json.

    Immediate only: superseded runs live under results/_historical_runs/ and used different data
    families and a different loss reduction, so they are not comparable with the current matrix.
    """
    return sorted(p.parent.name for p in results_dir.glob("*/summary.json"))


def main(argv=None) -> int:
    """Compare the requested runs with the reference experiment."""
    ap = argparse.ArgumentParser(
        description="Compare finished experiment arms against the reference run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The reference run is always included as the first column.",
    )
    ap.add_argument("runs", nargs="*",
                    help="run directory names under results/ (e.g. loss_uniform queries_6)")
    ap.add_argument("--all", action="store_true",
                    help="compare every results/*/summary.json found")
    ap.add_argument("--results-dir", default=None,
                    help="default: <repo>/results")
    ap.add_argument("--out-dir", default=None,
                    help="default: <results-dir>/_comparison")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    log = (lambda *a, **k: None) if args.quiet else print
    results_dir = Path(args.results_dir) if args.results_dir else REPO_ROOT / "results"
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "_comparison"

    if not results_dir.exists():
        print(f"error: no results directory at {results_dir}", file=sys.stderr)
        return 2

    names = list(args.runs)
    if args.all:
        names = discover_runs(results_dir)
        log(f"[compare] --all found {len(names)} finished run(s): {', '.join(names)}")
    if not names and not args.all:
        print("error: name at least one run, or pass --all.", file=sys.stderr)
        return 2

    # Reference column first, exactly once.
    names = [BASELINE_RUN] + [n for n in names if n != BASELINE_RUN]

    notes_extra: list[str] = []
    runs = []
    for n in names:
        try:
            runs.append(load_run(results_dir / n))
        except FileNotFoundError as e:
            if n == BASELINE_RUN:
                print(f"error: the reference run is required as the first column but "
                      f"could not be read.\n  {e}", file=sys.stderr)
                return 2
            log(f"[compare] skipping {n}: {e}")
            notes_extra.append(f"`{n}` was requested but could not be read: {e}")

    if len(runs) == 1:
        log("[compare] only the reference run is present. The table below is that one "
            "column alone, which is correct and not yet a comparison.")

    result = write_report(runs, out_dir, notes_extra)

    if not args.quiet:
        print()
        print(result["markdown"].read_text())
        print(f"[compare] wrote {result['markdown']}")
        print(f"[compare] wrote {result['metrics_csv']}")
        print(f"[compare] wrote {result['arms_csv']}")

    bad = [v["run"] for v in result["verdicts"] if "NOT INTERPRETABLE" in v["verdict"]]
    if bad:
        print(f"[compare] WARNING: not interpretable as single-change ablations: "
              f"{', '.join(bad)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

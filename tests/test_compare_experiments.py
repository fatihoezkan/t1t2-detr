"""Tests for evaluation/compare_experiments.py.

Covers the reference-first column, the single change / NOT INTERPRETABLE / DIFFERENT DATASET
verdicts, delta units, and the labelling of the median-named fields (no "MAE" beside a median).
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT))

import compare_experiments as ce  # noqa: E402

BASELINE_DIR = REPO_ROOT / "results" / ce.BASELINE_RUN


# ------------------------------------------------------------------------------ synthetic runs
def _minimal_config(name: str, data_family: str = "t1_3500_t2_500_100k", **overrides) -> dict:
    """A config with the same shape as a real one. Only the fields the script diffs matter."""
    cfg = {
        "name": name,
        "data": {
            "train_path": [f"data/{data_family}/n{i}/train.parquet" for i in (1, 2, 3)],
            "val_path": [f"data/{data_family}/n{i}/val.parquet" for i in (1, 2, 3)],
            "test_path": [f"data/{data_family}/n{i}/test.parquet" for i in (1, 2, 3)],
            "n_inputs": 64,
            "normalization": "log_minmax",
            "t1_min": 50.0, "t1_max": 3500.0, "t2_min": 5.0, "t2_max": 500.0,
            "signal_norm": "max",
        },
        "model": {"input_dim": 64, "hidden_dim": 512, "fs_dim": 256, "n_queries": 10,
                  "n_dlayers": 4, "n_heads": 4, "aux_loss": False},
        "loss": {"t1_weight": 1.0, "t2_weight": 1.0, "w_weight": 1.0, "exist_weight": 0.1,
                 "aux_weight": 1.0, "t1_t2_weighting": "signal_fraction"},
        "train": {"epochs": 500, "batch_size": 512, "lr": 1e-4, "seed": 20260724},
        "evaluation": {"calibrate_threshold": True, "threshold_objective": "parameter_set_error"},
        "notes": "synthetic test fixture",
    }
    for dotted, value in overrides.items():
        section, field = dotted.split(".", 1)
        cfg[section][field] = value
    return cfg


def _minimal_summary(name: str, count_accuracy=0.80, t1_rel=0.050) -> dict:
    """Build a small experiment summary for comparison tests."""
    return {
        "name": name,
        "epochs_run": 160, "epoch_budget": 500, "best_epoch": 120,
        "best_val": 0.008, "selection_metric": "parameter_loss",
        "best_total_val_loss": 0.0230, "best_parameter_val_loss": 0.0080,
        "early_stopped": True, "total_steps": 31000, "wall_seconds": 1500.0,
        "threshold_calibration": {"objective": "parameter_set_error",
                                  "selected_threshold": 0.70,
                                  "selection_split": "validation"},
        "detr": {
            "n_voxels": 9999,
            "count_accuracy": count_accuracy, "count_mae": 0.21,
            "existence_precision": 0.96, "existence_recall": 0.93, "existence_f1": 0.945,
            "false_positive_compartments_per_voxel": 0.07,
            "missed_compartments_per_voxel": 0.14,
            "t1_rel_median": t1_rel, "t2_rel_median": 0.065,
            "t1_abs_median_ms": 28.0, "t2_abs_median_ms": 2.5, "w_abs_median": 0.026,
            "t1_abs_mean_ms": 120.0, "t2_abs_mean_ms": 12.0, "w_abs_mean": 0.068,
            "count_accuracy_n1": 0.95, "count_accuracy_n2": 0.76, "count_accuracy_n3": 0.69,
            "n3_t1_rel_median": 0.118, "n3_t2_rel_median": 0.135,
            "existence_recall_n3": 0.89,
            "parameter_recovery": {
                "summary": {"recovered_signal_fraction": 0.976,
                            "parameter_set_error": 0.120},
                "bins": [{"label": "0.05–0.10", "match_rate": 0.70,
                          "t1_relative_error_median": 0.30,
                          "t2_relative_error_median": 0.33},
                         {"label": "0.75–1.00", "match_rate": 0.9996,
                          "t1_relative_error_median": 0.018,
                          "t2_relative_error_median": 0.023}],
            },
        },
    }


def _write_run(results_dir: Path, name: str, config: dict, summary: dict) -> Path:
    """Save a test run's configuration and summary."""
    d = results_dir / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (d / "summary.json").write_text(json.dumps(summary))
    return d


@pytest.fixture
def fake_results(tmp_path):
    """A synthetic results/ with a stand-in baseline plus four arms.

    The stand-in is written under the real baseline's name because the reference identity is
    hard-coded in the script.
    """
    results = tmp_path / "results"
    results.mkdir()
    _write_run(results, ce.BASELINE_RUN,
               _minimal_config(ce.BASELINE_RUN), _minimal_summary(ce.BASELINE_RUN))
    # one field differs -> interpretable
    _write_run(results, "loss_uniform",
               _minimal_config("loss_uniform", **{"loss.t1_t2_weighting": "uniform"}),
               _minimal_summary("loss_uniform", count_accuracy=0.79, t1_rel=0.044))
    # two fields differ -> not interpretable
    _write_run(results, "two_changes",
               _minimal_config("two_changes", **{"model.n_queries": 6, "model.n_dlayers": 2}),
               _minimal_summary("two_changes", count_accuracy=0.77))
    # different dataset family -> not a controlled comparison
    _write_run(results, "data_loguniform",
               _minimal_config("data_loguniform", data_family="t1_loguniform_100k"),
               _minimal_summary("data_loguniform", count_accuracy=0.785))
    # identical config under a different name -> the reproduction control. The name must differ
    # from ce.BASELINE_RUN, which is already written above.
    _write_run(results, "baseline_rerun",
               _minimal_config("baseline_rerun"),
               _minimal_summary("baseline_rerun", count_accuracy=0.784))
    return results


# ------------------------------------------------------------ against the real results/ tree
def test_runs_with_only_the_baseline_present(tmp_path):
    """--all on a results/ that holds only the real reference run gives a one-column table.

    Copies the reference run's files into tmp_path so the test does not depend on which other
    runs happen to be checked out, and checks the headline cells against the run's own summary
    so the table is wired to the right fields.
    """
    if not (BASELINE_DIR / "summary.json").exists():
        pytest.skip("reference run results not present in this checkout")
    results = tmp_path / "results"
    run_dir = results / ce.BASELINE_RUN
    run_dir.mkdir(parents=True)
    for name in ("config.yaml", "summary.json"):
        shutil.copy(BASELINE_DIR / name, run_dir / name)
    out = tmp_path / "real_baseline_out"
    rc = ce.main(["--all", "--quiet", "--results-dir", str(results), "--out-dir", str(out)])
    assert rc == 0
    md = (out / "comparison.md").read_text()
    assert ce.BASELINE_RUN in md
    assert "(0 arms besides the reference)" in md
    detr = json.loads((BASELINE_DIR / "summary.json").read_text())["detr"]
    assert ce.fmt(detr["count_accuracy"], "pct") in md
    assert ce.fmt(detr["t1_rel_median"], "pct") in md
    assert ce.fmt(detr["t1_abs_mean_ms"], "ms") in md
    for f in ("comparison.md", "comparison_metrics.csv", "comparison_arms.csv"):
        assert (out / f).exists()


def test_baseline_is_forced_into_the_comparison(fake_results, tmp_path):
    """Asking for one arm still yields the baseline as the first column."""
    out = tmp_path / "out"
    rc = ce.main(["loss_uniform", "--quiet",
                  "--results-dir", str(fake_results), "--out-dir", str(out)])
    assert rc == 0
    header = (out / "comparison_metrics.csv").read_text().splitlines()[0]
    assert header.split(",")[1].startswith(ce.BASELINE_RUN)


# ------------------------------------------------------------------------- the verdict logic
def test_single_change_arm_is_interpretable(fake_results, tmp_path):
    """Check that a single config change is reported as interpretable."""
    out = tmp_path / "out"
    ce.main(["loss_uniform", "--quiet",
             "--results-dir", str(fake_results), "--out-dir", str(out)])
    rows = list(csv.DictReader((out / "comparison_arms.csv").open()))
    arm = next(r for r in rows if r["run"] == "loss_uniform")
    assert arm["verdict"] == "single change"
    assert arm["n_config_diffs"] == "1"
    assert arm["differing_fields"] == "loss.t1_t2_weighting"
    assert "signal_fraction" in arm["detail"] and "uniform" in arm["detail"]


def test_two_changes_is_flagged_not_interpretable(fake_results, tmp_path):
    """Two differing fields must be named as uninterpretable, not silently tabulated."""
    out = tmp_path / "out"
    ce.main(["two_changes", "--quiet",
             "--results-dir", str(fake_results), "--out-dir", str(out)])
    rows = list(csv.DictReader((out / "comparison_arms.csv").open()))
    arm = next(r for r in rows if r["run"] == "two_changes")
    assert arm["verdict"] == "NOT INTERPRETABLE"
    assert arm["n_config_diffs"] == "2"
    assert "model.n_dlayers" in arm["differing_fields"]
    assert "model.n_queries" in arm["differing_fields"]
    md = (out / "comparison.md").read_text()
    assert "NOT INTERPRETABLE" in md
    # its numbers are still tabulated
    assert "two_changes" in md.split("## 1. Metrics")[1].split("## 2.")[0]


def test_different_dataset_arm_is_labelled(fake_results, tmp_path):
    """A cross-dataset arm is one conceptual change, but not a controlled comparison."""
    out = tmp_path / "out"
    ce.main(["data_loguniform", "--quiet",
             "--results-dir", str(fake_results), "--out-dir", str(out)])
    rows = list(csv.DictReader((out / "comparison_arms.csv").open()))
    arm = next(r for r in rows if r["run"] == "data_loguniform")
    assert arm["verdict"] == "DIFFERENT DATASET"
    assert arm["cross_dataset"] == "1"
    # three data.* paths changed, but the verdict must not read as "three changes"
    assert "NOT INTERPRETABLE" not in arm["verdict"]
    assert "not a controlled comparison" in arm["detail"]


def test_identical_config_is_the_control(fake_results, tmp_path):
    """Check that an unchanged config is labeled as a control."""
    out = tmp_path / "out"
    ce.main(["baseline_rerun", "--quiet",
             "--results-dir", str(fake_results), "--out-dir", str(out)])
    rows = list(csv.DictReader((out / "comparison_arms.csv").open()))
    arm = next(r for r in rows if r["run"] == "baseline_rerun")
    assert arm["verdict"] == "CONTROL"
    assert arm["n_config_diffs"] == "0"
    assert "noise floor" in arm["detail"]


def test_cross_dataset_plus_extra_change_is_both(fake_results, tmp_path):
    """A dataset swap AND another edit is flagged on both counts."""
    _write_run(fake_results, "data_and_queries",
               _minimal_config("data_and_queries", data_family="t1_loguniform_100k",
                               **{"model.n_queries": 6}),
               _minimal_summary("data_and_queries"))
    out = tmp_path / "out"
    ce.main(["data_and_queries", "--quiet",
             "--results-dir", str(fake_results), "--out-dir", str(out)])
    rows = list(csv.DictReader((out / "comparison_arms.csv").open()))
    arm = next(r for r in rows if r["run"] == "data_and_queries")
    assert "DIFFERENT DATASET" in arm["verdict"]
    assert "NOT INTERPRETABLE" in arm["verdict"]


# --------------------------------------------------------------------------- the naming trap
def test_median_fields_are_labelled_as_medians(fake_results, tmp_path):
    """The label "MAE" must not appear on a median field."""
    out = tmp_path / "out"
    ce.main(["--all", "--quiet",
             "--results-dir", str(fake_results), "--out-dir", str(out)])
    md = (out / "comparison.md").read_text()
    metric_labels = [line.split("|")[1].strip()
                     for line in md.splitlines()
                     if line.startswith("| ") and "|" in line[2:]]
    for axis in ("T1", "T2"):
        assert f"{axis} absolute error, median (ms)" in metric_labels
        assert f"{axis} absolute error, mean (ms)" in metric_labels
    # The only permitted "MAE" is the genuine count MAE, which really is a mean.
    maes = [l for l in metric_labels if "MAE" in l]
    assert maes == ["Count MAE (compartments)"], maes


def test_legacy_median_alias_is_used_and_announced(fake_results, tmp_path):
    """An older summary.json with only `t1_mae_ms` still reads, with a note that it is a median."""
    s = _minimal_summary("legacy_names")
    for new, old in ce.MEDIAN_ALIASES.items():
        s["detr"][old] = s["detr"].pop(new)
    _write_run(fake_results, "legacy_names", _minimal_config("legacy_names",
                                                            **{"model.n_dlayers": 6}), s)
    out = tmp_path / "out"
    ce.main(["legacy_names", "--quiet",
             "--results-dir", str(fake_results), "--out-dir", str(out)])
    md = (out / "comparison.md").read_text()
    assert "fell back to `t1_mae_ms`" in md
    assert "MEDIAN" in md


# ------------------------------------------------------------------------------- delta units
def test_percentage_deltas_are_percentage_points(fake_results, tmp_path):
    """0.80 -> 0.79 must read -1.00 pp, not -1.25 %."""
    out = tmp_path / "out"
    ce.main(["loss_uniform", "--quiet",
             "--results-dir", str(fake_results), "--out-dir", str(out)])
    rows = list(csv.reader((out / "comparison_metrics.csv").open()))
    row = next(r for r in rows if r and r[0] == "Compartment-count accuracy")
    assert row[1] == "80.00 %" and row[2] == "79.00 %"
    assert "pp" in row[3] and row[3].startswith("-1.00 pp")
    assert "worse" in row[3]


def test_lower_is_better_direction(fake_results, tmp_path):
    """A drop in median T1 relative error is annotated 'better', not 'worse'."""
    out = tmp_path / "out"
    ce.main(["loss_uniform", "--quiet",
             "--results-dir", str(fake_results), "--out-dir", str(out)])
    rows = list(csv.reader((out / "comparison_metrics.csv").open()))
    row = next(r for r in rows if r and r[0] == "T1 relative error, median")
    assert row[3].startswith("-0.60 pp") and "better" in row[3]


# ----------------------------------------------------------------------------- failure modes
def test_unfinished_arm_is_skipped_with_a_note(fake_results, tmp_path):
    """A directory with checkpoints but no summary.json is not a result."""
    (fake_results / "still_training").mkdir()
    out = tmp_path / "out"
    rc = ce.main(["still_training", "--quiet",
                  "--results-dir", str(fake_results), "--out-dir", str(out)])
    assert rc == 0
    md = (out / "comparison.md").read_text()
    assert "still_training" in md and "no summary.json" in md


def test_missing_baseline_is_a_hard_error(tmp_path):
    """Without the reference column there is nothing to compare against, so refuse."""
    empty = tmp_path / "results"
    empty.mkdir()
    assert ce.main(["--all", "--quiet", "--results-dir", str(empty),
                    "--out-dir", str(tmp_path / "o")]) == 2
    # Naming an arm explicitly does not help either: the reference column is mandatory.
    assert ce.main(["some_arm", "--quiet", "--results-dir", str(empty),
                    "--out-dir", str(tmp_path / "o2")]) == 2


def test_discover_runs_ignores_nested_historical_runs(fake_results):
    """results/_historical_runs/* predates the freeze and is not comparable."""
    hist = fake_results / "_historical_runs" / "old_run"
    hist.mkdir(parents=True)
    (hist / "summary.json").write_text("{}")
    found = ce.discover_runs(fake_results)
    assert "old_run" not in found
    assert ce.BASELINE_RUN in found


def test_footer_states_the_single_seed_limitation(fake_results, tmp_path):
    """Check that the report explains the limits of using one seed."""
    out = tmp_path / "out"
    ce.main(["--all", "--quiet",
             "--results-dir", str(fake_results), "--out-dir", str(out)])
    md = (out / "comparison.md").read_text()
    assert "One seed per arm" in md
    assert "run-to-run variation" in md

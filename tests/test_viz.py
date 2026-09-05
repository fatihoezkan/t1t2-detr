"""Tests for the figure module (t1t2.viz).

Each figure function must run on synthetic input, write a file, and pass its own text-overlap
check; label collisions are the failure mode of these figures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from t1t2.data import TargetNormalizer
from t1t2.viz import (
    apply_style,
    existence_distribution_figure,
    existence_metric_markdown,
    existence_metric_table,
    grouping_calibration_figure,
    prediction_stages_figure,
    query_histograms_figure,
    query_scatter_figure,
)


@pytest.fixture(scope="module", autouse=True)
def _style():
    apply_style()


@pytest.fixture
def fake():
    """A small synthetic run: 120 voxels, 10 queries, 1 to 3 compartments.

    Queries 0, 3 and 8 get high scores and clustered outputs (the duplicate-peak case), the
    rest low scores and diffuse outputs.
    """
    rng = np.random.default_rng(7)
    n, q = 120, 10
    normalizer = TargetNormalizer("log_minmax", 50.0, 3500.0, 5.0, 500.0)

    params_norm = rng.uniform(0.05, 0.95, size=(n, q, 3))
    # Queries 0, 3, 8 are the "active" ones and sit close together, mimicking redundancy.
    params_norm[:, [0, 3, 8], :2] = (params_norm[:, [0], :2]
                                     + rng.normal(0, 0.01, size=(n, 3, 2)))
    params_norm = np.clip(params_norm, 0.01, 0.99)
    score = rng.uniform(0.0, 0.2, size=(n, q))
    score[:, [0, 3, 8]] = rng.uniform(0.75, 0.999, size=(n, 3))

    physical = params_norm.copy()
    physical[..., 0] = normalizer.denormalize_t1(params_norm[..., 0])
    physical[..., 1] = normalizer.denormalize_t2(params_norm[..., 1])
    tbl = {"params_norm": params_norm, "params": physical, "exist_prob": score,
           "exist_logit": np.log(score / (1 - score))}

    n_comp = rng.integers(1, 4, size=n)
    trues = []
    for i in range(n):
        k = int(n_comp[i])
        w = rng.dirichlet(np.ones(k))
        trues.append([(float(normalizer.denormalize_t1(rng.uniform(0.1, 0.9))),
                       float(normalizer.denormalize_t2(rng.uniform(0.1, 0.9))),
                       float(w[j])) for j in range(k)])

    label = np.zeros((n, q), dtype=np.int8)
    for i in range(n):
        label[i, rng.choice(q, size=int(n_comp[i]), replace=False)] = 1
    diag = {"label": label, "cost": rng.exponential(0.05, size=(n, q)),
            "cost_matched": np.where(label == 1, rng.exponential(0.05, size=(n, q)), np.nan),
            "target_idx": np.where(label == 1, 0, -1).astype(np.int16)}
    return {"tbl": tbl, "trues": trues, "n_comp": n_comp, "diag": diag,
            "normalizer": normalizer}


def test_prediction_stages_figure_writes_and_has_no_overlapping_text(fake, tmp_path):
    """The four-panel figure; ten query labels in panel b) is the hard case.

    If this fails the greedy label placer has run out of candidate offsets: widen the figure or
    add offsets rather than dropping the assertion.
    """
    path, findings = prediction_stages_figure(
        3, fake["tbl"], fake["trues"][3], 0.69, 0.05, fake["normalizer"],
        tmp_path / "a.png",
    )
    assert (tmp_path / "a.png").exists()
    assert findings == [], findings


def test_prediction_stages_figure_handles_a_voxel_with_nothing_above_threshold(fake, tmp_path):
    """Panels c) and d) render as empty boxes, not raise; real runs contain such voxels."""
    tbl = {k: v.copy() for k, v in fake["tbl"].items()}
    tbl["exist_prob"][:] = 0.01
    path, findings = prediction_stages_figure(
        0, tbl, fake["trues"][0], 0.69, 0.05, fake["normalizer"], tmp_path / "empty.png",
    )
    assert (tmp_path / "empty.png").exists()
    assert findings == [], findings


def test_query_histograms_figure(fake, tmp_path):
    _, findings = query_histograms_figure(fake["tbl"], tmp_path / "c1.png", threshold=0.69)
    assert (tmp_path / "c1.png").exists()
    assert findings == [], findings


def test_query_scatter_figure_subsamples_without_error_when_asked_for_too_many(fake, tmp_path):
    """n_samples above the split size must clamp, not raise (rng.choice replace=False)."""
    _, findings = query_scatter_figure(fake["tbl"], tmp_path / "c2.png", n_samples=10_000,
                                       threshold=0.69)
    assert (tmp_path / "c2.png").exists()
    assert findings == [], findings


def test_existence_distribution_figure(fake, tmp_path):
    _, findings = existence_distribution_figure(
        fake["diag"], fake["tbl"]["exist_prob"], fake["n_comp"], tmp_path / "d.png",
        threshold=0.69,
    )
    assert (tmp_path / "d.png").exists()
    assert findings == [], findings


def test_grouping_calibration_figure(fake, tmp_path):
    """Uses a hand-built calibration dict so the panel is tested, not the sweep."""
    curve = [{"radius": r, "aggregate": a, "include_weight": False, "renormalize": True,
              "parameter_set_error": 0.12 + 0.05 * r, "parameter_set_accuracy": 0.88,
              "count_accuracy": 0.78, "recovered_signal_fraction": 0.97,
              "weight_set_l1_error_mean": 0.15, "mean_predicted_count": 1.93,
              "t1_fraction_weighted_relative_error_matched": 0.08,
              "t2_fraction_weighted_relative_error_matched": 0.09,
              "n_extra_predictions": 800, "matched_true_compartment_rate": 0.93}
             for a in ("mean", "weight", "confidence")
             for r in (0.0, 0.05, 0.10, 0.20)]
    calibration = {"objective": "parameter_set_error", "curve": curve,
                   "selected": curve[0], "n_evaluations": len(curve)}
    stage = pd.DataFrame({"threshold_only": [0.94, 0.74, 0.67, 0.79],
                          "grouped": [0.94, 0.75, 0.67, 0.79]},
                         index=["K = 1", "K = 2", "K = 3", "all"])
    _, findings = grouping_calibration_figure(calibration, stage, tmp_path / "b.png", 0.69)
    assert (tmp_path / "b.png").exists()
    assert findings == [], findings


def test_existence_metric_table_has_a_row_per_count_plus_pooled(fake):
    df = existence_metric_table(fake["diag"], fake["tbl"]["exist_prob"], fake["n_comp"], 0.69)
    assert list(df["K"]) == [1, 2, 3, "pooled"]
    assert df["n_voxels"].iloc[-1] == len(fake["n_comp"])
    # The pooled voxel count must be the sum of the per-count rows, or a K is being dropped.
    assert df["n_voxels"].iloc[:-1].sum() == df["n_voxels"].iloc[-1]


def test_existence_metric_markdown_renders_the_reference_row(fake):
    df = existence_metric_table(fake["diag"], fake["tbl"]["exist_prob"], fake["n_comp"], 0.69)
    md = existence_metric_markdown(df, 0.69, reference={"K": "prior thesis", "acc": 0.65})
    assert "prior thesis" in md
    assert md.count("\n|") >= len(df) + 2          # header + rule + rows + reference

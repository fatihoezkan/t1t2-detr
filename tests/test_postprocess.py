"""Tests for the query-grouping postprocessing (t1t2.postprocess).

Grouping sets the reported compartment count, so a bug here moves the headline count accuracy.
Covers empty voxels, fully collapsed voxels, and the radius-0 control.
"""
from __future__ import annotations

import numpy as np
import pytest

from t1t2.data import TargetNormalizer
from t1t2.eval import predictions_from_query_outputs
from t1t2.postprocess import (
    AGGREGATIONS,
    existence_score_metrics,
    group_peaks,
    grouped_predictions,
    peak_distances,
)


# --------------------------------------------------------------------------------------
# group_peaks: the single-voxel merge
# --------------------------------------------------------------------------------------

def test_no_peaks_returns_empty():
    """Nothing above the existence threshold gives an empty list (count 0), not an error."""
    assert group_peaks(np.empty((0, 3)), np.empty(0), radius=0.1) == []


def test_single_peak_is_returned_unchanged():
    """One peak survives untouched.

    It is returned without weight renormalization even though renormalize defaults to True;
    otherwise a lone peak would be rewritten to weight 1.0.
    """
    out = group_peaks(np.array([[0.4, 0.6, 0.3]]), np.array([0.9]), radius=0.1)
    assert out == [(0.4, 0.6, 0.3)]


def test_radius_zero_is_an_exact_identity():
    """Radius 0 must not merge, average, or renormalize.

    It is the control arm of every grouping comparison; a quiet renormalization here would put
    a calibration step into the measured effect of grouping.
    """
    p = np.array([[0.30, 0.50, 0.4], [0.3001, 0.5001, 0.4], [0.80, 0.20, 0.1]])
    out = group_peaks(p, np.array([0.9, 0.8, 0.7]), radius=0.0)
    assert len(out) == 3
    assert np.allclose(np.asarray(out), p)


def test_identical_peaks_collapse_to_one():
    """Peaks at one location become one peak; with renormalization it carries the full weight."""
    p = np.tile([0.5, 0.5, 0.2], (5, 1))
    out = group_peaks(p, np.full(5, 0.9), radius=0.05, renormalize=True)
    assert len(out) == 1
    assert out[0][:2] == (0.5, 0.5)
    assert out[0][2] == pytest.approx(1.0)


def test_identical_peaks_without_renormalization_keep_the_member_mean():
    """Without renormalization the merged weight is the member mean, not the sum.

    Duplicate queries each explain the whole compartment, so summing would double-count: five
    copies of weight 0.2 give 0.2.
    """
    p = np.tile([0.5, 0.5, 0.2], (5, 1))
    out = group_peaks(p, np.full(5, 0.9), radius=0.05, renormalize=False)
    assert len(out) == 1
    assert out[0][2] == pytest.approx(0.2)


def test_far_apart_peaks_are_never_merged():
    """A radius smaller than the separation must leave genuinely distinct pools alone."""
    p = np.array([[0.1, 0.1, 0.5], [0.9, 0.9, 0.5]])
    out = group_peaks(p, np.array([0.9, 0.9]), radius=0.2, renormalize=False)
    assert len(out) == 2


def test_radius_is_a_hard_bound_on_group_diameter():
    """Complete linkage bounds the group diameter, so a chain must not collapse.

    Three peaks 0.08 apart: neighbours are within radius 0.10 but the ends are 0.16 apart.
    Single linkage would merge the chain and let one radius swallow an arbitrarily wide region.
    """
    p = np.array([[0.30, 0.5, 0.3], [0.38, 0.5, 0.3], [0.46, 0.5, 0.3]])
    out = group_peaks(p, np.full(3, 0.9), radius=0.10, renormalize=False)
    assert len(out) > 1
    centres = np.asarray([o[0] for o in out])
    assert np.ptp(centres) > 0.10


def test_aggregation_modes_shift_the_centre_as_documented():
    """Weight- and confidence-weighting pull the centre toward the favoured member.

    Peaks at normalized T1 0.20 and 0.60, plain mean 0.40. Weighting by predicted weight
    (0.1 vs 0.9) pulls toward 0.60; weighting by confidence (0.99 vs 0.51) pulls toward 0.20.
    """
    p = np.array([[0.20, 0.5, 0.1], [0.60, 0.5, 0.9]])
    conf = np.array([0.99, 0.51])
    plain = group_peaks(p, conf, radius=0.9, aggregate="mean", renormalize=False)[0]
    by_w = group_peaks(p, conf, radius=0.9, aggregate="weight", renormalize=False)[0]
    by_c = group_peaks(p, conf, radius=0.9, aggregate="confidence", renormalize=False)[0]
    assert plain[0] == pytest.approx(0.40)
    assert by_w[0] == pytest.approx(0.20 * 0.1 + 0.60 * 0.9)
    assert by_c[0] < plain[0]
    # The merged weight is the plain member mean in all three modes.
    for out in (plain, by_w, by_c):
        assert out[2] == pytest.approx(0.5)


def test_degenerate_mixing_vector_falls_back_to_the_plain_mean():
    """All-zero weights (or confidences) must not produce NaN.

    An unmatched query gets no regression gradient, so a weight of exactly 0.0 is reachable.
    """
    p = np.array([[0.2, 0.5, 0.0], [0.4, 0.5, 0.0]])
    out = group_peaks(p, np.zeros(2), radius=0.9, aggregate="weight", renormalize=False)
    assert len(out) == 1
    assert out[0][0] == pytest.approx(0.3)
    assert np.isfinite(out[0]).all()


def test_include_weight_blocks_a_merge_that_differs_only_in_weight():
    """With the weight axis in the distance, same (T1, T2) but different weight stays split."""
    p = np.array([[0.5, 0.5, 0.05], [0.5, 0.5, 0.85]])
    assert len(group_peaks(p, np.full(2, .9), .1, include_weight=False, renormalize=False)) == 1
    assert len(group_peaks(p, np.full(2, .9), .1, include_weight=True, renormalize=False)) == 2


def test_renormalized_weights_sum_to_one():
    """The documented postcondition of renormalize=True, on a multi-group voxel."""
    p = np.array([[0.2, 0.3, 0.11], [0.21, 0.31, 0.12], [0.8, 0.7, 0.33]])
    out = group_peaks(p, np.full(3, 0.9), radius=0.05, renormalize=True)
    assert len(out) == 2
    assert sum(o[2] for o in out) == pytest.approx(1.0)


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        group_peaks(np.zeros((2, 3)), np.zeros(2), radius=0.1, aggregate="median")
    with pytest.raises(ValueError):
        group_peaks(np.zeros((2, 2)), np.zeros(2), radius=0.1)      # wrong width
    with pytest.raises(ValueError):
        group_peaks(np.zeros((3, 3)), np.zeros(2), radius=0.1)      # score count mismatch


def test_peak_distances_are_euclidean_in_normalized_space():
    d = peak_distances(np.array([[0.0, 0.0], [0.3, 0.4]]))
    assert d.shape == (1,)
    assert d[0] == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# grouped_predictions: the whole-split wrapper
# --------------------------------------------------------------------------------------

def _fake_query_table():
    """Two voxels: one with a duplicate pair plus a distinct pool, one with nothing above 0.5.

    Hand-built so the expected output is arithmetic rather than model behaviour.
    """
    params = np.array([
        [[0.30, 0.50, 0.40], [0.3002, 0.5002, 0.40], [0.80, 0.20, 0.20]],
        [[0.30, 0.50, 0.40], [0.60, 0.20, 0.30], [0.10, 0.90, 0.10]],
    ])
    probs = np.array([[0.95, 0.90, 0.85], [0.10, 0.20, 0.30]])
    return {"params_norm": params, "exist_prob": probs}


def test_radius_zero_matches_the_threshold_only_path_exactly():
    """Grouping at radius 0 must reproduce eval.predictions_from_query_outputs exactly.

    Every "grouping changed X by Y" figure is a difference against this path, so a mismatch at
    radius 0 would measure an implementation gap instead of the merge.
    """
    tbl = _fake_query_table()
    norm = TargetNormalizer("log_minmax", 50.0, 3500.0, 5.0, 500.0)
    physical = {"params": np.stack([
        np.stack([norm.denormalize_t1(tbl["params_norm"][..., 0]),
                  norm.denormalize_t2(tbl["params_norm"][..., 1]),
                  tbl["params_norm"][..., 2]], axis=-1)
    ])[0], "exist_prob": tbl["exist_prob"]}

    a = grouped_predictions(tbl, norm, threshold=0.5, radius=0.0)
    b = predictions_from_query_outputs(physical, exist_thresh=0.5)
    assert [len(x) for x in a] == [len(x) for x in b]
    for va, vb in zip(a, b):
        assert np.allclose(np.asarray(va).reshape(-1, 3), np.asarray(vb).reshape(-1, 3))


def test_grouping_reduces_the_count_and_returns_milliseconds():
    """The duplicate pair merges; the distinct pool survives; units come back physical."""
    tbl = _fake_query_table()
    norm = TargetNormalizer("log_minmax", 50.0, 3500.0, 5.0, 500.0)
    out = grouped_predictions(tbl, norm, threshold=0.5, radius=0.05, renormalize=False)
    assert [len(v) for v in out] == [2, 0]
    t1s = [p[0] for p in out[0]]
    assert all(50.0 <= t1 <= 3500.0 for t1 in t1s)
    # The merged peak's T1 is the geometric mean of its members, which for two nearly equal
    # members is (indistinguishably) their common value denormalized.
    assert min(t1s) == pytest.approx(norm.denormalize_t1(0.3001), rel=1e-3)


def test_grouping_never_increases_the_predicted_count():
    """Merging can only remove peaks, so no radius can rescue an under-counting model."""
    rng = np.random.default_rng(0)
    tbl = {"params_norm": rng.uniform(0.05, 0.95, size=(50, 10, 3)),
           "exist_prob": rng.uniform(0.0, 1.0, size=(50, 10))}
    norm = TargetNormalizer("log_minmax", 50.0, 3500.0, 5.0, 500.0)
    base = grouped_predictions(tbl, norm, 0.5, 0.0)
    for radius in (0.05, 0.15, 0.30):
        merged = grouped_predictions(tbl, norm, 0.5, radius)
        assert all(len(m) <= len(b) for m, b in zip(merged, base))


def test_all_aggregations_run_over_a_split():
    """Smoke coverage: no aggregation mode may crash or emit a non-finite compartment."""
    rng = np.random.default_rng(1)
    tbl = {"params_norm": rng.uniform(0.05, 0.95, size=(20, 10, 3)),
           "exist_prob": rng.uniform(0.4, 1.0, size=(20, 10))}
    norm = TargetNormalizer("log_minmax", 50.0, 3500.0, 5.0, 500.0)
    for aggregate in AGGREGATIONS:
        out = grouped_predictions(tbl, norm, 0.5, 0.1, aggregate=aggregate)
        assert all(np.isfinite(np.asarray(v).reshape(-1, 3)).all() for v in out if v)


# --------------------------------------------------------------------------------------
# existence_score_metrics: the ranking statistics
# --------------------------------------------------------------------------------------

def test_perfect_ranking_gives_pacc_and_scorr_of_one():
    """Score ordering exactly inverse to cost ordering: PAcc 1.0 and SCorr +1.0.

    SCorr is the correlation between quality (negative cost) and score, so good models score
    positive; a flipped sign convention would read -1 here.
    """
    cost = np.array([[0.1, 0.2, 0.3, 0.4]])
    prob = np.array([[0.9, 0.7, 0.5, 0.3]])
    diag = {"label": np.array([[1, 1, 0, 0]], dtype=np.int8), "cost": cost}
    m = existence_score_metrics(diag, prob, threshold=0.6)
    assert m["pacc"] == pytest.approx(1.0)
    assert m["scorr"] == pytest.approx(1.0)
    assert m["prec"] == pytest.approx(1.0)
    assert m["rec"] == pytest.approx(1.0)
    assert m["n_peak"] == pytest.approx(2.0)


def test_inverted_ranking_gives_pacc_zero():
    """The model is most confident about its worst prediction."""
    diag = {"label": np.array([[0, 0, 1, 1]], dtype=np.int8),
            "cost": np.array([[0.1, 0.2, 0.3, 0.4]])}
    m = existence_score_metrics(diag, np.array([[0.1, 0.3, 0.6, 0.9]]), threshold=0.5)
    assert m["pacc"] == pytest.approx(0.0)
    assert m["scorr"] == pytest.approx(-1.0)


def test_constant_score_voxels_are_skipped_not_counted_as_chance():
    """A voxel with a constant existence score has no defined ranking.

    Counting it as 0.5 would drag PAcc toward chance, so such voxels are excluded and the
    exclusion is reported via n_pacc_voxels.
    """
    diag = {"label": np.array([[1, 0, 0]], dtype=np.int8), "cost": np.array([[0.1, 0.2, 0.3]])}
    m = existence_score_metrics(diag, np.array([[0.5, 0.5, 0.5]]), threshold=0.4)
    assert m["n_pacc_voxels"] == 0
    assert m["n_scorr_voxels"] == 0
    assert np.isnan(m["pacc"]) and np.isnan(m["scorr"])


def test_cost_at_cuts_uses_the_documented_subsets():
    """cost@0.5 and cost@0.8 average the cost over queries above those score cuts."""
    diag = {"label": np.array([[1, 1, 0, 0]], dtype=np.int8),
            "cost": np.array([[0.10, 0.20, 0.30, 0.40]])}
    m = existence_score_metrics(diag, np.array([[0.95, 0.60, 0.40, 0.10]]), threshold=0.5)
    assert m["cost_at_0.5"] == pytest.approx(0.15)     # queries 1 and 2
    assert m["cost_at_0.8"] == pytest.approx(0.10)     # query 1 only

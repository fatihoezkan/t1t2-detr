"""Sampler tests: random compartments (T1 > T2, in range), weights, seeding and coverage."""
import numpy as np
import pytest
from scipy import stats

from voxel_simulator.sampler import (
    DEFAULT_SAMPLING,
    MAX_COMP,
    MIN_WEIGHT,
    SAMPLING_MODES,
    SAMPLING_REJECTION,
    SAMPLING_T1_LOG_UNIFORM,
    SPLIT_TRAIN,
    SPLIT_VAL,
    STREAM_PARAMS,
    T1_RANGE,
    T2_RANGE,
    sample_random_compartment,
    sample_voxel_spec,
    sample_weights,
    validate_ranges,
    validate_sampling,
    voxel_rng,
)


def test_random_compartment_keeps_t1_above_t2_and_in_range():
    """Check that sampled compartments respect their bounds and T1 exceeds T2."""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        t1, t2 = sample_random_compartment(rng)
        assert t1 > t2                                    # the single physical constraint
        assert T1_RANGE[0] <= t1 <= T1_RANGE[1]
        assert T2_RANGE[0] <= t2 <= t1


def test_voxel_spec_shapes_and_t1_gt_t2():
    """Check parameter sizes and valid relaxation times for each compartment count."""
    for n_comp in range(1, MAX_COMP + 1):
        for vid in range(50):
            spec = sample_voxel_spec(vid, n_comp=n_comp, base_seed=0)
            assert spec.n_comp == n_comp
            assert spec.t1.shape == spec.t2.shape == spec.w.shape == (n_comp,)
            assert np.all(spec.t1 > spec.t2)


def test_weights_sum_to_one_and_respect_floor():
    """Check that sampled weights sum to one and respect the minimum."""
    rng = np.random.default_rng(0)
    for n in range(1, MAX_COMP + 1):
        w = sample_weights(n, rng)
        assert abs(w.sum() - 1.0) < 1e-9
        assert w.min() >= MIN_WEIGHT - 1e-12


def test_n_comp_outside_range_raises():
    """Check that unsupported compartment counts are rejected."""
    for bad in (0, MAX_COMP + 1):
        with pytest.raises(ValueError, match="n_comp must be in"):
            sample_voxel_spec(0, n_comp=bad)


# --------------------------------------------------------------------------------------
# Seeding: reproducible, and no collisions of the kind the old arithmetic seed had.
# --------------------------------------------------------------------------------------

def test_same_key_is_bit_reproducible():
    """Check that the same sampling key produces identical voxel parameters."""
    a = sample_voxel_spec(5, n_comp=3, base_seed=0, split_code=SPLIT_TRAIN)
    b = sample_voxel_spec(5, n_comp=3, base_seed=0, split_code=SPLIT_TRAIN)
    np.testing.assert_array_equal(a.t1, b.t1)
    np.testing.assert_array_equal(a.w, b.w)
    assert a.snr == b.snr


def test_splits_and_seeds_and_counts_all_separate_the_stream():
    """Any change to the key must land on a different voxel."""
    base = sample_voxel_spec(5, n_comp=3, base_seed=0, split_code=SPLIT_TRAIN)
    for other in (
        sample_voxel_spec(5, n_comp=3, base_seed=0, split_code=SPLIT_VAL),    # split differs
        sample_voxel_spec(5, n_comp=3, base_seed=1, split_code=SPLIT_TRAIN),  # base seed differs
        sample_voxel_spec(6, n_comp=3, base_seed=0, split_code=SPLIT_TRAIN),  # voxel id differs
    ):
        assert not np.array_equal(base.t1, other.t1)


def test_old_arithmetic_seed_collision_cannot_return():
    """Regression: the old `master*1_000_003 + voxel_id` seed let voxel 10,000,030 of one split
    alias voxel 0 of the next. SeedSequence keys cannot alias like that.
    """
    a = sample_voxel_spec(10_000_030, n_comp=2, base_seed=0, split_code=SPLIT_TRAIN)
    b = sample_voxel_spec(0, n_comp=2, base_seed=10, split_code=SPLIT_TRAIN)
    assert not np.array_equal(a.t1, b.t1)


def test_param_stream_ignores_whether_snr_was_drawn():
    """Pinning the SNR must not disturb the parameter draw (SNR has its own stream)."""
    drawn = sample_voxel_spec(7, n_comp=3, base_seed=0)
    pinned = sample_voxel_spec(7, n_comp=3, base_seed=0, snr=20.0)
    np.testing.assert_array_equal(drawn.t1, pinned.t1)
    np.testing.assert_array_equal(drawn.t2, pinned.t2)
    np.testing.assert_array_equal(drawn.w, pinned.w)
    assert pinned.snr == 20.0 and drawn.snr != 20.0


def test_voxel_rng_streams_are_independent():
    """Check that parameter and noise streams produce different random draws."""
    from voxel_simulator.sampler import STREAM_NOISE, STREAM_PARAMS

    a = voxel_rng(0, 2, SPLIT_TRAIN, 3, STREAM_PARAMS).standard_normal(8)
    b = voxel_rng(0, 2, SPLIT_TRAIN, 3, STREAM_NOISE).standard_normal(8)
    assert not np.allclose(a, b)


# --------------------------------------------------------------------------------------
# Coverage of the (T1, T2) plane under the default rejection scheme.
# --------------------------------------------------------------------------------------

def test_validate_ranges_rejects_infeasible_and_inverted():
    """Check that invalid or physically impossible parameter ranges are rejected."""
    with pytest.raises(ValueError, match="no .T1, T2. with T2 < T1"):
        validate_ranges((50.0, 100.0), (200.0, 3000.0))     # t2_min above t1_max
    for bad in ((0.0, 100.0), (100.0, 50.0)):
        with pytest.raises(ValueError, match="0 < min < max"):
            validate_ranges(bad, T2_RANGE)
        with pytest.raises(ValueError, match="0 < min < max"):
            validate_ranges(T1_RANGE, bad)


def test_infeasible_range_raises_before_numpy_does():
    """A bad range must fail with the sampler's message, not numpy's `high - low < 0`."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="no .T1, T2. with T2 < T1"):
        sample_random_compartment(rng, (50.0, 100.0), (200.0, 3000.0))


def test_log_t2_is_uniform_within_its_feasible_row():
    """Rejection sampling gives a uniform joint over the feasible log region.

    Within any T1 band the normalised log-T2 position (log T2 - log t2_min) / (log min(t2_max, T1)
    - log t2_min) must be U(0, 1). A per-draw invariant (T1 > T2, in range) cannot detect a
    coverage distortion; this asserts a distribution instead, in two bands far apart in T1.
    """
    rng = np.random.default_rng(1)
    t1s, t2s = np.empty(20_000), np.empty(20_000)
    for i in range(20_000):
        t1s[i], t2s[i] = sample_random_compartment(rng)

    lo2 = np.log(T2_RANGE[0])
    for lo, hi in ((50.0, 200.0), (1000.0, 4000.0)):
        m = (t1s >= lo) & (t1s < hi)
        assert m.sum() > 500, f"too few samples in T1 band [{lo},{hi})"
        top = np.log(np.minimum(T2_RANGE[1], t1s[m]))
        u = (np.log(t2s[m]) - lo2) / (top - lo2)           # position within the feasible row
        assert 0.0 <= u.min() and u.max() <= 1.0
        for q in (0.25, 0.5, 0.75):                        # uniform => quantile q sits at q
            assert abs(np.quantile(u, q) - q) < 0.05, f"log-T2 not uniform in [{lo},{hi}) at q={q}"


def test_rejection_removes_short_t1_oversampling():
    """A uniform joint puts T1 < 200 ms near 21% of draws; the old clipping scheme gave about 31%."""
    rng = np.random.default_rng(2)
    t1s = np.array([sample_random_compartment(rng)[0] for _ in range(20_000)])
    frac_short = float((t1s < 200.0).mean())
    assert 0.18 < frac_short < 0.24, f"T1<200ms fraction {frac_short:.3f} (clipping gave ~0.31)"


# --------------------------------------------------------------------------------------
# Sampling modes: "rejection" (default, historical) vs "t1_log_uniform" (opt-in).
# These tests assert distributions against analytic references, not per-draw invariants.
# THESIS_T1 / THESIS_T2 are the ranges the thesis dataset uses. The module defaults
# (T1 50-4000, T2 5-3000) are wider and have a different acceptance rate; see the
# acceptance test.
# --------------------------------------------------------------------------------------

THESIS_T1 = (50.0, 3500.0)
THESIS_T2 = (5.0, 500.0)


def _log_bounds(t1_range, t2_range):
    """(a, b, c, d) = (log t1_min, log t1_max, log t2_min, log t2_max)."""
    return np.log(t1_range[0]), np.log(t1_range[1]), np.log(t2_range[0]), np.log(t2_range[1])


def _draw_many(n, sampling, t1_range=THESIS_T1, t2_range=THESIS_T2, seed=0):
    """Draw n compartments from one RNG; returns (t1, t2) arrays."""
    rng = np.random.default_rng(seed)
    t1 = np.empty(n)
    t2 = np.empty(n)
    for i in range(n):
        t1[i], t2[i] = sample_random_compartment(rng, t1_range, t2_range, sampling=sampling)
    return t1, t2


# --------------------------------------------------------------------------------------
# T1 marginal: log-uniform under t1_log_uniform, not under rejection.
# --------------------------------------------------------------------------------------

def test_t1_log_uniform_mode_has_log_uniform_t1_marginal():
    """Under t1_log_uniform, (log T1 - a) / (b - a) is U(0, 1) exactly: nothing downstream rejects.

    KS test against uniform; a large p is the pass (the test can only fail to reject).
    """
    a, b, _, _ = _log_bounds(THESIS_T1, THESIS_T2)
    t1, _ = _draw_many(100_000, SAMPLING_T1_LOG_UNIFORM, seed=101)
    u = (np.log(t1) - a) / (b - a)
    d_stat, p = stats.kstest(u, "uniform")
    assert p > 0.01, f"log T1 is not log-uniform under the new mode: D={d_stat:.5f}, p={p:.3g}"
    assert d_stat < 0.01, f"KS D={d_stat:.5f} is suspiciously large for n=100k"


def test_rejection_mode_t1_marginal_is_not_log_uniform():
    """Under rejection, log T1 is not log-uniform: p(log T1) is proportional to min(d, x) - c.

    The opposite outcome of the test above. If this ever passes log-uniformity, the default
    mode has changed and existing datasets no longer match the code.
    """
    a, b, _, _ = _log_bounds(THESIS_T1, THESIS_T2)
    t1, _ = _draw_many(100_000, SAMPLING_REJECTION, seed=101)
    u = (np.log(t1) - a) / (b - a)
    d_stat, p = stats.kstest(u, "uniform")
    assert p < 1e-6, f"rejection mode unexpectedly looks log-uniform in T1 (D={d_stat:.5f})"
    assert d_stat > 0.05, f"expected D near 0.083 for these ranges, got {d_stat:.5f}"


def test_rejection_mode_matches_its_analytic_post_rejection_marginal():
    """The rejection T1 marginal matches its analytic post-rejection CDF.

        F(x) = integral_a^x (min(d, t) - c) dt / integral_a^b (min(d, t) - c) dt

    Together with the previous test this shows the distortion is fully explained by the
    constraint plus rejection.
    """
    a, b, c, d = _log_bounds(THESIS_T1, THESIS_T2)

    def width_integral(x):
        """integral_a^x (min(d, t) - c) dt, vectorised over x in [a, b]."""
        x = np.asarray(x, dtype=float)
        below = np.minimum(x, d)                                  # part of the range under the diagonal
        tri = np.where(x > a, ((below - c) ** 2 - (a - c) ** 2) / 2.0, 0.0)
        rect = np.clip(x - d, 0.0, None) * (d - c)                # part where the full T2 range fits
        return tri + rect

    total = width_integral(b)
    t1, _ = _draw_many(100_000, SAMPLING_REJECTION, seed=202)
    d_stat, p = stats.kstest(np.log(t1), lambda x: width_integral(x) / total)
    assert p > 0.01, f"rejection T1 marginal does not match its analytic form: D={d_stat:.5f}, p={p:.3g}"


# --------------------------------------------------------------------------------------
# T2 marginal under t1_log_uniform: skewed toward small T2 by a known amount.
# --------------------------------------------------------------------------------------

def _analytic_log_t2_cdf(y, a, b, c, d):
    """Closed-form CDF of y = log T2 under sampling="t1_log_uniform".

    x = log T1 ~ U(a, b) and y | x ~ U(c, min(d, x)), so

        F(y) = (1 / (b - a)) * integral_a^b P(Y <= y | x) dx,
        P(Y <= y | x) = min(1, (y - c) / (min(d, x) - c)).

    Splitting the integral at y and at d gives three terms:

      * x <= y:      P = 1, contributes (y - a) when y > a, else 0.
      * y < x <= d:  P = (y - c) / (x - c), contributes (y - c) * ln((d - c) / (max(a, y) - c)).
      * x > d:       P = (y - c) / (d - c), contributes (b - d) * (y - c) / (d - c).

    Valid for c < a < d < b, the thesis geometry (5 < 50 < 500 < 3500 ms). F(c) = 0 and
    F(d) = 1 are checked in the test below. The middle term is what concentrates mass at small y.
    """
    y = np.asarray(y, dtype=float)
    tail = (b - d) * (y - c) / (d - c)
    below = np.where(y > a, y - a, 0.0)
    lo = np.maximum(a, y)
    mid = (y - c) * np.log((d - c) / (lo - c))
    return (below + mid + tail) / (b - a)


def test_analytic_log_t2_cdf_is_a_valid_cdf():
    """The analytic reference is a valid CDF: F(c) = 0, F(d) = 1, non-decreasing."""
    a, b, c, d = _log_bounds(THESIS_T1, THESIS_T2)
    assert abs(_analytic_log_t2_cdf(c, a, b, c, d)) < 1e-12
    assert abs(_analytic_log_t2_cdf(d, a, b, c, d) - 1.0) < 1e-12
    grid = np.linspace(c, d, 500)
    f = _analytic_log_t2_cdf(grid, a, b, c, d)
    assert np.all(np.diff(f) >= -1e-15), "reference CDF must be non-decreasing"


def test_t1_log_uniform_t2_marginal_matches_the_analytic_skew():
    """The t1_log_uniform T2 marginal matches the analytic CDF (KS, large p is the pass).

    Flattening T1 moves the coverage skew to T2; this checks the skew is exactly the predicted
    one and nothing more.
    """
    a, b, c, d = _log_bounds(THESIS_T1, THESIS_T2)
    _, t2 = _draw_many(100_000, SAMPLING_T1_LOG_UNIFORM, seed=303)
    d_stat, p = stats.kstest(np.log(t2), lambda y: _analytic_log_t2_cdf(y, a, b, c, d))
    assert p > 0.01, f"T2 marginal deviates from its analytic form: D={d_stat:.5f}, p={p:.3g}"


def test_new_mode_undersamples_long_t2_more_than_the_old_one():
    """Share of T2 in [100, 500): log-uniform 35%, rejection about 26%, t1_log_uniform about 24%.

    The new mode undersamples long T2 more than rejection does, because a short T1 caps T2.
    """
    _, _, c, d = _log_bounds(THESIS_T1, THESIS_T2)
    log_uniform_share = (np.log(500.0) - np.log(100.0)) / (d - c)
    _, t2_new = _draw_many(60_000, SAMPLING_T1_LOG_UNIFORM, seed=404)
    _, t2_old = _draw_many(60_000, SAMPLING_REJECTION, seed=404)
    share_new = float(((t2_new >= 100.0) & (t2_new < 500.0)).mean())
    share_old = float(((t2_old >= 100.0) & (t2_old < 500.0)).mean())
    assert share_new < share_old < log_uniform_share, (
        f"expected long-T2 share to fall (log-uniform {log_uniform_share:.3f} > "
        f"rejection {share_old:.3f} > new {share_new:.3f})"
    )
    assert 0.20 < share_new < 0.28, f"long-T2 share {share_new:.3f} outside the recorded band"


# --------------------------------------------------------------------------------------
# Per-draw invariants and RNG consumption in the new mode.
# --------------------------------------------------------------------------------------

def test_t1_log_uniform_respects_the_constraint_in_every_draw():
    """T2 < T1 strictly, and both inside their declared ranges, in every draw.

    exp(log(x)) can round up to exactly T1 at the top of the conditional interval; the sampler
    nudges that case down by one ULP, and nothing downstream would flag T2 == T1.
    """
    t1, t2 = _draw_many(200_000, SAMPLING_T1_LOG_UNIFORM, seed=505)
    assert np.all(t2 < t1), f"{int((t2 >= t1).sum())} draw(s) violated T2 < T1"
    assert np.all(t1 >= THESIS_T1[0]) and np.all(t1 <= THESIS_T1[1])
    assert np.all(t2 >= THESIS_T2[0]), f"T2 fell below t2_min: min {t2.min()}"
    assert np.all(t2 <= THESIS_T2[1]), f"T2 exceeded t2_max: max {t2.max()}"


def test_t1_log_uniform_consumes_exactly_two_uniforms_per_compartment():
    """The new mode draws exactly one uniform for T1 and one for T2, with no retry loop.

    The proxy counts `uniform` calls and forwards everything else. The rejection mode is
    measured alongside: about 2.32 uniforms per compartment on the thesis ranges, i.e. about
    1.16 (T1, T2) draw pairs.
    """
    class CountingRng:
        def __init__(self, gen):
            """Wrap the random generator and start counting uniform draws."""
            self._gen = gen
            self.uniform_calls = 0

        def uniform(self, *args, **kwargs):
            """Count a uniform draw and pass it to the wrapped generator."""
            self.uniform_calls += 1
            return self._gen.uniform(*args, **kwargs)

        def __getattr__(self, name):
            """Pass other attribute requests to the wrapped generator."""
            return getattr(self._gen, name)

    n = 5_000
    new = CountingRng(np.random.default_rng(606))
    for _ in range(n):
        sample_random_compartment(new, THESIS_T1, THESIS_T2, sampling=SAMPLING_T1_LOG_UNIFORM)
    assert new.uniform_calls == 2 * n, (
        f"new mode used {new.uniform_calls} uniforms for {n} compartments; expected exactly {2 * n}"
    )

    old = CountingRng(np.random.default_rng(606))
    for _ in range(n):
        sample_random_compartment(old, THESIS_T1, THESIS_T2, sampling=SAMPLING_REJECTION)
    assert old.uniform_calls > 2 * n, "rejection mode should sometimes redraw"
    draws_per_comp = old.uniform_calls / (2 * n)
    assert 1.10 < draws_per_comp < 1.22, (
        f"rejection cost {draws_per_comp:.3f} draws/compartment on the thesis ranges; "
        "the docstring says ~1.16 (the ~1.43 figure belongs to the module default ranges)"
    )


def test_t1_log_uniform_never_hits_the_max_tries_error():
    """max_tries does not apply to the new mode.

    With max_tries=1 the rejection branch raises as soon as a draw lands with T2 >= T1 (about
    14% of draws on these ranges). The new mode has no loop, so 50,000 calls must all succeed.
    """
    rng = np.random.default_rng(707)
    for _ in range(50_000):
        sample_random_compartment(
            rng, THESIS_T1, THESIS_T2, max_tries=1, sampling=SAMPLING_T1_LOG_UNIFORM
        )

    strict = np.random.default_rng(707)
    with pytest.raises(RuntimeError, match="rejection sampling found no T2 < T1"):
        for _ in range(200):
            sample_random_compartment(
                strict, THESIS_T1, THESIS_T2, max_tries=1, sampling=SAMPLING_REJECTION
            )


def test_analytic_acceptance_rate_differs_between_thesis_and_default_ranges():
    """Rejection acceptance rate is range-specific.

        P = integral_a^b (min(d, t) - c) dt / ((b - a) * (d - c))

    Thesis ranges (T1 50-3500, T2 5-500): 0.8645, i.e. 1.157 draws per compartment.
    Module default ranges (T1 50-4000, T2 5-3000): 0.700, i.e. 1.43.
    """
    def acceptance(t1_range, t2_range):
        """Calculate the expected fraction of accepted log-space draws."""
        a, b, c, d = _log_bounds(t1_range, t2_range)
        below = np.minimum(b, d)
        tri = ((below - c) ** 2 - (a - c) ** 2) / 2.0
        rect = max(b - d, 0.0) * (d - c)
        return float((tri + rect) / ((b - a) * (d - c)))

    p_thesis = acceptance(THESIS_T1, THESIS_T2)
    p_default = acceptance(T1_RANGE, T2_RANGE)
    assert abs(p_thesis - 0.8645) < 5e-4, f"thesis acceptance {p_thesis:.4f}"
    assert abs(1.0 / p_thesis - 1.157) < 5e-3, f"thesis draws/comp {1.0 / p_thesis:.3f}"
    assert abs(p_default - 0.700) < 5e-3, f"default acceptance {p_default:.4f}"
    assert abs(1.0 / p_default - 1.428) < 1e-2, f"default draws/comp {1.0 / p_default:.3f}"


# --------------------------------------------------------------------------------------
# Backward compatibility of the default sampling path.
#
# The baseline dataset (data/t1_3500_t2_500_100k, base seed 3500500) is only reproducible if
# the default path is untouched. The frozen values below were captured from the sampler before
# the two-mode change, under numpy 2.5.1. Reordering the two rng.uniform calls, moving the
# validation, or moving the rejection test would fail these tests.
#
# Tolerance: the baseline manifest records numpy 1.26.4, and the exp path is not bit-identical
# across numpy versions. Spot-checking 2000 baseline rows (n2/train) against numpy 2.5.1: 19.2%
# of T1 and 14.8% of T2 values differ, by up to 8 ULP (max relative difference 1.076e-15); SNR
# differs by at most 1 ULP; the Dirichlet weights are bit-identical. rtol 1e-12 is about four
# orders of magnitude above that drift and far below any change a reordered draw would cause
# (a factor, not a few ULP). Bit-exactness against the on-disk dataset is a claim about numpy
# 1.26.4, not about this code.
_RTOL = 1e-12
# --------------------------------------------------------------------------------------

# Captured pre-change; `default_rng(12345)`, thesis ranges, first three accepted pairs.
_FROZEN_REJECTION_PAIRS = [
    (131.3491461088766, 21.502387481690803),
    (1479.7469273170282, 112.58469246324051),
    (263.39375248191715, 23.152498251145378),
]

# Captured pre-change; the real dataset key (base_seed 3500500, n_comp 2, train, voxel 0).
_FROZEN_VOXEL_STREAM_PAIRS = [
    (2838.6018673499707, 16.771879732266406),
    (2812.8245692954974, 5.289941377452774),
    (98.40670217660086, 52.77778179112104),
]


def test_default_sampling_is_still_rejection():
    """The new mode is opt-in; the default must stay "rejection"."""
    assert DEFAULT_SAMPLING == SAMPLING_REJECTION == "rejection"
    assert SAMPLING_MODES == ("rejection", "t1_log_uniform")


def test_rejection_path_is_bit_exact_against_pre_change_values():
    """The default path returns the values it returned before the two-mode change.

    Tolerance rather than equality: see the block comment above.
    """
    rng = np.random.default_rng(12345)
    got = [sample_random_compartment(rng, THESIS_T1, THESIS_T2) for _ in range(3)]
    for i, ((t1, t2), (e1, e2)) in enumerate(zip(got, _FROZEN_REJECTION_PAIRS)):
        assert t1 == pytest.approx(e1, rel=_RTOL), f"pair {i}: T1 {t1!r} != frozen {e1!r}"
        assert t2 == pytest.approx(e2, rel=_RTOL), f"pair {i}: T2 {t2!r} != frozen {e2!r}"


def test_rejection_path_bit_exact_on_the_real_dataset_key():
    """Same regression through the SeedSequence key the reported dataset used.

    The test above pins the arithmetic; this one also pins the key derivation in voxel_rng.
    """
    rng = voxel_rng(3500500, 2, SPLIT_TRAIN, 0, STREAM_PARAMS)
    got = [sample_random_compartment(rng, THESIS_T1, THESIS_T2) for _ in range(3)]
    for i, ((t1, t2), (e1, e2)) in enumerate(zip(got, _FROZEN_VOXEL_STREAM_PAIRS)):
        assert t1 == pytest.approx(e1, rel=_RTOL), f"pair {i}: T1 {t1!r} != frozen {e1!r}"
        assert t2 == pytest.approx(e2, rel=_RTOL), f"pair {i}: T2 {t2!r} != frozen {e2!r}"


def test_frozen_voxel_spec_values_for_the_baseline_seed():
    """A whole VoxelSpec pinned: three compartments, weights, and SNR.

    Pins the composition: the order of the T1/T2 draws, the weights drawn from the same stream
    after them, and the SNR from its own stream.
    """
    spec = sample_voxel_spec(7, n_comp=3, base_seed=3500500, t1_range=THESIS_T1, t2_range=THESIS_T2)
    np.testing.assert_allclose(
        spec.t1, np.array([1953.2671423345857, 275.94470491546207, 358.4180990396131]),
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        spec.t2, np.array([62.01127139648379, 12.310300390916774, 242.1498247501432]),
        rtol=_RTOL,
    )
    np.testing.assert_allclose(
        spec.w, np.array([0.3450406558359255, 0.4390252979886396, 0.21593404617543477]),
        rtol=_RTOL,
    )
    assert spec.snr == pytest.approx(77.57756358600899, rel=_RTOL)


def test_explicit_rejection_argument_equals_the_default():
    """Passing sampling="rejection" explicitly must equal passing nothing."""
    implicit = sample_voxel_spec(11, n_comp=3, base_seed=3500500,
                                t1_range=THESIS_T1, t2_range=THESIS_T2)
    explicit = sample_voxel_spec(11, n_comp=3, base_seed=3500500,
                                 t1_range=THESIS_T1, t2_range=THESIS_T2,
                                 sampling=SAMPLING_REJECTION)
    np.testing.assert_array_equal(implicit.t1, explicit.t1)
    np.testing.assert_array_equal(implicit.t2, explicit.t2)
    np.testing.assert_array_equal(implicit.w, explicit.w)
    assert implicit.snr == explicit.snr


def test_the_two_modes_actually_produce_different_voxels():
    """The sampling argument is not silently ignored on the way to the draw."""
    old = sample_voxel_spec(3, n_comp=3, base_seed=3500501,
                           t1_range=THESIS_T1, t2_range=THESIS_T2)
    new = sample_voxel_spec(3, n_comp=3, base_seed=3500501,
                           t1_range=THESIS_T1, t2_range=THESIS_T2,
                           sampling=SAMPLING_T1_LOG_UNIFORM)
    assert not np.array_equal(old.t1, new.t1) or not np.array_equal(old.t2, new.t2)


# --------------------------------------------------------------------------------------
# Determinism and the weight contract, in both modes.
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("sampling", SAMPLING_MODES)
def test_same_key_reproduces_the_same_voxel_in_both_modes(sampling):
    """The key (base_seed, n_comp, split_code, voxel_id) fully determines the voxel in both modes."""
    kw = dict(t1_range=THESIS_T1, t2_range=THESIS_T2, sampling=sampling)
    a = sample_voxel_spec(42, n_comp=3, base_seed=3500501, split_code=SPLIT_VAL, **kw)
    b = sample_voxel_spec(42, n_comp=3, base_seed=3500501, split_code=SPLIT_VAL, **kw)
    np.testing.assert_array_equal(a.t1, b.t1)
    np.testing.assert_array_equal(a.t2, b.t2)
    np.testing.assert_array_equal(a.w, b.w)
    assert a.snr == b.snr


@pytest.mark.parametrize("sampling", SAMPLING_MODES)
def test_every_key_component_still_separates_the_stream(sampling):
    """Changing any one element of the key lands on a different voxel, in either mode."""
    kw = dict(t1_range=THESIS_T1, t2_range=THESIS_T2, sampling=sampling)
    base = sample_voxel_spec(5, n_comp=3, base_seed=3500501, split_code=SPLIT_TRAIN, **kw)
    for other in (
        sample_voxel_spec(5, n_comp=3, base_seed=3500501, split_code=SPLIT_VAL, **kw),
        sample_voxel_spec(5, n_comp=3, base_seed=3500502, split_code=SPLIT_TRAIN, **kw),
        sample_voxel_spec(6, n_comp=3, base_seed=3500501, split_code=SPLIT_TRAIN, **kw),
        sample_voxel_spec(5, n_comp=2, base_seed=3500501, split_code=SPLIT_TRAIN, **kw),
    ):
        n = min(len(base.t1), len(other.t1))
        assert not np.array_equal(base.t1[:n], other.t1[:n])


@pytest.mark.parametrize("sampling", SAMPLING_MODES)
def test_weight_contract_holds_in_both_modes(sampling):
    """Weights sum to 1 and each clears MIN_WEIGHT, whichever way (T1, T2) was drawn.

    The Dirichlet draw follows the compartment draws on the same stream, so the mode changes
    which numbers it sees but not the constraints it must satisfy.
    """
    for n_comp in range(1, MAX_COMP + 1):
        for vid in range(60):
            spec = sample_voxel_spec(vid, n_comp=n_comp, base_seed=3500501,
                                     t1_range=THESIS_T1, t2_range=THESIS_T2, sampling=sampling)
            assert abs(spec.w.sum() - 1.0) < 1e-9, f"weights sum to {spec.w.sum()!r}"
            assert spec.w.min() >= MIN_WEIGHT - 1e-12, f"min weight {spec.w.min()!r}"
            assert np.all(spec.t1 > spec.t2)


# --------------------------------------------------------------------------------------
# Validation: bad modes and range/mode combinations must fail loudly and early.
# --------------------------------------------------------------------------------------

def test_unknown_sampling_mode_raises():
    """An unknown mode raises; a typo must not fall back to the default."""
    with pytest.raises(ValueError, match="sampling must be one of"):
        validate_sampling("t1_loguniform")          # missing underscore, a plausible slip
    with pytest.raises(ValueError, match="sampling must be one of"):
        sample_random_compartment(np.random.default_rng(0), sampling="uniform")
    with pytest.raises(ValueError, match="sampling must be one of"):
        sample_voxel_spec(0, n_comp=1, sampling="")


def test_t1_log_uniform_requires_t2_min_below_t1_min():
    """t1_log_uniform needs t2_min < t1_min, checked before generation starts.

    Rejection only needs some feasible pair (t2_min < t1_max). With T1 in [50, 3500] and T2 in
    [100, 500], T1 = 50 admits no T2 and numpy would fail inside `uniform` with `high - low < 0`.
    """
    with pytest.raises(ValueError, match="requires t2_min"):
        validate_ranges((50.0, 3500.0), (100.0, 500.0), SAMPLING_T1_LOG_UNIFORM)
    with pytest.raises(ValueError, match="requires t2_min"):
        sample_random_compartment(
            np.random.default_rng(0), (50.0, 3500.0), (50.0, 500.0),
            sampling=SAMPLING_T1_LOG_UNIFORM,
        )
    # The same ranges are legal for rejection: T1 above 100 ms still has room for a T2.
    validate_ranges((50.0, 3500.0), (100.0, 500.0), SAMPLING_REJECTION)
    t1, t2 = sample_random_compartment(
        np.random.default_rng(0), (50.0, 3500.0), (100.0, 500.0), sampling=SAMPLING_REJECTION
    )
    assert t2 < t1


def test_validate_ranges_still_rejects_the_old_way_in_both_modes():
    """The basic range checks are mode-independent."""
    for mode in SAMPLING_MODES:
        with pytest.raises(ValueError, match="no .T1, T2. with T2 < T1"):
            validate_ranges((50.0, 100.0), (200.0, 3000.0), mode)
        for bad in ((0.0, 100.0), (100.0, 50.0)):
            with pytest.raises(ValueError, match="0 < min < max"):
                validate_ranges(bad, THESIS_T2, mode)
            with pytest.raises(ValueError, match="0 < min < max"):
                validate_ranges(THESIS_T1, bad, mode)

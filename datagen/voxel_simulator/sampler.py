"""Per-voxel ground truth: (T1, T2) per compartment, Dirichlet weights, and SNR.

Compartments are random points with the single constraint T1 > T2 and no attempt at tissue
realism. n_comp is given by the caller rather than drawn: every RNG stream is keyed on it, and
one file per count keeps the splits exactly balanced. Two (T1, T2) sampling modes exist; the
default, "rejection", is what the reported datasets used. sample_random_compartment derives
the marginals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Maximum compartments per voxel. Fixes the parquet schema width (T1/T2/w_1..MAX_COMP) and the
# valid range of --n-comp.
MAX_COMP = 4

# SNR window (uniform) for the random-SNR splits.
SNR_MIN = 30.0
SNR_MAX = 150.0

# Smallest weight that still counts as a compartment; below this it is not visible in the signal.
MIN_WEIGHT = 0.05

# Sampling ranges in ms. t1_min stays above t2_min so that T2 < T1 always has room.
T1_RANGE = (50.0, 4000.0)
T2_RANGE = (5.0, 3000.0)

# (T1, T2) draw schemes; the marginals are derived in sample_random_compartment.
#   "rejection"       draw log T1 and log T2 independently log-uniform and keep the pair only
#                     if T2 < T1. Uniform over the feasible region, so neither marginal is
#                     log-uniform. Default, and what the reported datasets used.
#   "t1_log_uniform"  draw log T1 log-uniform, then log T2 log-uniform on
#                     [log t2_min, log min(t2_max, T1)]. The log-T1 marginal is exactly
#                     log-uniform; the log-T2 marginal skews toward small values.
# No scheme can make both marginals log-uniform under T2 < T1.
SAMPLING_REJECTION = "rejection"
SAMPLING_T1_LOG_UNIFORM = "t1_log_uniform"
SAMPLING_MODES = (SAMPLING_REJECTION, SAMPLING_T1_LOG_UNIFORM)
DEFAULT_SAMPLING = SAMPLING_REJECTION

# Split code. Part of every RNG stream key, so splits cannot collide at any size (an arithmetic
# seed offset would eventually overlap).
SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2
SPLIT_SNR_LADDER = 3

# Independent RNG streams per voxel. The fixed-SNR ladder pins SNR and leaves the parameter and
# noise streams untouched, so every rung holds the same voxel with the same standardised noise
# and only the amplitude differs (a paired comparison).
STREAM_PARAMS = 1001
STREAM_NOISE = 2001
STREAM_SNR = 3001


@dataclass
class VoxelSpec:
    """The ground truth for one voxel: its compartments and the SNR it will be noised at."""
    n_comp: int
    snr: float
    t1: np.ndarray          # (n_comp,)
    t2: np.ndarray          # (n_comp,)
    w: np.ndarray           # (n_comp,)


def voxel_rng(
    base_seed: int,
    n_comp: int,
    split_code: int,
    voxel_id: int,
    stream_id: int,
) -> np.random.Generator:
    """Generator for one (voxel, stream) pair, keyed by SeedSequence on integer entropy.

    Distinct SeedSequence keys cannot collide the way arithmetic seed offsets can. Streams are
    reproducible for a pinned NumPy version only; the manifest records the version.
    """
    return np.random.default_rng(
        np.random.SeedSequence([int(base_seed), int(n_comp), int(split_code), int(voxel_id), int(stream_id)])
    )


def validate_sampling(sampling: str) -> str:
    """Raise ValueError for an unknown mode; a typo must not fall back to the default."""
    if sampling not in SAMPLING_MODES:
        raise ValueError(f"sampling must be one of {SAMPLING_MODES}; got {sampling!r}")
    return sampling


def validate_ranges(
    t1_range: tuple[float, float] = T1_RANGE,
    t2_range: tuple[float, float] = T2_RANGE,
    sampling: str = DEFAULT_SAMPLING,
) -> None:
    """Reject infeasible (T1, T2) ranges before generation starts.

    Rejection sampling only needs some feasible pair, so t2_min < t1_max suffices. t1_log_uniform
    has no retry, so every T1 must admit a T2, i.e. t2_min < t1_min; otherwise the conditional
    draw would fail deep inside numpy with "high - low < 0". The check is therefore mode-aware.
    """
    # each range must be a proper positive interval
    validate_sampling(sampling)
    lo1, hi1 = t1_range
    lo2, hi2 = t2_range
    if not (0 < lo1 < hi1):
        raise ValueError(f"t1_range must satisfy 0 < min < max; got {t1_range}")
    if not (0 < lo2 < hi2):
        raise ValueError(f"t2_range must satisfy 0 < min < max; got {t2_range}")
    # there must be room for T2 < T1
    if lo2 >= hi1:
        raise ValueError(
            f"t2_min ({lo2}) >= t1_max ({hi1}): no (T1, T2) with T2 < T1 exists in these ranges."
        )
    if sampling == SAMPLING_T1_LOG_UNIFORM and lo2 >= lo1:
        raise ValueError(
            f"sampling={SAMPLING_T1_LOG_UNIFORM!r} requires t2_min ({lo2}) < t1_min ({lo1}): "
            "without a rejection step every T1 in the range must admit at least one T2 < T1, "
            "and the smallest T1 is t1_min. Either lower t2_min, raise t1_min, or use "
            f"sampling={SAMPLING_REJECTION!r}."
        )


def sample_weights(n_comp: int, rng: np.random.Generator, min_weight: float = MIN_WEIGHT) -> np.ndarray:
    """Weights summing to one, none below min_weight: symmetric Dirichlet rescaled onto the floor."""
    if n_comp * min_weight >= 1.0:
        raise ValueError(f"n_comp * min_weight = {n_comp * min_weight} >= 1.")
    raw = rng.dirichlet(np.ones(n_comp))
    return raw * (1.0 - n_comp * min_weight) + min_weight


def sample_random_compartment(
    rng: np.random.Generator,
    t1_range: tuple[float, float] = T1_RANGE,
    t2_range: tuple[float, float] = T2_RANGE,
    max_tries: int = 1000,
    sampling: str = DEFAULT_SAMPLING,
) -> tuple[float, float]:
    """Draw one (T1, T2) pair with T2 < T1, in log space (relaxation times span decades).

    Let a, b = log t1_min, log t1_max and c, d = log t2_min, log t2_max. The feasible set is
    {(x, y): a <= x <= b, c <= y <= min(d, x)}. On the thesis ranges (T1 50 to 3500 ms, T2 5 to
    500 ms) this is a triangle below T1 = 500 ms joined to a rectangle above it.

    sampling="rejection" (default): draw log T1 ~ U(a, b) and log T2 ~ U(c, d) independently and
    retry until T2 < T1. Accepted pairs are uniform over the feasible set, so the log-T1 density
    is proportional to the feasible log-T2 width, min(d, x) - c, and neither marginal is
    log-uniform. Measured on the thesis ranges: T1 in [50, 100) is drawn at 0.66x the
    log-uniform expectation, T1 in [500, 3500) at 1.16x, T2 in [100, 500] at 0.75x; a KS test of
    log T1 against log-uniform gives D = 0.0835. Acceptance is 0.8645, about 1.16 draws per
    compartment; on the module default ranges (T1 50 to 4000, T2 5 to 3000) it is 0.700, about
    1.43 draws.

    sampling="t1_log_uniform": draw log T1 ~ U(a, b), then log T2 ~ U(c, min(d, x)). No retry, so
    exactly two uniforms per compartment. The log-T1 marginal is log-uniform by construction, but
    the log-T2 marginal piles up at small T2 because a small T1 caps T2 (at T1 = 60 ms, log T2
    is spread over [log 5, log 60]). The coverage skew moves from T1 to T2; it does not go away.

    Both marginals cannot be log-uniform at once: large T2 is reachable only from large T1, so a
    flat log-T2 marginal needs extra mass at large T1, which breaks a flat log-T1 marginal.
    Coverage confounds any error-versus-T1 or error-versus-T2 curve (rare region or hard region
    cannot be told apart), which is why the mode is recorded in the manifest under
    physics.sampling.
    """
    # ranges in ms, drawn in log space
    validate_ranges(t1_range, t2_range, sampling)
    lo1, hi1 = t1_range
    lo2, hi2 = t2_range

    if sampling == SAMPLING_T1_LOG_UNIFORM:
        # Two draws, no loop. validate_ranges guarantees t2_min < t1_min <= T1, so the interval
        # [log lo2, log hi2_eff] is never empty.
        t1 = float(np.exp(rng.uniform(np.log(lo1), np.log(hi1))))
        hi2_eff = min(hi2, t1)
        t2 = float(np.exp(rng.uniform(np.log(lo2), np.log(hi2_eff))))
        # exp(log(.)) can round to exactly t1 at the top of the interval; keep T2 < T1 strict.
        if t2 >= t1:
            t2 = float(np.nextafter(t1, 0.0))
        return t1, t2

    # Rejection path. Do not reorder or move these draws: the reported datasets depend on this
    # exact sequence of rng calls.
    log1, log2 = (np.log(lo1), np.log(hi1)), (np.log(lo2), np.log(hi2))
    for _ in range(max_tries):
        t1 = float(np.exp(rng.uniform(*log1)))
        t2 = float(np.exp(rng.uniform(*log2)))
        if t2 < t1:
            return t1, t2
    raise RuntimeError(
        f"rejection sampling found no T2 < T1 in {max_tries} tries for T1{t1_range}, T2{t2_range}; "
        "the feasible region is too small for these ranges."
    )


def sample_voxel_spec(
    voxel_id: int,
    n_comp: int,
    base_seed: int = 0,
    split_code: int = SPLIT_TRAIN,
    snr_min: float = SNR_MIN,
    snr_max: float = SNR_MAX,
    snr: float | None = None,
    t1_range: tuple[float, float] = T1_RANGE,
    t2_range: tuple[float, float] = T2_RANGE,
    sampling: str = DEFAULT_SAMPLING,
) -> VoxelSpec:
    """Draw the ground truth for one voxel: n_comp compartments, their weights, and an SNR.

    n_comp comes from the caller and is never drawn. `snr` pins the SNR (used by the fixed-SNR
    ladder); SNR has its own stream, so pinning it leaves the parameter draw identical.
    `sampling` is not part of the RNG key: the same key under two modes consumes the same
    uniforms and interprets them differently, so two such datasets are neither identical nor
    independent. Independent datasets need different base_seed values.
    """
    if not 1 <= n_comp <= MAX_COMP:
        raise ValueError(f"n_comp must be in 1..{MAX_COMP}; got {n_comp}")

    # parameter stream of this voxel
    rng = voxel_rng(base_seed, n_comp, split_code, voxel_id, STREAM_PARAMS)

    # one (T1, T2) pair per compartment
    t1 = np.empty(n_comp)
    t2 = np.empty(n_comp)
    for i in range(n_comp):
        t1[i], t2[i] = sample_random_compartment(rng, t1_range, t2_range, sampling=sampling)

    # signal fractions summing to one
    w = sample_weights(n_comp, rng)

    # SNR from its own stream unless pinned
    if snr is None:
        snr_rng = voxel_rng(base_seed, n_comp, split_code, voxel_id, STREAM_SNR)
        snr = float(snr_rng.uniform(snr_min, snr_max))

    return VoxelSpec(n_comp=n_comp, snr=float(snr), t1=t1, t2=t2, w=w)

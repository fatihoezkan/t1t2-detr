"""Gaussian noise tests."""
import numpy as np

from voxel_simulator.noise import (
    add_gaussian_noise,
    add_noise,
)


def _clean():
    """Create a simple noise-free signal for tests."""
    return np.linspace(0.1, 1.0, 64)


def test_shape_and_finite():
    """Check that adding noise preserves signal size and finite values."""
    noisy, sigma = add_noise(_clean(), snr=50, rng=np.random.default_rng(0))
    assert noisy.shape == (64,)
    assert np.all(np.isfinite(noisy)) and sigma > 0


def test_reproducible():
    """Check that the same random seed produces identical noise."""
    a, _ = add_noise(_clean(), 40, np.random.default_rng(7))
    b, _ = add_noise(_clean(), 40, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_direct_sigma_overrides_snr():
    """Passing sigma sets the absolute noise level and ignores SNR."""
    clean = np.linspace(-1.0, 1.0, 4000)
    noisy, sigma = add_gaussian_noise(clean, snr=999, rng=np.random.default_rng(0), sigma=0.1)
    assert sigma == 0.1
    assert abs((noisy - clean).std() - 0.1) < 0.02      # residual std close to requested sigma


def test_explicit_z_matches_legacy_normal_bit_for_bit():
    """sigma * standard_normal() equals rng.normal(0, sigma) bit for bit on the pinned NumPy.

    Bit identity is only claimed for the pinned environment; NumPy does not guarantee Generator
    streams across versions.
    """
    clean = _clean()
    noisy, sigma = add_gaussian_noise(clean, snr=40, rng=np.random.default_rng(11))
    legacy = clean + np.random.default_rng(11).normal(0.0, sigma, size=clean.shape)
    np.testing.assert_array_equal(noisy, legacy)


def test_same_rng_shares_z_across_sigma():
    """Same rng state, two sigmas: same standardised z, different amplitude (paired-ladder invariant)."""
    clean = _clean()
    a, sig_a = add_gaussian_noise(clean, snr=20, rng=np.random.default_rng(3))
    b, sig_b = add_gaussian_noise(clean, snr=150, rng=np.random.default_rng(3))
    assert sig_a > sig_b
    za, zb = (a - clean) / sig_a, (b - clean) / sig_b
    np.testing.assert_allclose(za, zb, rtol=0, atol=1e-12)

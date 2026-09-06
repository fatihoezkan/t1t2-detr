"""Additive Gaussian noise for the signed inversion-recovery signal.

The clean signal is real-valued and negative at short TI, so the noise is plain additive
Gaussian; Rician noise (the magnitude-image model) would rectify the sign away. sigma is
either given directly or derived from SNR as max(|S_clean|) / SNR. Noise is drawn as a
standardised z and scaled, S_clean + sigma * z, so the fixed-SNR ladder can reuse one z per voxel.
"""
from __future__ import annotations

import numpy as np


def _sigma_from_snr(signal_clean: np.ndarray, snr: float) -> float:
    """Noise standard deviation implied by an SNR: max(|S_clean|) / snr."""
    # sigma = peak / SNR
    if snr <= 0:
        raise ValueError("SNR must be positive.")
    peak = float(np.max(np.abs(signal_clean)))
    if peak == 0:
        raise ValueError("Clean signal is all zeros; cannot scale noise.")
    return peak / snr


def add_gaussian_noise(signal_clean, snr, rng, sigma=None):
    """Return (S_clean + sigma * z, sigma) with z standard normal; sigma overrides snr if given.

    z is drawn first and then scaled, rather than calling rng.normal(0, sigma): two sigmas from
    the same rng state then share one z, which the paired fixed-SNR ladder relies on. The two
    forms agree bit for bit on the pinned NumPy, but that is not guaranteed across versions.
    """
    # derive sigma from the SNR unless it is given
    if sigma is None:
        sigma = _sigma_from_snr(signal_clean, snr)
    # standardised draw, then scaled
    z = rng.standard_normal(signal_clean.shape)
    signal_noisy = signal_clean + sigma * z
    return signal_noisy, sigma


# Name used by generate.py.
add_noise = add_gaussian_noise

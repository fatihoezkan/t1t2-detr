"""Forward model: the noise-free signal of a multi-compartment voxel under the IR-MSE protocol.

Each compartment contributes an inversion-recovery factor in T1 times an exponential decay in
T2, weighted by its signal fraction. t1t2/physics.py holds the training-side implementation of
the same equation; the two must stay in step.
"""

from __future__ import annotations

import numpy as np

from .protocol import Protocol


def simulate_clean_signal(
    protocol: Protocol,
    t1: np.ndarray,
    t2: np.ndarray,
    w: np.ndarray,
    m0: float = 1.0,
) -> np.ndarray:
    """Noise-free signal for one voxel, one value per protocol point.

        S_p = M0 * sum_c w_c * (1 - 2 exp(-TI_p/T1_c) + exp(-TR/T1_c)) * exp(-TE_p/T2_c)

    t1, t2, w are (K,) arrays for K compartments; times in ms, weights non-negative and summing
    to one. m0 is the overall amplitude; the generator leaves it at 1.0 because the training
    pipeline normalises each signal anyway. Returns a signed (n_points,) float64 array (64 for
    the shipped protocol); it is negative at short TI.
    """
    t1 = np.asarray(t1, dtype=np.float64).flatten()
    t2 = np.asarray(t2, dtype=np.float64).flatten()
    w = np.asarray(w, dtype=np.float64).flatten()

    if not (t1.shape == t2.shape == w.shape):
        raise ValueError(f"t1, t2, w must have same shape; got {t1.shape}, {t2.shape}, {w.shape}")
    if np.any(t1 <= 0) or np.any(t2 <= 0):
        raise ValueError("T1 and T2 must be strictly positive (ms).")
    if np.any(w < 0):
        raise ValueError("Weights must be nonnegative.")
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1; got {w.sum():.6f}")

    ti = protocol.ti[:, None]   # (64, 1)
    te = protocol.te[:, None]   # (64, 1)
    tr = protocol.tr

    t1_row = t1[None, :]        # (1, K)
    t2_row = t2[None, :]        # (1, K)

    inv_recovery = 1.0 - 2.0 * np.exp(-ti / t1_row) + np.exp(-tr / t1_row)   # (64, K)
    t2_decay     = np.exp(-te / t2_row)                                      # (64, K)
    per_comp     = inv_recovery * t2_decay                                   # (64, K)

    signal = m0 * (per_comp * w[None, :]).sum(axis=1)                        # (64,)
    return signal

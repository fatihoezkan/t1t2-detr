"""Forward signal model: compartments in, 64-point signal out.

The inversion-recovery multi-echo equation of the data generator, restated on the training
side. Two implementations: numpy, checked against the generator by a parity test, and a
differentiable torch version for the signal-consistency loss.

    S_p = M0 * sum_c  w_c * (1 - 2 exp(-TI_p / T1_c) + exp(-TR / T1_c)) * exp(-TE_p / T2_c)

The bracket is the inversion recovery (negative for short TI, which is why the data is
signed) and the trailing exponential is the T2 decay.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.io as sio

# The protocol file lives next to the generator that also reads it.
_DEFAULT_MAT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "datagen", "data", "ti_te_dict.mat")
)


@dataclass(frozen=True)
class Protocol:
    """64 ordered (TI, TE) pairs in ms plus a single TR."""

    ti: np.ndarray
    te: np.ndarray
    tr: float

    @property
    def n_points(self) -> int:
        """Return the number of measurements in the protocol."""
        return int(self.ti.shape[0])


def load_protocol(mat_path: str | None = None) -> Protocol:
    """Read TI, TE and TR from ti_te_dict.mat in stored order.

    Position p has to mean the same (TI_p, TE_p) at training and at inference, so the
    arrays are never sorted or regrouped.
    """
    d = sio.loadmat(mat_path or _DEFAULT_MAT)
    ti = np.asarray(d["ti"], dtype=np.float64).flatten()
    te = np.asarray(d["te"], dtype=np.float64).flatten()
    tr = float(np.asarray(d["tr"]).flatten()[0])
    if ti.shape != te.shape:
        raise ValueError(f"TI/TE shape mismatch: {ti.shape} vs {te.shape}")
    return Protocol(ti=ti, te=te, tr=tr)


def forward_numpy(protocol: Protocol, t1, t2, w, m0: float = 1.0) -> np.ndarray:
    """Noise-free signal for one voxel, shape (n_points,). T1, T2 in ms."""
    t1 = np.asarray(t1, np.float64).ravel()
    t2 = np.asarray(t2, np.float64).ravel()
    w = np.asarray(w, np.float64).ravel()
    ti = protocol.ti[:, None]                                  # (P, 1)
    te = protocol.te[:, None]
    inv = 1.0 - 2.0 * np.exp(-ti / t1[None, :]) + np.exp(-protocol.tr / t1[None, :])   # (P, K)
    dec = np.exp(-te / t2[None, :])                            # (P, K)
    return m0 * ((inv * dec) * w[None, :]).sum(axis=1)         # (P,)


def forward_torch(protocol: Protocol, params, mask=None, m0: float = 1.0):
    """Differentiable, batched forward_numpy.

    params is (B, K, 3) holding [T1_ms, T2_ms, weight]; the result is (B, n_points).
    `mask` (B, K) zeroes compartments that should not contribute. T1 and T2 are clamped
    away from zero because early in training the model can propose values at the lower
    bound, and exp(-TI/0) is not recoverable.
    """
    import torch

    dev = params.device
    ti = torch.as_tensor(protocol.ti, dtype=params.dtype, device=dev)   # (P,)
    te = torch.as_tensor(protocol.te, dtype=params.dtype, device=dev)
    tr = float(protocol.tr)
    t1 = params[..., 0].clamp(min=1e-6).unsqueeze(1)           # (B, 1, K)
    t2 = params[..., 1].clamp(min=1e-6).unsqueeze(1)
    w = params[..., 2].unsqueeze(1)                            # (B, 1, K)
    if mask is not None:
        w = w * mask.unsqueeze(1)
    ti = ti.view(1, -1, 1)                                     # (1, P, 1)
    te = te.view(1, -1, 1)
    inv = 1.0 - 2.0 * torch.exp(-ti / t1) + torch.exp(-tr / t1)   # (B, P, K)
    dec = torch.exp(-te / t2)
    return m0 * (inv * dec * w).sum(dim=-1)                    # (B, P)

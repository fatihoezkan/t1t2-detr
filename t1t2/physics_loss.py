"""Signal-consistency loss.

The prediction is pushed back through the forward model that produced the data and the
mismatch with a target signal is penalised (signal -> model -> compartments -> physics ->
signal). It is added to the Hungarian loss rather than replacing it. The Hungarian term
scores parameters given a matching. This term scores the whole predicted set through the
signal it produces and never matches anything.

Design choices:
  * Gating is soft. Every query contributes w_q * sigmoid(exist_logit_q), so the term is
    differentiable end to end and the existence head gets a physics gradient. At
    initialisation every query is about half open and the signal is over-predicted, which is
    why train.py ramps the weight in.
  * The resynthesised signal goes through the same per-voxel normalisation as the input;
    otherwise the two sides differ by an arbitrary per-voxel factor.
  * The target is either the noisy input ("noisy", needs no labels) or the noise-free
    forward model of the true parameters ("clean", simulation only).
  * The metric is MSE. The simulated signals are signed, so Gaussian noise and MSE are the
    right likelihood; a Rician term is for magnitude data and is not implemented.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics import Protocol, forward_torch, load_protocol

_TARGETS = ("noisy", "clean")


def _denorm_torch(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Torch counterpart of TargetNormalizer._inv: [0, 1] back to milliseconds."""
    llo, lhi = math.log(lo), math.log(hi)
    return torch.exp(llo + x * (lhi - llo))


def _signal_norm_torch(s: torch.Tensor) -> torch.Tensor:
    """Torch counterpart of data._normalize_signal; the two must stay in step."""
    # per-voxel peak magnitude, guarding an all-zero signal
    m = s.abs().amax(dim=1, keepdim=True)
    m = torch.where(m == 0, torch.ones_like(m), m)
    return s / m


class SignalConsistencyLoss(nn.Module):
    """Resynthesise the signal from a prediction and return the mean squared mismatch.

    forward(y_pred, X, y_true) is unweighted; train.py applies the lambda schedule.

        y_pred : (B, n_queries, 4)  [t1n, t2n, w, exist_logit], t1n/t2n in [0, 1]
        X      : (B, P)             the input signal, already divided by its peak
        y_true : (B, max_comp * 3)  normalised ground truth, used only for target="clean"
    """

    def __init__(self, data_cfg, loss_cfg, protocol: Protocol | None = None):
        super().__init__()
        # which signal the resynthesis is compared against
        self.target = loss_cfg.signal_consistency_target
        if self.target not in _TARGETS:
            raise ValueError(f"signal_consistency_target must be one of {_TARGETS}; got {self.target!r}")
        # protocol and the normaliser bounds needed to go back to milliseconds
        self.protocol = protocol or load_protocol()
        self.t1_lo, self.t1_hi = float(data_cfg.t1_min), float(data_cfg.t1_max)
        self.t2_lo, self.t2_hi = float(data_cfg.t2_min), float(data_cfg.t2_max)

    def _params_ms(self, t1n, t2n, w):
        """Stack normalised (t1n, t2n, w) into the (B, K, 3) millisecond table for forward_torch."""
        # [0, 1] -> ms for T1 and T2; the weights stay as they are
        t1 = _denorm_torch(t1n, self.t1_lo, self.t1_hi)
        t2 = _denorm_torch(t2n, self.t2_lo, self.t2_hi)
        return torch.stack([t1, t2, w], dim=-1)

    def synthesize(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Soft-gated resynthesis: (B, Q, 4) prediction to (B, P) normalised signal."""
        w_eff = y_pred[..., 2] * torch.sigmoid(y_pred[..., 3])          # (B, Q), soft gate
        # forward model, then the same per-voxel normalisation as the input
        params = self._params_ms(y_pred[..., 0], y_pred[..., 1], w_eff)
        s_hat = forward_torch(self.protocol, params)                    # (B, P)
        return _signal_norm_torch(s_hat)

    def _clean_target(self, y_true: torch.Tensor) -> torch.Tensor:
        """Noise-free signal from the ground-truth table. Padded slots carry w=0 and drop out."""
        with torch.no_grad():
            # ground-truth table (B, max_comp, 3) -> noise-free signal
            B, W = y_true.shape
            yt = y_true.reshape(B, W // 3, 3)
            params = self._params_ms(yt[..., 0], yt[..., 1], yt[..., 2])
            s = forward_torch(self.protocol, params)
            return _signal_norm_torch(s)

    def forward(self, y_pred, X, y_true):
        # resynthesise, pick the target, compare
        s_hat = self.synthesize(y_pred)
        tgt = X if self.target == "noisy" else self._clean_target(y_true)
        return F.mse_loss(s_hat, tgt)

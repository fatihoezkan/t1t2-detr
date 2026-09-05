"""The acquisition protocol: 64 (TI, TE) pairs in scanner order, plus TR.

The arrays are used exactly as stored and never sorted or regrouped: input position p must mean
the same (TI_p, TE_p) at generation time, at training time and at inference on a real scan.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.io as sio


@dataclass(frozen=True)
class Protocol:
    """The acquisition schedule: 64 ordered (TI, TE) pairs and one TR, all in ms."""

    ti: np.ndarray   # shape (64,)
    te: np.ndarray   # shape (64,)
    tr: float        # ms

    @property
    def n_points(self) -> int:
        return int(self.ti.shape[0])

    def summary(self) -> str:
        """One-line description for log output."""
        return (
            f"Protocol: {self.n_points} points, "
            f"{len(np.unique(self.ti))} unique TI "
            f"[{self.ti.min():.1f}, {self.ti.max():.1f}] ms, "
            f"{len(np.unique(self.te))} unique TE "
            f"[{self.te.min():.1f}, {self.te.max():.1f}] ms, "
            f"TR={self.tr:.0f} ms"
        )


# Protocol file shipped with the repository, resolved relative to __file__ so it does not depend
# on the working directory. Module-level so the dataset manifest can checksum it.
DEFAULT_MAT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "ti_te_dict.mat")
)


def load_protocol(mat_path: str | None = None) -> Protocol:
    """Load TI, TE and TR from ti_te_dict.mat, preserving the acquisition order."""
    if mat_path is None:
        mat_path = DEFAULT_MAT_PATH

    d = sio.loadmat(mat_path)
    ti = np.asarray(d["ti"], dtype=np.float64).flatten()
    te = np.asarray(d["te"], dtype=np.float64).flatten()
    tr = float(np.asarray(d["tr"]).flatten()[0])

    if ti.shape != te.shape:
        raise ValueError(f"TI and TE shape mismatch: {ti.shape} vs {te.shape}")
    if ti.ndim != 1:
        raise ValueError(f"TI must be 1D after flatten, got {ti.shape}")
    if not np.all(ti > 0) or not np.all(te > 0) or tr <= 0:
        raise ValueError("TI, TE, TR must be strictly positive")

    return Protocol(ti=ti, te=te, tr=tr)

"""Parquet files from the generator, turned into the tensors the model consumes.

The generator writes one voxel per row: the signal (S_1..S_64), a fixed number of
compartment slots (T1_i, T2_i, w_i, NaN-padded), and n_comp. Three decisions here matter:
the number of slots is read from the columns and never configured (see infer_max_comp);
T1/T2 are mapped into [0, 1] because the heads end in a sigmoid (TargetNormalizer); and
empty slots are zero-filled, which is safe only because n_comp travels with the batch and
the loss slices the padding away. The raw millisecond targets stay on the dataset so that
evaluation can report errors in real units.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

_NORM_MODES = ("identity", "linear_minmax", "log_minmax")


class TargetNormalizer:
    """Maps T1 and T2 between milliseconds and the model's [0, 1] range.

    Log spacing is the default. Relaxation times span roughly 20 ms to a few thousand ms,
    so a linear map squeezes the short-T2 pools into the first few percent of the range.
    The log map spreads them evenly, matches the generator's log-uniform sampling, and puts
    the T1 and T2 loss terms on the same footing.
    """

    def __init__(self, mode="log_minmax", t1_min=100.0, t1_max=7000.0, t2_min=5.0, t2_max=4000.0):
        if mode not in _NORM_MODES:
            raise ValueError(f"unknown normalization {mode!r}; expected one of {_NORM_MODES}")
        self.mode = mode
        self.t1_min, self.t1_max = float(t1_min), float(t1_max)
        self.t2_min, self.t2_max = float(t2_min), float(t2_max)

    @classmethod
    def from_config(cls, data_cfg) -> "TargetNormalizer":
        return cls(data_cfg.normalization, data_cfg.t1_min, data_cfg.t1_max,
                   data_cfg.t2_min, data_cfg.t2_max)

    def _fwd(self, x, lo, hi, clip):
        """Milliseconds to [0, 1]."""
        x = np.asarray(x, dtype=np.float64)
        if self.mode == "identity":
            out = x
        elif self.mode == "linear_minmax":
            out = (x - lo) / (hi - lo)
        else:  # log_minmax
            out = (np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))
        # Targets are clamped: a target outside [0, 1] is one the sigmoid can never reach.
        # Predictions are never clamped here, that would distort them before inversion.
        if clip and self.mode != "identity":
            out = np.clip(out, 0.0, 1.0)
        return out

    def _inv(self, y, lo, hi):
        """[0, 1] back to milliseconds, the exact inverse of _fwd."""
        y = np.asarray(y, dtype=np.float64)
        if self.mode == "identity":
            return y
        if self.mode == "linear_minmax":
            return y * (hi - lo) + lo
        return np.exp(y * (np.log(hi) - np.log(lo)) + np.log(lo))

    def normalize_t1(self, t1, clip=True):
        return self._fwd(t1, self.t1_min, self.t1_max, clip)

    def normalize_t2(self, t2, clip=True):
        return self._fwd(t2, self.t2_min, self.t2_max, clip)

    def denormalize_t1(self, x):
        return self._inv(x, self.t1_min, self.t1_max)

    def denormalize_t2(self, x):
        return self._inv(x, self.t2_min, self.t2_max)


def _signal_columns(n_inputs: int) -> list[str]:
    """Signal column names in acquisition order. Column p must mean the same (TI_p, TE_p)
    at training time and later on a real scan, so the order is never changed."""
    return [f"S_{i + 1}" for i in range(n_inputs)]


def _apply_signal_norm(X: np.ndarray, mode: str) -> np.ndarray:
    """Per-voxel rescaling of the input signal.

    Real scans arrive at an arbitrary scale (receiver gain, coil sensitivity), so the same
    transform has to be applied to synthetic and real data. "max" divides each voxel by its
    peak magnitude, "first" by its first sample, "none" leaves it alone.
    """
    if mode == "none":
        return X
    if mode == "max":
        m = np.max(np.abs(X), axis=1, keepdims=True)
        m[m == 0] = 1.0
        return (X / m).astype(np.float32)
    if mode == "first":
        f = X[:, :1].copy()
        f[f == 0] = 1.0
        return (X / f).astype(np.float32)
    raise ValueError(f"unknown signal_norm {mode!r}; expected none|max|first")


def infer_max_comp(df: pd.DataFrame) -> int:
    """Read the width of the ground-truth table from the column names.

    This used to be a config field. A stale `max_comp: 3` on four-compartment data gave a
    model that could not count past three: slicing four columns out of a three-column array
    returns three without an error, and the metrics still looked plausible. Reading the width
    from the data removes that failure mode.
    """
    idx = {}
    for fam in ("T1", "T2", "w"):
        found = sorted(
            int(m.group(1))
            for m in (re.fullmatch(rf"{fam}_(\d+)", c) for c in df.columns)
            if m
        )
        if not found:
            raise ValueError(f"no {fam}_* ground-truth columns found; is this a voxel dataset?")
        idx[fam] = found

    if not (idx["T1"] == idx["T2"] == idx["w"]):
        raise ValueError(
            f"T1/T2/w column families disagree: T1{idx['T1']}, T2{idx['T2']}, w{idx['w']}. "
            "They must share identical indices."
        )
    k = len(idx["T1"])
    if idx["T1"] != list(range(1, k + 1)):
        raise ValueError(f"ground-truth columns must be contiguous 1..K; got {idx['T1']}")
    return k


def _read_frames(paths, limit: int | None) -> pd.DataFrame:
    """Read one or more parquet files into a single frame with a shared schema.

    `limit` is a total across all paths, split evenly between them. A plain head slice
    would spend the whole budget on the first file, and since the files are split by
    compartment count a smoke run would then see only single-compartment voxels.
    """
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not paths:
        raise ValueError("no dataset path given")

    per, extra = (None, 0) if limit is None else divmod(limit, len(paths))
    frames = []
    for i, p in enumerate(paths):
        df = pd.read_parquet(p)
        if limit is not None:
            take = per + (1 if i < extra else 0)
            df = df.iloc[:take]
        frames.append(df.reset_index(drop=True))

    cols = frames[0].columns
    for p, df in zip(paths[1:], frames[1:]):
        if not df.columns.equals(cols):
            raise ValueError(f"{p} has a different schema from {paths[0]}; cannot combine them.")
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


class VoxelDataset(Dataset):
    """One or more parquet splits held in memory as tensors.

    Everything is loaded up front; the splits fit in RAM and training is faster for it.
    A list of paths is concatenated, which is how the per-count files become one balanced
    dataset. `limit` caps the number of voxels for tests and smoke runs.
    """

    def __init__(self, path, cfg, normalizer: TargetNormalizer | None = None, limit: int | None = None):
        self.cfg = cfg
        self.normalizer = normalizer or TargetNormalizer.from_config(cfg)
        n_in = cfg.n_inputs

        df = _read_frames(path, limit)
        max_c = self.max_comp = infer_max_comp(df)

        observed = int(df["n_comp"].max()) if len(df) else 0
        if observed > max_c:
            raise ValueError(
                f"data has n_comp up to {observed} but only {max_c} ground-truth column slots; "
                "the loss would silently supervise only the first slots."
            )

        # copy=True: pandas may hand back a read-only view, and torch.from_numpy on a
        # non-writable buffer is undefined behaviour once anything writes to it.
        X = df[_signal_columns(n_in)].to_numpy(np.float32, copy=True)
        X = _apply_signal_norm(X, cfg.signal_norm)

        n_comp = df["n_comp"].to_numpy(np.int64)
        t1 = np.stack([df[f"T1_{i + 1}"].to_numpy(np.float64) for i in range(max_c)], axis=1)
        t2 = np.stack([df[f"T2_{i + 1}"].to_numpy(np.float64) for i in range(max_c)], axis=1)
        w = np.stack([df[f"w_{i + 1}"].to_numpy(np.float64) for i in range(max_c)], axis=1)

        t1n = self.normalizer.normalize_t1(t1)
        t2n = self.normalizer.normalize_t2(t2)
        target = np.stack([t1n, t2n, w], axis=2)            # (N, max_comp, 3)
        target = np.nan_to_num(target, nan=0.0)             # empty slots become zeros
        y = target.reshape(len(df), max_c * 3).astype(np.float32)

        # Raw millisecond targets for evaluation.
        self.raw_t1, self.raw_t2, self.raw_w = t1, t2, w

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.n_comp = torch.from_numpy(n_comp.astype(np.int16))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.n_comp[i]


def make_dataloader(path, cfg, batch_size, shuffle, normalizer=None, num_workers=0, limit=None):
    """Build a DataLoader and return it with its dataset (evaluation needs the raw targets)."""
    ds = VoxelDataset(path, cfg, normalizer, limit=limit)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, ds

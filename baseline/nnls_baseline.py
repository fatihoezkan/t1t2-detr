"""Classical reference: regularised NNLS T1-T2 spectrum (2D inverse Laplace transform) per voxel.

Ported from Sebastian Endt's `baseline/nnls_baseline.py` in DETR_T1T2 (October 2026) to this
repository's data format and physics module, so that the classical method and every DETR run are
scored by the same ND/mAP code on the same test voxels. The spectrum solver, the chi2 rule for the
regularisation weight and the peak segmentation are his; what changed is the data loader, the
kernel (built from t1t2.physics so the atoms are this repository's forward model) and the output
format (npz of per-slot params and confidences, the same table shape as eval.detr_query_outputs).

Per voxel:
  1. Spectrum f >= 0 on a log-spaced (T1, T2) grid restricted to T2 < T1 (the prior of the
     training data):  min ||K f - s||^2 + lam^2 ||f||^2, solved as NNLS on [K; lam I] f = [s; 0].
  2. lam by the chi2 rule of multi-exponential relaxometry (Whittall and MacKay 1989; DECAES):
     the misfit may rise by chi2_factor above that of unregularised NNLS. Needs neither the noise
     level nor the ground truth.
  3. Peaks: every grid cell climbs to the local maximum of its 3x3 neighbourhood; cells sharing a
     maximum form one peak, whose fraction is its share of the spectrum and whose (T1, T2) is the
     amplitude-weighted geometric mean.
  4. The n_slots largest peaks are written as slots with confidence = signal fraction, so mAP
     ranks NNLS peaks by fraction exactly as it ranks DETR queries by existence probability.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import nnls

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from t1t2.physics import load_protocol, forward_numpy  # noqa: E402

MAX_COMP = 4



def _portable(value):
    """Path recorded in info.json without the machine's home directory.

    Repo-relative when the path is inside the repository, otherwise the last two components
    prefixed with "external:", since an absolute data path on one machine is not reproducible
    information anyway (the family and the splits are what identify the data).
    """
    if not isinstance(value, str) or not os.path.isabs(value):
        return value
    try:
        rel = os.path.relpath(value, REPO_ROOT)
    except ValueError:
        rel = value
    if not rel.startswith(".."):
        return rel
    parts = [q for q in os.path.normpath(value).split(os.sep) if q]
    return "external:" + "/".join(parts[-2:])


# ---------------------------------------------------------------- spectrum
def build_kernel(protocol, t1_range, t2_range, n_grid):
    """Returns K (P, M) for the M grid atoms with T2 < T1, the validity mask and the axes.

    K[:, m] is this repository's forward signal of a single unit-weight compartment, so the
    column is asserted against t1t2.physics.forward_numpy for a few atoms.
    """
    t1 = np.geomspace(*t1_range, n_grid)
    t2 = np.geomspace(*t2_range, n_grid)
    T1, T2 = np.meshgrid(t1, t2, indexing="ij")
    valid = T2 < T1
    a1, a2 = T1[valid], T2[valid]
    ti = np.asarray(protocol.ti, dtype=np.float64)[:, None]
    te = np.asarray(protocol.te, dtype=np.float64)[:, None]
    inv = 1.0 - 2.0 * np.exp(-ti / a1[None, :]) + np.exp(-float(protocol.tr) / a1[None, :])
    K = inv * np.exp(-te / a2[None, :])
    for m in (0, len(a1) // 2, len(a1) - 1):
        ref = forward_numpy(protocol, np.array([a1[m]]), np.array([a2[m]]), np.array([1.0]))
        assert np.allclose(K[:, m], ref, rtol=1e-12, atol=1e-12), "kernel differs from forward_numpy"
    return K, valid, t1, t2


def solve(K, s, lam):
    """Tikhonov-regularised NNLS. Returns (f, chi2)."""
    M = K.shape[1]
    if lam > 0:
        f, _ = nnls(np.vstack([K, lam * np.eye(M)]), np.r_[s, np.zeros(M)], maxiter=50 * M)
    else:
        f, _ = nnls(K, s, maxiter=50 * M)
    r = K @ f - s
    return f, float(r @ r)


def solve_chi2(K, s, factor, lam_range=(1e-5, 10.0), n_iter=12, tol=0.005):
    """Largest lam with chi2(lam) <= factor * chi2(0), by bisection in log(lam)."""
    f0, c0 = solve(K, s, 0.0)
    target = factor * max(c0, 1e-30)
    lo, hi = np.log(lam_range[0]), np.log(lam_range[1])
    best = (f0, 0.0)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        f, c = solve(K, s, np.exp(mid))
        if c <= target:
            best, lo = (f, float(np.exp(mid))), mid
            if c >= target / (1 + tol):
                break
        else:
            hi = mid
    return best


# ------------------------------------------------------------------- peaks
def peaks(F, t1, t2, rel_floor=1e-3):
    """Steepest-ascent segmentation. Returns (n_peaks, 3) [T1, T2, fraction], fraction-sorted."""
    total = F.sum()
    if total <= 0:
        return np.zeros((0, 3))
    n1, n2 = F.shape
    mask = F > rel_floor * F.max()
    val = np.pad(np.where(mask, F, -1.0), 1, constant_values=-1.0)
    idx = np.pad(np.arange(n1 * n2).reshape(n1, n2), 1, constant_values=-1)
    shifts = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    vals = np.stack([val[1 + dx:1 + dx + n1, 1 + dy:1 + dy + n2] for dx, dy in shifts])
    ids = np.stack([idx[1 + dx:1 + dx + n1, 1 + dy:1 + dy + n2] for dx, dy in shifts])
    up = np.take_along_axis(ids, vals.argmax(0)[None], 0)[0]
    ptr = np.where(mask & (vals.max(0) > F), up, idx[1:-1, 1:-1]).ravel()
    while True:
        nxt = ptr[ptr]
        if np.array_equal(nxt, ptr):
            break
        ptr = nxt
    m = mask.ravel()
    roots, lab = np.unique(ptr[m], return_inverse=True)
    w = F.ravel()[m]
    L1, L2 = np.meshgrid(np.log(t1), np.log(t2), indexing="ij")
    amp = np.bincount(lab, weights=w)
    out = np.stack([np.exp(np.bincount(lab, weights=w * L1.ravel()[m]) / amp),
                    np.exp(np.bincount(lab, weights=w * L2.ravel()[m]) / amp),
                    amp / total], 1)
    return out[np.argsort(-out[:, 2])]


# ------------------------------------------------------------------ worker
_W = {}


def _init(cfg):
    K, valid, t1, t2 = build_kernel(cfg["protocol"], cfg["t1_range"], cfg["t2_range"], cfg["n_grid"])
    _W.update(cfg, K=K, valid=valid, t1=t1, t2=t2)


def _process(s):
    s = s / max(np.linalg.norm(s), 1e-30)          # scale-free; lam is relative to unit-norm data
    t = time.perf_counter()
    if _W["reg"] == "chi2":
        f, lam = solve_chi2(_W["K"], s, _W["chi2_factor"])
    else:
        lam = _W["lam"] if _W["reg"] == "fixed" else 0.0
        f, _ = solve(_W["K"], s, lam)
    F = np.zeros(_W["valid"].shape)
    F[_W["valid"]] = f
    pk = peaks(F, _W["t1"], _W["t2"])[:_W["n_slots"]]
    params = np.zeros((_W["n_slots"], 3))
    params[:, 0] = _W["t1_range"][0] * 1e-3        # unused slots far outside the range
    params[:, 1] = _W["t2_range"][0] * 1e-3
    prob = np.zeros(_W["n_slots"])
    params[:len(pk)] = pk
    prob[:len(pk)] = pk[:, 2]                      # confidence = signal fraction
    return params, prob, lam, time.perf_counter() - t


# -------------------------------------------------------------------- data
def load_split(data_root, family, split, n_samples=None):
    """Concatenate the per-compartment-count parquets in n1, n2, n3 order, as the thesis does."""
    frames = []
    for sub in ("n1", "n2", "n3"):
        p = os.path.join(data_root, family, sub, f"{split}.parquet")
        if os.path.exists(p):
            frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    if n_samples:
        df = df.iloc[:n_samples].reset_index(drop=True)
    S = df[[f"S_{i}" for i in range(1, 65)]].to_numpy(np.float64)
    T = np.full((len(df), MAX_COMP, 3), np.nan)
    for k in range(1, MAX_COMP + 1):
        T[:, k - 1, 0] = df[f"T1_{k}"].to_numpy()
        T[:, k - 1, 1] = df[f"T2_{k}"].to_numpy()
        T[:, k - 1, 2] = df[f"w_{k}"].to_numpy()
    return S, T, df["n_comp"].to_numpy(int), df["snr"].to_numpy(float)


def run_split(S, cfg, workers):
    with Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        res = pool.map(_process, S, chunksize=max(1, len(S) // (4 * workers)))
    return (np.stack([r[0] for r in res]), np.stack([r[1] for r in res]),
            np.array([r[2] for r in res]), np.array([r[3] for r in res]))


# -------------------------------------------------------------------- main
def get_args_parser():
    p = argparse.ArgumentParser("Regularised NNLS T1-T2 baseline on this repository's data")
    p.add_argument("--data_root", default=os.path.join(REPO_ROOT, "data"))
    p.add_argument("--family", default="t1_3500_t2_500_100k")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--n_samples", default=0, type=int, help="first N voxels of each split (0 = all)")
    p.add_argument("--reg", default="chi2", choices=["chi2", "fixed", "none"])
    p.add_argument("--chi2_factor", default=1.02, type=float)
    p.add_argument("--lam", default=0.05, type=float)
    p.add_argument("--n_grid", default=30, type=int, help="grid points per axis (log-spaced)")
    p.add_argument("--n_slots", default=10, type=int, help="max. peaks reported per voxel")
    p.add_argument("--t1_range", nargs=2, default=[50.0, 3500.0], type=float)
    p.add_argument("--t2_range", nargs=2, default=[5.0, 500.0], type=float)
    p.add_argument("--protocol_file", default=os.path.join(REPO_ROOT, "datagen", "data", "ti_te_dict.mat"))
    p.add_argument("--run_name", default=None)
    p.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results", "nnls"))
    p.add_argument("--workers", default=os.cpu_count(), type=int)
    return p


def main(args):
    run = args.run_name or f"nnls_{args.reg}_g{args.n_grid}"
    out = os.path.join(args.out_dir, run)
    os.makedirs(out, exist_ok=True)
    protocol = load_protocol(args.protocol_file)
    cfg = {"protocol": protocol, "t1_range": args.t1_range, "t2_range": args.t2_range,
           "n_grid": args.n_grid, "reg": args.reg, "chi2_factor": args.chi2_factor,
           "lam": args.lam, "n_slots": args.n_slots}
    n_atoms = int(build_kernel(protocol, args.t1_range, args.t2_range, args.n_grid)[0].shape[1])
    info = {k: _portable(v) if k.endswith(("_root", "_dir", "_file")) else v
            for k, v in vars(args).items()}
    info.update(method="nnls", n_atoms=n_atoms, ported_from="Endt DETR_T1T2 baseline/nnls_baseline.py")

    for split in args.splits:
        S, T, n_comp, snr = load_split(args.data_root, args.family, split, args.n_samples or None)
        t0 = time.time()
        params, prob, lam, sec = run_split(S, cfg, args.workers)
        np.savez_compressed(os.path.join(out, f"{split}_predictions.npz"),
                            params=params, exist_prob=prob, trues=T, n_comp=n_comp, snr=snr,
                            lam=lam, cpu_s=sec)
        info[split] = {"n_voxels": int(len(S)), "wall_s": round(time.time() - t0, 1),
                       "cpu_s_per_voxel": float(sec.mean()),
                       "lambda_median": float(np.median(lam)),
                       "peaks_per_voxel_mean": float((prob > 0).sum(1).mean())}
        print(f"{split}: {len(S)} voxels, {info[split]['wall_s']} s wall, "
              f"{info[split]['cpu_s_per_voxel']:.3f} s/voxel cpu, "
              f"{info[split]['peaks_per_voxel_mean']:.2f} peaks/voxel")
    with open(os.path.join(out, "info.json"), "w") as fh:
        json.dump(info, fh, indent=2, default=str)


if __name__ == "__main__":
    main(get_args_parser().parse_args())

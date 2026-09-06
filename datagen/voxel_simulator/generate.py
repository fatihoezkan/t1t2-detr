"""Synthetic voxel dataset generation (the CLI is run_generator.py, one level up).

    generate_voxel           one voxel  -> VoxelSpec plus noisy signal
    generate_one             one voxel  -> one dataset row (dict)
    generate_dataset         N voxels   -> DataFrame
    generate_dataset_family  one config -> train/val/test and fixed-SNR parquet files + manifest
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .noise import add_noise
from .physics import simulate_clean_signal
from .protocol import DEFAULT_MAT_PATH, Protocol, load_protocol
from .sampler import (
    DEFAULT_SAMPLING,
    MAX_COMP,
    MIN_WEIGHT,
    SAMPLING_MODES,
    SNR_MAX,
    SNR_MIN,
    SPLIT_SNR_LADDER,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    STREAM_NOISE,
    STREAM_PARAMS,
    STREAM_SNR,
    T1_RANGE,
    T2_RANGE,
    VoxelSpec,
    sample_voxel_spec,
    validate_ranges,
    validate_sampling,
    voxel_rng,
)


def _check_unique_names(names: list[str], what: str) -> None:
    """Raise if two ladder levels would map to the same output filename."""
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(
            f"{what} produces duplicate output names {sorted(dupes)}; each level must map to a "
            "distinct filename or one would silently overwrite the other."
        )


@dataclass(frozen=True)
class GeneratedVoxel:
    """One simulated voxel, before it is flattened into a dataset row."""

    voxel_id: int
    spec: VoxelSpec
    signal: np.ndarray
    sigma: float


def simulate_voxel_signal(
    protocol: Protocol,
    spec: VoxelSpec,
    base_seed: int,
    split_code: int,
    voxel_id: int,
    noise_sigma: float | None = None,
) -> tuple[np.ndarray, float]:
    """Simulate the noisy signal for one VoxelSpec; returns (signal, sigma).

    The noise stream is keyed like the parameter stream but with STREAM_NOISE, so the two are
    independent and both are reproducible from (base_seed, n_comp, split_code, voxel_id).
    """
    # clean forward signal, then noise from the voxel's own noise stream
    clean = simulate_clean_signal(protocol, spec.t1, spec.t2, spec.w)
    rng = voxel_rng(base_seed, spec.n_comp, split_code, voxel_id, STREAM_NOISE)
    return add_noise(clean, spec.snr, rng, sigma=noise_sigma)


def generate_voxel(
    voxel_id: int,
    n_comp: int,
    protocol: Protocol,
    base_seed: int = 0,
    split_code: int = SPLIT_TRAIN,
    snr_min: float = SNR_MIN,
    snr_max: float = SNR_MAX,
    snr: float | None = None,
    t1_range: tuple[float, float] = T1_RANGE,
    t2_range: tuple[float, float] = T2_RANGE,
    noise_sigma: float | None = None,
    sampling: str = DEFAULT_SAMPLING,
) -> GeneratedVoxel:
    """Sample the (T1, T2, w, SNR) targets for one voxel and simulate its measured signal.

    All arguments are forwarded by keyword; a positional call once dropped n_comp silently and
    made --n-comp a no-op. `sampling` affects only the parameter stream; the noise stream is
    keyed independently.
    """
    # ground truth first, then the noisy signal
    spec = sample_voxel_spec(
        voxel_id=voxel_id,
        n_comp=n_comp,
        base_seed=base_seed,
        split_code=split_code,
        snr_min=snr_min,
        snr_max=snr_max,
        snr=snr,
        t1_range=t1_range,
        t2_range=t2_range,
        sampling=sampling,
    )
    signal, sigma = simulate_voxel_signal(
        protocol, spec, base_seed=base_seed, split_code=split_code, voxel_id=voxel_id,
        noise_sigma=noise_sigma,
    )
    return GeneratedVoxel(voxel_id=voxel_id, spec=spec, signal=signal, sigma=sigma)


def voxel_to_row(voxel: GeneratedVoxel, protocol: Protocol, noise_sigma: float | None = None) -> dict:
    """Flatten a GeneratedVoxel into one row of the tabular schema.

    The ground-truth columns are always MAX_COMP wide, NaN-padded beyond n_comp, so per-count
    files share one schema and can be concatenated.
    """
    # voxel-level columns
    spec = voxel.spec
    row: dict = {
        "voxel_id": voxel.voxel_id,
        "snr": np.nan if noise_sigma is not None else spec.snr,
        "sigma": voxel.sigma,
        "n_comp": spec.n_comp,
    }
    # compartment slots 1..MAX_COMP, NaN beyond n_comp
    for i in range(MAX_COMP):
        if i < spec.n_comp:
            row[f"T1_{i+1}"] = float(spec.t1[i])
            row[f"T2_{i+1}"] = float(spec.t2[i])
            row[f"w_{i+1}"]  = float(spec.w[i])
        else:
            row[f"T1_{i+1}"] = np.nan
            row[f"T2_{i+1}"] = np.nan
            row[f"w_{i+1}"]  = np.nan
    # signal columns S_1..S_P as float32
    for p in range(protocol.n_points):
        row[f"S_{p+1}"] = np.float32(voxel.signal[p])
    return row


def generate_one(
    voxel_id: int,
    n_comp: int,
    protocol: Protocol,
    base_seed: int = 0,
    split_code: int = SPLIT_TRAIN,
    snr_min: float = SNR_MIN,
    snr_max: float = SNR_MAX,
    snr: float | None = None,
    t1_range: tuple[float, float] = T1_RANGE,
    t2_range: tuple[float, float] = T2_RANGE,
    noise_sigma: float | None = None,
    sampling: str = DEFAULT_SAMPLING,
) -> dict:
    """Build one dataset row.

    If noise_sigma is given it overrides the per-voxel SNR and the snr column is written as NaN.
    `sampling` is not stored per row; it is constant per family and recorded in the manifest
    under physics.sampling.
    """
    # one voxel -> one row
    voxel = generate_voxel(
        voxel_id=voxel_id,
        n_comp=n_comp,
        protocol=protocol,
        base_seed=base_seed,
        split_code=split_code,
        snr_min=snr_min,
        snr_max=snr_max,
        snr=snr,
        t1_range=t1_range,
        t2_range=t2_range,
        noise_sigma=noise_sigma,
        sampling=sampling,
    )
    return voxel_to_row(voxel, protocol, noise_sigma)


def generate_dataset(
    n_voxels: int,
    n_comp: int,
    base_seed: int = 0,
    split_code: int = SPLIT_TRAIN,
    protocol: Optional[Protocol] = None,
    snr_min: float = SNR_MIN,
    snr_max: float = SNR_MAX,
    snr: float | None = None,
    t1_range: tuple[float, float] = T1_RANGE,
    t2_range: tuple[float, float] = T2_RANGE,
    noise_sigma: float | None = None,
    sampling: str = DEFAULT_SAMPLING,
) -> pd.DataFrame:
    """Generate n_voxels rows, each with exactly n_comp compartments.

    Noise level: SNR drawn in [snr_min, snr_max], pinned by `snr`, or absolute via noise_sigma.
    Two datasets that differ only in `sampling` are not independent samples: the same key
    consumes the same uniforms in both. Independent families need different base_seed values.
    """
    # one row per voxel id, all with the same n_comp
    if protocol is None:
        protocol = load_protocol()
    rows = [
        generate_one(
            voxel_id=i,
            n_comp=n_comp,
            protocol=protocol,
            base_seed=base_seed,
            split_code=split_code,
            snr_min=snr_min,
            snr_max=snr_max,
            snr=snr,
            t1_range=t1_range,
            t2_range=t2_range,
            noise_sigma=noise_sigma,
            sampling=sampling,
        )
        for i in range(n_voxels)
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Dataset family: turn one config into the full set of train/val/test + fixed-SNR files.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetFamilyConfig:
    """Everything needed to generate one train/val/test family plus its fixed-SNR ladder."""

    out_dir: str | Path = "output/data"
    # Every voxel in the family has exactly this many compartments. One family per count keeps
    # the per-count splits exactly balanced.
    n_comp: int = 1
    # Base seed for the whole family. Different base seeds give independent datasets.
    seed: int = 0
    overwrite: bool = False
    n_train: int = 250_000
    n_val: int = 25_000
    n_test: int = 25_000
    n_per_snr: int = 12_500
    # Train-SNR bounds default to the sampler constants so the two cannot drift apart.
    snr_min: float = SNR_MIN
    snr_max: float = SNR_MAX
    # The ladder deliberately reaches below snr_min: SNR 20 is an extrapolation test.
    snr_ladder: tuple[float, ...] = (20, 40, 60, 100, 150)
    t1_range: tuple[float, float] = T1_RANGE
    t2_range: tuple[float, float] = T2_RANGE
    # (T1, T2) draw scheme under T2 < T1; see sampler.sample_random_compartment. "rejection" is
    # what the reported runs used and must stay the default. Recorded in the manifest.
    sampling: str = DEFAULT_SAMPLING
    # Absolute-noise mode: if set, the whole family uses Gaussian noise of this std (SNR is
    # ignored) and the robustness test sets come from sigma_ladder instead of snr_ladder.
    noise_sigma: float | None = None
    sigma_ladder: tuple[float, ...] = (0.05, 0.1, 0.2)

    def __post_init__(self) -> None:
        """Check dataset settings before generating any voxels."""
        # Validate up front so a bad config fails before any voxel is generated.
        if min(self.n_train, self.n_val, self.n_test, self.n_per_snr) < 0:
            raise ValueError("dataset sizes must be nonnegative")
        if not 1 <= self.n_comp <= MAX_COMP:
            raise ValueError(f"n_comp must be in 1..{MAX_COMP}; got {self.n_comp}")
        if self.n_comp * MIN_WEIGHT >= 1.0:
            raise ValueError(
                f"n_comp={self.n_comp} x MIN_WEIGHT={MIN_WEIGHT} >= 1: no valid weights exist."
            )
        # The range check is mode-aware: "t1_log_uniform" additionally needs t2_min < t1_min.
        validate_sampling(self.sampling)
        validate_ranges(self.t1_range, self.t2_range, self.sampling)
        if self.noise_sigma is not None:
            if self.noise_sigma <= 0 or any(s <= 0 for s in self.sigma_ladder):
                raise ValueError("sigma values must be positive")
            _check_unique_names([f"test_sigma{s:g}" for s in self.sigma_ladder], "sigma_ladder")
        else:
            if self.snr_min <= 0 or self.snr_max <= 0:
                raise ValueError("SNR bounds must be positive")
            if self.snr_min > self.snr_max:
                raise ValueError("snr_min must be <= snr_max")
            if any(snr <= 0 for snr in self.snr_ladder):
                raise ValueError("all fixed SNR values must be positive")
            # Filenames use int(snr), so 20.5 and 20.7 would both become test_snr20.
            _check_unique_names([f"test_snr{int(s)}" for s in self.snr_ladder], "snr_ladder")


@dataclass(frozen=True)
class DatasetJob:
    """One split to generate: output name, size, split code and noise setting."""

    name: str
    n_voxels: int
    split_code: int
    snr_min: float = SNR_MIN
    snr_max: float = SNR_MAX
    snr: float | None = None            # pinned SNR (fixed-SNR ladder); None => drawn per voxel
    noise_sigma: float | None = None


def smoke_config(config: DatasetFamilyConfig) -> DatasetFamilyConfig:
    """Shrink a config to something that finishes in seconds, for a local check."""
    return replace(config, n_train=2_000, n_val=500, n_test=500, n_per_snr=300)


def build_dataset_jobs(config: DatasetFamilyConfig) -> list[DatasetJob]:
    """Build the list of split jobs for one config.

    Splits are separated by split_code, which is part of every RNG key, so they cannot overlap
    at any size. All fixed-SNR rungs share SPLIT_SNR_LADDER and pin their SNR; since SNR has its
    own stream, every rung holds the same voxels with the same standardised noise and differs
    only in amplitude (a paired comparison).

    Checking that property from the written files needs a tolerance: ground truth is float64
    and matches exactly, but signals are float32, and z = (S - S_clean) / sigma amplifies the
    storage rounding by 1/sigma (about 2e-5 at SNR 150). Use atol around 1e-4.
    """
    # absolute-noise mode: the same sigma for train/val/test, one test set per ladder sigma
    if config.noise_sigma is not None:      # absolute-noise mode
        s = config.noise_sigma
        jobs = [
            DatasetJob("train", config.n_train, SPLIT_TRAIN, noise_sigma=s),
            DatasetJob("val", config.n_val, SPLIT_VAL, noise_sigma=s),
            DatasetJob("test", config.n_test, SPLIT_TEST, noise_sigma=s),
        ]
        for sig in config.sigma_ladder:
            jobs.append(
                DatasetJob(f"test_sigma{sig:g}", config.n_per_snr, SPLIT_SNR_LADDER,
                           noise_sigma=float(sig))
            )
        return jobs

    # SNR mode: random SNR for train/val/test, one pinned-SNR test set per rung
    jobs = [
        DatasetJob("train", config.n_train, SPLIT_TRAIN, config.snr_min, config.snr_max),
        DatasetJob("val", config.n_val, SPLIT_VAL, config.snr_min, config.snr_max),
        DatasetJob("test", config.n_test, SPLIT_TEST, config.snr_min, config.snr_max),
    ]
    for snr in config.snr_ladder:
        jobs.append(
            DatasetJob(f"test_snr{int(snr)}", config.n_per_snr, SPLIT_SNR_LADDER,
                       config.snr_min, config.snr_max, snr=float(snr))
        )
    return jobs


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write to a temporary file and rename, so a crash never leaves a partial parquet in place."""
    # write next to the target, then rename
    tmp = path.with_name(path.name + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)           # atomic within a filesystem
    finally:
        if tmp.exists():
            tmp.unlink()


def _git_state(repo_dir: Path) -> dict:
    """Git commit and dirty flag of repo_dir; fields are None if git is unavailable."""
    import subprocess

    def _run(*args: str) -> str | None:
        """Read command output, returning None if the command fails."""
        try:
            out = subprocess.run(args, cwd=repo_dir, capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    # HEAD commit and whether the tree has uncommitted changes
    commit = _run("git", "rev-parse", "HEAD")
    status = _run("git", "status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _dependency_versions() -> dict:
    """Interpreter and library versions.

    NumPy does not guarantee Generator bit streams across versions, so reproducibility is only
    claimed for the recorded environment.
    """
    import platform

    import pyarrow

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
    }


def build_manifest(config: DatasetFamilyConfig, jobs: list[DatasetJob], rows: dict[str, int]) -> dict:
    """Manifest dict: sizes, seeds, streams, ranges, sampling mode, protocol checksum, git, versions."""
    return {
        "n_comp": config.n_comp,
        "base_seed": config.seed,
        "max_comp": MAX_COMP,
        "splits": {
            j.name: {
                "rows": rows[j.name],
                "split_code": j.split_code,
                "snr": j.snr,
                "snr_min": None if j.noise_sigma is not None else j.snr_min,
                "snr_max": None if j.noise_sigma is not None else j.snr_max,
                "noise_sigma": j.noise_sigma,
            }
            for j in jobs
        },
        "streams": {"params": STREAM_PARAMS, "noise": STREAM_NOISE, "snr": STREAM_SNR},
        "physics": {
            "t1_range": list(config.t1_range),
            "t2_range": list(config.t2_range),
            "min_weight": MIN_WEIGHT,
            "noise": "additive_gaussian_signed",
            # The (T1, T2) marginals depend on this. Manifests written before the field existed
            # are all "rejection", the only scheme at the time.
            "sampling": config.sampling,
        },
        "protocol_sha256": _sha256(Path(DEFAULT_MAT_PATH)),
        "git": _git_state(Path(__file__).resolve().parents[2]),
        "dependencies": _dependency_versions(),
    }


def _sha256(path: Path) -> str:
    """SHA-256 of a file; used to pin the protocol file the data came from."""
    # hash in 64 KB chunks
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_dataset_family(config: DatasetFamilyConfig, *, verbose: bool = True) -> list[Path]:
    """Generate the train, val, test and fixed-SNR parquet files for one compartment count."""
    # protocol, output folder and the list of splits to write
    protocol = load_protocol()
    out_dir = Path(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    jobs = build_dataset_jobs(config)

    # Refuse to overwrite before generating anything.
    if not config.overwrite:
        existing = [out_dir / f"{j.name}.parquet" for j in jobs]
        existing = [p for p in existing if p.exists()]
        if existing:
            raise FileExistsError(
                f"{len(existing)} output file(s) already exist in {out_dir} "
                f"(e.g. {existing[0].name}). Pass overwrite=True / --overwrite to replace them."
            )

    # one line describing the family
    level = (f"sigma={config.noise_sigma:g}" if config.noise_sigma is not None
             else f"snr=[{config.snr_min:g},{config.snr_max:g}]")
    if verbose:
        print(protocol.summary())
        print(
            f"n_comp={config.n_comp} | base_seed={config.seed} | noise=gaussian | {level} | "
            f"T1{config.t1_range} T2{config.t2_range} | sampling={config.sampling}"
        )

    # generate and write each split
    written: list[Path] = []
    rows: dict[str, int] = {}
    total, t0 = 0, time.time()
    for job in jobs:
        t = time.time()
        df = generate_dataset(
            job.n_voxels,
            n_comp=config.n_comp,
            base_seed=config.seed,
            split_code=job.split_code,
            protocol=protocol,
            snr_min=job.snr_min,
            snr_max=job.snr_max,
            snr=job.snr,
            t1_range=config.t1_range,
            t2_range=config.t2_range,
            noise_sigma=job.noise_sigma,
            sampling=config.sampling,
        )

        # write atomically and record the row count
        path = out_dir / f"{job.name}.parquet"
        _write_parquet_atomic(df, path)
        written.append(path)
        rows[job.name] = len(df)
        total += job.n_voxels

        # per-split progress line
        if verbose:
            mb = os.path.getsize(path) / 1e6
            job_level = (f"sigma={job.noise_sigma:g}" if job.noise_sigma is not None
                         else (f"snr={job.snr:g}" if job.snr is not None
                               else f"snr=[{job.snr_min:g},{job.snr_max:g}]"))
            print(
                f"  {job.name:<14} n={job.n_voxels:>9,} split={job.split_code} "
                f"{job_level} -> {path} ({mb:.0f} MB, {time.time() - t:.1f}s)"
            )

    # Written last and only on full success, so the manifest doubles as a completion marker.
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(build_manifest(config, jobs, rows), f, indent=2)
    written.append(manifest_path)

    if verbose:
        print(f"done: {total:,} voxels in {len(written) - 1} files, "
              f"{time.time() - t0:.0f}s total -> {out_dir}")
    return written

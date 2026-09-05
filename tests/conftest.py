"""Shared setup: generates the small development dataset in data/dev/ on first use.

Several tests need real generated voxels (physics parity, parquet schema round trip), so the
family is generated once, fully seeded, and reused. Delete data/dev/ to force regeneration.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "datagen"))

DEV_DIR = ROOT / "data" / "dev"

# Small enough to generate in seconds, large enough that per-compartment-count metrics are not
# pure noise. The full experiments use 100k voxels per count.
DEV_SIZES = dict(n_train=2_000, n_val=500, n_test=500, n_per_snr=300)


def ensure_dev_data() -> Path:
    """Generate data/dev/n1..n4 if they are not already there, and return the directory."""
    from voxel_simulator.generate import DatasetFamilyConfig, generate_dataset_family

    for n_comp in (1, 2, 3, 4):
        out = DEV_DIR / f"n{n_comp}"
        if (out / "train.parquet").exists():
            continue
        generate_dataset_family(
            DatasetFamilyConfig(
                out_dir=str(out),
                n_comp=n_comp,
                seed=0,
                overwrite=False,
                **DEV_SIZES,
            )
        )
    return DEV_DIR


# Generated at import time rather than in a fixture: the test modules build their path lists at
# module level, before any fixture runs.
ensure_dev_data()

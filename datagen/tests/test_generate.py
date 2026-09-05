"""Generation tests: voxel pipeline, row schema, fixed-n families, paired ladder, safe writes."""

import json

import numpy as np
import pandas as pd
import pytest

from voxel_simulator.generate import (
    DatasetFamilyConfig,
    build_dataset_jobs,
    generate_dataset,
    generate_dataset_family,
    generate_one,
    generate_voxel,
    voxel_to_row,
)
from voxel_simulator.protocol import load_protocol
from voxel_simulator.sampler import (
    DEFAULT_SAMPLING,
    MAX_COMP,
    SAMPLING_MODES,
    SPLIT_SNR_LADDER,
)


def test_generate_voxel_exposes_model_input_signal():
    proto = load_protocol()
    voxel = generate_voxel(0, n_comp=2, protocol=proto, noise_sigma=0.1)

    assert voxel.signal.shape == (proto.n_points,)
    assert voxel.sigma == 0.1
    assert np.all(np.isfinite(voxel.signal))


def test_generate_one_matches_voxel_to_row_schema():
    proto = load_protocol()
    voxel = generate_voxel(1, n_comp=2, protocol=proto, noise_sigma=0.1)

    row_from_parts = voxel_to_row(voxel, proto, noise_sigma=0.1)
    row_direct = generate_one(1, n_comp=2, protocol=proto, noise_sigma=0.1)

    assert row_from_parts.keys() == row_direct.keys()
    for key in row_direct:
        if isinstance(row_direct[key], float) and np.isnan(row_direct[key]):
            assert np.isnan(row_from_parts[key])
        else:
            assert row_from_parts[key] == row_direct[key]


def test_n_comp_reaches_every_row():
    """Regression: generate_voxel was once called positionally, so n_comp never reached the sampler."""
    proto = load_protocol()
    for n_comp in range(1, MAX_COMP + 1):
        df = generate_dataset(20, n_comp=n_comp, protocol=proto)
        assert (df.n_comp == n_comp).all(), f"asked for n_comp={n_comp}, got {set(df.n_comp)}"


def test_schema_width_is_fixed_regardless_of_n_comp():
    """Per-n files must share one schema or they cannot be concatenated into one dataset."""
    proto = load_protocol()
    frames = {n: generate_dataset(5, n_comp=n, protocol=proto) for n in range(1, MAX_COMP + 1)}
    widths = {n: df.shape[1] for n, df in frames.items()}
    assert len(set(widths.values())) == 1, f"schema width varies with n_comp: {widths}"
    assert next(iter(widths.values())) == 4 + 3 * MAX_COMP + proto.n_points

    for n, df in frames.items():
        for i in range(1, MAX_COMP + 1):
            filled = df[[f"T1_{i}", f"T2_{i}", f"w_{i}"]].notna().all(axis=1)
            assert filled.all() if i <= n else (~filled).all(), f"padding wrong at n={n}, slot {i}"


def test_weights_and_t1_gt_t2_in_rows():
    df = generate_dataset(200, n_comp=3, protocol=load_protocol())
    for i in range(1, 4):
        assert (df[f"T1_{i}"] > df[f"T2_{i}"]).all()
    w = df[[f"w_{i}" for i in range(1, MAX_COMP + 1)]].sum(axis=1)
    np.testing.assert_allclose(w, 1.0, atol=1e-9)


# --------------------------------------------------------------------------------------
# Paired fixed-SNR ladder
# --------------------------------------------------------------------------------------

def test_ladder_jobs_share_split_and_pin_snr():
    jobs = {j.name: j for j in build_dataset_jobs(DatasetFamilyConfig(n_comp=2))}
    rungs = [j for name, j in jobs.items() if name.startswith("test_snr")]
    assert rungs, "no ladder jobs built"
    assert all(j.split_code == SPLIT_SNR_LADDER for j in rungs)
    assert all(j.snr is not None for j in rungs)
    assert jobs["train"].snr is None                       # train draws its SNR per voxel


def test_ladder_rungs_share_ground_truth_and_z(tmp_path):
    """Ladder rungs hold the same voxels and the same standardised noise; only sigma differs."""
    cfg = DatasetFamilyConfig(out_dir=tmp_path, n_comp=2, n_train=0, n_val=0, n_test=0,
                              n_per_snr=40, snr_ladder=(20, 150))
    generate_dataset_family(cfg, verbose=False)

    a = pd.read_parquet(tmp_path / "test_snr20.parquet")
    b = pd.read_parquet(tmp_path / "test_snr150.parquet")

    gt = ["n_comp"] + [f"{p}_{i}" for p in ("T1", "T2", "w") for i in range(1, MAX_COMP + 1)]
    pd.testing.assert_frame_equal(a[gt], b[gt])            # ground truth is float64: exact

    # Signals are stored float32, so recovering z amplifies the storage rounding by 1/sigma
    # (~2e-5 at SNR 150). Compare with a tolerance.
    from voxel_simulator.physics import simulate_clean_signal
    proto = load_protocol()
    cols = [f"S_{i+1}" for i in range(proto.n_points)]
    for r in range(len(a)):
        n = int(a.n_comp.iloc[r])
        t1 = a[[f"T1_{i+1}" for i in range(n)]].iloc[r].to_numpy(float)
        t2 = a[[f"T2_{i+1}" for i in range(n)]].iloc[r].to_numpy(float)
        w = a[[f"w_{i+1}" for i in range(n)]].iloc[r].to_numpy(float)
        clean = simulate_clean_signal(proto, t1, t2, w)
        za = (a[cols].iloc[r].to_numpy(float) - clean) / a.sigma.iloc[r]
        zb = (b[cols].iloc[r].to_numpy(float) - clean) / b.sigma.iloc[r]
        np.testing.assert_allclose(za, zb, rtol=0, atol=1e-4)

    assert (a.sigma > b.sigma).all()                       # lower SNR => more noise


# --------------------------------------------------------------------------------------
# Config validation and safe writes
# --------------------------------------------------------------------------------------

def test_config_rejects_bad_n_comp_and_ranges():
    with pytest.raises(ValueError, match="n_comp must be in"):
        DatasetFamilyConfig(n_comp=MAX_COMP + 1)
    with pytest.raises(ValueError, match="no .T1, T2. with T2 < T1"):
        DatasetFamilyConfig(n_comp=1, t1_range=(50.0, 100.0), t2_range=(200.0, 3000.0))


def test_config_rejects_colliding_ladder_names():
    """Filenames use int(snr), so 20.2 and 20.8 would both be test_snr20."""
    with pytest.raises(ValueError, match="duplicate output names"):
        DatasetFamilyConfig(n_comp=1, snr_ladder=(20.2, 20.8))


def test_existing_output_is_not_clobbered(tmp_path):
    cfg = DatasetFamilyConfig(out_dir=tmp_path, n_comp=1, n_train=5, n_val=0, n_test=0,
                              n_per_snr=0, snr_ladder=())
    generate_dataset_family(cfg, verbose=False)
    before = (tmp_path / "train.parquet").read_bytes()

    with pytest.raises(FileExistsError, match="already exist"):
        generate_dataset_family(cfg, verbose=False)
    assert (tmp_path / "train.parquet").read_bytes() == before      # untouched

    generate_dataset_family(cfg.__class__(**{**cfg.__dict__, "overwrite": True}), verbose=False)
    assert (tmp_path / "train.parquet").exists()


def test_no_tmp_files_left_behind(tmp_path):
    cfg = DatasetFamilyConfig(out_dir=tmp_path, n_comp=1, n_train=5, n_val=0, n_test=0,
                              n_per_snr=0, snr_ladder=())
    generate_dataset_family(cfg, verbose=False)
    assert not list(tmp_path.glob("*.tmp")), "atomic write left a temp file behind"


def test_manifest_records_provenance(tmp_path):
    cfg = DatasetFamilyConfig(out_dir=tmp_path, n_comp=2, n_train=5, n_val=0, n_test=0,
                              n_per_snr=0, snr_ladder=())
    generate_dataset_family(cfg, verbose=False)
    m = json.loads((tmp_path / "manifest.json").read_text())

    assert m["n_comp"] == 2 and m["max_comp"] == MAX_COMP
    assert m["splits"]["train"]["rows"] == 5
    assert len(m["protocol_sha256"]) == 64
    assert set(m["dependencies"]) == {"python", "numpy", "pandas", "pyarrow"}
    assert "commit" in m["git"]


# --------------------------------------------------------------------------------------
# Sampling mode: must survive the call chain from config to draw, and land in the manifest.
# test_sampler.py shows the two modes differ at the draw; these tests show the argument is
# not dropped on the way there (as --n-comp once was at a positional call site).
# --------------------------------------------------------------------------------------

def test_config_defaults_to_rejection_sampling():
    """The default must stay the historical scheme; existing datasets are labelled by it."""
    assert DatasetFamilyConfig(n_comp=1).sampling == DEFAULT_SAMPLING == "rejection"


def test_config_rejects_unknown_sampling_mode():
    """An unknown mode fails at config construction."""
    with pytest.raises(ValueError, match="sampling must be one of"):
        DatasetFamilyConfig(n_comp=1, sampling="t1_loguniform")


def test_config_rejects_range_mode_combination_only_the_new_mode_forbids():
    """t1_log_uniform needs t2_min < t1_min; the same ranges are legal under rejection."""
    with pytest.raises(ValueError, match="requires t2_min"):
        DatasetFamilyConfig(n_comp=1, t1_range=(50.0, 3500.0), t2_range=(100.0, 500.0),
                            sampling="t1_log_uniform")
    DatasetFamilyConfig(n_comp=1, t1_range=(50.0, 3500.0), t2_range=(100.0, 500.0))


@pytest.mark.parametrize("sampling", SAMPLING_MODES)
def test_manifest_records_the_sampling_mode(tmp_path, sampling):
    """The manifest records the sampling mode; nothing else in it distinguishes the two coverages."""
    cfg = DatasetFamilyConfig(out_dir=tmp_path, n_comp=2, n_train=5, n_val=0, n_test=0,
                              n_per_snr=0, snr_ladder=(), t1_range=(50.0, 3500.0),
                              t2_range=(5.0, 500.0), sampling=sampling)
    generate_dataset_family(cfg, verbose=False)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["physics"]["sampling"] == sampling


def test_sampling_mode_reaches_the_rows(tmp_path):
    """Two families identical except for sampling must differ in the parquet contents.

    Same seed, ranges and sizes. Equal contents would mean the argument is dropped somewhere
    between the config and sample_random_compartment.
    """
    kw = dict(n_comp=2, seed=3500501, n_train=200, n_val=0, n_test=0, n_per_snr=0,
              snr_ladder=(), t1_range=(50.0, 3500.0), t2_range=(5.0, 500.0))
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    generate_dataset_family(DatasetFamilyConfig(out_dir=old_dir, sampling="rejection", **kw),
                            verbose=False)
    generate_dataset_family(DatasetFamilyConfig(out_dir=new_dir, sampling="t1_log_uniform", **kw),
                            verbose=False)
    old = pd.read_parquet(old_dir / "train.parquet")
    new = pd.read_parquet(new_dir / "train.parquet")

    assert not np.array_equal(old["T1_1"].to_numpy(), new["T1_1"].to_numpy())
    # The constraint holds in both, and SNR (its own stream) is identical row for row.
    for df in (old, new):
        assert (df["T1_1"] > df["T2_1"]).all() and (df["T1_2"] > df["T2_2"]).all()
    np.testing.assert_array_equal(old["snr"].to_numpy(), new["snr"].to_numpy())


def test_generate_one_forwards_sampling():
    """Single-row version of the test above, through generate_one -> ... -> sample_random_compartment.

    The two modes agree exactly on some voxels, and that is correct: when T1 >= t2_max (500 ms
    here) the conditional T2 interval equals the rejection interval, so if the first rejection
    draw is accepted both modes consume the same two uniforms and return identical values.
    Voxel 0 of this family is such a case, so several ids are scanned.
    """
    proto = load_protocol()
    kw = dict(n_comp=1, protocol=proto, base_seed=3500501,
              t1_range=(50.0, 3500.0), t2_range=(5.0, 500.0))
    rows = [(generate_one(v, sampling="rejection", **kw),
             generate_one(v, sampling="t1_log_uniform", **kw)) for v in range(40)]

    differing = [v for v, (a, b) in enumerate(rows)
                 if a["T1_1"] != b["T1_1"] or a["T2_1"] != b["T2_1"]]
    assert differing, "sampling= never changed the draw over 40 voxels: the keyword is being dropped"

    for a, b in rows:
        assert a["snr"] == b["snr"], "SNR lives in its own stream and must not move"
        # Coinciding voxels must be the predicted ones: first draw accepted and T1 >= t2_max.
        if a["T1_1"] == b["T1_1"] and a["T2_1"] == b["T2_1"]:
            assert a["T1_1"] >= 500.0, (
                f"modes agreed at T1={a['T1_1']:.1f} < t2_max, which the cap argument forbids"
            )

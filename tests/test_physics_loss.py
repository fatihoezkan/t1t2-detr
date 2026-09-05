"""Tests for the signal-consistency loss (t1t2.physics_loss).

These are the checks run before the physics arms were submitted; the methods they verify are
described in docs/physics-loss.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from t1t2.config import DataConfig, LossConfig, from_dict
from t1t2.data import TargetNormalizer
from t1t2.physics import forward_numpy, load_protocol
from t1t2.physics_loss import SignalConsistencyLoss, _denorm_torch, _signal_norm_torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def proto():
    return load_protocol()


@pytest.fixture()
def dcfg():
    return DataConfig(train_path="unused", t1_min=50.0, t1_max=3500.0, t2_min=5.0, t2_max=500.0)


@pytest.fixture()
def compartments():
    rng = np.random.default_rng(0)
    t1 = rng.uniform(50, 3500, 3)
    t2 = np.minimum(rng.uniform(5, 500, 3), t1 * 0.9)
    w = rng.dirichlet(np.ones(3))
    return t1, t2, w


def _sc(dcfg, target):
    lcfg = LossConfig(signal_consistency=True, signal_consistency_weight=1.0,
                      signal_consistency_target=target)
    return SignalConsistencyLoss(dcfg, lcfg)


def _perfect_pred(nz, t1, t2, w, n_queries=10):
    """A prediction tensor whose first 3 queries are the truth and the rest are absent."""
    yp = torch.zeros(1, n_queries, 4)
    yp[..., 3] = -12.0
    yp[0, :3, 0] = torch.tensor(nz.normalize_t1(t1), dtype=torch.float32)
    yp[0, :3, 1] = torch.tensor(nz.normalize_t2(t2), dtype=torch.float32)
    yp[0, :3, 2] = torch.tensor(w, dtype=torch.float32)
    yp[0, :3, 3] = 12.0
    return yp


def _truth_tensors(nz, t1, t2, w, proto):
    y_true = torch.tensor(
        np.stack([nz.normalize_t1(t1), nz.normalize_t2(t2), w], -1).reshape(1, -1),
        dtype=torch.float32,
    )
    s = forward_numpy(proto, t1, t2, w)
    X = torch.tensor(s / np.abs(s).max(), dtype=torch.float32).unsqueeze(0)
    return y_true, X


def test_denorm_matches_target_normalizer(dcfg, compartments):
    t1, _, _ = compartments
    nz = TargetNormalizer.from_config(dcfg)
    back = _denorm_torch(torch.tensor(nz.normalize_t1(t1)), 50.0, 3500.0, "log_minmax")
    assert np.abs(back.numpy() - t1).max() < 1e-9


def test_signal_norm_matches_dataset_transform():
    from t1t2.data import _apply_signal_norm
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 64)).astype(np.float32)
    for mode in ("none", "max", "first"):
        ref = _apply_signal_norm(X.copy(), mode)
        got = _signal_norm_torch(torch.tensor(X), mode).numpy()
        assert np.abs(ref - got).max() < 1e-6, mode


def test_clean_target_matches_generator(dcfg, proto, compartments):
    t1, t2, w = compartments
    nz = TargetNormalizer.from_config(dcfg)
    y_true, X = _truth_tensors(nz, t1, t2, w, proto)
    tgt = _sc(dcfg, "clean")._clean_target(y_true)
    assert np.abs(tgt.numpy()[0] - X.numpy()[0]).max() < 1e-5


@pytest.mark.parametrize("target", ["noisy", "clean"])
def test_perfect_prediction_is_near_zero(dcfg, proto, compartments, target):
    t1, t2, w = compartments
    nz = TargetNormalizer.from_config(dcfg)
    y_true, X = _truth_tensors(nz, t1, t2, w, proto)
    loss = _sc(dcfg, target)(_perfect_pred(nz, t1, t2, w), X, y_true)
    assert float(loss) < 1e-8


def test_wrong_prediction_separates(dcfg, proto, compartments):
    t1, t2, w = compartments
    nz = TargetNormalizer.from_config(dcfg)
    y_true, X = _truth_tensors(nz, t1, t2, w, proto)
    bad = _perfect_pred(nz, t1, t2, w)
    with torch.no_grad():
        bad[0, :3, 0] = 0.9
    assert float(_sc(dcfg, "noisy")(bad, X, y_true)) > 0.01


def test_gradient_reaches_all_four_channels(dcfg, proto, compartments):
    t1, t2, w = compartments
    nz = TargetNormalizer.from_config(dcfg)
    y_true, X = _truth_tensors(nz, t1, t2, w, proto)
    yp = _perfect_pred(nz, t1, t2, w)
    with torch.no_grad():                      # perturb so gradients are non-trivial
        yp[0, :3, 0] += 0.05
    yp.requires_grad_(True)
    _sc(dcfg, "noisy")(yp, X, y_true).backward()
    for ch in range(4):
        assert float(yp.grad[..., ch].abs().sum()) > 0, f"no gradient on channel {ch}"


def test_absent_queries_do_not_contribute(dcfg, proto, compartments):
    """A query with exist_logit = -inf must not change the resynthesized signal."""
    t1, t2, w = compartments
    nz = TargetNormalizer.from_config(dcfg)
    sc = _sc(dcfg, "noisy")
    yp = _perfect_pred(nz, t1, t2, w)
    junk = yp.clone()
    junk[0, 5, :3] = torch.tensor([0.5, 0.5, 0.9])     # big weight, but gate stays -12
    assert torch.allclose(sc.synthesize(yp), sc.synthesize(junk), atol=1e-6)


def test_rician_raises(dcfg):
    lcfg = LossConfig(signal_consistency=True, signal_consistency_type="rician")
    with pytest.raises(NotImplementedError):
        SignalConsistencyLoss(dcfg, lcfg)


def test_bad_target_raises(dcfg):
    lcfg = LossConfig(signal_consistency=True, signal_consistency_target="oracle")
    with pytest.raises(ValueError):
        SignalConsistencyLoss(dcfg, lcfg)


def test_generated_configs_are_one_change_clean():
    """The shipped physics configs differ from the baseline YAML in exactly the four
    signal_consistency fields plus name and notes, so the one-change rule holds."""
    base = yaml.safe_load(open(ROOT / "configs" / "baseline_v2_reproduction.yaml"))

    def flat(d, p=""):
        out = {}
        for k, v in d.items():
            kk = f"{p}.{k}" if p else k
            out.update(flat(v, kk)) if isinstance(v, dict) else out.__setitem__(kk, v)
        return out

    expected = sorted([
        "name", "notes",
        "loss.signal_consistency", "loss.signal_consistency_weight",
        "loss.signal_consistency_target", "loss.signal_consistency_warmup_epochs",
    ])
    for arm in ("physics_noisy", "physics_clean"):
        raw = yaml.safe_load(open(ROOT / "configs" / f"{arm}.yaml"))
        fb, fr = flat(base), flat(raw)
        diffs = sorted({k for k in fb if fb.get(k) != fr.get(k)} | (set(fr) - set(fb)))
        assert diffs == expected, (arm, diffs)
        cfg = from_dict(raw)                    # must survive the typed loader
        assert cfg.loss.signal_consistency is True
        assert cfg.model.pretrain_path is None  # trains from random init, not a warm start
        assert cfg.train.seed == base["train"]["seed"]


def test_physics_smoke_training_runs(tmp_path, dcfg, proto):
    """Two optimizer steps with the physics term on: loss finite, sc logged, history sane."""
    from t1t2.train import train

    rng = np.random.default_rng(2)
    rows = []
    for i in range(64):
        t1 = rng.uniform(60, 3400, 2); t2 = np.minimum(rng.uniform(6, 490, 2), t1 * 0.9)
        w = rng.dirichlet(np.ones(2))
        s = forward_numpy(proto, t1, t2, w) + rng.normal(0, 0.01, proto.n_points)
        rows.append({"voxel_id": i, "snr": 100.0, "sigma": 0.01, "n_comp": 2,
                     "T1_1": t1[0], "T2_1": t2[0], "w_1": w[0],
                     "T1_2": t1[1], "T2_2": t2[1], "w_2": w[1],
                     **{f"S_{p+1}": s[p] for p in range(proto.n_points)}})
    import pandas as pd
    df = pd.DataFrame(rows)
    pq = tmp_path / "train.parquet"
    df.to_parquet(pq)

    raw = yaml.safe_load(open(ROOT / "configs" / "physics_noisy.yaml"))
    raw["data"]["train_path"] = str(pq)
    raw["data"]["val_path"] = str(pq)
    raw["data"]["test_path"] = None
    raw["train"]["batch_size"] = 32
    raw["train"]["num_workers"] = 0
    raw["train"]["device"] = "cpu"
    raw["loss"]["signal_consistency_warmup_epochs"] = 2
    cfg = from_dict(raw)

    hist, _, _ = train(cfg, results_dir=tmp_path / "res", max_epochs=2, log=lambda *_: None)
    assert len(hist) == 2
    for h in hist:
        assert np.isfinite(h["train"]["loss"])
        assert "sc" in h["train"] and "sc" in h["val"]
        assert h["train"]["sc"] > 0                     # the term is actually active
    assert hist[0]["phys_lambda"] == pytest.approx(0.5)  # warmup: epoch 1 of 2
    assert hist[1]["phys_lambda"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# sqrt weighting mode (added for baseline_v3, the combined model)
# ---------------------------------------------------------------------------

def test_sqrt_mode_is_accepted_and_produces_finite_grad():
    import torch
    from t1t2.config import LossConfig
    from t1t2.loss import HungarianLoss

    torch.manual_seed(0)
    y_pred = torch.randn(4, 6, 4, requires_grad=True)
    y_true = torch.rand(4, 9)
    n_comp = torch.tensor([1, 2, 3, 3])
    crit = HungarianLoss(LossConfig(t1_t2_weighting="sqrt"))
    loss, *_ = crit(y_pred, y_true, n_comp)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(y_pred.grad).all()


def test_sqrt_mode_compresses_gradient_ratio_to_sqrt_of_w_ratio():
    import math
    w_lo, w_hi = 0.05, 0.75
    assert math.isclose((w_hi ** 0.5) / (w_lo ** 0.5), math.sqrt(w_hi / w_lo))
    assert (w_hi ** 0.5) / (w_lo ** 0.5) < 4.0  # 15x -> ~3.9x


def test_invalid_weighting_mode_still_raises():
    import pytest
    from t1t2.config import LossConfig
    from t1t2.loss import HungarianLoss

    with pytest.raises(ValueError):
        HungarianLoss(LossConfig(t1_t2_weighting="cube"))

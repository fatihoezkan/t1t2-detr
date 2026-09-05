"""Experiment configuration as typed dataclasses.

A run is defined by its YAML file and nothing else. Data, model, loss, training and
evaluation settings are all fields here, and every run writes its resolved config back
next to its results, so two runs that differ have configs that differ.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    """Dataset paths and the transforms applied on the way into the model."""

    # One parquet path or a list to concatenate. The per-count files n1/n2/n3 are combined
    # this way into one balanced training set.
    train_path: str | list[str]
    val_path: Optional[str | list[str]] = None
    test_path: Optional[str | list[str]] = None

    n_inputs: int = 64            # 8 TI x 8 TE, fixed by the protocol

    # Cap on training voxels per path. Validation and test are never limited, so all runs
    # are scored on the same voxels.
    train_limit_per_path: Optional[int] = None

    # There is deliberately no max_comp field. The width of the compartment table is read
    # from the parquet columns (data.infer_max_comp); a stale config value once produced a
    # model that could not count past three without any error being raised.

    # Bounds of the log-min-max map from T1/T2 in ms onto the [0, 1] sigmoid outputs
    # (data.TargetNormalizer). They must cover the ranges the generator sampled from,
    # otherwise targets are clamped and the model can never reach the edges of the space.
    # The input signal is always divided by its own peak magnitude (data._normalize_signal).
    t1_min: float = 50.0
    t1_max: float = 4000.0
    t2_min: float = 5.0
    t2_max: float = 3000.0


@dataclass
class ModelConfig:
    """Network shape."""

    input_dim: int = 64
    hidden_dim: int = 512
    fs_dim: int = 256             # width of the feature and query vectors
    n_queries: int = 10           # upper bound on the number of proposed compartments
    n_dlayers: int = 4            # transformer decoder depth
    n_heads: int = 4
    aux_loss: bool = False        # also supervise the intermediate decoder layers

    # Existence head wiring, see model.T1T2DETR.
    #   joint   one head sees all query states at once and can suppress a duplicate.
    #           The baseline used this, so it stays the default.
    #   shared  one head applied per query with shared weights; no query sees another.
    # The two wirings differ in parameter count (about 2.6 %), so the exist_head_shared
    # arm changes capacity as well as wiring.
    exist_head: str = "joint"     # joint | shared


@dataclass
class LossConfig:
    """Loss term weights and the optional signal-consistency term.

    With log-normalised targets the three regression weights can all sit at 1.0; the
    existence term is down-weighted to 0.1.
    """

    t1_weight: float = 1.0
    t2_weight: float = 1.0
    w_weight: float = 1.0
    exist_weight: float = 0.1
    aux_weight: float = 1.0       # scale of the per-layer auxiliary losses

    # How a compartment's own signal weight scales its T1/T2 error; see HungarianLoss.
    t1_t2_weighting: str = "signal_fraction"   # signal_fraction | sqrt | uniform

    # Signal-consistency term (physics_loss.py): resynthesise the signal from the prediction
    # and penalise the mean squared mismatch with a target signal (the simulated signal is
    # signed, so Gaussian noise and MSE are the right likelihood).
    #   target   "noisy" compares against the network input (no labels needed, would
    #            transfer to real scans); "clean" compares against forward(true params),
    #            which only a simulation can provide.
    #   warmup   ramps the weight from zero, because at initialisation every query is
    #            about half open and the gated resynthesis is meaningless.
    signal_consistency: bool = False
    signal_consistency_weight: float = 0.0
    signal_consistency_target: str = "noisy"   # noisy | clean
    signal_consistency_warmup_epochs: int = 0


@dataclass
class TrainConfig:
    """Optimiser, schedule and run mechanics."""

    epochs: int = 200             # an upper bound; early stopping usually ends the run first
    batch_size: int = 256

    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    opt_betas: tuple = (0.9, 0.98)

    # Early stopping and best-model selection watch the validation parameter loss
    # (train.SELECTION_METRIC) and need a validation split; without one both are disabled.
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1.0e-4

    lr_scheduler: str = "constant"              # constant | reduce_on_plateau
    scheduler_factor: float = 0.5
    scheduler_patience: int = 8
    scheduler_min_lr: float = 1.0e-6
    gradient_clip_norm: Optional[float] = None

    device: Optional[str] = None      # None = auto-detect, see device.py
    seed: int = 0
    num_workers: int = 0


@dataclass
class EvaluationConfig:
    """Test-split reporting and the validation-only threshold search."""

    # The search minimises the bounded parameter-set error on the validation split; see
    # eval.calibrate_existence_threshold. False keeps the fixed threshold below.
    calibrate_threshold: bool = False
    fixed_threshold: float = 0.5
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_steps: int = 91


@dataclass
class ExperimentConfig:
    """The five sub-configs plus a name and free-text notes."""

    name: str
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    notes: str = ""

    def to_dict(self) -> dict:
        """Turn the experiment settings into a dictionary."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write the config as YAML, keeping the section order (sort_keys=False) so a saved
        config diffs cleanly against its source."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def load_config(path: str | Path) -> ExperimentConfig:
    """Read a YAML file into an ExperimentConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return from_dict(raw)


# Settings retired after the thesis runs. Every run set each of them to the one value that is
# still implemented, and results/<run>/config.yaml records them, so they are accepted at that
# value and refused at any other: a saved config never claims a behaviour the code lacks.
_RETIRED = {
    ("data", "normalization"): "log_minmax",
    ("data", "signal_norm"): "max",
    ("model", "pretrain_path"): None,
    ("model", "freeze_encoder"): False,
    ("loss", "signal_consistency_type"): "mse",
    ("train", "selection_metric"): "parameter_loss",
    ("train", "ckpt_every"): 1,
    ("evaluation", "threshold_objective"): "parameter_set_error",
}


def from_dict(raw: dict) -> ExperimentConfig:
    """Build the nested dataclasses from a plain dict.

    Written out explicitly so that a misspelled key raises a TypeError instead of being
    dropped. Retired keys are dropped after their value is checked (see _RETIRED).
    opt_betas comes back from YAML as a list and the optimiser wants a tuple.
    """
    raw = {k: (dict(v) if isinstance(v, dict) else v) for k, v in raw.items()}
    for (section, key), only in _RETIRED.items():
        if key in raw.get(section, {}):
            value = raw[section].pop(key)
            if value != only:
                raise ValueError(
                    f"{section}.{key}={value!r} is no longer supported; every reported run "
                    f"used {only!r}, which is now the only behaviour."
                )
    data = DataConfig(**raw["data"])
    model = ModelConfig(**raw.get("model", {}))
    loss = LossConfig(**raw.get("loss", {}))
    train_raw = raw.get("train", {})
    if train_raw.get("opt_betas") is not None:
        train_raw["opt_betas"] = tuple(train_raw["opt_betas"])
    train = TrainConfig(**train_raw)
    evaluation = EvaluationConfig(**raw.get("evaluation", {}))
    return ExperimentConfig(
        name=raw["name"],
        data=data,
        model=model,
        loss=loss,
        train=train,
        evaluation=evaluation,
        notes=raw.get("notes", ""),
    )

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

    # Mapping of T1/T2 in ms onto the [0, 1] sigmoid outputs. Log spacing is the default:
    # relaxation times span more than a decade and a linear map leaves the short end with
    # almost no resolution. The bounds must cover the ranges the generator sampled from,
    # otherwise targets are clamped and the model can never reach the edges of the space.
    normalization: str = "log_minmax"     # identity | linear_minmax | log_minmax
    t1_min: float = 50.0
    t1_max: float = 4000.0
    t2_min: float = 5.0
    t2_max: float = 3000.0

    # Per-voxel input rescaling. Real scans come at an arbitrary scale, so the same
    # transform has to be applied to synthetic and real data. "max" also divides out M0.
    signal_norm: str = "max"              # none | max | first


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
    pretrain_path: Optional[str] = None   # optional warm start for the encoder
    freeze_encoder: bool = False


@dataclass
class LossConfig:
    """Loss term weights and the optional signal-consistency term.

    With log-normalised targets the four base weights can all sit near 1.0.
    """

    t1_weight: float = 1.0
    t2_weight: float = 1.0
    w_weight: float = 1.0
    exist_weight: float = 0.1
    aux_weight: float = 1.0       # scale of the per-layer auxiliary losses

    # How a compartment's own signal weight scales its T1/T2 error; see HungarianLoss.
    # "legacy" reproduces the very first baseline and is kept so those runs stay repeatable.
    t1_t2_weighting: str = "legacy"       # legacy | signal_fraction | sqrt | uniform

    # Signal-consistency term (physics_loss.py): resynthesise the signal from the prediction
    # and penalise the mismatch with a target signal.
    #   target   "noisy" compares against the network input (no labels needed, would
    #            transfer to real scans); "clean" compares against forward(true params),
    #            which only a simulation can provide.
    #   type     "mse" because the simulated signal is signed; a Rician likelihood is for
    #            magnitude data and is not implemented.
    #   warmup   ramps the weight from zero, because at initialisation every query is
    #            about half open and the gated resynthesis is meaningless.
    signal_consistency: bool = False
    signal_consistency_weight: float = 0.0
    signal_consistency_type: str = "mse"       # mse | rician
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

    # Early stopping and best-model selection need a validation split; without one both
    # are disabled.
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1.0e-4
    # "parameter_loss" selects on T1 + T2 + weight only, so a checkpoint is not chosen for
    # a good existence head at the cost of worse parameters. "total_loss" is the old
    # behaviour.
    selection_metric: str = "total_loss"        # total_loss | parameter_loss

    lr_scheduler: str = "constant"              # constant | reduce_on_plateau
    scheduler_factor: float = 0.5
    scheduler_patience: int = 8
    scheduler_min_lr: float = 1.0e-6
    gradient_clip_norm: Optional[float] = None

    device: Optional[str] = None      # None = auto-detect, see device.py
    seed: int = 0
    num_workers: int = 0
    # last.pt also carries the training history, so a coarser cadence loses epochs on resume.
    ckpt_every: int = 1


@dataclass
class EvaluationConfig:
    """Test-split reporting and the validation-only threshold search."""

    calibrate_threshold: bool = False   # False keeps the fixed threshold below
    fixed_threshold: float = 0.5
    threshold_objective: str = "count_accuracy"  # count_accuracy | parameter_set_error
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


def from_dict(raw: dict) -> ExperimentConfig:
    """Build the nested dataclasses from a plain dict.

    Written out explicitly so that a misspelled key raises a TypeError instead of being
    dropped. opt_betas comes back from YAML as a list and the optimiser wants a tuple.
    """
    data = DataConfig(**raw["data"])
    model = ModelConfig(**raw.get("model", {}))
    loss = LossConfig(**raw.get("loss", {}))
    train_raw = dict(raw.get("train", {}))
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

"""A finished run on disk, loaded for evaluation.

Every script under evaluation/ starts the same way: read results/<name>/config.yaml, build the
model, load checkpoints/best.pt, build the target normaliser, and take the log spans that the
Normalised Distance tau is a fraction of. This is the one place that does it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import ExperimentConfig, load_config
from .data import TargetNormalizer, VoxelDataset
from .eval import detr_query_outputs, true_compartments
from .model import build_model
from .nd_metrics import log_spans


@dataclass
class Run:
    """Config, model in eval mode, normaliser and log spans of one results/<name>/ directory."""

    dir: Path
    cfg: ExperimentConfig
    model: torch.nn.Module
    normalizer: TargetNormalizer
    spans: tuple[float, float]      # log spans of T1 and T2, what tau is a fraction of
    epoch: int                       # 1-based epoch of best.pt
    device: torch.device

    @property
    def fitted_threshold(self) -> float:
        """The existence threshold the run's own evaluation selected on validation."""
        summary = json.loads((self.dir / "summary.json").read_text())
        return float(summary["threshold_calibration"]["selected_threshold"])

    def dataset(self, split="test", limit=None) -> VoxelDataset:
        """One of the run's own splits by name, or explicit parquet path(s)."""
        d = self.cfg.data
        named = {"train": d.train_path, "val": d.val_path, "test": d.test_path}
        paths = named[split] if isinstance(split, str) and split in named else split
        return VoxelDataset(paths, d, self.normalizer, limit=limit)

    def predict(self, split="test", limit=None):
        """(query table, true compartments) for a split: the two inputs every metric takes."""
        ds = self.dataset(split, limit)
        return detr_query_outputs(self.model, ds, self.device, self.normalizer), true_compartments(ds)


def load_run(run_dir, device="cpu") -> Run:
    """Load results/<name>/ with its best checkpoint; the run directory is only read."""
    rd = Path(run_dir)
    cfg = load_config(rd / "config.yaml")
    ckpt = torch.load(rd / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
    model = build_model(cfg.model)
    model.load_state_dict(ckpt["model"])
    device = torch.device(device)
    model.to(device).eval()
    d = cfg.data
    return Run(dir=rd, cfg=cfg, model=model, normalizer=TargetNormalizer.from_config(d),
               spans=log_spans(d.t1_min, d.t1_max, d.t2_min, d.t2_max),
               epoch=int(ckpt["epoch"]) + 1, device=device)

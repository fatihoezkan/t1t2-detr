"""Training loop with checkpointing and resume.

The optimisation itself is ordinary. The file is built for running unattended on a shared
cluster: it writes its state every epoch and continues from the last checkpoint if the job
dies. A run leaves the resolved config, the loss history, the checkpoints and the metrics
in results/<name>/.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, load_config
from .data import TargetNormalizer, make_dataloader
from .device import device_info, get_device
from .loss import HungarianLoss
from .model import build_model
from .physics_loss import SignalConsistencyLoss

# Loss components in logging order. "sc" is the signal-consistency term, 0.0 when it is off.
_LOSS_KEYS = ("loss", "t1", "t2", "wt", "ex", "sc")


def _total_limit(data_cfg) -> int | None:
    """Per-path training cap times the number of paths, which is what the loader expects.

    The cap is stated per path so that a reduced-data run stays balanced across compartment
    counts instead of starving the later files.
    """
    per = data_cfg.train_limit_per_path
    if per is None:
        return None
    paths = data_cfg.train_path
    n = 1 if isinstance(paths, str) else len(paths)
    return per * n


def _fingerprint(cfg: ExperimentConfig) -> dict:
    """The parts of a config a resume has to agree on."""
    return {"data": asdict(cfg.data), "model": asdict(cfg.model),
            "loss": asdict(cfg.loss), "train": asdict(cfg.train)}


def _check_resume_compatible(cfg: ExperimentConfig, results_dir: Path, resume: bool, log) -> None:
    """Refuse to resume into a results directory that belongs to a different config.

    Checkpoints are keyed by directory alone. Pointing an edited config at an existing
    directory would continue from those weights and produce a model that is half one
    experiment and half another, with a saved config claiming it was entirely the second.
    """
    prev_path = results_dir / "config.yaml"
    if not (resume and prev_path.exists() and (results_dir / "checkpoints" / "last.pt").exists()):
        return

    prev = load_config(prev_path)
    old, new = _fingerprint(prev), _fingerprint(cfg)
    changed = {k: (old[k], new[k]) for k in new if old[k] != new[k]}
    if changed:
        detail = ", ".join(sorted(changed))
        raise ValueError(
            f"{results_dir} holds a checkpoint from a different config (differs in: {detail}). "
            "Resuming would blend two experiments. Use a new results dir / cfg.name, or pass "
            "resume=False to start over."
        )
    log(f"[{cfg.name}] resume fingerprint matches the existing run")


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch. Makes two runs on the same machine match; it does not
    guarantee bit-identical results across GPUs or torch versions."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_epoch(model, loader, crit, device, opt=None, aux_weight=1.0,
               gradient_clip_norm=None, phys_crit=None, phys_lambda=0.0) -> dict:
    """One pass over a loader. Trains if an optimiser is given, evaluates otherwise.

    aux_weight scales the per-layer auxiliary losses (later layers count fully, earlier ones
    are scaled down), the deep-supervision recipe from DETR. The signal-consistency term is
    applied to the final prediction only. The logged "sc" value is the raw mismatch before
    lambda, so runs with different lambdas stay comparable.
    """
    train = opt is not None
    model.train() if train else model.eval()
    agg = {k: [] for k in _LOSS_KEYS}
    with torch.enable_grad() if train else torch.no_grad():
        for X, y, nc in loader:
            X, y, nc = X.to(device), y.to(device), nc.to(device)
            out = model(X)
            if isinstance(out, dict):                          # aux_loss enabled
                pred = out["pred"]
                loss, l1, l2, lw, le = crit(pred, y, nc)
                for i, aux in enumerate(out["aux"]):
                    al, *_ = crit(aux, y, nc)
                    loss = loss + al * min((i + 1) * aux_weight, 1.0)
            else:
                pred = out
                loss, l1, l2, lw, le = crit(out, y, nc)
            if phys_crit is not None:
                sc = phys_crit(pred, X, y)
                loss = loss + phys_lambda * sc
            else:
                sc = torch.zeros((), device=device)
            if train:
                opt.zero_grad()
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                opt.step()
            for k, v in zip(_LOSS_KEYS, (loss, l1, l2, lw, le, sc)):
                agg[k].append(float(v.item()))
    result = {k: float(np.mean(v)) for k, v in agg.items()}
    result["parameter_loss"] = result["t1"] + result["t2"] + result["wt"]
    return result


def _selection_value(metrics: dict, selection_metric: str) -> float:
    """The validation scalar used for best-checkpoint selection and early stopping."""
    if selection_metric == "total_loss":
        return float(metrics["loss"])
    if selection_metric == "parameter_loss":
        return float(metrics["parameter_loss"])
    raise ValueError(
        f"selection_metric must be total_loss|parameter_loss; got {selection_metric!r}"
    )


def _build_scheduler(opt, train_cfg):
    mode = train_cfg.lr_scheduler
    if mode == "constant":
        return None
    if mode == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=train_cfg.scheduler_factor,
            patience=train_cfg.scheduler_patience,
            min_lr=train_cfg.scheduler_min_lr,
        )
    raise ValueError(f"lr_scheduler must be constant|reduce_on_plateau; got {mode!r}")


def train(cfg: ExperimentConfig, results_dir=None, max_epochs=None, resume=True, limit=None, log=print):
    """Train from a config and return (history, results_dir, best model).

    max_epochs overrides the config's epoch count and limit caps the voxels loaded; both are
    for smoke runs. `log` is injectable so tests can silence the output.
    """
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    log(f"[{cfg.name}] device={device} | {device_info()}")

    results_dir = Path(results_dir) if results_dir else Path("results") / cfg.name
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Must run before the config below is written, or there is nothing to compare against.
    _check_resume_compatible(cfg, results_dir, resume, log)
    cfg.save(results_dir / "config.yaml")

    # One normaliser for both splits.
    normalizer = TargetNormalizer.from_config(cfg.data)
    # train_limit_per_path reduces the training set only; `limit` is the smoke-run cap and
    # applies to everything. Validation stays identical across arms so best_val is comparable.
    train_loader, _ = make_dataloader(
        cfg.data.train_path, cfg.data, cfg.train.batch_size, True,
        normalizer, cfg.train.num_workers,
        limit=limit if limit is not None else _total_limit(cfg.data),
    )
    val_loader = None
    if cfg.data.val_path:
        val_loader, _ = make_dataloader(
            cfg.data.val_path, cfg.data, cfg.train.batch_size, False,
            normalizer, cfg.train.num_workers, limit=limit,
        )

    model = build_model(cfg.model).to(device)
    crit = HungarianLoss(cfg.loss)
    phys_crit = None
    if cfg.loss.signal_consistency:
        phys_crit = SignalConsistencyLoss(cfg.data, cfg.loss).to(device)
        log(f"[{cfg.name}] signal-consistency ON: target={phys_crit.target} "
            f"lambda={cfg.loss.signal_consistency_weight} "
            f"warmup={cfg.loss.signal_consistency_warmup_epochs} epochs")

    def _phys_lambda(epoch: int) -> float:
        """Ramp linearly from 0 to the configured weight over the warmup epochs, then hold."""
        if phys_crit is None:
            return 0.0
        w = float(cfg.loss.signal_consistency_weight)
        wu = int(cfg.loss.signal_consistency_warmup_epochs)
        if wu <= 0 or epoch >= wu:
            return w
        return w * (epoch + 1) / wu
    # Filtering on requires_grad keeps a frozen encoder out of the optimiser.
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, betas=tuple(cfg.train.opt_betas),
    )
    scheduler = _build_scheduler(opt, cfg.train)
    # A checkpoint carries model, optimiser, epoch and the selection state, so a resume
    # continues the run rather than restarting momentum and the patience counter. It is not
    # a bit-identical replay: the dataloader's shuffle order is not restored.
    start_epoch, history = 0, []
    best_val, best_epoch, bad_epochs = float("inf"), -1, 0
    last_ckpt, best_ckpt = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    if resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        history = state["history"]
        best_val = state.get("best_val", float("inf"))
        best_epoch = state.get("best_epoch", -1)
        bad_epochs = state.get("bad_epochs", 0)
        log(f"[{cfg.name}] resumed at epoch {start_epoch} (best val {best_val:.5f} @ ep {best_epoch + 1})")

    # Without a validation split there is nothing to select on.
    early_stop = cfg.train.early_stopping and val_loader is not None
    if cfg.train.early_stopping and val_loader is None:
        log(f"[{cfg.name}] no val split -> early stopping disabled, final epoch is the result")

    epochs = max_epochs if max_epochs is not None else cfg.train.epochs
    steps = sum(h.get("steps", 0) for h in history)
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        lam = _phys_lambda(epoch)
        tr = _run_epoch(
            model, train_loader, crit, device, opt, cfg.loss.aux_weight,
            gradient_clip_norm=cfg.train.gradient_clip_norm,
            phys_crit=phys_crit, phys_lambda=lam,
        )
        # Validation always uses the full physics weight, not the warmup value, so every
        # epoch is scored against the same objective.
        va = (_run_epoch(model, val_loader, crit, device,
                         phys_crit=phys_crit,
                         phys_lambda=float(cfg.loss.signal_consistency_weight))
              if val_loader else {})
        selection_value = _selection_value(va, cfg.train.selection_metric) if va else None
        steps += len(train_loader)
        history.append({
            "epoch": epoch, "train": tr, "val": va,
            "selection_metric": cfg.train.selection_metric,
            "selection_value": selection_value,
            "phys_lambda": lam,
            "lr": float(opt.param_groups[0]["lr"]),
            "seconds": round(time.time() - t0, 2),
            "steps": len(train_loader),
            "cum_steps": steps,
        })

        improved = bool(va) and selection_value < best_val - cfg.train.early_stopping_min_delta
        if improved:
            # Plain Python scalars only: torch 2.6+ loads with weights_only=True by default,
            # and a numpy scalar in here breaks the resume on the cluster.
            best_val, best_epoch, bad_epochs = float(selection_value), epoch, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val": best_val,
                    "val_loss": float(va["loss"]),
                    "parameter_loss": float(va["parameter_loss"]),
                    "selection_metric": cfg.train.selection_metric,
                },
                best_ckpt,
            )
        elif va:
            bad_epochs += 1

        if scheduler is not None and va:
            scheduler.step(selection_value)

        # Checkpoint on the configured cadence and always on the final epoch. history.json
        # is rewritten every epoch so a running job's curves can be watched.
        if epoch % cfg.train.ckpt_every == 0 or epoch == epochs - 1:
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                 "history": history, "best_val": best_val, "best_epoch": best_epoch,
                 "bad_epochs": bad_epochs,
                 "scheduler": None if scheduler is None else scheduler.state_dict()},
                last_ckpt,
            )
        with open(results_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        msg = f"[{cfg.name}] ep {epoch + 1}/{epochs} train {tr['loss']:.5f}"
        if va:
            msg += (f" | val {va['loss']:.5f} (t1 {va['t1']:.4f} t2 {va['t2']:.4f} "
                    f"wt {va['wt']:.4f} ex {va['ex']:.4f}")
            msg += f" sc {va['sc']:.5f})" if phys_crit is not None else ")"
            if cfg.train.selection_metric != "total_loss":
                msg += f" | select parameter {selection_value:.5f}"
            msg += "  *best*" if improved else f"  (no gain {bad_epochs}/{cfg.train.early_stopping_patience})"
        log(msg)

        if early_stop and bad_epochs >= cfg.train.early_stopping_patience:
            log(f"[{cfg.name}] early stop at epoch {epoch + 1}: no val gain for "
                f"{bad_epochs} epochs. Best {best_val:.5f} @ epoch {best_epoch + 1}.")
            break

    # Return the best model, not the last epoch.
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])
        log(
            f"[{cfg.name}] loaded best.pt (epoch {best_epoch + 1}, "
            f"{cfg.train.selection_metric} {best_val:.5f}) for evaluation"
        )

    return history, str(results_dir), model

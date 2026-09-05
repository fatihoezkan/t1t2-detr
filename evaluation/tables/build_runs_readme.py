"""results/README.md: the training procedure and one row per trained run.

Everything is read from results/<run>/config.yaml, summary.json and history.json, so the
table cannot drift from the runs. Parameter counts come from building the model the config
describes. Usage: PYTHONPATH=.:datagen python3 evaluation/tables/build_runs_readme.py
"""
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from t1t2.config import load_config
from t1t2.model import build_model

RES = ROOT / "results"
runs = sorted(p.parent.name for p in RES.glob("*/summary.json") if (p.parent / "config.yaml").exists())


def config_path(run):
    for p in (ROOT / "configs" / f"{run}.yaml", ROOT / "configs" / "seeds" / f"{run}.yaml",
              ROOT / "configs" / "combined" / f"{run}.yaml"):
        if p.exists():
            return p.relative_to(ROOT)
    return None


rows = []
for run in runs:
    cfg = load_config(RES / run / "config.yaml")
    s = json.load(open(RES / run / "summary.json"))
    n_params = sum(p.numel() for p in build_model(cfg.model).parameters())
    ckpt = RES / run / "checkpoints" / "best.pt"
    rows.append((run, config_path(run), cfg.train.seed, cfg.model.n_queries, cfg.model.n_dlayers,
                 cfg.model.exist_head, cfg.loss.t1_t2_weighting,
                 "on" if cfg.loss.signal_consistency else "off", n_params / 1e6,
                 s["epochs_run"], s["best_epoch"], s["early_stopped"], s["wall_seconds"] / 60,
                 ckpt.stat().st_size / 1e6 if ckpt.exists() else None))

ref = load_config(RES / "baseline_v2_reproduction" / "config.yaml")
t = ref.train
lines = [
    "# Trained runs",
    "",
    "One directory per run. `config.yaml` is the configuration exactly as it was trained (the",
    "same file as under `configs/`, with every default written out), `history.json` the",
    "per-epoch training and validation losses, `summary.json` the run summary, `metrics_detr.json`",
    "and `parameter_recovery_detr.json` the test metrics, `metrics_snr_ladder.json` the fixed-SNR",
    "test sets, and `threshold_calibration.json` the validation threshold search. The",
    "checkpoint `checkpoints/best.pt` is the model of the epoch with the lowest validation",
    "parameter loss.",
    "",
    "## Procedure, shared by every run",
    "",
    *textwrap.wrap(
        f"All runs use the same optimiser and schedule, read from the reference config: AdamW with "
        f"learning rate {t.lr:g}, weight decay {t.weight_decay:g}, betas {tuple(t.opt_betas)}, batch "
        f"size {t.batch_size}, gradient clipping at {t.gradient_clip_norm:g}, and the learning rate "
        f"halved after {t.scheduler_patience} epochs without improvement down to {t.scheduler_min_lr:g}. "
        f"The budget is {t.epochs} epochs; training stops after {t.early_stopping_patience} epochs "
        f"without an improvement of more than {t.early_stopping_min_delta:g} in the validation "
        "parameter loss (T1 + T2 + weight terms), and the checkpoint of the best such epoch is the "
        "model that is evaluated. Every run trains on the same 99,999 voxels and is scored on the "
        "same 9,999 test voxels; `data_loguniform` uses its own family. The runs were trained on one "
        "A100 each. What a run changes relative to the reference is stated in the `notes` field of "
        "its config.", width=96),
    "",
    "## The runs",
    "",
    "| run | config | seed | queries | decoder layers | existence head | loss weighting | consistency term | parameters | epochs run | best epoch | early stopped | wall (min) | best.pt (MB) |",
    "|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---|---:|---:|",
]
for (run, cfgp, seed, q, nl, head, weighting, sc, mp, ep, best, early, wall, size) in rows:
    lines.append(f"| `{run}` | `{cfgp}` | {seed} | {q} | {nl} | {head} | {weighting} | {sc} | "
                 f"{mp:.2f} M | {ep} | {best} | {'yes' if early else 'no'} | {wall:.0f} | "
                 f"{'' if size is None else f'{size:.0f}'} |")
lines += [
    "",
    "## Loading a checkpoint",
    "",
    "```python",
    "import torch",
    "from t1t2.config import load_config",
    "from t1t2.model import build_model",
    "",
    'cfg = load_config("results/loss_uniform/config.yaml")',
    "model = build_model(cfg.model)",
    'state = torch.load("results/loss_uniform/checkpoints/best.pt", map_location="cpu", weights_only=True)',
    'model.load_state_dict(state["model"])',
    "model.eval()",
    "```",
    "",
    "`best.pt` holds `model` (the state dict), `epoch`, `val` (the selection value), `val_loss`,",
    "`parameter_loss` and `selection_metric`. The model expects a batch of 64-point signals",
    "normalised by their own peak magnitude (`signal_norm: max`) and returns `(batch, queries, 4)`:",
    "T1 and T2 in the normalised [0, 1] space of `t1t2.data.TargetNormalizer`, the signal fraction,",
    "and the existence logit. `t1t2.eval.detr_query_outputs` does the conversion back to",
    "milliseconds.",
    "",
    "The checkpoints of `loss_uniform` and `baseline_v2_reproduction` are in the git repository.",
    "The other 24 are attached to the GitHub release as `checkpoints_best.zip`; unpacked at the",
    "repository root they land under `results/<run>/checkpoints/best.pt`, where every script and",
    "the notebook look for them (the README has the download command).",
    "",
]
(RES / "README.md").write_text("\n".join(lines))
print(f"wrote results/README.md with {len(rows)} runs")

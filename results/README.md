# Trained runs

One directory per run: `config.yaml` as it was trained (the file under `configs/` with every
default written out), `history.json` with the per-epoch losses, `summary.json`, the test
metrics `metrics_detr.json`, `parameter_recovery_detr.json` and `metrics_snr_ladder.json`,
`threshold_calibration.json` from the validation threshold search, and `checkpoints/best.pt`,
the epoch with the lowest validation parameter loss. `_comparison/`, `nd_evaluation/`,
`threshold_val/`, `threshold_sweep/` and `snr_ladder/` hold the outputs of the scripts in
`evaluation/`.

Directly under `results/`, `paired_tests.json` comes from `evaluation/paired_tests.py`,
`error_distribution_summary.json` from `evaluation/figures/make_error_distribution.py`, and the
two parquet files from the two noise scripts under `evaluation/figures/`. `review_stats.json`
and `nd_evaluation/tables_2d_3d.json` are frozen outputs; the scripts that built them are no
longer in the repository. The notebook reads the first for the per-count accuracies of the
reference and final families, and nothing reads the second.

Every run shares the same procedure: AdamW at learning rate 1e-4, weight decay 1e-4, betas
(0.9, 0.98), batch size 512, gradient clipping at 1, learning rate halved after 7 epochs
without improvement down to 1e-6, a budget of 500 epochs, and early stopping after 35 epochs
without an improvement of more than 1e-5 in the validation parameter loss. Every run trains
on the same 99,999 voxels and is scored on the same 9,999 test voxels (`data_loguniform` on
its own family), one A100 each. The `notes` field of each config says what the run changes.

| run | config | seed | queries | decoder layers | existence head | loss weighting | consistency term | parameters | epochs run | best epoch | early stopped | wall (min) | best.pt (MB) |
|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---|---:|---:|
| `aux_loss` | `configs/aux_loss.yaml` | 20260724 | 10 | 4 | joint | signal_fraction | off | 4.51 M | 181 | 146 | yes | 68 | 18 |
| `baseline_seed20260725` | `configs/seeds/baseline_seed20260725.yaml` | 20260725 | 10 | 4 | joint | signal_fraction | off | 4.51 M | 187 | 152 | yes | 30 | 18 |
| `baseline_seed20260726` | `configs/seeds/baseline_seed20260726.yaml` | 20260726 | 10 | 4 | joint | signal_fraction | off | 4.51 M | 204 | 169 | yes | 32 | 18 |
| `baseline_seed20260727` | `configs/seeds/baseline_seed20260727.yaml` | 20260727 | 10 | 4 | joint | signal_fraction | off | 4.51 M | 166 | 131 | yes | 26 | 18 |
| `baseline_v2_reproduction` | `configs/baseline_v2_reproduction.yaml` | 20260724 | 10 | 4 | joint | signal_fraction | off | 4.51 M | 199 | 164 | yes | 31 | 18 |
| `baseline_v3` | `configs/combined/baseline_v3.yaml` | 20260724 | 6 | 2 | shared | sqrt | on | 2.81 M | 151 | 116 | yes | 16 | 11 |
| `baseline_v3_no_physics` | `configs/combined/baseline_v3_no_physics.yaml` | 20260724 | 6 | 2 | shared | sqrt | off | 2.81 M | 183 | 148 | yes | 18 | 11 |
| `baseline_v3_no_sqrt` | `configs/combined/baseline_v3_no_sqrt.yaml` | 20260724 | 6 | 2 | shared | signal_fraction | on | 2.81 M | 152 | 117 | yes | 16 | 11 |
| `baseline_v4` | `configs/combined/baseline_v4.yaml` | 20260724 | 6 | 4 | shared | signal_fraction | on | 4.39 M | 158 | 123 | yes | 21 | 18 |
| `data_loguniform` | `configs/data_loguniform.yaml` | 20260724 | 10 | 4 | joint | signal_fraction | off | 4.51 M | 152 | 117 | yes | 25 | 18 |
| `decoder_2` | `configs/decoder_2.yaml` | 20260724 | 10 | 2 | joint | signal_fraction | off | 2.93 M | 169 | 134 | yes | 20 | 12 |
| `decoder_6` | `configs/decoder_6.yaml` | 20260724 | 10 | 6 | joint | signal_fraction | off | 6.09 M | 222 | 187 | yes | 45 | 24 |
| `exist_head_shared` | `configs/exist_head_shared.yaml` | 20260724 | 10 | 4 | shared | signal_fraction | off | 4.39 M | 169 | 133 | yes | 27 | 18 |
| `exist_weight_03` | `configs/exist_weight_03.yaml` | 20260724 | 10 | 4 | joint | signal_fraction | off | 4.51 M | 246 | 211 | yes | 40 | 18 |
| `final_uniform_q6_seed20260724` | `configs/seeds/final_uniform_q6_seed20260724.yaml` | 20260724 | 6 | 4 | joint | uniform | off | 4.44 M | 147 | 112 | yes | 20 | 18 |
| `final_uniform_q6_seed20260725` | `configs/seeds/final_uniform_q6_seed20260725.yaml` | 20260725 | 6 | 4 | joint | uniform | off | 4.44 M | 140 | 105 | yes | 19 | 18 |
| `final_uniform_q6_seed20260726` | `configs/seeds/final_uniform_q6_seed20260726.yaml` | 20260726 | 6 | 4 | joint | uniform | off | 4.44 M | 104 | 69 | yes | 14 | 18 |
| `final_uniform_q6_seed20260727` | `configs/seeds/final_uniform_q6_seed20260727.yaml` | 20260727 | 6 | 4 | joint | uniform | off | 4.44 M | 131 | 96 | yes | 17 | 18 |
| `loss_uniform` | `configs/loss_uniform.yaml` | 20260724 | 10 | 4 | joint | uniform | off | 4.51 M | 162 | 126 | yes | 26 | 18 |
| `loss_uniform_seed20260725` | `configs/seeds/loss_uniform_seed20260725.yaml` | 20260725 | 10 | 4 | joint | uniform | off | 4.51 M | 128 | 93 | yes | 21 | 18 |
| `loss_uniform_seed20260726` | `configs/seeds/loss_uniform_seed20260726.yaml` | 20260726 | 10 | 4 | joint | uniform | off | 4.51 M | 121 | 86 | yes | 19 | 18 |
| `loss_uniform_seed20260727` | `configs/seeds/loss_uniform_seed20260727.yaml` | 20260727 | 10 | 4 | joint | uniform | off | 4.51 M | 129 | 94 | yes | 21 | 18 |
| `physics_clean` | `configs/physics_clean.yaml` | 20260724 | 10 | 4 | joint | signal_fraction | on | 4.51 M | 254 | 219 | yes | 43 | 18 |
| `physics_noisy` | `configs/physics_noisy.yaml` | 20260724 | 10 | 4 | joint | signal_fraction | on | 4.51 M | 186 | 151 | yes | 31 | 18 |
| `queries_4` | `configs/queries_4.yaml` | 20260724 | 4 | 4 | joint | signal_fraction | off | 4.41 M | 160 | 125 | yes | 20 | 18 |
| `queries_6` | `configs/queries_6.yaml` | 20260724 | 6 | 4 | joint | signal_fraction | off | 4.44 M | 186 | 151 | yes | 25 | 18 |

## Loading a checkpoint

```python
import torch
from t1t2.config import load_config
from t1t2.model import build_model

cfg = load_config("results/loss_uniform/config.yaml")
model = build_model(cfg.model)
state = torch.load("results/loss_uniform/checkpoints/best.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state["model"])
model.eval()
```

`best.pt` holds `model` (the state dict), `epoch`, `val`, `val_loss`, `parameter_loss` and
`selection_metric`. The model expects a batch of 64-point signals normalised by their own
peak magnitude and returns `(batch, queries, 4)`: T1 and T2 in the normalised [0, 1] space of
`t1t2.data.TargetNormalizer`, the signal fraction and the existence logit;
`t1t2.eval.detr_query_outputs` converts back to milliseconds. `t1t2.runs.load_run` does all
of the above in one call. Only two checkpoints are in the repository. The root README says
where the other 24 come from.

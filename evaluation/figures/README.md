# Thesis figure scripts

Each script writes one thesis figure to `figures/` at the repository root. Run from there
with `PYTHONPATH=.:datagen python evaluation/figures/<script>.py`. Scripts that load a
checkpoint run inference on the CPU over the 9,999 test voxels, about a minute per model, at
the run's fitted threshold from `results/<run>/summary.json` (0.77 for the reference, 0.65
for `loss_uniform`).

| script | figure | needs |
|---|---|---|
| `make_relaxation_figure.py` | `00_relaxation.png` | nothing, analytic curves |
| `make_query_figure.py` | `11_queries.png` | `baseline_v2_reproduction` checkpoint, test parquets |
| `make_scatter_figure.py` | `12_pred_true_scatter.png` | both checkpoints, test parquets |
| `make_error_map.py` | `13_error_map.png` | `loss_uniform` checkpoint, test parquets |
| `make_found_scatter.py` | `14_found_missed.png` | both checkpoints, test parquets |
| `make_found_scatter.py --map7` | `15_found_missed_map7.png` | same, existence threshold dropped |
| `make_t2_profile.py` | `16_t2_profile.png` | both checkpoints, test parquets |
| `make_missed_dist.py` | `17_missed_dist.png` | both checkpoints, test parquets |
| `make_missed_scatter.py` | `17_missed_scatter.png` | both checkpoints, test parquets |
| `make_error_distribution.py` | `19_error_distribution.png`, `results/error_distribution_summary.json` | both checkpoints, test parquets |
| `make_noise_ratio_table.py` | `results/compartment_noise_ratio_test.parquet`, one row per true compartment | `results/nd_evaluation/<run>.json` for both runs, test parquets |
| `make_noise_effect_figure.py` | `20_noise_small_compartments.png`, `results/separability_k2_test.parquet` | the parquet from the previous script, `n2/test.parquet` |
| `plot_threshold_sweep.py` | `fig_threshold_sweep.png` | `results/threshold_sweep/<run>.json` for 12 runs |

"Both checkpoints" means `baseline_v2_reproduction` and `loss_uniform`, both in the
repository. Figure 18 (SNR ladder) comes from `evaluation/snr_ladder.py`; figures 01 to 10,
21 and `figures/arms/` are written by `notebooks/thesis.ipynb`, whose last cell runs every
script here in the right order.

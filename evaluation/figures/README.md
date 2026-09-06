# Thesis figure scripts

Each script writes one thesis figure to `figures/` at the repository root. Run from the
repository root: `PYTHONPATH=.:datagen python3 evaluation/figures/<script>.py`. Scripts that
load a checkpoint run inference on the CPU over the 9 999 test voxels named in the run's
`config.yaml`, about one minute per model. The fitted threshold they use is
`results/<run>/summary.json`, key `threshold_calibration.selected_threshold`
(0.77 for the reference, 0.65 for `loss_uniform`).

| Script | Figure | Needs |
|---|---|---|
| `make_relaxation_figure.py` | `00_relaxation.png` | nothing (analytic curves) |
| `make_query_figure.py` | `11_queries.png` | `baseline_v2_reproduction` checkpoint, config, summary.json, test parquets |
| `make_scatter_figure.py` | `12_pred_true_scatter.png` | checkpoints of `baseline_v2_reproduction` and `loss_uniform`, test parquets |
| `make_error_map.py` | `13_error_map.png` | `loss_uniform` checkpoint, test parquets |
| `make_found_scatter.py` | `14_found_missed.png` | both checkpoints, test parquets |
| `make_found_scatter.py --map7` | `15_found_missed_map7.png` | same, existence threshold dropped |
| `make_t2_profile.py` | `16_t2_profile.png` | both checkpoints, test parquets |
| `make_missed_dist.py` | `17_missed_dist.png` | both checkpoints, test parquets |
| `make_missed_scatter.py` | `17_missed_scatter.png` | both checkpoints, test parquets |
| `make_error_distribution.py` | `19_error_distribution.png`, `results/error_distribution_summary.json` | both checkpoints, summary.json, metrics_detr.json, test parquets |
| `make_noise_ratio_table.py` | `results/compartment_noise_ratio_test.parquet` (one row per true compartment: amplitude, noise, found flags) | `results/nd_evaluation/<run>.json` for both runs, `threshold_calibration.json`, test parquets, protocol |
| `make_noise_effect_figure.py` | `20_noise_small_compartments.png`, `results/separability_k2_test.parquet` | `results/compartment_noise_ratio_test.parquet`, `data/t1_3500_t2_500_100k/n2/test.parquet`, protocol |
| `plot_threshold_sweep.py` | `fig_threshold_sweep.png` | `results/threshold_sweep/<run>.json` for 12 runs |

Figure 18 (SNR ladder) comes from `evaluation/snr_ladder.py`; figures 01 to 10, 21 and
`figures/arms/` are written by `notebooks/thesis.ipynb` itself. Run
`make_noise_ratio_table.py` before `make_noise_effect_figure.py`; the notebook's last cell
runs every script in the right order when the checkpoints are present.

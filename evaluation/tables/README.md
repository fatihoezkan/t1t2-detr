# Thesis tables

Every LaTeX table of the thesis is written to `tables/` from the files under `results/`; nothing is typed by hand. Run all commands from the repository root with `PYTHONPATH=.:datagen`.

Order:

1. `python3 evaluation/tables/build_2d_3d_tables.py $(ls -d results/*/checkpoints | cut -d/ -f2)` writes `results/nd_evaluation/tables_2d_3d.json` (mAP@7 in 2D and 3D per run). Needs `results/<run>/checkpoints/best.pt`, `results/<run>/config.yaml` and the test parquet files named in the config. About 10 s per run on a CPU. Add `--verify` to compare against the stored entry without writing.
2. `python3 evaluation/tables/build_review_stats.py` writes `results/review_stats.json` (paired count and strict accuracy per K at the calibrated and the declared threshold, physical plausibility of the reported set, ground-truth separation). Needs the checkpoints of the three four-seed families and the test split; the raw query tables are cached in `results/_review_cache_<run>.npz` after the first run.
3. `python3 evaluation/tables/build_strict_tables.py` writes `tables/arms/<run>.tex` for the eleven single-change runs and `tables/tab_matrix.tex`.
4. `python3 evaluation/tables/build_baseline_tables.py` writes `tables/tab_baseline_perK.tex` and `tables/tab_seed_spread.tex`. It also writes `tables/tab_baseline.tex` and `tables/tab_baseline_own.tex` when the frozen v1 baseline `t1_3500_t2_500_weighted_long` is under `results/`; that run is not part of this repository, so these two are skipped here.
5. `python3 evaluation/tables/build_progression_table.py` writes `tables/tab_progression.tex`.
6. `python3 evaluation/tables/build_final_model_table.py` writes `tables/tab_final_model.tex`.

Steps 3 to 6 read only JSON: `results/threshold_sweep/`, `results/threshold_val/`, `results/nd_evaluation/<run>.json` (the per-voxel ND dumps), `tables_2d_3d.json`, `review_stats.json` and the per-run `metrics_detr.json`, `parameter_recovery_detr.json`, `summary.json`.

Two scripts print and write nothing: `seed_spread.py <run> <run> ...` reports the run-to-run spread of a group of runs, and `final_model_compare.py` applies the criteria fixed in `configs/seeds/final_uniform_q6_seed*.yaml` to the final model against the reference.

7. `python3 evaluation/tables/build_criteria_table.py` writes `tables/tab_criteria.tex`, the pre-specified criteria of the combined models v3 and v4 re-checked against `metrics_detr.json` and `parameter_recovery_detr.json`.
8. `python3 evaluation/tables/build_nd_table.py` writes `tables/tab_nd.tex`, mAP and exact F1 of every main-matrix run in the 2D and 3D forms, from `tables_2d_3d.json`. The frozen v1 baseline row of the thesis version is written only when that run is under `results/`.
9. `python3 evaluation/tables/build_runs_readme.py` writes `results/README.md`, the training procedure and one row per run, from the stored configs and summaries.

`tab_combined.tex`, `tab_physics.tex` and `tab_matrix_full.tex` exist in the thesis folder but are not used by the thesis and have no generator here; `tab_snr.tex` comes from `evaluation/snr_ladder.py` in a different layout.

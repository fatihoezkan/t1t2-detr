# The classical reference

A regularised NNLS T1-T2 spectrum (2D inverse Laplace transform) per voxel, segmented into peaks
and written in the same table shape as a DETR run, so that the classical method and every trained
model are scored by the same code on the same test voxels. The thesis reported no classical
comparison; this closes that gap for the conference paper.

The solver, the chi2 rule for the regularisation weight and the peak segmentation come from
Sebastian Endt's `baseline/nnls_baseline.py` in DETR_T1T2 (October 2026). What is new here is the
data loader for this repository's parquet families, a kernel built from `t1t2.physics` so the grid
atoms are this repository's forward model, and the npz output format.

## Method

Per voxel, on a log-spaced (T1, T2) grid restricted to T2 < T1, the same prior as the training
data:

1. Spectrum `f >= 0` from `min ||K f - s||^2 + lam^2 ||f||^2`, solved as NNLS on the augmented
   system `[K; lam I] f = [s; 0]` (zero-order Tikhonov). `K[:, m]` is the forward signal of one
   unit-weight compartment and is asserted against `t1t2.physics.forward_numpy`.
2. `lam` by the chi2 rule of multi-exponential relaxometry (Whittall and MacKay 1989; DECAES): the
   largest `lam` whose misfit stays within `--chi2_factor` (1.02) of unregularised NNLS, found by
   bisection in `log lam`. It needs neither the noise level nor the ground truth, so the same
   setting transfers to measured data.
3. Peaks by steepest ascent: every grid cell above a floor climbs to the largest value in its 3x3
   neighbourhood; cells sharing a maximum form one peak, whose signal fraction is its share of the
   spectrum and whose (T1, T2) is the amplitude-weighted geometric mean.
4. The `--n_slots` largest peaks are written as slots with confidence equal to the signal
   fraction, so mAP ranks NNLS peaks by fraction exactly as it ranks DETR queries by existence
   probability. The signal is normalised to unit L2 norm first, which makes `lam` scale-free.

## Scoring

`score_nnls.py` applies the functions the DETR tables use: `nd_metrics.dataset_records`,
`map_101_from_records`, `stratified_map` and `evaluation.threshold_sweep.score` / `score_by_k`.
Two details differ from a DETR run and both are properties of the method, not choices:

* Empty slots are dropped (`exist_thresh = 1e-12`). A slot with zero fraction is not a prediction,
  whereas all ten DETR queries carry a genuine existence probability.
* The threshold grid runs from 0.01 to 0.50. Confidence here is a signal fraction, so at three
  compartments no true peak can exceed about 0.33 and the DETR grid (0.05 to 0.95 on a
  probability) would discard correct answers by construction.

As for every arm, the operating point is calibrated on the validation split and applied unchanged
to test, once for best strict voxel accuracy and once for best F1. mAP needs no threshold.

## Results

Both grids on the same 9,999 test voxels, DETR families averaged over four seeds (range in
brackets for the two headline columns):

| method | mAP@5 | mAP@7 | mAP@10 | theta | strict acc | count acc |
|---|---:|---:|---:|---:|---:|---:|
| DETR reference | 0.5445 | 0.6642 [0.0168] | 0.7709 | 0.90 | 57.63 % [0.81] | 76.41 % |
| DETR `loss_uniform` | 0.6402 | 0.7635 [0.0136] | 0.8556 | 0.82 | 61.43 % [1.25] | 73.12 % |
| NNLS, grid 30 | 0.5957 | 0.6917 | 0.7781 | 0.09 | 53.29 % | 63.26 % |
| NNLS, grid 50 | 0.6024 | 0.6936 | 0.7793 | 0.09 | 52.86 % | 63.10 % |

Raising the grid from 771 to 2,149 atoms (15.8 % to 9.1 % quantisation per axis) moves mAP@7 by
0.0019 and leaves the weight-stratified values unchanged to three decimals, so the classical
result is converged rather than grid-limited.

The classical method beats the DETR reference on mAP at all three tolerances, by more than the
four-seed spread, and the margin is widest at the strictest tolerance: when it finds a compartment
it places it accurately. It is much weaker at deciding how many compartments a voxel holds (count
accuracy 63 % against 76 %), which is the familiar failure mode of inverse Laplace inversion.
`loss_uniform` beats it everywhere, and by the largest margin where the loss change helped: faint
compartments (stratified mAP@7 0.6673 against 0.4446) and three-compartment voxels (0.7569 against
0.6224).

## Running it

```bash
PYTHONPATH=.:datagen python baseline/nnls_baseline.py --n_grid 30 --run_name nnls_chi2_g30
PYTHONPATH=.:datagen python baseline/score_nnls.py nnls_chi2_g30
PYTHONPATH=.:datagen python baseline/compare_methods.py
```

Outputs go to `results/nnls/<run>/`: `<split>_predictions.npz` (regenerated, not versioned),
`info.json` with the settings and timings, and `nd_scores.json` with every metric.

Cost on an M3 Max with 12 workers and BLAS threads pinned to one per worker
(`OMP_NUM_THREADS=1` and the OPENBLAS, MKL and VECLIB equivalents): 0.24 s per voxel at grid 30,
about three minutes per 9,999-voxel split, and 5.0 s per voxel at grid 50, about 75 minutes.
Grid 70 costs 17 s per voxel and is not worth it. The solver is a CPU active-set method, so a GPU
does not help; without the thread pinning the workers oversubscribe the cores and the split takes
twice as long.

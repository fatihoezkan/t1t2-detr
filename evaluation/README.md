# Evaluation

Scripts that turn a finished run into numbers that can be compared between runs. `main.py`
at the root runs them in order; each also runs on its own from the repository root with
`PYTHONPATH=.:datagen`.

| script | what it writes |
|---|---|
| `run_nd_evaluation.py results/<run>` | `results/nd_evaluation/<run>.json`: mAP, exact metrics and the per-voxel ND records of one run |
| `calibrate_threshold.py <run> ...` | `results/threshold_val/<run>.json`: the validation-calibrated threshold and the test accuracy at it |
| `threshold_sweep.py <run> ...` | `results/threshold_sweep/<run>.json`: every metric at every threshold, in 2D and 3D |
| `summarize_nd_evaluation.py` | the combined ND table across runs and the paired mAP deltas |
| `compare_experiments.py --all` | the one-change comparison against the reference, with a verdict per arm |
| `paired_tests.py` | McNemar and paired-bootstrap tests of every arm against the reference |
| `snr_ladder.py` | the fixed-SNR figure and table; `--replot` redraws from `results/snr_ladder/summary.json` |
| `query_analysis.py <run>` | per-query usage of one run |
| `figures/` | one script per thesis figure |

Only `compare_experiments.py --all` runs on a fresh clone as it is. The first three need a
checkpoint and the generated data; `summarize_nd_evaluation.py` and `paired_tests.py` read
the per-voxel ND dumps that `run_nd_evaluation.py` writes, which are not shipped.

## What counts as a hit

Intersection over union has no meaning for a point in (T1, T2) space, so this work uses the
Normalised Distance (ND) criterion from Wirth's thesis. A prediction may be accepted for a
true compartment only if every feature deviates by less than tau, as a fraction of that
feature's range in log space:

    ND_T1 = |log T1_pred - log T1_true| / (log T1_max - log T1_min)

Among the candidates that pass, the assignment goes to the smallest ND sum; the sum ranks
but never accepts, so one badly wrong dimension cannot be averaged away. Log space makes tau
a relative error budget and matches the log-space costs of the training loss. The weight is
excluded from the test and reported separately.

## The metrics

- **mAP at tau** (5, 7 and 10 %): COCO-style 101-point average precision with the existence
  probability as confidence and no threshold. It measures how well the existence head ranks
  its own queries and is the primary metric.
- **Strict voxel accuracy**: the right number of compartments and every one inside the ND
  tolerance. Harsher than either half alone, and what an application would care about.
- **Precision, recall, F1 and errors** at one existence threshold per run. Errors are over
  true positives only, so they improve as recall falls and must be read next to the recall.

Every metric is also broken down by compartment count, because one-, two- and
three-compartment voxels are very different problems and an aggregate describes none of
them. CSF gets its own line: the longest echo time is 150 ms, so a 2000 ms T2 is close to
unconstrained.

## The existence threshold

Each run is reported at its own threshold, swept on the validation split for strict
accuracy and applied unchanged to test (`calibrate_threshold.py`). Models peak between 0.65
and 0.95, so one fixed threshold would handicap whichever sits furthest from it; the two
protocols agree on every arm except `queries_6` and `queries_4`, whose order swaps. Strict
accuracy is the right quantity to calibrate on because it has an interior optimum, unlike
F1, which saturates at the edge of the grid. Nothing is tuned on the test split.

Three thresholds live in `results/` and mean different things:

| where | field | what it is |
|---|---|---|
| `results/threshold_val/<run>.json` | `val_theta` | swept on validation for strict accuracy, 0.65 to 0.95; the one the thesis reports |
| `results/<run>/threshold_calibration.json` | `selected_threshold` | the parameter-error pipeline's threshold, 0.22 to 0.77 |
| `results/nd_evaluation/<run>.json` | `existence_threshold` | F1-calibrated on the grid 0.25 to 0.75; used for the ND precision, recall and F1 columns |

## How large a difference has to be

The reference retrained at four seeds gives the ruler: 0.81 pp on strict accuracy and
0.0168 on mAP@7 ([configs/README.md](../configs/README.md)). `paired_tests.py` goes further
where the data allows: two runs score the same test voxels, so strict and count accuracy
get McNemar's exact test, mAP a paired bootstrap over voxels, both Holm-Bonferroni
corrected, all from the stored ND records without any inference.

## Misnamed fields

`t1_mae_ms`, `t2_mae_ms` and `w_mae` in `metrics_detr.json` are medians. The finished runs
cannot be renamed, so the aliases `t1_abs_median_ms`, `t2_abs_median_ms` and `w_abs_median`
exist and `compare_experiments.py` reads those. The means are separate fields, and the gap
is itself a finding: on the reference the median T1 error is 28.5 ms against a mean of
115.4 ms, so the error is concentrated in a minority of hard voxels.

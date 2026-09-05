# Evaluation

The model emits a fixed number of candidate compartments per voxel, most of which it is
rejecting. Turning that into a number that can be compared between runs takes three
decisions: what counts as a hit, where the operating point sits, and how large a difference
has to be before it means anything.

## What counts as a hit

Object detection normally uses intersection over union, which has no meaning for a point in
(T1, T2) space. This work uses the Normalised Distance criterion from Wirth's
diffusion-correlation thesis instead. A prediction may be accepted for a ground-truth
compartment only if every feature deviates by less than tau, expressed as a fraction of that
feature's global range. Among the candidates that pass, the assignment goes to the smallest
ND sum. The sum ranks; it never accepts. Using it to accept would let one badly wrong
dimension be averaged away by the others.

The normalisation is in log space:

    ND_T1 = |log T1_pred - log T1_true| / (log T1_max - log T1_min)

T1 and T2 are log-distributed, so on a linear span a 35 ms error would be negligible at
T1 = 2000 ms and a different tissue at T2 = 30 ms. In log space tau is a relative error
budget, which also matches the log-space costs the training loss uses.

As in the original, the weight is excluded from the acceptance test and reported
separately.

## The metrics

mAP at tau is COCO-style 101-point interpolated average precision with the existence
probability as the confidence score, computed at tau = 5, 7 and 10 % with no existence
threshold. Being threshold-free, it measures how well the existence head ranks its own
queries, which is a property of the model rather than of the operating point. It is the
primary metric in [experiments.md](experiments.md).

Strict voxel accuracy is the fraction of voxels where the model reported the right number
of compartments and every one of them fell inside the ND tolerance. It is harsher than
either half alone, and it is what an application would care about.

Precision, recall, F1 and the mean errors are reported at a fixed existence threshold. The
mean absolute errors are over true positives only, which is safe here because every
true-positive error is bounded by the ND gate and gross misses are counted as false
positives rather than inflating the average. The true-positive mean error improves as
recall falls, so it must never be read without the recall next to it.

## Where the operating point sits

Two protocols are defensible and both are implemented.

Reporting every run at one declared threshold is comparable by construction, but not
neutral. Models peak at different thresholds, between 0.65 and 0.95 in this matrix, so a
single fixed value handicaps whichever models sit furthest from it.

Calibrating per run is fairer, provided it is done on validation.
`evaluation/calibrate_threshold.py` sweeps the threshold on the validation split, takes the
argmax of strict voxel accuracy, and applies that value unchanged to test. No test data
enters the choice. Strict accuracy is the right quantity to calibrate on because it punishes
over-reporting and under-reporting alike and therefore has an interior optimum, unlike F1,
which rises almost monotonically and saturates at the edge of the search grid.

The two protocols agree on every arm except `queries_6` and `queries_4`, whose ordering
swaps.

Nothing is tuned on the test split: not the existence threshold, and not anything else with
a knob.

## Three thresholds live in the results

They are easy to confuse and mean different things.

| where | field | range | what it is |
|---|---|---|---|
| `results/threshold_val/<run>.json` | `val_theta` | 0.65 to 0.95 | swept on validation for strict accuracy; the one the thesis reports |
| `results/<run>/threshold_calibration.json` | `selected_threshold` | 0.22 to 0.77 | the parameter-error pipeline's threshold, a different quantity |
| `results/nd_evaluation/<run>.json` | `existence_threshold` | 0.75 | fixed, used for the ND precision/recall columns |

## How large a difference has to be

A single training run gives a point estimate and no interval. The reference configuration
was therefore retrained at four seeds, and the spread across them is the ruler: 0.81 pp on
strict accuracy and 0.0168 on mAP at 7 %. A delta smaller than that is run-to-run variation.

`evaluation/paired_tests.py` goes further where the data allows. Two runs are scored on the
same test voxels, so the comparison is paired rather than independent: strict and count
accuracy get McNemar's exact test, mAP gets a paired bootstrap over voxels with both runs
scored on the same resample, and both families are corrected for multiplicity with
Holm-Bonferroni. No retraining or inference is needed, because the stored ND records
contain every prediction with its existence probability.

## Report per compartment count

The dataset balances one-, two- and three-compartment voxels, which are very different
problems. A single-compartment voxel is close to a curve fit; a three-compartment voxel is
ill-posed. An aggregate over the three describes none of them well, so every metric is also
broken down by count, and any claim about quality should be read that way.

CSF is reported on its own line for the same reason. The longest echo time in the protocol
is 150 ms, so exp(-TE/T2) barely moves for a 2000 ms pool and its T2 is close to
unconstrained. Leaving it in the aggregate misrepresents both halves.

## Three misnamed fields

`t1_mae_ms`, `t2_mae_ms` and `w_mae` in `metrics_detr.json` are computed with a median. The
finished runs cannot be renamed without invalidating them, so the aliases
`t1_abs_median_ms`, `t2_abs_median_ms` and `w_abs_median` exist and
`compare_experiments.py` reads those instead, labelling every such row as a median.

The means are separate fields, and the gap between them is itself a finding: on the
reference run the median T1 absolute error is 28.5 ms against a mean of 115.4 ms, a factor
of four, which says the error is concentrated in a minority of hard voxels.

## Running it

```bash
# per-run ND / mAP evaluation, writes results/nd_evaluation/<run>.json (needs the checkpoint)
PYTHONPATH=.:datagen python evaluation/run_nd_evaluation.py results/<run>

# combined table across every evaluated run
PYTHONPATH=.:datagen python evaluation/summarize_nd_evaluation.py

# validation-calibrated threshold and the test accuracy at it
PYTHONPATH=.:datagen python evaluation/calibrate_threshold.py <run> [<run> ...]

# paired significance tests against the reference
PYTHONPATH=.:datagen python evaluation/paired_tests.py

# the full comparison table, arm verdicts and footer
PYTHONPATH=.:datagen python evaluation/compare_experiments.py --all

# the reference and the final model on the fixed-SNR test sets, as a figure and a table
PYTHONPATH=.:datagen python evaluation/snr_ladder.py

# every thesis figure and every thesis table (see the README in each folder for the order)
PYTHONPATH=.:datagen python evaluation/figures/<script>.py
PYTHONPATH=.:datagen python evaluation/tables/<script>.py

# the executed notebook: data figures, paired voxel examples, every results table
PYTHONPATH=.:datagen python notebooks/build_thesis_notebook.py
```

`notebooks/thesis.ipynb` recomputes every table of this document from `results/` and shows
the figures in the order of the thesis.

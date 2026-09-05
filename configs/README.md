# The experiments

`configs/` holds the reference run, eleven arms that each change exactly one thing, and a
smoke config for exercising the pipeline. `configs/combined/` holds five models that change
several things at once, and `configs/seeds/` the ten seed replicates. Every run has a
directory under `results/` with the config as it was trained, the training curve and the
metrics.

The one-change rule is what makes the main table readable: when two things change at once a
difference cannot be attributed to either. `evaluation/compare_experiments.py` checks the
rule mechanically instead of trusting the file names.

Eight arms differ from the reference in a single config field. Three touch more than one
field and still count as one change:

| arm | fields touched | why it is still one change |
|---|---|---|
| `data_loguniform` | 3 | `train_path`, `val_path`, `test_path`, all pointing at the same new dataset |
| `physics_noisy` | 3 | `signal_consistency`, its weight and its warmup; switching the term on needs all three |
| `physics_clean` | 4 | the same three plus the target. It is one change relative to `physics_noisy`, which is how it should be read |

The comparison script collapses such groups into one diff, and only when the feature switch
itself flipped. Anything else that differs in more than one field is marked NOT
INTERPRETABLE; its numbers are printed but not treated as an ablation result.

## The reference

`baseline_v2_reproduction` is the same configuration as the original baseline, retrained
under a new name on the current code at seed 20260724. It exists so that every delta is
measured against a run produced by exactly the code in this repository.

## The ruler

A single run gives a point estimate with no interval, so a small delta means nothing on its
own. The reference was therefore retrained at four seeds, 20260724 through 20260727, and the
spread across them is what an effect has to clear. The seed replicates are
`configs/seeds/baseline_seed<seed>.yaml`, identical to the reference config except for
`train.seed` and the name.

| quantity | 20260724 | 20260725 | 20260726 | 20260727 | range |
|---|---:|---:|---:|---:|---:|
| strict voxel accuracy, calibrated threshold | 57.98 % | 57.64 % | 57.17 % | 57.75 % | 0.81 pp |
| count accuracy, calibrated threshold | 76.61 % | 76.69 % | 76.09 % | 76.24 % | 0.60 pp |
| mAP at tau = 7 % | 0.6671 | 0.6687 | 0.6520 | 0.6688 | 0.0168 |
| validation parameter loss (the selection metric) | 0.00810 | 0.00824 | 0.00833 | 0.00814 | 0.00023 |

The four runs are equally good on the metric they were selected by and still span 0.0168 in
mAP@7, so the selection metric does not track detection quality. Anything smaller than the
spread is run-to-run variation and is reported as flat.

## The single-change arms

Strict voxel accuracy requires the right number of compartments and every one of them
inside the ND tolerance. The threshold is calibrated per run on validation, which is why it
differs between rows; [the evaluation page](../evaluation/README.md) explains why that is the fairer
protocol. mAP is threshold-free. Every arm is a single run at seed 20260724 and is compared
against the reference at the same seed.

| arm | what changed | theta | strict acc | mAP@7 | precision | recall | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `baseline_v2_reproduction` | reference | 0.90 | 57.98 % | 0.6671 | 0.779 | 0.737 | reference |
| `loss_uniform` | `t1_t2_weighting` -> uniform | 0.85 | 62.13 % | 0.7719 | 0.846 | 0.767 | better |
| `physics_noisy` | signal-consistency loss on | 0.95 | 58.05 % | 0.6751 | 0.783 | 0.743 | flat |
| `aux_loss` | supervise every decoder layer | 0.90 | 58.23 % | 0.6721 | 0.784 | 0.741 | flat |
| `physics_clean` | signal consistency, clean target | 0.90 | 57.86 % | 0.6672 | 0.781 | 0.739 | flat |
| `queries_6` | `n_queries` 10 -> 6 | 0.90 | 57.76 % | 0.6604 | 0.803 | 0.727 | flat |
| `exist_head_shared` | existence head joint -> shared | 0.95 | 57.72 % | 0.6643 | 0.776 | 0.736 | flat |
| `decoder_6` | `n_dlayers` 4 -> 6 | 0.90 | 57.33 % | 0.6604 | 0.775 | 0.732 | flat |
| `queries_4` | `n_queries` 10 -> 4 | 0.85 | 57.31 % | 0.6529 | 0.810 | 0.719 | flat |
| `decoder_2` | `n_dlayers` 4 -> 2 | 0.95 | 57.21 % | 0.6581 | 0.778 | 0.732 | flat |
| `exist_weight_03` | `exist_weight` 0.1 -> 0.3 | 0.95 | 55.66 % | 0.6705 | 0.749 | 0.713 | worse |
| `data_loguniform` | log-uniform T1 dataset | 0.90 | 58.08 % | 0.6684 | 0.785 | 0.735 | not comparable |

One arm helped, one hurt, and nine stayed inside the seed spread.

`data_loguniform` is scored on a different voxel family, so its delta mixes the sampling
change with the difference between two independent draws. It is listed for completeness
and should not be read as an ablation result.

## loss_uniform at four seeds

`loss_uniform` changes a single config field and moves mAP at 7 % by 0.105 at the shared
seed, more than six times the seed spread. It was then retrained at the other three seeds
(`configs/seeds/loss_uniform_seed<seed>.yaml`), so the headline comparison is four runs
against four:

| quantity | reference, 4 seeds | `loss_uniform`, 4 seeds | delta | seed spread |
|---|---:|---:|---:|---:|
| strict voxel accuracy | 57.63 % [range 0.81] | 61.43 % [range 1.25] | +3.80 pp | 0.81 pp |
| mAP at tau = 7 % | 0.6642 [range 0.0168] | 0.7635 [range 0.0137] | +0.0993 | 0.0168 |
| count accuracy | 76.41 % | 73.12 % | -3.29 pp | 0.60 pp |
| strict accuracy, K = 2 | 65.49 % | 70.45 % | +4.96 pp | 1.02 pp |
| strict accuracy, K = 3 | 9.50 % | 16.16 % | +6.66 pp | 2.49 pp |

The per-seed strict accuracies of `loss_uniform` are 62.13, 61.00, 60.88 and 61.72 %, and
its per-seed mAP@7 values 0.7719, 0.7624, 0.7582 and 0.7615. Neither band comes close to
the reference's.

The field controls how much a compartment's own signal weight scales its contribution to
the T1/T2 loss. Under `signal_fraction`, a 5 % pool carries roughly fifteen times less
gradient than a 75 % pool. The reference model recovers small compartments poorly, and the
natural reading of that is the information limit of a 64-measurement protocol. Removing the
weighting shows that a substantial part of it was the loss instead. Among the faintest
compartments the median relative T1 error falls from 34.23 to 22.17 % and the T2 error from
39.41 to 27.56 %, both well past their seed spreads of 6.47 and 5.08 pp, while the share of
those compartments the model finds at all falls from 63.49 to 55.23 %. The loss weighting
governs where a compartment is placed; it does not decide whether a faint one is detected.

The switch is one config field but three code paths, all expressing the same idea: the
Hungarian assignment cost is no longer scaled by the true weight, matched per-pair T1/T2
errors are no longer scaled by it, and the per-voxel reduction becomes a plain mean over
matched pairs rather than `sum(w*e)/sum(w)`. It should be described as removing
signal-fraction weighting, not as changing one line.

## Combined models

Five models in `configs/combined/` change more than one thing. Each carries, in its `notes`
field, the success criteria that were written down before it was trained and how it did
against them. None of them beat the single change.

| model | what changed | runs | theta | strict acc | delta vs reference |
|---|---|---:|---:|---:|---:|
| `baseline_v3` | 2 decoder layers, 6 queries, shared existence head, sqrt(w) loss weighting, signal-consistency term | 1 | 0.85 | 59.35 % | +1.72 pp |
| `baseline_v3_no_sqrt` | v3 with the baseline's signal-fraction weighting back | 1 | 0.85 | 57.62 % | -0.02 pp |
| `baseline_v3_no_physics` | v3 without the signal-consistency term | 1 | 0.85 | 58.44 % | +0.81 pp |
| `baseline_v4` | 6 queries, shared existence head, signal-consistency term | 1 | 0.90 | 57.90 % | +0.27 pp |
| `final_uniform_q6` | uniform loss weighting and 6 queries | 4 | 0.69 | 60.76 % [range 1.58] | +3.13 pp |

v3 produced the lowest pooled median absolute T1 error of any single-seed run, 25.45 ms,
but failed two of its three pre-specified criteria: count accuracy 77.49 % against a
required 78.1, and a smallest-compartment T1 error of 29.62 % against a required 25. It was
rejected on those grounds. The two decomposition runs locate the cause. Putting the
signal-fraction weighting back costs 1.73 pp of strict accuracy, twice the seed spread, so
the flattened weighting was carrying most of what v3 gained and was also what cost the
counting. Switching the consistency term off instead moves the median T1 error from 25.45
to 30.03 ms while costing only 0.91 pp of strict accuracy: that term sharpens parameters
without solving many more voxels, as it did on its own.

v4 kept what the decomposition had cleared and dropped what it implicated. It passed two of
its three criteria, missed the T1 target by 0.41 ms, and gained 0.27 pp of strict accuracy,
inside the seed spread. Every component it kept had measured as flat on its own, and the
one it dropped, the flattened weighting, was the only part of v3 doing work.

`final_uniform_q6` combined the two changes that had earned a place and was trained at all
four seeds (`configs/seeds/final_uniform_q6_seed<seed>.yaml`). It reaches 60.76 % strict
accuracy against 61.43 % for `loss_uniform`, a difference of 0.67 pp inside the 0.81 pp
spread, and an mAP@7 of 0.7452 against 0.7635, a loss of 0.0183 against a spread of
0.0168. Per seed: 61.59, 60.82, 60.62 and 60.01 % strict accuracy, and 0.7550, 0.7487,
0.7407 and 0.7362 mAP@7. The combination is not additive: the query cut had gained against
the weighted loss and adds nothing once the weighting is fixed. The final model of the
thesis is therefore `loss_uniform` alone, because it is the simpler recipe.

## The physics term

Both physics arms are flat on the headline measures (58.05 % and 0.6751 for the noisy
target, 57.86 % and 0.6672 for the clean one, against 57.98 % and 0.6671). They do sharpen
the parameters of the compartments the signal already pins down: the pooled median absolute
T1 error moves from 28.50 ms to 26.71 ms with the measured target and to 26.94 ms with the
noise-free one, single runs each. The faint compartments the term was meant for do not
benefit, and the choice of target makes no difference. The
[signal-consistency section of the t1t2 README](../t1t2/README.md#the-signal-consistency-loss)
discusses the design and the reading.

## The fixed-SNR ladder

Every family ships five fixed-SNR test sets that hold the same voxels with the same
standardised noise draw, so the noise amplitude is the only thing that changes between
rungs. Means over the four seeds of each model, with the range over seeds in brackets;
SNR 20 is below the training range of 30 to 150 and is an extrapolation.

| SNR | reference strict acc. | final strict acc. | reference count acc. | final count acc. | reference rel. T1 err. | final rel. T1 err. |
|---:|---:|---:|---:|---:|---:|---:|
| 20 (extrapolation) | 35.30 [2.82] | 37.62 [1.80] | 62.45 [1.00] | 61.06 [1.40] | 12.37 [0.45] | 12.41 [0.09] |
| 40 | 49.08 [0.70] | 51.12 [1.12] | 72.14 [0.52] | 69.15 [1.44] | 7.62 [0.10] | 7.44 [0.21] |
| 60 | 55.90 [0.42] | 58.78 [0.70] | 75.82 [0.88] | 72.60 [1.32] | 6.05 [0.18] | 5.72 [0.09] |
| 100 | 61.07 [0.78] | 64.85 [1.42] | 77.88 [1.04] | 74.42 [1.86] | 4.95 [0.31] | 4.44 [0.08] |
| 150 | 62.34 [0.98] | 67.10 [1.60] | 78.02 [1.42] | 74.80 [1.32] | 4.53 [0.27] | 3.96 [0.09] |

The final model's gain in strict accuracy grows with the signal-to-noise ratio, from about
2 pp at SNR 40 to about 5 pp at SNR 150. Its count accuracy sits below the reference at
every rung, the same trade seen on the main test set. `evaluation/snr_ladder.py` reproduces
this table and the matching figure from `results/snr_ladder/summary.json`.

## Confounders

Three arms change more than their name suggests, and the qualification belongs in the same
sentence as the result.

`queries_4` and `queries_6` change the parameter count and the existence class balance
along with the query budget. At four queries with three compartments, the existence
`pos_weight` reaches its clamp floor of 0.50, where the term stops up-weighting positives
and starts down-weighting them. Both arms also show the pattern that comes with fewer
slots: precision rises (0.810 and 0.803 against the reference's 0.779) while recall falls,
because there are fewer queries available to propose a duplicate but also fewer to catch a
third compartment.

`exist_head_shared` changes the parameter count by about 2.6 %, so it is a capacity change
as well as a wiring change.

## Two threshold protocols

The tables above report each run at its own validation-calibrated threshold. Reporting
every run at one declared threshold is also defensible and is comparable by construction,
but it is not neutral: models peak at different thresholds, between 0.65 and 0.95 here, so
a single fixed value handicaps whichever models sit furthest from it. The two protocols
agree on every arm except `queries_6` and `queries_4`, whose ordering swaps. Both are
computable from the stored results, and `evaluation/calibrate_threshold.py` produces the
first.

## Reproducing an arm

```bash
PYTHONPATH=.:datagen python -m t1t2.experiment --config configs/loss_uniform.yaml
PYTHONPATH=.:datagen python evaluation/run_nd_evaluation.py results/loss_uniform
PYTHONPATH=.:datagen python evaluation/calibrate_threshold.py loss_uniform
```

`python main.py train evaluate --runs loss_uniform` does the same from the repository root.
The configs reference dataset paths under `data/`. Generate the families first with the
commands in the [root README](../README.md#data) or `python main.py data`, or edit the paths
to point at your own.

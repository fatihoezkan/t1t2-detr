# The experiments

One YAML per trained run. `configs/*.yaml` holds the reference, eleven arms that each change
exactly one thing, and `smoke.yaml`; `combined/` holds five models that change several
things at once and `seeds/` the ten seed replicates. Every run has a directory under
`results/` with its config, training curve and metrics, and the `notes` field of each config
says what it changes and why.

`evaluation/compare_experiments.py` checks the one-change rule mechanically. Three arms touch
more than one field and still count as one change: `data_loguniform` (three dataset paths),
`physics_noisy` (the term, its weight and its warmup) and `physics_clean` (the same plus the
target). Anything else that differs in more than one field is marked NOT INTERPRETABLE.

## The reference and the ruler

`baseline_v2_reproduction` is the original baseline retrained on the current code at seed
20260724, so that every delta is measured against this repository's code. It was retrained
at three more seeds (`seeds/baseline_seed<seed>.yaml`), and the spread across the four is
what a change has to clear. Anything smaller is run-to-run variation and reported as flat.

| quantity | 20260724 | 20260725 | 20260726 | 20260727 | range |
|---|---:|---:|---:|---:|---:|
| strict voxel accuracy, calibrated threshold | 57.98 % | 57.64 % | 57.17 % | 57.75 % | 0.81 pp |
| count accuracy, calibrated threshold | 76.61 % | 76.69 % | 76.09 % | 76.24 % | 0.60 pp |
| mAP at tau = 7 % | 0.6671 | 0.6687 | 0.6520 | 0.6688 | 0.0168 |
| validation parameter loss (the selection metric) | 0.00810 | 0.00824 | 0.00833 | 0.00814 | 0.00023 |

The four runs tie on the metric they were selected by and still span 0.0168 in mAP@7, so
the selection metric does not track detection quality.

## The single-change arms

Every arm is one run at seed 20260724, scored on the same 9,999 test voxels at its own
validation-calibrated threshold theta ([evaluation/README.md](../evaluation/README.md)
explains the protocol). Strict accuracy needs the right number of compartments and every one
inside the ND tolerance; mAP is threshold-free.

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

One arm helped, one hurt, nine stayed inside the seed spread. `data_loguniform` is scored on
a different voxel family, so it is not an ablation result.

`queries_4` and `queries_6` also change the parameter count and the existence class balance;
at four queries the existence `pos_weight` hits its clamp floor of 0.50. Both show the
pattern of fewer slots, higher precision and lower recall. `exist_head_shared` changes the
parameter count by about 2.6 %, so it is a capacity change as well.

## loss_uniform at four seeds

`loss_uniform` moves mAP@7 by 0.105 at the shared seed, six times the spread, and was
retrained at the other three seeds (`seeds/loss_uniform_seed<seed>.yaml`):

| quantity | reference, 4 seeds | `loss_uniform`, 4 seeds | delta | seed spread |
|---|---:|---:|---:|---:|
| strict voxel accuracy | 57.63 % [range 0.81] | 61.43 % [range 1.25] | +3.80 pp | 0.81 pp |
| mAP at tau = 7 % | 0.6642 [range 0.0168] | 0.7635 [range 0.0137] | +0.0993 | 0.0168 |
| count accuracy | 76.41 % | 73.12 % | -3.29 pp | 0.60 pp |
| strict accuracy, K = 2 | 65.49 % | 70.45 % | +4.96 pp | 1.02 pp |
| strict accuracy, K = 3 | 9.50 % | 16.16 % | +6.66 pp | 2.49 pp |

Per seed, `loss_uniform` reaches 62.13, 61.00, 60.88 and 61.72 % strict accuracy and
0.7719, 0.7624, 0.7582 and 0.7615 mAP@7.

`t1_t2_weighting` controls how much a compartment's signal weight scales its T1/T2 loss.
Under `signal_fraction` a 5 % pool carries about fifteen times less gradient than a 75 %
pool. Removing the weighting places the faintest compartments much better (median relative
T1 error 34.23 % to 22.17 %, T2 error 39.41 % to 27.56 %) but finds fewer of them (63.49 %
to 55.23 %) and costs count accuracy. The weighting governs where a compartment is placed,
not whether a faint one is detected. It is one config field but three code paths: the
matching cost, the per-pair errors and the per-voxel reduction all stop being scaled by the
true weight.

## Combined models

Each config in `combined/` carries in its `notes` the success criteria written down before
training and how it did against them. None beat the single change.

| model | what changed | runs | theta | strict acc | delta vs reference |
|---|---|---:|---:|---:|---:|
| `baseline_v3` | 2 decoder layers, 6 queries, shared existence head, sqrt(w) loss weighting, signal-consistency term | 1 | 0.85 | 59.35 % | +1.72 pp |
| `baseline_v3_no_sqrt` | v3 with the baseline's signal-fraction weighting back | 1 | 0.85 | 57.62 % | -0.02 pp |
| `baseline_v3_no_physics` | v3 without the signal-consistency term | 1 | 0.85 | 58.44 % | +0.81 pp |
| `baseline_v4` | 6 queries, shared existence head, signal-consistency term | 1 | 0.90 | 57.90 % | +0.27 pp |
| `final_uniform_q6` | uniform loss weighting and 6 queries | 4 | 0.69 | 60.76 % [range 1.58] | +3.13 pp |

`baseline_v3` had the lowest median T1 error of any single-seed run (25.45 ms) but failed
two of its three criteria. Its two decomposition runs show that the flattened weighting did
the work and also cost the counting, while the consistency term sharpened parameters without
solving more voxels. `baseline_v4` kept only the components that had measured flat and
gained nothing. `final_uniform_q6` (uniform weighting plus 6 queries, four seeds) reaches
60.76 % against 61.43 % for `loss_uniform` and 0.7452 mAP@7 against 0.7635: the query cut
adds nothing once the weighting is fixed, so `loss_uniform` alone is the final model.

## The physics term

Both physics arms are flat on the headline measures. They sharpen the parameters of the
compartments the signal already pins down (pooled median absolute T1 error 28.50 ms to
26.71 ms with the measured target and 26.94 ms with the clean one) but not the faint
compartments the term was meant for, and the target makes no difference. The clean arm is
the informative half: with the best target the term could have, it still does not help, so
the Hungarian loss already constrains the prediction enough. The design is in
[t1t2/README.md](../t1t2/README.md#the-signal-consistency-loss).

## The fixed-SNR ladder

Five fixed-SNR test sets hold the same voxels with the same standardised noise draw, so only
the noise amplitude changes between rungs. Means over four seeds, range in brackets; SNR 20
is below the training range of 30 to 150.

| SNR | reference strict acc. | final strict acc. | reference count acc. | final count acc. | reference rel. T1 err. | final rel. T1 err. |
|---:|---:|---:|---:|---:|---:|---:|
| 20 (extrapolation) | 35.30 [2.82] | 37.62 [1.80] | 62.45 [1.00] | 61.06 [1.40] | 12.37 [0.45] | 12.41 [0.09] |
| 40 | 49.08 [0.70] | 51.12 [1.12] | 72.14 [0.52] | 69.15 [1.44] | 7.62 [0.10] | 7.44 [0.21] |
| 60 | 55.90 [0.42] | 58.78 [0.70] | 75.82 [0.88] | 72.60 [1.32] | 6.05 [0.18] | 5.72 [0.09] |
| 100 | 61.07 [0.78] | 64.85 [1.42] | 77.88 [1.04] | 74.42 [1.86] | 4.95 [0.31] | 4.44 [0.08] |
| 150 | 62.34 [0.98] | 67.10 [1.60] | 78.02 [1.42] | 74.80 [1.32] | 4.53 [0.27] | 3.96 [0.09] |

The final model's gain grows with SNR, from about 2 pp at SNR 40 to 5 pp at SNR 150, and
its count accuracy sits below the reference at every rung. `evaluation/snr_ladder.py`
reproduces the table and the figure.

## Reproducing an arm

```bash
PYTHONPATH=.:datagen python -m t1t2.experiment --config configs/loss_uniform.yaml
PYTHONPATH=.:datagen python evaluation/run_nd_evaluation.py results/loss_uniform
PYTHONPATH=.:datagen python evaluation/calibrate_threshold.py loss_uniform
```

or `python main.py --force train evaluate --runs loss_uniform` (`--force` because the
shipped `results/loss_uniform/` already counts as finished). Generate the data first with
`python main.py data`.

# t1t2

The model, its loss, the training loop and the scoring code. One run is one YAML under
`configs/`; `python -m t1t2.experiment --config <yaml>` trains it, scores it on the test
split and writes `results/<name>/`. Run from the repository root with `PYTHONPATH=.:datagen`.

| file | what it does |
|---|---|
| `config.py` | the run definition as dataclasses; a misspelled key is an error |
| `data.py` | parquet to tensors: log-min-max normalisation of T1/T2, per-voxel peak normalisation of the signal |
| `model.py` | MLP encoder, learned queries, transformer decoder, heads for T1, T2, weight and existence |
| `loss.py` | Hungarian matching, regression on the matched pairs, existence BCE; the `t1_t2_weighting` switch |
| `physics.py` | the forward model in numpy and differentiable torch; same equation as `datagen/voxel_simulator/physics.py` |
| `physics_loss.py` | the signal-consistency term, below |
| `train.py` | the training loop: early stopping on validation parameter loss, resumable checkpoints, `history.json` |
| `experiment.py` | train, calibrate the existence threshold on validation, score test and the SNR ladder, write `summary.json` |
| `eval.py` | predictions to metrics: matching, count and parameter errors, per-count breakdown, threshold search |
| `nd_metrics.py` | the Normalised Distance criterion, mAP and strict voxel accuracy |
| `runs.py` | `load_run("results/<name>")`: config, best checkpoint and normaliser, used by every script in `evaluation/` |
| `device.py` | CUDA, then Apple MPS, then CPU |

The docstring at the top of each file explains the non-obvious choices.

## The signal-consistency loss

`physics_loss.py` pushes the predicted compartments back through the forward model and
penalises the mean squared error against a target signal:

    signal -> model -> compartments -> physics -> signal

Every query contributes `w_q * sigmoid(exist_logit_q)`, so no matching is involved and the
existence head receives a physics gradient of its own. At initialisation every query is half
open and the signal comes out over-predicted, so the weight is ramped in over the first
epochs. The resynthesised signal gets the same per-voxel peak normalisation as the input.
The target is `noisy` (the measured input, needs no labels) or `clean` (the noise-free
signal of the true parameters, an upper bound only a simulation can provide). MSE is the
right likelihood because the signals keep their sign; the Rician caveat applies to
magnitude data only.

Both arms, `configs/physics_noisy.yaml` and `configs/physics_clean.yaml`, came out flat;
the numbers are in [configs/README.md](../configs/README.md#the-physics-term).

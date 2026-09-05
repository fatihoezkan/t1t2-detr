# t1t2

The model, its loss, the training loop and the scoring code. One run is one YAML under
`configs/`; `python -m t1t2.experiment --config <yaml>` trains it, scores it on the test
split and writes `results/<name>/`. Run from the repository root with `PYTHONPATH=.:datagen`.

| file | what it does |
|---|---|
| `config.py` | the run definition as typed dataclasses; a misspelled key is an error, and a retired key is accepted only at the value every run used |
| `data.py` | parquet to tensors: log-min-max normalisation of T1/T2 for the sigmoid heads, per-voxel peak normalisation of the signal, the table width read from the columns |
| `model.py` | MLP encoder, learned queries, transformer decoder, per-query heads for T1, T2, weight and existence; two existence-head wirings |
| `loss.py` | Hungarian matching, regression on the matched pairs, existence BCE; the `t1_t2_weighting` switch the thesis is about |
| `physics.py` | the forward model on the training side, in numpy and in differentiable torch |
| `physics_loss.py` | the signal-consistency term, described below |
| `train.py` | the training loop: early stopping on the validation parameter loss, resumable checkpoints, `history.json` |
| `experiment.py` | train, calibrate the existence threshold on validation, score the test split and the fixed-SNR ladder, write `summary.json` |
| `eval.py` | predictions to metrics: matching to the truth, count and parameter errors, per-count breakdown, physics checks, the threshold search |
| `nd_metrics.py` | the Normalised Distance criterion, mAP and the exact metrics the thesis reports |
| `runs.py` | `load_run("results/<name>")`: config, best checkpoint, normaliser and log spans, for every script under `evaluation/` |
| `device.py` | CUDA, then Apple MPS, then CPU |

The docstring at the top of each file says why the non-obvious choices were made. The
evaluation protocol is in [evaluation/README.md](../evaluation/README.md) and the experiment
matrix in [configs/README.md](../configs/README.md).

## The signal-consistency loss

The Hungarian loss supervises parameters given a matching. This term supervises the
predicted set as a whole, through the physics, with no matching involved: the prediction is
pushed back through the same inversion-recovery forward model that produced the data, and
the mismatch with the measured signal is penalised.

    signal -> model -> compartments -> physics -> signal

The appeal is that it rules out predictions that are physically inconsistent with the
measurement even when they look reasonable in parameter space, and that in its
self-supervised form it needs no ground truth, so it would carry over unchanged to real
scans.

The implementation is `t1t2/physics_loss.py`, the differentiable forward model is
`t1t2/physics.py`, and the two arms are `configs/physics_noisy.yaml` and
`configs/physics_clean.yaml`.

### Design choices

Gating is soft. Every query contributes `w_q * sigmoid(exist_logit_q)` to the resynthesis,
with no Hungarian indices involved. That keeps the term differentiable end to end and gives
the existence head a physics gradient of its own: a query that claims to exist while ruining
the resynthesis gets pushed down. The price is that at initialisation every query is half
open and the signal comes out over-predicted, which is why the weight is ramped in linearly
over the first epochs.

The resynthesised signal goes through the same per-voxel normalisation as the input. The
dataset divides every signal by its own peak magnitude before the model sees it, and the
weights the model predicts live in that rescaled space. Without the transform the two sides
of the comparison would differ by an arbitrary per-voxel factor.

The target is configurable. `noisy` compares against the input signal itself, which needs
no labels; the noise is zero-mean, so the gradient stays unbiased and the loss floor is the
per-voxel noise power. `clean` compares against the noise-free forward model of the true
parameters, which only a simulation can provide, and is the upper bound the noisy arm is
measured against. Comparing the two says how much of the noisy arm's behaviour comes from
its noisy target rather than from the physics term.

The metric is mean squared error, not a Rician likelihood. The simulated signals keep the
sign of the inversion recovery, so Gaussian noise and MSE are the correct likelihood. The
Rician caveat from the relaxometry literature applies to magnitude data, which this is not.

The term is applied to the final prediction only, never to the auxiliary heads. Asking the
first decoder layer to already explain the measured signal is not a useful constraint.

### Result

Both arms came out flat. `physics_noisy` reaches mAP at 7 % of 0.6751 and strict accuracy
of 58.05 %, against the reference's 0.6671 and 57.98 %; `physics_clean` reaches 0.6672 and
57.86 %. The seed spread on those quantities is 0.0168 and 0.81 pp, so neither arm clears
it.

The clean arm is the informative half of the result. It has the noise-free target, which is
the best case the term could ever have, and it does not help either. That points at the term
itself rather than at the noise in the self-supervised target: the Hungarian loss already
constrains the prediction enough that re-imposing the forward model adds little.

The term does sharpen the parameters of the compartments the signal already pins down (the
pooled median absolute T1 error moves from 28.50 ms to 26.71 ms with the measured target
and 26.94 ms with the clean one), but not the faint compartments it was meant for. It is a
result about this problem at this protocol, not about physics-informed losses in general.

### Reproducing

```bash
PYTHONPATH=.:datagen python -m t1t2.experiment --config configs/physics_noisy.yaml
PYTHONPATH=.:datagen python -m t1t2.experiment --config configs/physics_clean.yaml
```

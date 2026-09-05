# The signal-consistency loss

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

## Design choices

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
Rician caveat from the relaxometry literature applies to magnitude data, which this is not,
and requesting `rician` raises an error rather than quietly doing the wrong thing.

The term is applied to the final prediction only, never to the auxiliary heads. Asking the
first decoder layer to already explain the measured signal is not a useful constraint.

## Result

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

## Reproducing

```bash
PYTHONPATH=.:datagen python -m t1t2.experiment --config configs/physics_noisy.yaml
PYTHONPATH=.:datagen python -m t1t2.experiment --config configs/physics_clean.yaml
```

The checks that were run before those arms were submitted are kept as tests in
`tests/test_physics_loss.py`: the denormalisation and signal-normalisation round trips
against the dataset's own transforms, agreement between the clean target and the generator,
a near-zero loss when the prediction is exactly right, a clear separation when it is wrong,
gradient flow into all four output channels, gated-off queries contributing nothing, and an
error when a Rician term is requested.

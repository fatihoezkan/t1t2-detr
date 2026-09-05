# Data generation

There are no public T1-T2 correlation datasets at the density this work needs, so all
training data is simulated. This document describes what is simulated, what is deliberately
not, and the one sampling choice that affects how the results may be read.

## The forward model

For a protocol point p with inversion time TI_p and echo time TE_p, and compartments c with
relaxation times (T1_c, T2_c) and signal weights w_c:

    S_p = M0 * sum_c  w_c * (1 - 2 exp(-TI_p / T1_c) + exp(-TR / T1_c)) * exp(-TE_p / T2_c)

The bracket is the inversion recovery in T1 and the trailing exponential is the T2 decay.
The protocol is 64 points, 8 inversion times by 8 echo times, read from
`datagen/data/ti_te_dict.mat` and used exactly as stored: position p has to mean the same
(TI_p, TE_p) at generation, at training and at inference, so the arrays are never sorted or
regrouped.

The same equation is implemented twice, in `datagen/voxel_simulator/physics.py` for
generation and in `t1t2/physics.py` for the signal-consistency loss, and a test checks that
the two agree.

## Noise

The signal keeps its sign. The inversion-recovery curve goes negative just after the
inversion pulse, and this data is not a magnitude image, so the correct noise model is
additive Gaussian. Rician noise, which is standard for magnitude data, would rectify the
negative values away and change the problem.

The noise level is set either by SNR, where sigma is the peak clean signal divided by the
SNR, or by an absolute sigma. Noise is drawn as a standardised z and then scaled, rather
than drawn directly at the target scale. That split is what makes the fixed-SNR ladder a
paired comparison: every rung gets the same voxels with the same z and differs only in
amplitude, so SNR is the only variable that changes between rungs.

## What is not modelled

The compartments are random. A compartment is a point in (T1, T2) space subject only to
T1 > T2, with no tissue prototypes and no attempt at anatomical realism. This is deliberate.
The task being posed is finding compartments under noise, and a realistic tissue prior would
let the model succeed by reciting the prior rather than by reading the signal.

The consequence is stated in the thesis and repeated here: results on this data say what
the method can do on this problem. They do not by themselves establish in-vivo performance.

## The sampling constraint

A compartment's transverse relaxation time cannot exceed its longitudinal one. This is a
physical constraint, and everything below follows from enforcing it.

Working in logs, with a = log T1_min, b = log T1_max, c = log T2_min and d = log T2_max,
the feasible region is everything under the diagonal:

    {(x, y) : a <= x <= b,  c <= y <= min(d, x)}

On the ranges used for the reported experiments, T1 from 50 to 3500 ms and T2 from 5 to
500 ms, that is a triangle joined to a rectangle: below T1 = 500 ms the diagonal cuts the
T2 range short, and above it the full range is available.

Two samplers are implemented, and the choice between them decides the training-data
coverage. Coverage is a confounder for any plot of error against T1 or T2, which is why the
choice is exposed at all: if short-T1 compartments are rarer in training, a large error at
short T1 could mean that short T1 is hard to estimate, or only that the model saw fewer of
them.

`rejection` is the default. Draw log T1 and log T2 independently log-uniform and keep the
pair only if T2 < T1. Accepted pairs are uniform over the feasible region, so the log-T1
marginal density is proportional to the feasible log-T2 width at that T1, and neither
marginal is log-uniform. Measured on the ranges above, T1 in [50, 100) is drawn at 0.66
times the log-uniform expectation, T1 in [500, 3500) at 1.16 times, and T2 in [100, 500] at
0.75 times; a KS test against log-uniform rejects log T1 at D = 0.0835. Acceptance is
0.8645, about 1.16 draws per compartment.

`t1_log_uniform` is opt-in. Draw log T1 log-uniform over its full range, then log T2
log-uniform on [c, min(d, log T1)]. There is no rejection step and the log-T1 marginal is
exactly log-uniform, which removes T1 coverage as an explanation for a T1-dependent error
trend.

Neither mode can flatten both marginals, and that is a property of the constraint rather
than a shortcoming of either sampler. If log T2 were flat over [log 5, log 500], a voxel at
T1 = 50 ms could never carry a T2 above 50 ms, while a compartment at T2 = 400 ms would
force T1 above 400 ms. The upper part of the T2 range is reachable only from the upper part
of the T1 range, so a flat log-T2 marginal needs extra mass at large T1, which is what
breaks a flat log-T1 marginal.

The second mode also makes the T2 marginal worse: when T1 is small the conditional T2 range
is narrow, so the density piles up at small T2. The confounder moves from T1 to T2 rather
than disappearing. The `data_loguniform` arm trains on such a dataset; its results are in
[experiments.md](experiments.md).

The mode is recorded in each dataset's `manifest.json` under `physics.sampling`, because
which error curve can be trusted depends on which sampler produced the data.

## Weights and counts

Weights come from a symmetric Dirichlet, rescaled so that every compartment clears a floor
of 5 %. Below that a compartment is invisible in the signal, and asking a model to find it
is asking it to guess.

The compartment count is an input, not a draw. Each file is generated at one fixed count,
which is what makes the per-count splits exactly balanced. It also has to work that way
because the RNG streams are keyed on the count.

## Reproducibility

Every random number comes from `SeedSequence([base_seed, n_comp, split_code, voxel_id,
stream_id])`. SeedSequence rather than seed arithmetic guarantees that two different keys
cannot collide on the same state however large a split grows; an earlier scheme spaced
master seeds by a fixed stride and silently collided once a split outgrew it.

The parameter, noise and SNR streams are separate. That is what lets the fixed-SNR ladder
pin the SNR without disturbing the parameter or noise draws.

Every family writes a `manifest.json` recording the seeds, the split sizes, the ranges, the
sampling mode, the git commit, a SHA-256 of the protocol file and the library versions.
Writes go through a temporary file and a rename, so a crash cannot leave a truncated parquet
under a name that looks finished.

NumPy does not guarantee that Generator bit streams stay stable across versions, which is
why `numpy` is pinned exactly in `requirements.txt` and why the manifest records the version
that produced each dataset. The manifests of the two families used in the thesis are kept
under `data/`.

## Usage

The exact commands for the two families used in the thesis are in the README. In general:

```bash
# one family per compartment count
for n in 1 2 3; do
  PYTHONPATH=.:datagen python datagen/run_generator.py --n-comp $n --out-dir data/<family>/n$n
done

# tiny files, for a quick check
PYTHONPATH=.:datagen python datagen/run_generator.py --n-comp 2 --smoke --out-dir data/smoke/n2
```

Each family writes `train`, `val` and `test` splits plus `test_snr{20,40,60,100,150}`. The
SNR 20 rung is below the training range of 30 to 150 on purpose and is reported as
extrapolation.

`--help` lists every option. The ones to know are `--sampling`, described above,
`--n-train`, `--n-val` and `--n-test` for the split sizes, and `--seed`, which has to differ
between two families that are meant to be independent draws.

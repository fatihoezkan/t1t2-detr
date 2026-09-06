# Data generation

All training data is simulated; no public T1-T2 correlation dataset exists at this density.
The compartments are random points in (T1, T2) space subject only to T1 > T2, with no tissue
prototypes, on purpose: a realistic tissue prior would let the model recite the prior instead
of reading the signal. Results on this data therefore do not establish in-vivo performance.

| file | what it does |
|---|---|
| `voxel_simulator/protocol.py` | the 8 x 8 acquisition protocol, read from `data/ti_te_dict.mat` exactly as stored, never sorted |
| `voxel_simulator/physics.py` | the inversion-recovery multi-echo forward model |
| `voxel_simulator/sampler.py` | random compartments with T1 > T2, Dirichlet weights and SNR, seeded per voxel |
| `voxel_simulator/noise.py` | signed additive Gaussian noise, drawn as a standardised z and scaled |
| `voxel_simulator/generate.py` | one family: train, val, test, the fixed-SNR ladder and the manifest |
| `run_generator.py` | the command-line entry point; `--help` lists every option |

## Forward model and noise

    S_p = M0 * sum_c  w_c * (1 - 2 exp(-TI_p / T1_c) + exp(-TR / T1_c)) * exp(-TE_p / T2_c)

for the 64 protocol points p and compartments c with (T1_c, T2_c, w_c). The same equation
lives in `t1t2/physics.py` for the signal-consistency loss; the two must stay in step.

The signal keeps its sign (the curve goes negative after the inversion pulse), so the noise
is additive Gaussian rather than Rician. Sigma is the peak clean signal divided by the SNR,
or an absolute value with `--noise-sigma`. Noise is drawn as a standardised z and then
scaled, so the fixed-SNR test sets hold the same voxels with the same z and differ only in
amplitude.

## Sampling

T2 cannot exceed T1, so in log space the feasible region is a triangle joined to a
rectangle (T1 from 50 to 3500 ms, T2 from 5 to 500 ms). Two samplers, chosen with
`--sampling`:

- `rejection` (default): draw log T1 and log T2 independently log-uniform and keep the pair
  if T2 < T1. Uniform over the feasible region, so neither marginal is log-uniform: T1 in
  [50, 100) is drawn at 0.66 times the log-uniform rate, T1 in [500, 3500) at 1.16 times.
- `t1_log_uniform`: draw log T1 log-uniform, then log T2 log-uniform on
  [log T2_min, min(log T2_max, log T1)]. The T1 marginal is exactly log-uniform, but the T2
  density piles up at small T2.

Neither mode can flatten both marginals; the constraint forbids it. Coverage confounds any
plot of error against T1 or T2, which is why the mode is recorded in each `manifest.json`
under `physics.sampling`. The shipped `t1_3500_t2_500_100k` family predates that field and
used `rejection`; the `data_loguniform` arm trains on the other mode.

Weights come from a symmetric Dirichlet rescaled so that every compartment has at least 5 %;
below that it is invisible in the signal. The compartment count is fixed per file, which is
what makes the per-count splits exactly balanced.

## Reproducibility

Every random number comes from `SeedSequence([base_seed, n_comp, split_code, voxel_id,
stream_id])`, with separate parameter, noise and SNR streams, so keys cannot collide and the
SNR ladder can pin the SNR without disturbing the other draws. Each family writes a
`manifest.json` with seeds, sizes, ranges, sampling mode, git commit, protocol checksum and
library versions. Writes go through a temporary file and a rename, so a crash cannot leave a
truncated parquet. NumPy bit streams are not stable across versions, which is why `numpy` is
pinned in `requirements.txt`.

## Usage

```bash
# one family per compartment count; the thesis families are in the root README and data/README.md
for n in 1 2 3; do
  PYTHONPATH=.:datagen python datagen/run_generator.py --n-comp $n --out-dir data/<family>/n$n
done

# tiny files, for a quick check
PYTHONPATH=.:datagen python datagen/run_generator.py --n-comp 2 --smoke --out-dir data/smoke/n2
```

Each family writes `train`, `val`, `test` and `test_snr{20,40,60,100,150}`; SNR 20 is below
the training range of 30 to 150 and is reported as extrapolation. `--seed` has to differ
between two families that are meant to be independent draws.

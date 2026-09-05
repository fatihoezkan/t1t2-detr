# Data

The two dataset families the thesis uses, as manifests only. The parquet files are not in
the repository; they are regenerated with

```bash
python main.py data
```

which runs the `datagen/run_generator.py` commands given in the
[root README](../README.md#data). [datagen/README.md](../datagen/README.md) describes what
is simulated.

| family | used by | sampler | seed |
|---|---|---|---|
| `t1_3500_t2_500_100k/` | every run except `data_loguniform` | `rejection` | 3500500 |
| `t1_loguniform_100k/` | `data_loguniform` | `t1_log_uniform` | 3500501 |

Each family has one folder per compartment count, `n1`, `n2` and `n3`, holding the `train`,
`val` and `test` splits and five fixed-SNR test sets `test_snr{20,40,60,100,150}`; SNR 20 is
below the training range of 30 to 150 and is an extrapolation test. `manifest.json` records
the seeds, sizes, ranges, protocol checksum, git commit and library versions of the files
that were actually used; the sampling mode appears only in the log-uniform family's
manifests, the main family predates that field and used `rejection`. The main family regenerates exactly under the pinned
numpy; the log-uniform family was generated under a newer numpy and regenerates in
distribution but not voxel for voxel.

`dev/` is the small development dataset for `configs/smoke.yaml`, generated with the
`--smoke` command in the [root README](../README.md#setup). It is not versioned.

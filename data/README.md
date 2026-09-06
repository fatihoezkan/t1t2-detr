# Data

Manifests of the two dataset families the thesis uses. The parquet files (about 60 MB per
family) are not in the repository; `python main.py data` regenerates them in about a minute.

| family | used by | sampler | seed |
|---|---|---|---|
| `t1_3500_t2_500_100k/` | every run except `data_loguniform` | `rejection` | 3500500 |
| `t1_loguniform_100k/` | `data_loguniform` | `t1_log_uniform` | 3500501 |

Each family has one folder per compartment count (`n1`, `n2`, `n3`) holding `train`, `val`,
`test` and the fixed-SNR sets `test_snr{20,40,60,100,150}`: 33,333 / 3,333 / 3,333 voxels
per count and 1,667 per SNR rung. `manifest.json` records the seeds, sizes, ranges, protocol
checksum, git commit and library versions of the files that were actually used.

The main family is the data command in the root README. The second is the same command with
`--out-dir data/t1_loguniform_100k/n$n --seed 3500501 --sampling t1_log_uniform`. The main
family regenerates exactly under the pinned numpy. The log-uniform family was generated under
a newer numpy, so it regenerates in distribution but not voxel for voxel.

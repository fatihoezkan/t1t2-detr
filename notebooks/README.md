# The thesis notebook

`thesis.ipynb` is the executed notebook that walks through the data, paired voxel examples
under the reference and the final model, and every results table of the thesis recomputed
from `results/`. It is the place to start reading the repository, and it is the only source:
edit it in Jupyter and re-execute it in place with

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/thesis.ipynb
```

or `python main.py --force notebook` from the repository root. The first cell moves to the
repository root by itself, so the command works from either directory.

Parts 1 and 2 run inference and need `results/<run>/checkpoints/best.pt` for
`baseline_v2_reproduction` and `loss_uniform`; the raw query tables are cached under
`.cache_visuals/` after the first run, delete it to force a re-run. Part 3 reads only
`results/` and works on a fresh clone. Figures 01 to 10 and 21 of the thesis and
`figures/arms/` are written by the notebook itself, and its last cell reruns every script
under `evaluation/figures/` and `evaluation/tables/` when the checkpoints are present, so an
execution also refreshes `figures/` and `tables/`.

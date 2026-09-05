# The thesis notebook

`thesis.ipynb` is the executed notebook that walks through the data, paired voxel examples
under the reference and the final model, and every results table of the thesis recomputed
from `results/`. It is the place to start reading the repository.

`build_thesis_notebook.py` writes and executes it:

```bash
PYTHONPATH=.:datagen python notebooks/build_thesis_notebook.py
```

Parts 1 and 2 run inference and need `results/<run>/checkpoints/best.pt` for
`baseline_v2_reproduction` and `loss_uniform`; the raw query tables are cached under
`.cache_visuals/` after the first run. Part 3 reads only `results/` and works on a fresh
clone. Figures 01 to 10 and 21 of the thesis and `figures/arms/` are written by the notebook
itself, and its last cell reruns every script under `evaluation/figures/` and
`evaluation/tables/` when the checkpoints are present, so a rebuild also refreshes
`figures/` and `tables/`. `python main.py notebook` does the same from the repository root.

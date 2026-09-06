# The thesis notebook

`thesis.ipynb` walks through the data (Part 1), paired voxel examples under the reference and
the final model (Part 2) and every results table of the thesis recomputed from `results/`
(Part 3). Part 3 works on a fresh clone. Parts 1 and 2 run inference and need
`results/<run>/checkpoints/best.pt`; the per-arm galleries of section 2.9 need every
checkpoint from the release zip (root README, Checkpoints). Inference results are cached
under `.cache_visuals/`; delete it to force a rerun.

Re-execute it in place with

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/thesis.ipynb
```

or `python main.py --force notebook`. The first cell moves to the repository root by itself.
The notebook writes figures 01 to 10, 21 and `figures/arms/` itself, and its last cell reruns
`evaluation/snr_ladder.py` and every script under `evaluation/figures/` when the checkpoints
are present.

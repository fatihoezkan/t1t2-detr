# Tests

59 tests for the `t1t2` package; the generator has its own 68 under `datagen/tests/`.

```bash
PYTHONPATH=.:datagen python -m pytest tests -q
```

`conftest.py` generates a small development dataset under `data/dev/` on the first run
(2,000 training voxels per compartment count, fully seeded) and reuses it afterwards; delete
`data/dev/` to regenerate it. `configs/smoke.yaml` trains on the same files.

| file | what it covers |
|---|---|
| `test_smoke.py` | config round trip, the normaliser, dataset shapes and peak normalisation, forward and loss shapes, the auxiliary-loss path, the compartment-table width read from the columns |
| `test_pipeline.py` | forward-model parity with the generator, training and resume, early stopping, refusal to resume into a foreign config, the metrics and the fixed-SNR ladder |
| `test_parameter_first_training.py` | the loss weighting rules, the validation threshold search, checkpoint and scheduler metadata |
| `test_physics_loss.py` | the signal-consistency term against the dataset transforms and the generator, gradients and gating; the physics configs are one change from the reference |
| `test_compare_experiments.py` | the one-change verdicts, the metric table, median labelling and the footer, on synthetic results |

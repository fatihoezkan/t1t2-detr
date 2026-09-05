# Running on a cluster

The reported runs were trained on a single A100. Nothing in the code requires that: the
device is auto-detected, so the same config runs on CPU or Apple MPS, only slower.

`train.slurm` is a generic template. It does not set a partition, QoS, account or GPU type,
because those are site-specific. Add the `#SBATCH` lines your cluster needs or pass them on
the command line.

```bash
mkdir -p slurm/logs
sbatch --job-name=loss_uniform slurm/train.slurm configs/loss_uniform.yaml
```

Set `T1T2_VENV` to your virtual environment if it needs activating inside the job.

## Resuming

Training is resumable. A checkpoint carries the model, the optimiser state, the epoch
counter and the early-stopping patience, so resubmitting the same command continues the run
instead of restarting it.

The resume is stateful, not a bit-identical replay: the dataloader's shuffle order is not
restored, so a resumed run and an uninterrupted one diverge from the point of interruption.
A resume into a results directory whose stored config differs from the incoming one is
refused, because it would produce a model that is half one experiment and half another.

## Installing torch

`requirements.txt` does not list torch, so that pip cannot replace a working CUDA build with
a CPU-only wheel. Install it separately, matching the CUDA version on the node:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Generating data on the node

The datasets are large enough that generating them where they will be used beats copying
them. The generator is deterministic given its seed, so regenerating from the same seed on
another machine gives the same voxels, provided the numpy version matches the pin in
`requirements.txt`. Each family's `manifest.json` records the version that produced it. The
exact commands for the two families used in the thesis are in the README.

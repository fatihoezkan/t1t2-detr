# T1T2-DETR

Code and results for the bachelor's thesis *Detection Transformer for Microstructure
Quantification from T1-T2 Correlation MRI* (Fatih Özkan, Technische Hochschule Ingolstadt,
2026, supervised by Sebastian Endt in the group of Prof. Dr. Marion Menzel).

An MRI voxel holds water in several compartments, each relaxing with its own T1 and T2. A
T1-T2 correlation scan records 64 numbers per voxel (8 inversion times x 8 echo times), and
working back from those 64 numbers to the compartments is an ill-posed inverse problem. The
thesis treats it as object detection. A Detection Transformer (DETR) reads the 64
measurements and returns a fixed set of candidate compartments, each with a T1, a T2, a
signal fraction and an existence score. During training the candidates are matched to the
true compartments with the Hungarian algorithm. The architecture follows DETR (Carion et al.
2020) and the diffusion-correlation DETRs of Schlund and Wirth. [CREDITS.md](CREDITS.md)
says what was inherited and what is new.

## Layout

```
main.py                 runs the pipeline: data, train, evaluate, aggregate, figures, notebook
datagen/                synthetic data generator: forward model, sampler, noise
t1t2/                   model, loss, training loop, scoring
evaluation/             scripts that score finished runs and compare them
evaluation/figures/     one script per thesis figure
configs/                one YAML per trained run, and the experiment matrix with its results
results/                config, training curve and metrics of every run; two checkpoints
data/                   dataset manifests (the parquet files are regenerated)
figures/                the thesis figures
notebooks/thesis.ipynb  executed walk-through: data, example voxels, every results table
```

Start with `notebooks/thesis.ipynb`. Its Part 3 runs on a fresh clone. Every folder has a
short README. The experiments and their numbers are in [configs/README.md](configs/README.md),
the evaluation protocol in [evaluation/README.md](evaluation/README.md).

![Training: from a YAML file to a scored run](docs/img/training.png)

![Inference: from one voxel to its compartments](docs/img/inference.png)

## Setup

```bash
pip install -r requirements.txt
pip install torch          # or the CUDA wheel from pytorch.org
```

Python 3.12 was used; 3.9 or newer should work. Torch is not in `requirements.txt` so that pip
cannot replace a CUDA build with a CPU one.
`numpy` is pinned because the generator's random streams depend on the version. Every
command below runs from the repository root with `PYTHONPATH=.:datagen`.

## Trained models

The trained models are on the release page of this repository,
https://github.com/fatihoezkan/t1t2-detr/releases. `checkpoints_best.zip` holds one `best.pt`
per run, the epoch with the lowest validation parameter loss, 11 to 24 MB each. Two of them,
the final model `loss_uniform` and the reference `baseline_v2_reproduction`, are also
committed under `results/`, so a plain clone can run the notebook's paired examples and every
figure script; the per-arm galleries and the SNR ladder need the rest.

Download the zip and unpack it at the repository root. Every file lands in
`results/<run>/checkpoints/best.pt`, where all scripts look for it; downloading the zip from
the release page in a browser and unzipping it in the repository folder does the same.

```bash
curl -L -o checkpoints_best.zip https://github.com/fatihoezkan/t1t2-detr/releases/download/v1.1.0/checkpoints_best.zip
unzip -o checkpoints_best.zip
python main.py --dry-run        # reports no missing checkpoint once every run has one
```

To use a model, load its run and give it a 64-point signal in the protocol's stored order,
divided by its own peak. The model returns ten candidates; the existence threshold the run
chose on its validation split decides which ones count. The example takes the first
two-compartment test voxel (`python main.py data` writes the file); a real scan's 64
measurements go in the same way.

```python
import numpy as np, pandas as pd, torch
from t1t2.runs import load_run

run = load_run("results/loss_uniform")                        # config, best.pt and normaliser, on the CPU
voxel = pd.read_parquet("data/t1_3500_t2_500_100k/n2/test.parquet").iloc[0]
signal = voxel[[f"S_{i}" for i in range(1, 65)]].to_numpy(float)          # S_1 .. S_64
x = torch.tensor(signal / np.abs(signal).max(), dtype=torch.float32)[None]  # peak-normalised, batch axis
with torch.no_grad():
    out = run.model(x)[0]                                     # (10, 4): T1, T2, weight in [0, 1], existence logit
keep = torch.sigmoid(out[:, 3]) > run.fitted_threshold        # the run's validation-calibrated threshold
t1 = run.normalizer.denormalize_t1(out[keep, 0].numpy())      # back to milliseconds
t2 = run.normalizer.denormalize_t2(out[keep, 1].numpy())
w = out[keep, 2].numpy()                                      # signal fractions
for a, b, c in zip(t1, t2, w):
    print(f"T1 {a:.0f} ms, T2 {b:.0f} ms, weight {c:.2f}")
# T1 119 ms, T2 102 ms, weight 0.31 and T1 417 ms, T2 27 ms, weight 0.66;
# the truth is 129 / 107 / 0.28 and 402 / 29 / 0.72
```

`run.predict("test")` does the same for a whole split, and [results/README.md](results/README.md)
shows what a checkpoint file holds and the lower-level way to load it.

## Reproducing

`main.py` runs the stages in order and skips every step whose outputs already exist:

```bash
python main.py                                        # all stages
python main.py evaluate figures --runs loss_uniform   # some stages, some runs
python main.py --force figures                        # redo even if the outputs exist
python main.py --dry-run                              # print the plan
```

The same stages as single commands:

```bash
# data: one file per compartment count (data/README.md has the second family)
for n in 1 2 3; do
  PYTHONPATH=.:datagen python datagen/run_generator.py --n-comp $n \
      --out-dir data/t1_3500_t2_500_100k/n$n --seed 3500500 \
      --n-train 33333 --n-val 3333 --n-test 3333 --n-per-snr 1667 \
      --t1-min 50 --t1-max 3500 --t2-min 5 --t2-max 500
done

# train: one run, about 30 min on one A100, writes results/<name>/; resumes from last.pt
PYTHONPATH=.:datagen python -m t1t2.experiment --config configs/loss_uniform.yaml

# evaluate: evaluation/README.md lists every script
PYTHONPATH=.:datagen python evaluation/run_nd_evaluation.py results/loss_uniform
PYTHONPATH=.:datagen python evaluation/calibrate_threshold.py loss_uniform
PYTHONPATH=.:datagen python evaluation/compare_experiments.py --all

# notebook
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/thesis.ipynb
```

Training all 26 runs takes about a day on a GPU node. Submit one config per cluster job and
run the other stages afterwards.

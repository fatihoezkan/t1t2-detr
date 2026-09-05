#!/usr/bin/env bash
#SBATCH --job-name=t1t2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.out
#
# Train one experiment arm. Submit with the config as the only argument:
#
#     sbatch --job-name=loss_uniform slurm/train.slurm configs/loss_uniform.yaml
#
# Partition, QoS, account and the exact GPU type are site-specific and are deliberately not
# set here. Add the #SBATCH lines your cluster needs, or pass them on the sbatch command line.
#
# Wall time: a full run early-stops well inside the 500-epoch budget and takes about half an
# hour on an A100 for the datasets used here. Four hours covers the full budget. If the wall is
# hit anyway, resubmit: training resumes from last.pt.
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
CFG="${1:?usage: sbatch slurm/train.slurm <config.yaml>}"

cd "${REPO}"
[[ -f "${CFG}" ]] || { echo "no such config: ${CFG}" >&2; exit 1; }

# Activate a virtual environment if T1T2_VENV is set.
if [[ -n "${T1T2_VENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "${T1T2_VENV}/bin/activate"
fi

export PYTHONPATH="${REPO}:${REPO}/datagen:${PYTHONPATH:-}"
mkdir -p slurm/logs

# Provenance: which commit and which library versions produced this results directory.
echo "host        $(hostname)"
echo "commit      $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
echo "dirty       $(git status --porcelain 2>/dev/null | wc -l) file(s)"
echo "config      ${CFG}"
python - <<'PY'
import torch, numpy, sys
print(f"python      {sys.version.split()[0]}")
print(f"torch       {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"numpy       {numpy.__version__}")
if torch.cuda.is_available():
    print(f"gpu         {torch.cuda.get_device_name(0)}")
PY

srun python -m t1t2.experiment --config "${CFG}"

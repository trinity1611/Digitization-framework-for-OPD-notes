#!/bin/bash
# Phase 3: create the isolated conda environment "rxbench" (cluster guide v3.0, section 4).
#
# Run it on a GPU node, not on the master node, so that conda-forge selects the CUDA build of PyTorch:
#
#     srun --partition=normal --gres=gpu:h200_nvl:1 --cpus-per-task=8 --mem=32G --time=01:00:00 \
#          bash ~/prescription_vlm_baseline/jobs/01_create_env.sh
#
# It never runs `conda init`, creates one environment, keeps package caches on /scratch, and makes no
# system-wide change. Creating an environment writes many small files to /home and takes several
# minutes; do not start a second one at the same time.

module purge
module load Miniforge3/26.1.1-3
set -eo pipefail
source "$HOME/prescription_vlm_baseline/jobs/common.sh"
cd "$RXB_PROJECT"

EXISTING_ENVS="$(conda env list | awk '{print $1}')"
if grep -qx "$RXB_ENV" <<< "$EXISTING_ENVS"; then
    echo "Environment '$RXB_ENV' already exists; reusing it."
else
    conda create -y -n "$RXB_ENV" -c conda-forge python=3.11 pytorch torchvision
fi
# shellcheck disable=SC1091
source activate "$RXB_ENV"

python - <<'PY'
import sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA build", torch.version.cuda,
      "| CUDA available", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("ERROR: this PyTorch build cannot use CUDA. Do not continue; paste this output to the agent.")
print("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
PY

pip install -r requirements.txt

# Record exactly what was installed (small files, written once into the project).
mkdir -p docs/environment
pip freeze > docs/environment/cluster_pip_freeze.txt
conda list --explicit > docs/environment/cluster_conda_explicit.txt
python -m rxbench hardware-report --out docs/environment/HARDWARE_REPORT_cluster.md \
    --role-note "Slurm cluster compute node, inside the rxbench environment." > /dev/null
python -m rxbench models list
python -m pytest -q --ignore=tests/test_app_pages.py 2>&1 | tail -n 5 || true
echo "Environment '$RXB_ENV' is ready."

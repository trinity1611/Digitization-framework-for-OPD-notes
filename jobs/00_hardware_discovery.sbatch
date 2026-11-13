#!/bin/bash
#SBATCH --job-name=rxb_hw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --partition=normal
#SBATCH --time=00:15:00
#SBATCH --output=slurm_%x_%j.out

# Phase 1: inspect the compute node BEFORE any environment is created or any package installed.
# Uses only the Python that comes with the Miniforge3 module (standard library). Installs nothing.
# To inspect the other GPU type instead: sbatch --gres=gpu:pro_6000:1 jobs/00_hardware_discovery.sbatch

module purge
module load Miniforge3/26.1.1-3
set -euo pipefail
source "$HOME/prescription_vlm_baseline/jobs/common.sh"

WORK="$RXB_SCRATCH/$SLURM_JOB_ID"
mkdir -p "$WORK"
cd "$RXB_PROJECT"

echo "===== uname -a";            uname -a
echo "===== /etc/os-release";     head -n 4 /etc/os-release
echo "===== lscpu";               lscpu | head -n 25 || true
echo "===== free -h";             free -h
echo "===== df -h";               df -h /scratch "$HOME" /data 2>/dev/null || true
echo "===== nvidia-smi";          nvidia-smi
echo "===== python";              which python; python --version
echo "===== modules";             module avail 2>&1 | head -n 60 || true
echo "===== groups";              id -Gn

python -m rxbench.hardware.diagnostics \
    --out "$WORK/HARDWARE_REPORT_pre_env.md" --json "$WORK/hardware_pre_env.json" \
    --path "scratch=$RXB_SCRATCH" --path "home=$HOME" --path "data=/data/$RXB_TEAM" \
    --role-note "Slurm cluster compute node, inspected inside Slurm job $SLURM_JOB_ID before any environment was created." \
    > /dev/null

mkdir -p "$RXB_RESULTS/$SLURM_JOB_ID"
cp "$WORK/HARDWARE_REPORT_pre_env.md" "$WORK/hardware_pre_env.json" "$RXB_RESULTS/$SLURM_JOB_ID/"
rm -rf "$WORK"
echo "Report: $RXB_RESULTS/$SLURM_JOB_ID/HARDWARE_REPORT_pre_env.md"

#!/bin/bash
#SBATCH --job-name=rxb_offline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%x_%j.out

# Phase 11: offline verification with real inference.
#
# Two layers of isolation:
#   1. If the node allows it, the whole check runs in a private network namespace (`unshare -rn`),
#      which has no network interface at all: a real "network disabled" test.
#   2. Inside Python, a guard blocks and records every attempt to connect or resolve a host name.
# The report says which layer(s) applied, and lists a model as verified only if real inference
# completed under them.
#
#   sbatch jobs/04_offline_check.sbatch
#   sbatch --export=ALL,RXB_RUN=medgemma_27b jobs/04_offline_check.sbatch

source "$HOME/prescription_vlm_baseline/jobs/common.sh"
rxb_activate
set -uo pipefail
cd "$RXB_PROJECT"

MODELS="${RXB_RUN:-primary}"
WORK="$RXB_SCRATCH/$SLURM_JOB_ID"
DATA="$RXB_SCRATCH/datasets/dataset_english"
mkdir -p "$WORK/reports" "$DATA"
export RXBENCH_EXPERIMENTS_DIR="$WORK/experiments" RXBENCH_STATE_DIR="$WORK/state" RXBENCH_LOG_DIR="$WORK/logs"
export RXBENCH_HF_CACHE_DIR="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

(
    flock 9
    if [ ! -f "$DATA/.complete" ]; then
        tar -xf "$RXB_DATASET_TAR" -C "$DATA" --strip-components=1
        touch "$DATA/.complete"
    fi
) 9>"$DATA.lock"
rxb_stage_models || exit 1
IMAGE="$(ls "$DATA"/images/* | head -n 1)"

CHECK=(python -m rxbench offline-check --models "$MODELS" --image "$IMAGE" --inference
       --out "$WORK/reports/OFFLINE_VERIFICATION.md")
if unshare -rn true 2>/dev/null; then
    echo "Running in a private network namespace (no network interfaces)."
    unshare -rn "${CHECK[@]}"
else
    echo "Network namespaces are not available to this user; using the in-process guard only."
    "${CHECK[@]}"
fi
RC=$?

mkdir -p "$RXB_RESULTS"
tar -cf "$RXB_RESULTS/rxbench_offline_${SLURM_JOB_ID}.tar" -C "$WORK" experiments reports logs state 2>/dev/null
rm -rf "$WORK"
echo "Results: $RXB_RESULTS/rxbench_offline_${SLURM_JOB_ID}.tar"
exit $RC

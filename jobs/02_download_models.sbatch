#!/bin/bash
#SBATCH --job-name=rxb_dl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%x_%j.out

# Phase 4: download official model checkpoints (pinned revisions) as a Slurm job.
# Downloads go to /scratch and are copied once to /data/<team>/$USER/rxbench_models.
#
# Submitting with RXB_CONFIRM_DOWNLOAD=yes is the explicit confirmation of the download:
#
#   sbatch --export=ALL,RXB_CONFIRM_DOWNLOAD=yes jobs/02_download_models.sbatch                  # 3 primary models, about 184 GB
#   sbatch --export=ALL,RXB_CONFIRM_DOWNLOAD=yes,RXB_GET=medgemma_27b jobs/02_download_models.sbatch
#   sbatch --export=ALL,RXB_CONFIRM_DOWNLOAD=yes,RXB_GET=optional jobs/02_download_models.sbatch # 3 optional models, about 116 GB
#
# Before the first run: accept the MedGemma terms on huggingface.co and run `hf auth login` once
# (see docs/MODEL_DOWNLOAD_GUIDE.md).

source "$HOME/prescription_vlm_baseline/jobs/common.sh"
rxb_activate
set -uo pipefail
set -e
cd "$RXB_PROJECT"

MODELS="${RXB_GET:-primary}"
WORK="$RXB_SCRATCH/$SLURM_JOB_ID"
mkdir -p "$WORK"
export RXBENCH_STATE_DIR="$WORK/state" RXBENCH_LOG_DIR="$WORK/logs" RXBENCH_EXPERIMENTS_DIR="$WORK/experiments"
export RXBENCH_HF_CACHE_DIR="$HF_HOME/hub"

echo "===== plan"
python -m rxbench models plan "$MODELS"
echo "===== access"
python -m rxbench models access "$MODELS"

if [ "${RXB_CONFIRM_DOWNLOAD:-no}" != "yes" ]; then
    echo "Download NOT started: resubmit with --export=ALL,RXB_CONFIRM_DOWNLOAD=yes to confirm."
    rm -rf "$WORK"
    exit 2
fi

echo "===== download (resumes if this job is submitted again)"
python -m rxbench models download "$MODELS" --yes --workers 8
echo "===== verify"
python -m rxbench models verify "$MODELS" --check-remote

echo "===== copy once to $RXB_MODEL_STORE"
mkdir -p "$RXB_MODEL_STORE"
for src in "$HF_HOME"/hub/models--*; do
    [ -d "$src/snapshots" ] || continue
    name="$(basename "$src")"
    rsync -a --partial "$src/" "$RXB_MODEL_STORE/$name/"
    echo "copied $name"
done

mkdir -p "$RXB_RESULTS/$SLURM_JOB_ID"
cp -r "$WORK/state" "$WORK/logs" "$RXB_RESULTS/$SLURM_JOB_ID/" 2>/dev/null || true
rm -rf "$WORK"
du -sh "$HF_HOME/hub"
echo "Done. Models are in $RXB_MODEL_STORE"

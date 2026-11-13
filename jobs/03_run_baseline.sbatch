#!/bin/bash
#SBATCH --job-name=rxb_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=slurm_%x_%j.out

# Phases 5-10: run the zero-shot baseline. Models run one after another on one GPU:
# load -> infer every page -> save -> unload -> verify memory -> next model.
#
#   sbatch jobs/03_run_baseline.sbatch                                   # pilot: 3 images, 3 primary models
#   sbatch --export=ALL,RXB_LIMIT=1,RXB_RUN=medgemma_27b jobs/03_run_baseline.sbatch          # single-model check
#   sbatch --export=ALL,RXB_LIMIT=0,RXB_NAME=full_v1 jobs/03_run_baseline.sbatch              # all images
#   sbatch --export=ALL,RXB_LIMIT=0,RXB_NAME=full_v1 jobs/03_run_baseline.sbatch              # same line again = resume
#
# RXB_LIMIT   number of images (0 = all)                      default 3
# RXB_RUN     'primary', 'optional', 'all' or model keys      default primary
# RXB_NAME    name of a resumable run kept on /scratch        default: none (one-off run)
# RXB_ARGS    extra arguments for `rxbench run`, e.g. "--max-new-tokens 16384"
#
# Inference runs with the Hub switched off (HF_HUB_OFFLINE=1) and loads local snapshots only.

source "$HOME/prescription_vlm_baseline/jobs/common.sh"
rxb_activate
set -uo pipefail
cd "$RXB_PROJECT"

LIMIT="${RXB_LIMIT:-3}"
MODELS="${RXB_RUN:-primary}"
if [ -n "${RXB_NAME:-}" ]; then
    WORK="$RXB_SCRATCH/rxbench_runs/$RXB_NAME"         # kept between jobs so a long run can resume
else
    WORK="$RXB_SCRATCH/$SLURM_JOB_ID"
fi
DATA="$RXB_SCRATCH/datasets/dataset_english"
mkdir -p "$WORK/reports" "$DATA"

export RXBENCH_EXPERIMENTS_DIR="$WORK/experiments" RXBENCH_STATE_DIR="$WORK/state" RXBENCH_LOG_DIR="$WORK/logs"
export RXBENCH_HF_CACHE_DIR="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

pack_results() {
    # One archive back to /home. Runs at the end and also if Slurm is about to stop the job.
    mkdir -p "$RXB_RESULTS"
    python -m rxbench report --out "$WORK/reports/RUNTIME_REPORT.md" > /dev/null 2>&1 || true
    python -m rxbench export metadata --out "$WORK/reports/metadata.csv" > /dev/null 2>&1 || true
    tar -cf "$RXB_RESULTS/rxbench_${SLURM_JOB_ID}.tar" -C "$WORK" experiments reports logs state 2>/dev/null
    echo "Results: $RXB_RESULTS/rxbench_${SLURM_JOB_ID}.tar"
}
trap 'echo "Time limit close: packing partial results"; pack_results; exit 143' TERM

# Dataset: unpack the archive once per node.
(
    flock 9
    if [ ! -f "$DATA/.complete" ]; then
        tar -xf "$RXB_DATASET_TAR" -C "$DATA" --strip-components=1
        touch "$DATA/.complete"
    fi
) 9>"$DATA.lock"

rxb_stage_models || exit 1

echo "===== hardware"
python -m rxbench hardware-report --out "$WORK/reports/HARDWARE_REPORT.md" --json "$WORK/reports/hardware.json" \
    --role-note "Slurm cluster compute node, Slurm job $SLURM_JOB_ID (the machine the experiment ran on)." > /dev/null
sed -n '1,30p' "$WORK/reports/HARDWARE_REPORT.md"
echo "===== models"
python -m rxbench models list
python -m rxbench models verify "$MODELS" || echo "WARNING: at least one model failed file verification (see above)"

echo "===== run"
LIMIT_ARG=""
[ "$LIMIT" != "0" ] && LIMIT_ARG="--limit $LIMIT"
if [ -n "$(ls -d "$WORK"/experiments/EXP_* 2>/dev/null | head -n 1)" ]; then
    echo "Existing experiments found in $WORK: resuming, completed runs are skipped."
    # shellcheck disable=SC2086
    python -m rxbench rerun all --models "$MODELS" --skip-completed ${RXB_ARGS:-} &
else
    # shellcheck disable=SC2086
    python -m rxbench run --input-dir "$DATA/images" $LIMIT_ARG --models "$MODELS" ${RXB_ARGS:-} &
fi
wait $!                       # run in the background so the TERM trap can fire
RC=$?

echo "===== comparison"
python -m rxbench compare all | tail -n 20
pack_results
[ -z "${RXB_NAME:-}" ] && rm -rf "$WORK"
exit $RC

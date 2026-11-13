#!/bin/bash
#SBATCH --job-name=rxb_ui
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%x_%j.out

# The Streamlit research interface on a GPU node. It listens on the node's loopback address only,
# so it is reached through an SSH tunnel and is not visible to other cluster users.
#
#   sbatch jobs/05_interface.sbatch                                  # a new, empty workspace
#   sbatch --export=ALL,RXB_NAME=full_v1 jobs/05_interface.sbatch    # browse / extend a named run
#
# When the job is running, the log prints the node name. Then, on the laptop:
#   ssh -t -L 8501:127.0.0.1:8501 <user>@<cluster-master> ssh -L 8501:127.0.0.1:8501 <node>
# and open http://localhost:8501 . Stop with: scancel <jobid>

source "$HOME/prescription_vlm_baseline/jobs/common.sh"
rxb_activate
set -uo pipefail
cd "$RXB_PROJECT"

WORK="$RXB_SCRATCH/rxbench_runs/${RXB_NAME:-ui_$SLURM_JOB_ID}"
mkdir -p "$WORK"
export RXBENCH_EXPERIMENTS_DIR="$WORK/experiments" RXBENCH_STATE_DIR="$WORK/state" RXBENCH_LOG_DIR="$WORK/logs"
export RXBENCH_ANNOTATIONS_DIR="$WORK/annotations"
export RXBENCH_HF_CACHE_DIR="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

rxb_stage_models || echo "WARNING: no staged models; the interface will show them as not downloaded."

pack_results() {
    mkdir -p "$RXB_RESULTS"
    tar -cf "$RXB_RESULTS/rxbench_ui_${SLURM_JOB_ID}.tar" -C "$WORK" . 2>/dev/null
    echo "Workspace saved: $RXB_RESULTS/rxbench_ui_${SLURM_JOB_ID}.tar (also kept in $WORK)"
}
trap 'pack_results; exit 0' TERM INT

echo "Interface starting on node $(hostname), port 8501 (loopback only)."
streamlit run app/main.py --server.address 127.0.0.1 --server.port 8501 --server.headless true &
wait $!
pack_results

#!/bin/bash
# Shared settings for the Slurm cluster job scripts (cluster guide v3.0).
# Sourced by every script in this folder; not run directly.
#
# Storage rules followed here (guide sections 6 and 7.2):
#   - everything a job writes while it runs goes to /scratch/<team>/$USER/...
#   - model weights persist as a few large files under /data/<team>/$USER/rxbench_models
#     and are only read by jobs
#   - results are copied back once, as one archive, to ~/results
#   - caches (Hugging Face, pip, conda, temporary files) point at /scratch

# Team folder: group "oc_tm" means /data/oc and /scratch/oc. Override with RXB_TEAM if needed.
TEAM="${RXB_TEAM:-$(id -Gn | tr ' ' '\n' | grep '_tm$' | head -n 1 | sed 's/_tm$//')}"
if [ -z "$TEAM" ]; then
    echo "ERROR: could not find a *_tm group in 'id -Gn'. Submit with --export=ALL,RXB_TEAM=<team>." >&2
    exit 1
fi

export RXB_TEAM="$TEAM"
export RXB_PROJECT="${RXB_PROJECT:-$HOME/prescription_vlm_baseline}"
export RXB_ENV="${RXB_ENV:-rxbench}"
export RXB_SCRATCH="/scratch/$TEAM/$USER"
export RXB_MODEL_STORE="/data/$TEAM/$USER/rxbench_models/hub"       # persistent copy of the model snapshots
export RXB_DATASET_TAR="${RXB_DATASET_TAR:-/data/$TEAM/$USER/rxbench_data/dataset_english.tar}"
export RXB_RESULTS="$HOME/results"

# Caches and temporary files on the node's local disk.
export HF_HOME="$RXB_SCRATCH/huggingface"
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"               # written once by `hf auth login`
export PIP_CACHE_DIR="$RXB_SCRATCH/pip-cache"
export CONDA_PKGS_DIRS="$RXB_SCRATCH/conda-pkgs"
export TORCH_HOME="$RXB_SCRATCH/torch"
export TMPDIR="$RXB_SCRATCH/tmp"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
mkdir -p "$RXB_SCRATCH" "$HF_HOME/hub" "$TMPDIR"

rxb_activate() {
    # Module and conda scripts are not written for `set -u`; switch it off while they run.
    set +u
    module purge
    module load Miniforge3/26.1.1-3
    # shellcheck disable=SC1091
    source activate "$RXB_ENV"
}

# Stage the persistent model snapshots onto this node's scratch disk (large sequential copies), so that
# model loading reads local files. rsync copies only what is missing or changed, which also repairs a
# scratch copy that the weekly clean-up has thinned out. flock stops two jobs on one node copying at once.
rxb_stage_models() {
    local src name
    if [ ! -d "$RXB_MODEL_STORE" ]; then
        echo "ERROR: no models in $RXB_MODEL_STORE. Run jobs/02_download_models.sbatch first." >&2
        return 1
    fi
    for src in "$RXB_MODEL_STORE"/models--*; do
        [ -d "$src" ] || continue
        name="$(basename "$src")"
        (
            flock 9
            echo "staging $name"
            mkdir -p "$HF_HOME/hub/$name"
            rsync -a "$src/" "$HF_HOME/hub/$name/"
        ) 9>"$HF_HOME/hub/$name.lock"
    done
}

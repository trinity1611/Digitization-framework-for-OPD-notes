# Running on the Slurm cluster

Written to the cluster guide v3.0 and its storage rules. The cluster takes a password at login, so
these commands are run by you; paste the output back when something looks wrong.

- `<user>` is your cluster user name and `<cluster-master>` is the address of the cluster's login
  (master) node. These are the only two things to fill in.
- The scripts in `jobs/` were written for one specific Slurm cluster (module `Miniforge3/26.1.1-3`,
  GPU types `h200_nvl` and `pro_6000`, `/scratch/<team>/<user>` and `/data/<team>` storage). On another
  cluster, adapt `jobs/common.sh` and the `#SBATCH` lines first.
- The team folder is worked out by the scripts from `id -Gn` (group `oc_tm` → `/data/oc`, `/scratch/oc`).
- Each block says where it runs: **on the laptop** (PowerShell) or **on the cluster** (after `ssh`).

Nothing in these steps runs computation on the master node: Python work happens inside `sbatch` or `srun`.

## 0. What goes where

| what | where | written by |
|---|---|---|
| code, conda environment, final result archives | `~/prescription_vlm_baseline`, `~/results` | you, the env script, the end of each job |
| dataset archive | `/data/<team>/<user>/rxbench_data/dataset_english.tar` | you, once |
| model weights (few large files) | `/data/<team>/<user>/rxbench_models/hub` | download job, once |
| everything a job writes while running, caches, staged models, unpacked dataset | `/scratch/<team>/<user>/...` | jobs |

`/scratch` is per node and is cleaned of files unused for 30 days; the scripts re-stage what is missing.

## 1. Upload (on the laptop)

```powershell
cd C:\path\to\Baseline-OCR
tar -cf prescription_vlm_baseline.tar --exclude=prescription_vlm_baseline/experiments --exclude=prescription_vlm_baseline/logs --exclude=prescription_vlm_baseline/.state --exclude=__pycache__ prescription_vlm_baseline
tar -cf dataset_english.tar dataset_english
scp prescription_vlm_baseline.tar <user>@<cluster-master>:~/
scp dataset_english.tar <user>@<cluster-master>:~/
```

Both archives are small (the dataset is about 8 MB).

From a fresh clone of the repository instead: the job scripts expect the project in a folder named
`prescription_vlm_baseline` in your cluster home directory, and the dataset as
`dataset_english.tar`. Unzip `data/dataset_english.zip`, then run
`tar -cf dataset_english.tar -C data dataset_english` and upload that archive as above.

## 2. Unpack and inspect the hardware (on the cluster)

```bash
ssh <user>@<cluster-master>
tar -xf ~/prescription_vlm_baseline.tar -C ~
sed -i 's/\r$//' ~/prescription_vlm_baseline/jobs/*.sbatch ~/prescription_vlm_baseline/jobs/*.sh ~/prescription_vlm_baseline/scripts/*.sh
id -Gn
TEAM=$(id -Gn | tr ' ' '\n' | grep '_tm$' | head -n 1 | sed 's/_tm$//'); echo "team folder: $TEAM"
mkdir -p /data/$TEAM/$USER/rxbench_data && mv ~/dataset_english.tar /data/$TEAM/$USER/rxbench_data/
cd ~/prescription_vlm_baseline
sbatch jobs/00_hardware_discovery.sbatch
squeue --me
```

When the job has left the queue (it takes about a minute once it starts):

```bash
cat ~/prescription_vlm_baseline/slurm_rxb_hw_<jobid>.out
cat ~/results/<jobid>/HARDWARE_REPORT_pre_env.md
```

Paste both. This is Phase 1: nothing has been installed yet, and the report is what the later choices
(PyTorch build, GPU type, memory) are checked against.

## 3. Create the environment (on the cluster)

One command, on a GPU node, about 10 to 20 minutes:

```bash
srun --partition=normal --gres=gpu:h200_nvl:1 --cpus-per-task=8 --mem=32G --time=01:00:00 bash ~/prescription_vlm_baseline/jobs/01_create_env.sh
```

It ends with "Environment 'rxbench' is ready." after printing the PyTorch/CUDA line, the model list and
the test summary. If it prints "this PyTorch build cannot use CUDA", stop and paste the output.

## 4. Hugging Face access for MedGemma (once)

In a browser: sign in at huggingface.co, open https://huggingface.co/google/medgemma-27b-it and accept
the terms. Then **on the cluster**:

```bash
module load Miniforge3/26.1.1-3 && source activate rxbench
hf auth login
```

Paste a read token when asked (it is not echoed). It is stored in `~/.cache/huggingface/token`.

## 5. Download the models (on the cluster)

About 184 GB for the three primary models. Submit in the evening: the job ends with one large copy onto
the shared array. `RXB_CONFIRM_DOWNLOAD=yes` is your confirmation of the download.

```bash
cd ~/prescription_vlm_baseline
sbatch --export=ALL,RXB_CONFIRM_DOWNLOAD=yes jobs/02_download_models.sbatch
squeue --me
tail -n 30 slurm_rxb_dl_<jobid>.out
```

Without the variable the job prints the plan and the access check and stops. If the job is interrupted,
submit the same line again; it resumes. It finishes with a `verify` block (every line should say `ok`)
and "Done. Models are in ...".

## 6. First real run: one model, one page (on the cluster)

```bash
cd ~/prescription_vlm_baseline
sbatch --export=ALL,RXB_LIMIT=1,RXB_RUN=medgemma_27b jobs/03_run_baseline.sbatch
squeue --me
tail -n 40 slurm_rxb_run_<jobid>.out
sacct -j <jobid> --format=JobID,State,Elapsed,NodeList,MaxRSS
```

The first run on a node stages the models from `/data` to `/scratch` (several minutes). Look at the GPU
while it runs: `srun --jobid=<jobid> --overlap --pty nvidia-smi`.

## 7. Pilot and full run (on the cluster)

```bash
sbatch jobs/03_run_baseline.sbatch                                              # 3 pages × 3 primary models
sbatch --export=ALL,RXB_LIMIT=0,RXB_NAME=full_v1 jobs/03_run_baseline.sbatch    # all 77 pages
```

A named run (`RXB_NAME`) keeps its working folder on `/scratch`. If it reaches the 24-hour limit it packs
what is finished; submitting the same line again continues on the **same node type** and skips completed
runs. Because `/scratch` is per node, add `--nodelist=<node>` (the node shown by `sacct`) when resuming.

Other options: `RXB_RUN=primary|optional|all|<keys>`, `RXB_ARGS="--max-new-tokens 16384"`,
and `--gres=gpu:pro_6000:1` on the `sbatch` line to use the 96 GB cards instead.

## 8. Offline verification (on the cluster)

```bash
sbatch jobs/04_offline_check.sbatch
tail -n 40 slurm_rxb_offline_<jobid>.out
```

The log says whether a private network namespace was available, and the report lists a model as
verified offline only if real inference completed with no network use.

## 9. Fetch results

**On the cluster**, each job prints the archive it wrote, for example `~/results/rxbench_<jobid>.tar`.
**On the laptop**, fetch to a folder outside OneDrive (the archive contains prescription images and
model outputs):

```powershell
mkdir C:\Users\<you>\rxbench_results
scp <user>@<cluster-master>:~/results/rxbench_<jobid>.tar C:\Users\<you>\rxbench_results\
tar -xf C:\Users\<you>\rxbench_results\rxbench_<jobid>.tar -C C:\Users\<you>\rxbench_results
```

To browse them in the interface on the laptop:

```powershell
$env:RXBENCH_EXPERIMENTS_DIR = "C:\Users\<you>\rxbench_results\experiments"
C:\Users\<you>\.venvs\rxbench\Scripts\streamlit run C:\path\to\Baseline-OCR\prescription_vlm_baseline\app\main.py
```

`reports/RUNTIME_REPORT.md` in the archive is the memory and inference-time report;
`reports/HARDWARE_REPORT.md` describes the node the run used.

## 10. Interface on the cluster (optional)

```bash
sbatch --export=ALL,RXB_NAME=full_v1 jobs/05_interface.sbatch
squeue --me                       # note the node, e.g. n2
```

**On the laptop**, tunnel through the master node to that node (two password prompts), then open
http://localhost:8501:

```powershell
ssh -t -L 8501:127.0.0.1:8501 <user>@<cluster-master> ssh -L 8501:127.0.0.1:8501 <node>
```

The interface listens on the node's loopback address only. Stop it with `scancel <jobid>`; the
workspace is packed to `~/results` when the job ends.

## 11. Housekeeping

```bash
du -sh /scratch/<team>/$USER        # fine on /scratch; never run du or find over /home or /data
```

Delete finished run folders under `/scratch/<team>/$USER/rxbench_runs/` by full path once their
archives are safely fetched. If a job stalls at start or while saving, or `ls` hangs, the shared array
is busy: do not resubmit in a loop.

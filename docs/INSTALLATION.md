# Installation

The application is plain Python. The only platform-specific part is PyTorch, which must be the CUDA
build for the machine. Three situations are covered:

| target | status | section |
|---|---|---|
| Slurm cluster (x86-64, H200 141 GB / RTX Pro 6000 96 GB, Slurm) | job scripts written to the cluster guide v3.0; not yet run | A |
| Stand-alone NVIDIA GB10 system (DGX Spark class, ARM64, unified memory) | script provided; not verified on a device | B |
| Development laptop (tests, mock runs, processor preflight, small-model check) | used to build and test this project | C |

In every case the order is: **inspect the machine first, then create an isolated environment, then
install.** Nothing is installed system-wide.

## A. Slurm cluster

Follow `docs/CLUSTER_RUNBOOK.md`; it gives each command and where to run it. In outline:

1. `jobs/00_hardware_discovery.sbatch` inspects a compute node with the Python of the Miniforge3 module
   only (standard library). It installs nothing and writes `HARDWARE_REPORT_pre_env.md`.
2. `jobs/01_create_env.sh`, run with `srun` on a GPU node, creates the conda environment `rxbench`
   exactly as the guide prescribes: `module load Miniforge3/26.1.1-3`, package cache on `/scratch`,
   `conda create -n rxbench -c conda-forge python=3.11 pytorch torchvision`, then
   `pip install -r requirements.txt`. It stops if the resulting PyTorch cannot use CUDA, and writes
   the exact package lists to `docs/environment/`.
3. `jobs/02_download_models.sbatch` downloads the models.
4. `jobs/03_run_baseline.sbatch`, `jobs/04_offline_check.sbatch`, `jobs/05_interface.sbatch` run the
   experiment, the offline verification and the interface.

The environment lives in `/home` (shared by all nodes); everything a job writes while running is on
`/scratch/<team>/<user>`.

## B. Stand-alone GB10 system

Not verified: no GB10 machine was available. `scripts/setup_gb10.sh` is written to fail safely.

```
bash scripts/setup_gb10.sh                         # inspect only; installs nothing
bash scripts/setup_gb10.sh install <pytorch-index> # create .venv and install
```

Read the report from the first command before the second. On a GB10 system expect: `aarch64`, the GPU
named "NVIDIA GB10", `nvidia-smi` printing `[N/A]` for memory, and the application reporting
"unified memory: yes". Points to check on the device before trusting a run:

- PyTorch must be an **aarch64 CUDA** build matching the system's CUDA version. A CPU-only or x86-64
  wheel installs without complaint and then cannot run the models; the script checks
  `torch.cuda.is_available()` and stops if it is false.
- `bitsandbytes` (INT8/INT4) is listed for x86-64 only in `requirements.txt`. BF16 does not need it.
- The memory budget is the system's available memory (`free -h`), shared with everything else running.
  The application's pre-load guard uses that figure on unified memory.
- Do not upgrade the NVIDIA driver, CUDA or system Python for this project.

## C. Development laptop (Windows or Linux)

Used for the unit tests, mock runs, the interface and processor-level preflights. An 8 GB laptop GPU
cannot run the primary models; the application refuses to load a model that does not fit rather than
spilling onto CPU or disk.

```
python -m venv --system-site-packages C:\Users\<you>\.venvs\rxbench     # outside OneDrive
C:\Users\<you>\.venvs\rxbench\Scripts\python -m pip install -r requirements.txt
```

`--system-site-packages` reuses an existing PyTorch; omit it and install PyTorch first if there is none.
Keep virtual environments outside cloud-synchronised folders.

## Commands

```
python -m rxbench hardware-report                 # writes docs/HARDWARE_REPORT.md
python -m rxbench models list
python -m rxbench models download primary --config-only
python -m rxbench preflight --image <page.jpg>    # processors only: image and prompt reach each model's input
python -m rxbench models download primary         # weights; asks for confirmation
python -m rxbench run --input <page.jpg> --models primary
python -m rxbench run --input-dir <folder> --limit 5 --models medgemma_27b,gemma4_31b
python -m rxbench rerun EXP_0001 --models qwen3vl_32b --max-new-tokens 16384
python -m rxbench compare EXP_0001
python -m rxbench report --out reports/RUNTIME_REPORT.md
python -m rxbench offline-check --image <page.jpg> --inference
python -m pytest
streamlit run app/main.py                         # interface on http://localhost:8501
```

`python -m rxbench run --mock ...` exercises the whole pipeline with a software test double. Its
outputs are labelled MOCK everywhere and are not model results.

## Configuration

`config.yaml` holds paths, default generation settings, per-model precision and native image-processor
options. Environment variables override the paths: `RXBENCH_EXPERIMENTS_DIR`, `RXBENCH_STATE_DIR`,
`RXBENCH_LOG_DIR`, `RXBENCH_ANNOTATIONS_DIR`, `RXBENCH_HF_CACHE_DIR`, and `RXBENCH_CONFIG` for the file
itself. `--precision <model>=<precision>` overrides a model's precision for one run.

`.streamlit/config.toml` binds the interface to `127.0.0.1` and switches off Streamlit's usage statistics.

## Tested versions

The test suite and the checks in `docs/STATUS_REPORT.md` were run on the development laptop with:

| component | version |
|---|---|
| Python | 3.13.13 (source also parses and its core tests pass on 3.10) |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| transformers | 5.12.1 |
| accelerate | 1.14.0 |
| huggingface_hub | 1.20.1 |
| safetensors | 0.8.0 |
| tokenizers | 0.22.2 |
| bitsandbytes | 0.49.2 |
| pillow | 12.2.0 |
| pypdfium2 | 5.14.0 |
| PyYAML | 6.0.3 |
| psutil | 7.2.2 |
| rapidfuzz | 3.14.5 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| streamlit | 1.67.0 |
| pytest | 9.1.1 |

`transformers` is pinned to 5.12.1 in `requirements.txt` because the processors and model classes of
all six registered checkpoints were checked against that release. Gemma 4 needs at least 5.5.

The cluster's versions are not known until `jobs/01_create_env.sh` has run there; it records them in
`docs/environment/cluster_pip_freeze.txt` and `cluster_conda_explicit.txt`.

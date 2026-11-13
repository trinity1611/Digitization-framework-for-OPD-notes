# Model download guide

All facts in this guide (repository names, commit hashes, file sizes, parameter counts, licences, gating)
were read from the Hugging Face Hub on 2026-11-13. The application downloads exactly the repository and
revision listed here and never substitutes another checkpoint.

## 1. Models

### Primary models

| key | model | official repository | pinned revision | download | parameters | licence | access |
|---|---|---|---|---|---|---|---|
| `medgemma_27b` | MedGemma 27B Multimodal IT | https://huggingface.co/google/medgemma-27b-it | `2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f` | 54.90 GB (12 weight files) | 27.43 B | [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms) | gated, automatic approval |
| `gemma4_31b` | Gemma 4 31B IT | https://huggingface.co/google/gemma-4-31B-it | `842da3794eaa0b77d5f08bae87a17459d91ff475` | 62.58 GB (2 weight files) | 31.27 B | Apache-2.0 ([Gemma 4 licence page](https://ai.google.dev/gemma/docs/gemma_4_license)) | public |
| `qwen3vl_32b` | Qwen3-VL 32B Instruct | https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct | `0cfaf48183f594c314753d30a4c4974bc75f3ccb` | 66.73 GB (14 weight files) | 33.36 B | Apache-2.0 | public |

Total for the three primary models: **184.2 GB**.

### Optional models

| key | model | official repository | pinned revision | download | parameters | licence | access |
|---|---|---|---|---|---|---|---|
| `gemma4_26b_a4b` | Gemma 4 26B A4B IT (mixture of experts) | https://huggingface.co/google/gemma-4-26B-A4B-it | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | 51.64 GB | 25.81 B total, about 4 B active per token | Apache-2.0 | public |
| `qwen3_5_27b` | Qwen3.5 27B | https://huggingface.co/Qwen/Qwen3.5-27B | `fc05daec18b0a78c049392ed2e771dde82bdf654` | 55.59 GB | 27.78 B | Apache-2.0 | public |
| `medgemma_1_5_4b` | MedGemma 1.5 4B IT | https://huggingface.co/google/medgemma-1.5-4b-it | `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` | 8.64 GB | 4.30 B | Health AI Developer Foundations Terms of Use | gated, automatic approval |

Total for the three optional models: 115.9 GB. All six: 300.1 GB.

Points that are easy to get wrong:

- **MedGemma 27B has two variants.** `google/medgemma-27b-it` is the multimodal one used here.
  `google/medgemma-27b-text-it` is text-only; the registry lists it as forbidden and the adapter refuses it.
- **MedGemma 27B is MedGemma 1** (Gemma 3 architecture, model card v1:
  https://developers.google.com/health-ai-developer-foundations/medgemma/model-card-v1).
  MedGemma 1.5 exists only as the 4B multimodal model.
- **Gemma 4 26B A4B is a mixture-of-experts model.** About 4 B parameters are active per token, but all
  25.8 B are stored and loaded: 51.6 GB on disk and about 48 GB of weights in memory at BF16.
- `google/gemma-4-31B` (without `-it`) is the base model, and `Qwen/Qwen3-VL-32B-Thinking` is a different
  checkpoint. Neither is used.
- All checkpoints store BF16 weights; the download size is essentially the BF16 memory footprint of the weights.

Official documentation: [Gemma 4 model card](https://gemma.ai/models/gemma-4/model-card/),
[Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL).

## 2. Licence and access

| model family | what you must do before downloading |
|---|---|
| MedGemma (27B and 1.5 4B) | Sign in at huggingface.co, open the model page, review and accept the Health AI Developer Foundations terms (approval is immediate). The terms govern use of the model and its outputs; read them before using results outside research. |
| Gemma 4 (31B, 26B A4B) | Nothing. Public repositories, Apache-2.0. |
| Qwen3-VL, Qwen3.5 | Nothing. Public repositories, Apache-2.0. |

Authentication (needed for MedGemma only): create a token with read access at
https://huggingface.co/settings/tokens and store it once with

```
hf auth login
```

The token is saved in `~/.cache/huggingface/token`. Do not put it in scripts, job files or command lines.
Check access without downloading anything:

```
python -m rxbench models access primary
```

## 3. Downloading with the application

```
python -m rxbench models list                    # registry, sizes, memory estimates, local state
python -m rxbench models plan primary            # what would be downloaded and whether there is room
python -m rxbench models download primary        # shows the details and asks for confirmation
python -m rxbench models download medgemma_27b   # one model
python -m rxbench models verify primary --check-remote
```

What the download manager does:

1. Checks whether the pinned revision is already complete locally (and then does nothing).
2. Checks free disk space against the remaining size.
3. Prints repository, revision, licence, access terms, size and destination.
4. Asks for explicit confirmation. Without a terminal (a batch job) it refuses unless `--yes` is given.
5. Downloads weights and configuration with progress bars. An interrupted download resumes when the
   same command is run again.
6. Records the resolved commit hash in `.state/models_state.json`.

`python -m rxbench models download primary --config-only` fetches only processor, tokenizer and
configuration files (tens of MB per model, no confirmation). That is enough for
`python -m rxbench preflight --image <file>`, which shows how each model's processor will tokenise a page.

In the interface: **Model management**, "Prepare download", tick the confirmation box, "Download now".

### On the Slurm cluster

Downloads run as a Slurm job, to `/scratch`, with one copy to `/data/<team>/<user>/rxbench_models`
(`jobs/02_download_models.sbatch`; see `docs/CLUSTER_RUNBOOK.md`). The confirmation is the
`RXB_CONFIRM_DOWNLOAD=yes` variable on the `sbatch` line. The copy of 184 GB onto the shared array is a
large transfer: submit the job in the evening.

## 4. Manual download

The same result with the Hugging Face CLI (`pip install -U huggingface_hub`):

```
hf download google/medgemma-27b-it      --revision 2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f
hf download google/gemma-4-31B-it       --revision 842da3794eaa0b77d5f08bae87a17459d91ff475
hf download Qwen/Qwen3-VL-32B-Instruct  --revision 0cfaf48183f594c314753d30a4c4974bc75f3ccb

hf download google/gemma-4-26B-A4B-it   --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52
hf download Qwen/Qwen3.5-27B            --revision fc05daec18b0a78c049392ed2e771dde82bdf654
hf download google/medgemma-1.5-4b-it   --revision 91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b
```

Files go to the Hugging Face cache (`$HF_HOME/hub`, by default `~/.cache/huggingface/hub`). The
application finds them there; to use another location set `paths.hf_cache_dir` in `config.yaml` (or
`RXBENCH_HF_CACHE_DIR`) and add `--cache-dir <dir>` to the commands above. Afterwards run
`python -m rxbench models verify primary`.

To move a model to a machine without internet access, copy its whole cache folder
(for example `models--google--gemma-4-31B-it`) into the other machine's `hub` directory, keeping the
`blobs`, `refs` and `snapshots` sub-folders and the symbolic links inside `snapshots`
(`rsync -a` or `cp -a`).

## 5. Storage and memory estimates

Estimated memory = total parameters × bytes per parameter + a runtime allowance (activations, vision
features, key-value cache for a long answer). These are planning figures; measured values are written
to every experiment's `metadata.json`.

| model | disk | BF16 | INT8 | INT4 |
|---|---|---|---|---|
| MedGemma 27B | 54.9 GB | about 59 GB | about 37 GB | about 23 GB |
| Gemma 4 31B | 62.6 GB | about 66 GB | about 41 GB | about 25 GB |
| Qwen3-VL 32B | 66.7 GB | about 70 GB | about 44 GB | about 26 GB |
| Gemma 4 26B A4B | 51.6 GB | about 56 GB | about 35 GB | about 22 GB |
| Qwen3.5 27B | 55.6 GB | about 60 GB | about 38 GB | about 23 GB |
| MedGemma 1.5 4B | 8.6 GB | about 11 GB | about 8 GB | about 5 GB |

Every primary model fits at BF16 on a 141 GB H200 and on a 96 GB RTX Pro 6000, and is expected to fit
in the 128 GB of a standard GB10 system when little else is running (to be confirmed on the device).
Models run one at a time, so the requirement is the largest single model, not the sum.

FP16 uses the same memory as BF16 but the Gemma-family checkpoints are published for BF16. FP32 doubles
the memory. FP8 is not available in this backend (it needs separately converted checkpoints, which the
baseline does not use). INT8/INT4 are quantised on load with bitsandbytes and change model behaviour;
treat them as separate experimental conditions.

## 6. Verification

`python -m rxbench models verify <models>` checks, without loading onto a GPU:

- the pinned revision is present and matches the registry;
- every weight file listed in `model.safetensors.index.json` exists;
- every weight file opens as a valid safetensors file;
- `config.json` names the registered architecture;
- the processor loads and is the registered processor class;
- with `--check-remote`, file sizes equal the Hub's; with `--deep-hash`, SHA-256 too (slow: reads every byte).

Whether a model actually loads and generates is recorded separately as its *load status*, after a real run.

## 7. Troubleshooting

| symptom | cause and fix |
|---|---|
| `GatedRepoError` / 403 for MedGemma | Terms not accepted for this account, or no token. Accept the terms on the model page, run `hf auth login`, then `python -m rxbench models access medgemma_27b`. |
| `Not enough disk space` | Free space or point `paths.hf_cache_dir` at a larger disk. The check needs the remaining size plus 2 GB. |
| Download interrupted | Run the same command again; completed files are kept and partial files resume. |
| Status `partial` | Some weight files are missing. Run the download again. |
| Status `config_only` | Only processor files are present (from `--config-only`). Run the full download. |
| Status `revision_mismatch` | A different commit is in the cache. Run the download again to fetch the pinned revision; the application will not load the other one. |
| Windows: `OSError [WinError 1314] A required privilege is not held` | Windows could not create a symbolic link in the cache. Run the command again (the cache falls back to copying), or enable Developer Mode. |
| `LocalEntryNotFoundError` / "offline mode" during download | `HF_HUB_OFFLINE=1` is set in the shell, or there is no network. Unset it for downloads; inference sets it on purpose. |
| Cluster: job stalls while copying to `/data` | The shared array is busy. Do not resubmit in a loop; wait, and tell the admins if it lasts (cluster guide 6.4). |
| Model loads on another machine but not here | Compare `python -m rxbench hardware-report` and `python -m rxbench models info <key>`: Transformers version, CUDA availability and free memory are the usual differences. |

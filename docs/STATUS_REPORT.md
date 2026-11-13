# Status report (2026-11-13)

What exists, what was actually run, and what was not. Mock runs are software tests and are never
counted as model results anywhere in this report.

## 1. Summary

| | state |
|---|---|
| Application source, interface, evaluation, job scripts, documentation | complete |
| Software tests | 127 passed, 0 failed (`docs/TEST_RESULTS.md`) |
| The three primary models (MedGemma 27B, Gemma 4 31B, Qwen3-VL 32B) | **not downloaded, not loaded, not run** |
| Real inference | one optional model, MedGemma 1.5 4B at 4-bit, on one dataset page, twice |
| Target hardware (GB10 system or the Slurm cluster) | **not accessed**; everything below was measured on a laptop |
| Ground truth | none exists yet, so **no accuracy figure is reported** |

The primary experiment still has to be run on a large-memory GPU. `docs/CLUSTER_RUNBOOK.md` contains
the commands.

## 2. Model checkpoints

| model | weights downloaded | processor files | loaded | processed a prescription image |
|---|---|---|---|---|
| MedGemma 27B Multimodal IT (`google/medgemma-27b-it`) | no | yes | **no** | **no** |
| Gemma 4 31B IT (`google/gemma-4-31B-it`) | no | yes | **no** | **no** |
| Qwen3-VL 32B Instruct (`Qwen/Qwen3-VL-32B-Instruct`) | no | yes | **no** | **no** |
| Gemma 4 26B A4B IT (`google/gemma-4-26B-A4B-it`) | no | yes | no | no |
| Qwen3.5 27B (`Qwen/Qwen3.5-27B`) | no | yes | no | no |
| MedGemma 1.5 4B IT (`google/medgemma-1.5-4b-it`) | already in the local cache before this project (8.6 GB) | yes | **yes** | **yes** |

"Processor files" are the official tokenizer, chat template, image-processor and configuration files
of the pinned revision (about 130 MB in total), fetched with `models download all --config-only`. No
model weights were downloaded by this project.

For all six repositories the following was verified on the Hugging Face Hub: the repository exists
under the exact name, it is an image-text-to-text model, its current commit (now pinned in
`model_registry.yaml`), file list and sizes, parameter count, licence and gating. The Hugging Face
account logged in on the laptop already has access to both gated MedGemma repositories.

## 3. What was verified for the primary models without weights

With each checkpoint's real processor, on the sample page `dataset_english/images/0022.jpg`
(960×1280) and the full master prompt (`docs/reports/preflight_0022.json`):

| model | input tokens | of which prompt | of which image | pixel tensor | settings |
|---|---|---|---|---|---|
| MedGemma 27B | 4,729 | 4,462 | 256 | 1×3×896×896 | checkpoint defaults |
| Gemma 4 31B | 5,540 | 4,462 | 1,064 | 10,080 patches | visual token budget 1120, thinking off |
| Qwen3-VL 32B | 5,269 | 4,059 | 1,200 | 4,800 patches (960×1280 kept) | checkpoint defaults |
| Gemma 4 26B A4B | 5,540 | 4,462 | 1,064 | 10,080 patches | as Gemma 4 31B |
| Qwen3.5 27B | 5,736 | 4,523 | 1,200 | 4,800 patches | thinking off |
| MedGemma 1.5 4B | 4,729 | 4,462 | 256 | 1×3×896×896 | checkpoint defaults |

This shows that, for every model, the page and the complete prompt arrive in the model input through
the model's own chat template, that the installed Transformers release has each architecture's model
and processor classes, and that the image-processor options take effect. It does **not** show that the
weights load or what the models generate.

## 4. Real inference performed

| | |
|---|---|
| Model | MedGemma 1.5 4B IT, `google/medgemma-1.5-4b-it`, revision `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` |
| Backend | Hugging Face Transformers 5.12.1, in-process, PyTorch 2.6.0+cu124 |
| Precision | **4-bit** (bitsandbytes NF4, double quantisation, BF16 compute). Not BF16: the laptop GPU has 8 GB |
| Hardware | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, discrete), Windows 11, x86-64 |
| Input | `0022.jpg`, SHA-256 `fb03cb2eebfe…`, 960×1280, no transformation before the processor |
| Prompt | the master prompt, SHA-256 `d85b9b4ad985…`, sent in full |
| Generation | greedy, maximum 8192 new tokens, repetition penalty 1.0, seed 0 |
| Stored as | `experiments/EXP_0001` and `experiments/EXP_0002` |

| run | status | output tokens | model load | generation | speed | peak GPU memory, load | peak GPU memory, inference | memory released |
|---|---|---|---|---|---|---|---|---|
| EXP_0001 | complete (end of sequence reached) | 3,761 | 14.4 s | 432.1 s | 8.7 tokens/s | 3.01 GB | 4.05 GB | yes |
| EXP_0002 (under the network guard) | complete | 3,761 | 12.7 s | 425.1 s | 8.9 tokens/s | 3.01 GB | 4.05 GB | yes |

Peak memory is `torch.cuda.max_memory_allocated`. Device memory in use by all processes peaked at
5.3 GB. System memory available fell from 5.1 GB to a minimum of 1.5 GB during the run, on a machine
with 15.4 GB. Image preprocessing took 0.46 s and added about 5 MB on the GPU. The full table is
`docs/reports/RUNTIME_REPORT_laptop.md`.

What these two runs establish about the software:

- the complete path works with real weights: local snapshot → processor → generation → exact raw
  output → parse → metadata → unload → memory check;
- the pixel hash and prompt hash given to the model equal the experiment's recorded values;
- the two runs produced **byte-identical** output (same SHA-256), as expected for greedy decoding on
  one machine;
- the adapter code that ran is the Gemma 3 family adapter, which MedGemma 27B also uses.

What they showed about the model, as observations rather than measurements:

- MedGemma 1.5 4B opened an **unrequested reasoning block** (`<unused94>thought … <unused95>`), about
  8,100 of the 13,100 output characters. It is kept in the raw output and excluded from the structured
  parse, and the run is flagged with a warning. (EXP_0001 was run before that warning was added; its
  structured output was rebuilt from the stored raw text with `rxbench reparse`.)
- After that block the model followed the requested A to E layout, and all eleven categories were located.
- The transcription can be read in `experiments/EXP_0001/medgemma_1_5_4b/raw_output.txt` next to the
  image. It visibly departs from the page in places. No error rate is given because no human-verified
  transcription of this page exists; a 4 B model at 4-bit is in any case not one of the models under study.

## 5. Inference backend, precision and memory per model

| model | backend | precision | memory | inference time |
|---|---|---|---|---|
| MedGemma 27B | Transformers (planned) | BF16 (planned) | not measured; estimate about 59 GB | not measured |
| Gemma 4 31B | Transformers (planned) | BF16 (planned) | not measured; estimate about 66 GB | not measured |
| Qwen3-VL 32B | Transformers (planned) | BF16 (planned) | not measured; estimate about 70 GB | not measured |
| MedGemma 1.5 4B | Transformers 5.12.1 | 4-bit NF4 | measured: 4.05 GB peak | measured: 425 to 432 s for 3,761 tokens |

Estimates are total parameters × 2 bytes plus a runtime allowance; they are for planning only.

## 6. Offline verification

`docs/reports/OFFLINE_VERIFICATION_laptop.md`: real inference with MedGemma 1.5 4B completed with the
in-process network guard active and **no attempt to open a connection or resolve a host name**.

Scope of that statement: the guard isolates the Python process; the laptop itself was still connected
(the report says so). Offline inference is therefore verified at process level for MedGemma 1.5 4B
only. It has **not** been verified for any primary model, and not on a machine with networking
physically disabled. `jobs/04_offline_check.sbatch` does both on the cluster (a private network
namespace where the node permits it).

## 7. Hardware

`docs/HARDWARE_REPORT.md` describes the development laptop, and says at the top that it is not the
target. The specification names an NVIDIA GB10 system as the target; the cluster guide supplied with
the project describes a Slurm cluster with x86-64 nodes carrying H200 141 GB and RTX Pro 6000 96 GB
GPUs. Neither could be reached from here (the cluster needs a password login). The application detects
which it is running on and accounts for memory accordingly; the detection of GB10 and unified memory
is tested with simulated inputs only.

## 8. Deviations from the requested protocol

1. **Primary models not run** and **target hardware not inspected**, for the reason above. Phases 1 to 4
   and 7 to 9 are complete; phases 5, 6, 10 and 11 are complete only as far as software tests, real
   processors and one small model allow.
2. **Real inference used 4-bit quantisation** on an optional model, not BF16 on a primary model.
3. **Structured output.** The master prompt asks for Markdown sections, not JSON. Section 15 of the
   specification would mark every such response "structured parsing unsuccessful". Instead a
   deterministic, rule-based reader locates the requested sections and fills `structured_output.json`
   with text copied from the raw output; JSON is still parsed as JSON when a model produces it. No
   model is involved and nothing is corrected.
4. **Gemma 4 visual token budget 1120** rather than the checkpoint default 280, and **Qwen3.5 reasoning
   off** rather than its default on. Both are configurable and recorded.
5. **Greedy decoding** replaces every checkpoint's sampling defaults, as the specification asks;
   temperature 0 is expressed as `do_sample=False`.
6. **EXIF orientation** is applied before the processors when a file carries a rotation tag
   (configurable, recorded). None of the 77 dataset images was checked for such a tag.
7. **Folder structure.** `models/`, `inference/`, `hardware/`, `evaluation/`, `storage/` are sub-packages
   of `rxbench/` instead of top-level folders, to avoid name collisions.
8. **Simultaneous model serving is not implemented**; the option only exists as an explicit switch that
   raises an error. Execution is always sequential.
9. **One backend** (Transformers). FP8 is not available in it; INT8/INT4 use bitsandbytes.
10. **Additions not in the specification:** annotation and evaluation screens, a processor-only
    `preflight` command, a `reparse` command, a resumable batch mode, and Slurm job scripts.

## 9. Changes made to the development laptop

- A virtual environment `C:\Users\<you>\.venvs\rxbench` (outside OneDrive) with Streamlit and its
  dependencies. The base Python installation was not modified.
- About 130 MB of processor and configuration files, and the model cards, in the Hugging Face cache
  `C:\Users\<you>\.cache\huggingface\hub`.
- No system-wide configuration was changed and no model weights were downloaded.

## 10. Next steps

1. Run `docs/CLUSTER_RUNBOOK.md` steps 1 to 6: hardware discovery, environment, download, then one
   model on one page.
2. Pilot (3 pages × 3 models), read the runtime report, adjust the token limit if runs are truncated.
3. Annotate a first set of pages (Annotation screen) and have them verified, so that evaluation can
   report accuracy.
4. Offline verification job; full run.

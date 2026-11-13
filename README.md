# rxbench: zero-shot VLM baseline for handwritten prescriptions

A local research application that measures how well large open-weight vision-language models read
handwritten medical prescriptions and OPD notes **with no help at all**: no OCR, no medicine catalogue,
no retrieval, no agents, no fine-tuning, no model collaboration and no post-correction.

Every model receives the same original page and the same fixed master prompt, once. Its raw response is
stored exactly as generated, with everything needed to reproduce the run. Models can then be compared
with each other, and evaluated against human-verified ground truth.

## Status

| | |
|---|---|
| Application, interface, evaluation, tests, documentation | complete |
| Test suite | 127 tests passing |
| Primary models run with weights | **none yet** (they need a GPU with roughly 60 to 70 GB of memory at BF16) |
| Real inference so far | MedGemma 1.5 4B at 4-bit, one page, on an 8 GB laptop GPU |
| Ground truth | none yet, so no accuracy figures exist |

`docs/STATUS_REPORT.md` states precisely what has and has not been exercised. Nothing in this repository
should be read as a result for the primary models.

## Models

| tier | model | repository | parameters | download |
|---|---|---|---|---|
| primary | MedGemma 27B Multimodal IT | [google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it) | 27.4 B | 54.9 GB |
| primary | Gemma 4 31B IT | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | 31.3 B | 62.6 GB |
| primary | Qwen3-VL 32B Instruct | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | 33.4 B | 66.7 GB |
| optional | Gemma 4 26B A4B IT (mixture of experts) | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | 25.8 B total | 51.6 GB |
| optional | Qwen3.5 27B | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | 27.8 B | 55.6 GB |
| optional | MedGemma 1.5 4B IT | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) | 4.3 B | 8.6 GB |

Models are declared in `model_registry.yaml` and pinned to exact commits. The MedGemma repositories are
gated: accept the terms on the model page and run `hf auth login` first. Adding a model is one entry in
that file. Details, licences and troubleshooting: `docs/MODEL_DOWNLOAD_GUIDE.md`.

## Requirements

- Python 3.10 or newer.
- A CUDA build of PyTorch for your machine (not in `requirements.txt` on purpose).
- For the primary models: one GPU with about 70 GB free at BF16 (for example an H200 or RTX Pro 6000
  96 GB), or an NVIDIA GB10 system with 128 GB of unified memory. INT8 and INT4 loading are available
  as separate experimental conditions.
- Without such hardware you can still run the tests, the interface, mock runs, processor-only
  preflights, and the 4 B model on a small GPU.

## Setup

```bash
git clone <this repository> && cd <repository folder>

python -m rxbench.hardware.diagnostics --out docs/HARDWARE_REPORT.md   # inspect first; standard library only

python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url <the PyTorch CUDA index for your machine>
pip install -r requirements.txt

python -m pytest                                       # 127 tests, about 1.5 minutes
```

More detail, including a Slurm cluster and a GB10 system: `docs/INSTALLATION.md`.

## Using it

```bash
python -m rxbench models list                              # what is registered and what is local
python -m rxbench models download primary --config-only    # processor files only (small)
python -m rxbench preflight --image page.jpg               # shows how each model will tokenise the page
python -m rxbench models download primary                  # weights, about 184 GB; asks for confirmation

python -m rxbench run --input page.jpg --models primary    # one model after another
python -m rxbench run --input-dir data/dataset_english/images --limit 5 --models medgemma_27b,gemma4_31b
python -m rxbench rerun EXP_0001 --models qwen3vl_32b --max-new-tokens 16384

python -m rxbench compare EXP_0001                         # where the models differ
python -m rxbench annotation-template EXP_0001             # blank ground-truth file for a person to fill in
python -m rxbench evaluate EXP_0001                        # needs a verified annotation
python -m rxbench report --out reports/RUNTIME_REPORT.md   # memory and inference time
python -m rxbench offline-check --image page.jpg --inference

streamlit run app/main.py                                  # interface at http://localhost:8501
```

To try the whole pipeline without any weights:

```bash
python -m rxbench run --mock --input page.jpg
```

Mock output comes from a software test double. It is labelled MOCK in metadata, reports and the
interface, and says nothing about any model.

### The interface

Eight screens: hardware and environment, model management, new experiment, results (one tab per model),
comparison, experiment history, annotation, evaluation. It binds to `127.0.0.1` and sends nothing out.

### What a run produces

```
experiments/EXP_0001/
    experiment.json                         hashes, dimensions, transformations, protocol flags
    input/original_prescription.jpg         the exact original bytes
    prompt/master_prompt_snapshot.txt       the exact prompt bytes, with prompt_hash.txt
    <model>/raw_output.txt                  the authoritative record
    <model>/structured_output.json          rule-based view of the raw output
    <model>/metadata.json                   model revision, settings, tokens, timing, memory, environment
    comparison/comparison_report.{json,md}
    evaluation/evaluation_report.{json,md}  only when ground truth exists
```

## What the code enforces

- The original bytes are stored and hashed; the page is never cropped, enhanced or resized by the
  application. What each model's own processor does to it is recorded.
- The master prompt is sent in full and unchanged to every model. A run is refused if the rule to
  ignore pre-printed text is missing from it.
- Models run one after another and never see each other's output. One failure does not stop the rest.
- Raw output is stored byte-exact. Parsing reads it and changes nothing; no model repairs or corrects it.
- A registered model that is unavailable is reported as not run. Nothing is substituted.
- Truncated output is marked incomplete, and a rerun keeps the earlier result.
- Comparison shows differences only. Accuracy is reported only against human-verified ground truth.
- Inference loads local files only and makes no network request.

## Things to know before reading results

- The models see the page at **different effective resolutions** by design: MedGemma uses one 896×896
  view (256 image tokens), Gemma 4 up to 1120 tokens, Qwen3-VL close to native resolution.
- Three settings depart from checkpoint defaults, deliberately and with a record in every run:
  greedy decoding, Gemma 4's visual token budget (1120, not 280), Qwen3.5 reasoning off.
- MedGemma can open a reasoning block without being asked. It is kept in the raw output and left out of
  the structured parse.
- The master prompt asks for Markdown, not JSON. `structured_output.json` is built by fixed rules from
  that Markdown; a model that ignores the layout is reported as not parsed.

`docs/EXPERIMENT_PROTOCOL.md` and `docs/KNOWN_LIMITATIONS.md` cover these in full.

## Data and privacy

`experiments/`, `annotations/*.json`, `logs/` and `.state/` are git-ignored because they can contain
patient documents and model outputs about them. Keep them out of version control and out of
cloud-synchronised folders, and rename files that carry patient identifiers before loading them.
None of these models is a medical device, and nothing here is for clinical decision-making.

`data/dataset_english.zip` holds the 77 prescription photographs used during development; read
`data/README.md` before using or sharing them. It has no verbatim ground truth.

## Licence

The code and documentation in this repository are released under the MIT License (see `LICENSE`).

That licence does not extend to things this project uses but does not own:

- the images in `data/dataset_english.zip`, which remain under the terms of their original collection;
- the model weights, each under its own licence (MedGemma under the Health AI Developer Foundations
  terms; Gemma 4, Qwen3-VL and Qwen3.5 under Apache-2.0). See `docs/MODEL_DOWNLOAD_GUIDE.md`.

## Documentation

| file | contents |
|---|---|
| `docs/STATUS_REPORT.md` | what was downloaded, loaded and run; measured time and memory; deviations |
| `docs/INSTALLATION.md` | installation on a Slurm cluster, a GB10 system and a laptop; tested versions |
| `docs/MODEL_DOWNLOAD_GUIDE.md` | official links, revisions, sizes, licences, access, troubleshooting |
| `docs/EXPERIMENT_PROTOCOL.md` | protocol, message packaging, image processing, stored files |
| `docs/EVALUATION_GUIDE.md` | annotation format and the definition of every metric |
| `docs/KNOWN_LIMITATIONS.md` | what is unverified and what to keep in mind when reading results |
| `docs/TEST_RESULTS.md` | test results and what kind of test each file is |
| `docs/CLUSTER_RUNBOOK.md` | step-by-step commands for the Slurm cluster the `jobs/` scripts were written for |
| `CLAUDE.md` | working rules for Claude Code in this repository |

## Layout

```
app/                    Streamlit interface (presentation only)
rxbench/
  hardware/             machine discovery, memory measurement, hardware report
  models/               adapter interface, shared Transformers adapter, one adapter per model,
                        registry, download manager, backend manager, mock test double
  inference/            protocol, sequential engine, image and prompt loading, generation settings,
                        output parser
  evaluation/           CER, WER, per-category, medicine, printed-text and uncertainty metrics;
                        model comparison; annotation format
  storage/              experiment folders, metadata, export
  offline.py  cli.py  context.py  config.py
prompts/master_prescription_prompt.txt     the master prompt, verbatim
model_registry.yaml     models and pinned revisions
config.yaml             paths, generation defaults, per-model options
data/                   dataset_english.zip (77 prescription photographs) and its description
jobs/                   Slurm job scripts (written for one specific cluster; adapt paths and modules)
scripts/setup_gb10.sh   environment setup for a stand-alone GB10 system (unverified)
tests/                  test suite
```

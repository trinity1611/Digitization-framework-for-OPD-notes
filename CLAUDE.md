# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`rxbench`: a local research benchmark that measures how well large open-weight vision-language models
transcribe and structure **handwritten medical prescriptions, zero-shot**. Each model gets the same
original page and the same fixed master prompt once; the raw response is stored exactly and compared
or evaluated afterwards. It is a baseline, so the value of the project is in what it refuses to do.

## Rules that must not be broken

These are the experiment. A change that violates one is wrong even if it improves results.

- **No help for the models.** No OCR, cropping, layout or text-region detection, image enhancement,
  medicine catalogue, retrieval, internet lookup, agents, model-to-model communication, consensus or
  automatic correction. `tests/test_hardware_offline_cli.py` fails if such a library is imported.
- **The master prompt is fixed.** `prompts/master_prescription_prompt.txt` is the user's text, verbatim.
  Never shorten, reword, reformat or "fix" it, and never add a system prompt or per-model wording.
  Every model receives the identical text; only chat-template packaging differs.
- **The original image is the input.** The application never resizes or crops the page. Only options
  of a model's *own* image processor may be set (each adapter has an `ALLOWED_IMAGE_OPTIONS` whitelist).
- **Raw output is authoritative.** `raw_output.txt` is written byte-exact and never edited. Parsing
  reads it and may only copy text out of it. No model is ever used to repair or correct output.
- **Never substitute a checkpoint.** Registered repositories and pinned revisions only. A model that is
  not available is recorded as not run. `google/medgemma-27b-text-it` (text-only) is forbidden.
- **Models run sequentially and independently.** A failure of one model or page is recorded and the
  rest continue. Outputs are never merged or ranked.
- **No accuracy without human-verified ground truth.** Model agreement is not correctness. Annotations
  are written by people and are never pre-filled from model output.
- **Mock is not a model.** `rxbench/models/mock.py` is a test double. Its results are flagged `is_mock`
  and labelled MOCK everywhere. Never describe a model as tested, loaded or run on the basis of mock
  tests or processor-only checks. Say exactly which of the three levels was exercised.
- **Local only.** Inference loads local snapshot directories with `local_files_only=True`. No network
  calls, telemetry or remote logging. Logs carry ids, hash prefixes and timings only: never file names,
  prompt text or model output.
- **Weights are never downloaded without explicit confirmation** from the user (tens of GB each).

## Commands

```
python -m pytest                                  # full suite (about 1.5 min)
python -m pytest tests/test_output_parser.py      # one file
python -m pytest -k truncated                     # one test by name

python -m rxbench hardware-report                 # inspect the machine (do this before installing anything)
python -m rxbench models list                     # registry and local availability
python -m rxbench models download primary --config-only   # processor files only, small, no confirmation
python -m rxbench preflight --image page.jpg      # real processors, no weights
python -m rxbench run --mock --input page.jpg     # whole pipeline with the test double
python -m rxbench run --input page.jpg --models primary
python -m rxbench reparse all                     # rebuild structured_output.json from stored raw outputs
streamlit run app/main.py                         # interface
```

PyTorch is intentionally absent from `requirements.txt`: it must be the CUDA build for the machine.
`transformers` is pinned to 5.12.1 because every registered architecture was checked against it
(Gemma 4 needs at least 5.5). Do not change the pin without re-running `tests/test_real_processors.py`.

## Architecture

```
app/        Streamlit pages. Presentation only; each page ends with render() and must stay importable alone.
rxbench/
  context.py        builds AppContext (config, registry, download manager, experiment manager)
  cli.py            every command; heavy imports happen inside the command functions
  inference/        protocol.py (the protocol as one call) -> engine.py (sequential loop)
                    image_loader.py, prompt_loader.py, generation.py, output_parser.py
  models/           base.py (adapter interface) -> transformers_adapter.py (shared implementation)
                    -> one small file per model; registry.py reads model_registry.yaml
  hardware/         detect.py (stdlib only), memory.py (unified vs discrete accounting), diagnostics.py
  evaluation/       cer, wer, field, medicine, printed-text, uncertainty metrics; comparison; annotation
  storage/          experiment_manager.py (on-disk layout), metadata.py, export.py
  offline.py        NetworkGuard and the offline verification
model_registry.yaml   models, pinned revisions, licences     config.yaml   paths, defaults, per-model options
jobs/                 Slurm scripts for one specific cluster (site-specific paths and module names)
data/                 dataset_english.zip: 77 prescription photographs; unzip to data/dataset_english/ (git-ignored)
```

Flow: `protocol.run_protocol` stores the document and a prompt snapshot (one experiment per page), then
`InferenceEngine.run` loops models on the outside and experiments on the inside: load once, run every
page, unload, check memory. Adapters receive only a `PreparedImage` and the prompt text.

Adding a model: add an entry to `model_registry.yaml`. Write an adapter class only if the architecture
needs a different model class, image options or chat-template arguments (subclasses are about 20 lines).

## Things that are easy to get wrong

- **Tests come in three kinds**, listed in `docs/TEST_RESULTS.md`: mock-adapter software tests,
  real-processor tests without weights (skipped when processor files are not cached), and none that run
  model weights. Real inference is always a separate, reported activity.
- **`rxbench.hardware` must import with the standard library only** (it runs before an environment
  exists). Do not add `yaml`, `torch` or `psutil` imports at module top level there.
- **Memory reporting has two bases.** Discrete GPUs use `torch.cuda` peaks; unified-memory machines
  (NVIDIA GB10) use the drop in system available memory. Keep `primary_peak_basis` truthful.
- **The parser is deliberately dumb.** It locates the A to E sections the prompt asks for. A leading
  reasoning block (`<think>`, Gemma 4 channel, MedGemma `<unused94>…<unused95>`) is split off for
  parsing only. If you change it, bump `PARSER_VERSION` and keep the test that every structured value
  is a substring of the raw output.
- **Reruns archive, never overwrite**: earlier results move to `<model>/history/<timestamp>/`.
- **Deviations from checkpoint defaults are deliberate and documented** (greedy decoding, Gemma 4 visual
  token budget 1120, Qwen3.5 reasoning off). Record any new one in `docs/EXPERIMENT_PROTOCOL.md`.
- **`jobs/*.sbatch` and `*.sh` must keep LF line endings** (`.gitattributes` enforces it).
- **The dataset's CSV descriptions are paraphrases, not transcriptions.** Never use them as ground
  truth for CER, WER or medicine matching (see `data/README.md`).
- **`experiments/`, `annotations/*.json`, `logs/`, `.state/` are git-ignored** and may hold patient
  documents. Never commit them, and never paste their contents into issues, commits or external tools.
- On Windows, pass multi-line Python to the shell through a file, not a heredoc with nested quotes.

## Current state

Read `docs/STATUS_REPORT.md` before making claims about what works. In short: the software is complete
and its test suite passes; the three primary models (MedGemma 27B, Gemma 4 31B, Qwen3-VL 32B) have not
been run with weights by this project; only MedGemma 1.5 4B at 4-bit was run for real, on one page.
GB10 detection and the Slurm scripts are unverified on real hardware. When you run something for real,
update `docs/STATUS_REPORT.md` with what was measured, on which machine, at which precision.

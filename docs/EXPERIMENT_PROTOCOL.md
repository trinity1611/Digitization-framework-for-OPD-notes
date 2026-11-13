# Experiment protocol

This is a zero-shot baseline. Each model receives the original prescription page and one fixed master
prompt, once, and its response is stored unmodified.

## 1. What is and is not in the pipeline

| in | not in (and tested not to be) |
|---|---|
| the original document bytes, stored and hashed | OCR, handwriting-region cropping, layout or text-region detection, line recognition |
| decoding to an upright RGB page | image enhancement, binarisation, denoising, manual cropping |
| each model's own processor and chat template | medicine catalogues, retrieval, internet search, external clinical knowledge |
| one generation per model per page | agents, iteration, model-to-model communication, consensus, automatic correction |
| deterministic parsing of the response for display | another model repairing JSON or correcting handwriting |

`tests/test_hardware_offline_cli.py` scans the source for OCR, API, retrieval and telemetry libraries and
fails if one is imported.

## 2. Steps

For every page:

1. **Store the original.** The uploaded bytes are written unchanged to `input/` and their SHA-256 is
   recorded. For a PDF the original file is kept and each selected page is rendered once to a lossless
   PNG (pypdfium2, 200 dpi by default); page order is preserved and each page is its own experiment.
2. **Snapshot the prompt.** The exact bytes of `prompts/master_prescription_prompt.txt` are copied to
   `prompt/master_prompt_snapshot.txt` with their SHA-256. A run is refused if the prompt lacks the
   printed / pre-filled text exclusion rule.
3. **Decode the page once**, identically for all models: apply the EXIF orientation tag if present
   (`image.apply_exif_orientation`), convert to RGB if needed. Both are recorded as transformations when
   they change anything. The page is not resized.
4. **For each selected model, in turn:** record memory, load the model, run it on every page, save the
   output and metadata after each page, unload, release memory, check that memory returned to the
   pre-load level, then move to the next model.
5. **Write a comparison** of the outputs for the page. It lists differences; it does not merge, rank or
   choose.

A model's run reads only the stored page and the prompt snapshot. No output of any model is ever an
input to another. A failure to load a model, or an error on one page, is written to that run's
`metadata.json` and the remaining models and pages continue.

## 3. Message packaging

The content is the same for every model: one user turn containing the page followed by the complete
master prompt. There is no system prompt, no few-shot example and no added wording. Only the
chat-template tokens differ, because each architecture defines its own.

| model | model class (Transformers) | processor | reasoning mode |
|---|---|---|---|
| MedGemma 27B | `AutoModelForImageTextToText` → `Gemma3ForConditionalGeneration` | `Gemma3Processor` | no switch in the chat template (see note below) |
| Gemma 4 31B | `AutoModelForMultimodalLM` → `Gemma4ForConditionalGeneration` | `Gemma4Processor` | off (`enable_thinking=False`, the checkpoint default) |
| Qwen3-VL 32B | `Qwen3VLForConditionalGeneration` | `Qwen3VLProcessor` | none in the Instruct checkpoint |
| Gemma 4 26B A4B | as Gemma 4 31B | `Gemma4Processor` | off |
| Qwen3.5 27B | `Qwen3_5ForConditionalGeneration` | `Qwen3VLProcessor` | off (`enable_thinking=False`; the checkpoint default is on) |
| MedGemma 1.5 4B | as MedGemma 27B | `Gemma3Processor` | no switch in the chat template (see note below) |

**Unrequested reasoning blocks.** Gemma-3-based MedGemma models have no reasoning switch, but they can
open a reasoning block on their own, delimited by `<unused94>` and `<unused95>`. MedGemma 1.5 4B did so
in the real run on 2026-11-13. The application does not suppress it (that would mean changing the
prompt): the block is kept in the raw output, a warning is written to the metadata, and the structured
parse reads only the answer that follows. Whether MedGemma 27B does the same is not yet known.

After loading, the adapter checks that the loaded class is the registered architecture and stops
otherwise. Before generating, it checks that the processed input contains a pixel tensor and image
tokens; if not, the run fails with `ImageNotDeliveredError` rather than sending text only. The SHA-256
of the pixels handed to the processor and of the prompt text are stored per run and compared with the
experiment's values.

## 4. Image processing by each model's own processor

No resizing or tiling is done by the application. What each model's native processor does is recorded
in `metadata.json` under `image_processing`. For the 960×1280 sample page `0022.jpg` (measured with the
real processors, see `docs/STATUS_REPORT.md`):

| model | native behaviour | setting used | image tokens |
|---|---|---|---|
| MedGemma 27B, MedGemma 1.5 4B | page resized to one 896×896 view (aspect ratio not kept), 256 tokens per view; optional "pan and scan" adds crops | checkpoint default (pan and scan off) | 256 |
| Gemma 4 31B, 26B A4B | aspect-ratio-preserving resize to fit a visual token budget of 70, 140, 280, 560 or 1120 | **1120** (checkpoint default is 280) | 1064 |
| Qwen3-VL 32B, Qwen3.5 27B | dynamic resolution, one token per 32×32 pixels, between 65,536 and 16,777,216 pixels | checkpoint default | 1200 (page kept at 960×1280) |

The models therefore see the same page at different effective resolutions. That is a property of the
models, and part of what the baseline measures; the settings are in `config.yaml` under `models:` and
only options of the native processors are accepted there.

## 5. Generation settings

Defaults (`config.yaml`, shown and editable on the New experiment screen, overridable on the command line):

| setting | default | note |
|---|---|---|
| decoding | greedy (`do_sample=False`) | temperature 0 is expressed this way in Transformers |
| temperature, top-p, top-k | not used | passed as `None` so a checkpoint's sampling defaults cannot apply |
| maximum new tokens | 8192 | the one real run so far (MedGemma 1.5 4B, sample page) needed 3,761 tokens, of which roughly 40 % was an unrequested reasoning block; raise the limit if a run is truncated |
| repetition penalty | 1.0 | |
| seed | 0 | set before each generation; irrelevant for greedy decoding but recorded |
| stop sequences | none | |
| reasoning mode | off | where a model has one |

Every checkpoint ships sampling defaults (for example temperature 1.0, top-p 0.95, top-k 64 for Gemma 4;
0.7 / 0.8 / 20 for Qwen3-VL). The baseline overrides them with greedy decoding for reproducibility. The
requested settings, the arguments actually passed to `generate`, and the checkpoint defaults that were
overridden are all written to `metadata.json`.

**Truncation.** If generation stops at the token limit without an end-of-sequence token, the run is
stored with status `incomplete_truncated`, the partial text is kept, and the interface offers a rerun
with a higher limit. A rerun moves the earlier result to `history/<timestamp>/`; nothing is deleted.

## 6. What is stored

```
experiments/EXP_0001/
    experiment.json                          input hashes, dimensions, transformations, prompt hash, protocol flags
    input/original_prescription.jpg          exact original bytes
    prompt/master_prompt_snapshot.txt        exact prompt bytes
    prompt/prompt_hash.txt
    medgemma_27b/raw_output.txt              authoritative record: the generated text, exactly
    medgemma_27b/raw_output_with_special_tokens.txt
    medgemma_27b/structured_output.json      rule-based view of the raw output (sections A to E)
    medgemma_27b/metadata.json
    gemma4_31b/...   qwen3vl_32b/...
    comparison/comparison_report.json, comparison_report.md
    evaluation/...                           only when ground truth exists
experiments/_batches/BATCH_*.json            which models ran on which experiments, in what order
```

`metadata.json` holds: experiment and batch identifiers; original file name, image SHA-256 and
dimensions; model identifier and revision; prompt hash; backend and version; model and processor class;
precision and quantisation; device placement; image-processing configuration; requested and actual
generation settings; load time, preprocessing time, generation time, tokens per second; input, image and
output token counts; truncation flag; raw-output hash; parse status; memory (before load, peak during
load, before inference, peak during inference, preprocessing overhead, after unload, release check);
Python, PyTorch, CUDA and driver versions; GPU and unified-memory information; errors and warnings.

The application log (`logs/rxbench.log`) contains identifiers, hash prefixes, counts and timings only:
no file names, prompt text or model output.

## 7. Structured output

The master prompt asks for Markdown sections A to E, not JSON. So:

- If the output is valid JSON (whole output or one fenced block) it is stored as parsed.
- Otherwise a fixed set of rules locates the sections, the eleven categories and their
  "Field: value" items. Every value in `structured_output.json` is text copied from the raw output.
- If neither applies, parsing is reported as unsuccessful and the raw output is shown as it is.

Nothing is repaired, corrected, normalised or completed, and no model is involved. `parse_status`
records which method applied and which sections or categories were not found.

## 8. Memory measurement

Peaks are sampled every 0.25 s in a background thread and reported on the basis that fits the machine:

- **discrete GPU** (H200, RTX Pro 6000): `torch.cuda.max_memory_allocated`, plus device memory in use
  from `torch.cuda.mem_get_info`;
- **unified memory** (GB10): the largest drop in the operating system's available memory, because the
  GPU allocates from system RAM and `nvidia-smi` may report no total.

Both views are stored; `primary_peak_basis` says which one the headline figure uses.

## 9. Documented departures from checkpoint defaults

These are deliberate, recorded in every run's metadata, and can be changed in `config.yaml`:

1. Greedy decoding instead of each checkpoint's sampling defaults (reproducibility).
2. Gemma 4 visual token budget 1120 instead of 280 (the official model card recommends higher budgets
   for OCR and document parsing; 280 would hide most handwriting detail).
3. Qwen3.5 reasoning mode off instead of on (so that all models answer directly; turning it on is a
   separate condition).
4. The whole prompt is sent in the user turn; no system prompt is used, although Gemma 4 and Qwen
   support one (identical packaging across models).

MedGemma is run with its default single 896×896 view. Pan and scan is a native option that may help on
dense pages; using it is a separate condition, not the baseline.

## 10. Inference backend

Hugging Face Transformers, in-process, for every model. Reasons and the alternatives considered are in
`rxbench/models/backend_manager.py` and on the Hardware screen; in short: it is the reference
implementation on each model card, it lets the application inspect the processed input and prove the
image is present, it runs on x86-64 and ARM64 without a server, and throughput is not a goal here.

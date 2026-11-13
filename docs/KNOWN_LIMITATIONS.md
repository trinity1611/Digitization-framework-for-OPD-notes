# Known limitations

## What has not been run

- **The three primary models have not been run.** Their weights were not downloaded: the machine this
  project was built on is a laptop with an 8 GB GPU. Their adapters were checked with the official
  processors (tokenizer, chat template, image processor) but not with weights. See `docs/STATUS_REPORT.md`
  for exactly what was and was not exercised.
- **No GB10 system was available.** GB10 detection, unified-memory accounting and `scripts/setup_gb10.sh`
  are written from the platform's documented behaviour and tested only with simulated inputs.
- **The Slurm job scripts have not been executed.** They follow the cluster guide v3.0 and pass a shell
  syntax check. Things that can only be confirmed on the cluster: that conda-forge resolves a CUDA build
  of PyTorch on Rocky Linux 8.10, that CPU-only jobs are accepted for the download, that `unshare -rn`
  is permitted, and the exact GRES names.
- **Offline inference is verified only where it has been run.** The offline check never reports a model
  as verified unless real inference completed under the network guard.

## Experimental design

- **Different effective resolutions.** MedGemma sees one 896×896 view (256 tokens, aspect ratio not
  kept); Gemma 4 is given its largest visual budget (up to 1120 tokens); Qwen3-VL works near native
  resolution (1200 tokens for a 960×1280 page). This is how the models are built, but it means a
  difference in transcription quality may be a difference in resolution. MedGemma's pan-and-scan option
  can be switched on as a separate condition.
- **Gemma 4's visual budget is not the checkpoint default** (1120 instead of 280), following the model
  card's advice for document parsing. Qwen3.5's reasoning mode is switched off although its default is
  on. Both are recorded and configurable.
- **One fixed prompt for all models.** The prompt is long (about 4,100 to 4,500 tokens, depending on the tokenizer) and asks for a long,
  highly structured answer. Models differ in how well they follow long instructions; that is part of
  what is measured, and it is not separated from handwriting-reading ability.
- **Greedy decoding** gives one deterministic answer per model on a given software and hardware stack.
  It does not measure variability, and bit-identical output across different GPUs or library versions
  is not guaranteed.
- **Long outputs.** The requested format can exceed the default 8192 new tokens on dense pages. Such
  runs are marked incomplete rather than silently accepted; rerun them with a higher limit.
- **Quantised runs are different models in effect.** INT8/INT4 results must not be pooled with BF16.

## Parsing and metrics

- **Structured parsing is rule-based.** It recognises the section layout the prompt asks for (and JSON).
  A model that answers in another layout is reported as "not parsed" even if its content is good; the
  raw output is always available. Parsing failure therefore partly measures format compliance.
- **Bounding boxes and location labels** in model output are stored but not scored.
- **Alignment heuristics.** Uncertainty metrics align annotated regions to transcription lines in
  reading order, and medicine and category metrics pair items by string similarity. A model that
  reorders content heavily, or merges lines, can be penalised by the alignment rather than by its reading.
- **Printed-text leakage is detected by string containment.** A printed word that the doctor also wrote
  by hand has to be marked `also_handwritten` in the annotation or it is counted as a leak.
- **"Normalised" means string normalisation only.** Brand and generic names are never equated, by
  design; a model that writes the generic for a handwritten brand name is scored as wrong.
- **CER/WER depend on the reference convention** (one region per line, crossed-out text excluded).
  Keep the convention fixed across annotators.

## Data

- The dataset's CSV descriptions are paraphrases and are not used as ground truth. Verbatim annotations
  do not exist yet; until they do, the application can compare models but cannot report accuracy.
- The dataset images are small photographs (about 960×1280, 30 to 140 KB, some blurred). Results on
  them may not transfer to scanned OPD forms; the printed-form subset of the original collection
  contains ticks only and was excluded from `dataset_english`.

## Software

- **Simultaneous model serving is not implemented.** The option exists only as an explicit switch that
  currently raises an error; models always run one after another.
- **One backend.** Only Hugging Face Transformers is implemented. FP8 is unavailable in it. vLLM and
  SGLang were assessed and left out (see `rxbench/models/backend_manager.py`).
- **The interface blocks while a model runs** and shows progress per page. For long runs use the
  command line or the Slurm job; the interface is best for single pages and for reviewing results.
- **The interface was tested with Streamlit's headless test harness** (every page renders, key elements
  present) but not reviewed visually in a browser.
- **PDF rendering** uses pypdfium2 at 200 dpi. Scanned PDFs are rendered, not extracted; any embedded
  text layer is ignored on purpose.
- **Memory peaks are sampled** every 0.25 s; very short spikes can be missed. On unified memory the
  figure is the drop in system available memory, which includes anything else that allocated at the
  same time.
- **The in-process network guard** blocks Python-level socket use. Native code opening sockets directly
  would bypass it; the cluster offline job therefore also uses a network namespace where permitted.

## Privacy

- All data stays on local disks; the application makes no network request during inference, and
  Streamlit's usage statistics are disabled in `.streamlit/config.toml`.
- **Cloud-synchronised folders.** If the project sits inside OneDrive, Dropbox or a similar folder,
  anything written to `experiments/`, `annotations/` or `logs/` is uploaded by that service, outside
  the application's control.
  For real patient documents set `RXBENCH_EXPERIMENTS_DIR`, `RXBENCH_ANNOTATIONS_DIR` and
  `RXBENCH_LOG_DIR` to a folder that is not synchronised, or move the project.
- Original file names are stored in `experiment.json` and `metadata.json` (not in logs). Rename files
  that carry patient identifiers before loading them.
- MedGemma's licence terms apply to its use and outputs; none of the models is a medical device, and
  nothing here is for clinical decision-making.

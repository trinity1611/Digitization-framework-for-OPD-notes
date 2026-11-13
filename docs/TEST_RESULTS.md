# Test results

Run on 2026-11-13 on the development laptop (Windows 11, Python 3.13.13, PyTorch 2.6.0+cu124,
transformers 5.12.1, Streamlit 1.67.0) with `python -m pytest`.

```
============================= test session starts =============================
platform win32 -- Python 3.13.13, pytest-9.1.1, pluggy-1.6.0
configfile: pyproject.toml
testpaths: tests
collected 127 items

tests\test_app_pages.py ....................                             [ 15%]
tests\test_evaluation.py .....................                           [ 32%]
tests\test_hardware_offline_cli.py .............                         [ 42%]
tests\test_output_parser.py .........                                    [ 49%]
tests\test_prompt_and_images.py .................                        [ 62%]
tests\test_real_processors.py ...........                                [ 71%]
tests\test_registry_and_models.py ...................                    [ 86%]
tests\test_storage_and_engine.py .................                       [100%]

======================= 127 passed in 87.11s (0:01:27) ========================
```

**127 passed, 0 failed, 0 skipped.**

## What kind of test each file is

| file | tests | kind | uses |
|---|---|---|---|
| `test_prompt_and_images.py` | 17 | unit | the real master prompt file; synthetic images and PDFs |
| `test_registry_and_models.py` | 19 | unit | the real registry; no network (Hub calls are replaced) |
| `test_output_parser.py` | 9 | unit | sample outputs, including the shape of a real MedGemma response |
| `test_evaluation.py` | 21 | unit | hand-made annotation and output with known answers |
| `test_storage_and_engine.py` | 17 | software integration | **mock adapters** |
| `test_hardware_offline_cli.py` | 13 | unit and integration | this machine's hardware; **mock adapters**; simulated GB10 inputs |
| `test_app_pages.py` | 20 | interface smoke tests | Streamlit's headless test harness; **mock adapters** |
| `test_real_processors.py` | 11 | real processors, no weights | the official tokenizer, chat template and image processor of all six checkpoints |

**None of these 127 tests is model inference.** Tests that use the mock adapter exercise the
application's own logic with a software test double; they say nothing about any model. The
real-processor tests load official processor files but no weights and generate nothing. Real inference
is reported separately in `docs/STATUS_REPORT.md`.

`test_real_processors.py` skips a model whose processor files are not in the local Hugging Face cache
(`python -m rxbench models download all --config-only` fetches them). In this run all six were present.

Additionally, every source file was parsed with Python 3.10, and the parser, evaluation, prompt/image
and storage/engine tests were run on Python 3.10.20: all passed except the two PDF tests, which need
`pypdfium2` and that interpreter did not have it installed.

## Coverage of the specification's testing requirements (section 22)

| # | requirement | how it is checked | status |
|---|---|---|---|
| 1 | hardware correctly detected | `test_detect_all_on_this_machine`; GB10 logic with simulated inputs (`test_gb10_is_identified_from_evidence_not_assumed`) | checked on the laptop; **not on a GB10 or the cluster** |
| 2 | CPU architecture supported | compatibility assessment in the hardware report | x86-64 checked; ARM64 not available |
| 3 | backend supports GB10 | none possible without the device | **not verified** |
| 4 | all three primary models can be downloaded or loaded | repositories, revisions, access and file lists verified on the Hub; processors load | downloadable: verified; **loading weights: not done** |
| 5 | each model accepts an image input | `test_real_processor_receives_image_and_full_prompt` (image tokens and pixel tensor present for all six); the adapter refuses to generate without them | processor level: verified; with weights: MedGemma 1.5 4B only |
| 6 | same master prompt to every model | `test_all_models_get_the_same_image_and_prompt`, `test_models_run_strictly_one_after_another_and_never_see_each_other`; prompt text read back from each real tokenizer | verified |
| 7 | image hash identical across models | same tests; pixel hash recorded per run and compared | verified |
| 8 | no external OCR | `test_no_ocr_api_retrieval_or_telemetry_libraries_are_imported` | verified |
| 9 | no external medicine database | same test; metrics use string comparison only | verified |
| 10 | printed / pre-filled exclusion in the prompt | `test_printed_text_exclusion_rule_present`; runs are refused without it (`test_prompt_without_exclusion_rule_is_refused`) | verified |
| 11 | raw outputs preserved exactly | `test_raw_output_is_preserved_exactly_and_parsing_does_not_touch_it`, `test_text_is_written_and_read_back_exactly` | verified |
| 12 | parsing does not modify raw output | `test_parsing_does_not_modify_the_raw_string`, `test_every_structured_value_is_a_substring_of_the_raw_output` | verified |
| 13 | execution and memory recorded | metadata fields asserted in `test_all_models_get_the_same_image_and_prompt`; real values in the MedGemma 1.5 4B run | verified |
| 14 | one model's failure does not stop the others | `test_one_model_failing_does_not_stop_the_others`, `test_unavailable_real_model_is_reported_not_substituted` | verified |
| 15 | offline execution | `test_network_guard_*`, `test_offline_check_*`; real run under the guard reported in `STATUS_REPORT.md` | guard verified; real offline inference for MedGemma 1.5 4B only |
| 16 | no patient information in logs | `test_logs_contain_no_document_content_or_file_names` | verified |

Re-run after any change with `python -m pytest`. On the cluster, `jobs/01_create_env.sh` runs the suite
(without the interface tests) at the end of environment creation.

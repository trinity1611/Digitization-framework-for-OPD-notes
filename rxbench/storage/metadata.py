"""Assemble metadata.json for one model run on one experiment."""
from __future__ import annotations

import hashlib
from typing import Any

from .. import __version__

STATUS_COMPLETE = "complete"
STATUS_TRUNCATED = "incomplete_truncated"
STATUS_FAILED_LOAD = "failed_model_load"
STATUS_FAILED_INFERENCE = "failed_inference"
STATUS_NOT_AVAILABLE = "model_not_available_locally"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_metadata(*, experiment: dict[str, Any], model: dict[str, Any], status: str,
                   generation_config: dict[str, Any], hardware: dict[str, Any], batch_id: str | None,
                   started_at: str, finished_at: str, result: dict[str, Any] | None = None,
                   runtime: dict[str, Any] | None = None, memory: dict[str, Any] | None = None,
                   parse_status: dict[str, Any] | None = None, errors: list[str] | None = None,
                   warnings: list[str] | None = None, is_mock: bool = False) -> dict[str, Any]:
    result = result or {}
    runtime = runtime or {}
    inp = experiment["input"]
    raw = result.get("raw_text")
    meta: dict[str, Any] = {
        "schema_version": 1,
        "application_version": __version__,
        "experiment_id": experiment["experiment_id"],
        "batch_id": batch_id,
        "completion_status": status,
        "output_complete": status == STATUS_COMPLETE,
        "is_mock": is_mock,
        "started_at": started_at,
        "finished_at": finished_at,
        "input": {
            "original_filename": inp["original_filename"],
            "image_sha256": inp["original_sha256"],
            "model_input_file_sha256": inp["model_input_file_sha256"],
            "original_width": inp["image"]["original_width"],
            "original_height": inp["image"]["original_height"],
            "pixel_sha256_given_to_model": result.get("input_pixel_sha256"),
            "pixel_sha256_expected": inp["image"]["model_input_pixel_sha256"],
            "transformations_before_processor": inp["transformations"],
        },
        "prompt": {"sha256": experiment["prompt"]["sha256"],
                   "sha256_given_to_model": result.get("prompt_sha256"),
                   "n_chars": experiment["prompt"]["n_chars"],
                   "rendered_chat_prompt_sha256": result.get("rendered_prompt_sha256")},
        "model": {
            "key": model.get("key"),
            "identifier": model.get("repo_id"),
            "display_name": model.get("display_name"),
            "revision": runtime.get("revision") or model.get("revision"),
            "architecture": model.get("architecture"),
            "parameters_total": model.get("parameters_total"),
            "parameters_active": model.get("parameters_active"),
            "licence": model.get("licence"),
        },
        "backend": {"name": runtime.get("backend") or model.get("backend"),
                    "version": runtime.get("backend_version"),
                    "model_class": runtime.get("model_class"),
                    "processor_class": runtime.get("processor_class")},
        "precision": runtime.get("precision"),
        "quantisation": runtime.get("quantisation"),
        "device_map": runtime.get("device_map"),
        "image_processing": result.get("image_processing"),
        "generation": {
            "requested": generation_config,
            "actual_backend_arguments": result.get("actual_generation_kwargs"),
            "checkpoint_defaults_overridden": runtime.get("checkpoint_generation_defaults"),
            "thinking": result.get("thinking"),
        },
        "timing": {"model_load_seconds": runtime.get("load_seconds"),
                   "image_preprocessing_seconds": result.get("preprocess_seconds"),
                   "generation_seconds": result.get("generation_seconds"),
                   "tokens_per_second": result.get("tokens_per_second")},
        "tokens": {"input": result.get("n_input_tokens"), "image": result.get("n_image_tokens"),
                   "output": result.get("n_output_tokens"),
                   "max_new_tokens": generation_config.get("max_new_tokens"),
                   "stopped_on_end_of_sequence": result.get("stopped_on_eos"),
                   "truncated": result.get("truncated")},
        "output": {"raw_output_sha256": text_sha256(raw) if raw is not None else None,
                   "raw_output_chars": len(raw) if raw is not None else None,
                   "parse_status": parse_status},
        "memory": memory,
        "environment": hardware,
        "errors": errors or [],
        "warnings": (warnings or []) + (result.get("warnings") or []),
    }
    if is_mock:
        meta["mock_notice"] = ("MOCK RUN: produced by a software test double, not by a model. "
                               "It is not evidence of any model's behaviour.")
    return meta

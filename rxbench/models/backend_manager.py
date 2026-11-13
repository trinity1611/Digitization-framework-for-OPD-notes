"""Inference backend selection and status.

One backend is implemented: Hugging Face Transformers, in-process. The other
runtimes were assessed and are listed with the reason they are not used, so the
choice is explicit and can be revisited per model without touching the
experiment protocol or the interface.
"""
from __future__ import annotations

import importlib.metadata as ilm
from typing import Any

from .registry import ModelSpec
from .transformers_adapter import _version_tuple

BACKENDS: dict[str, dict[str, Any]] = {
    "transformers": {
        "implemented": True,
        "summary": "Hugging Face Transformers, in-process (PyTorch).",
        "why": [
            "It is the reference implementation named on every registered model's official card, with the "
            "model's own processor class and chat template.",
            "The processed input can be inspected before generation, so the application can prove the image "
            "is present (pixel tensor and image-token count) instead of trusting a server.",
            "The same code runs on x86-64 and ARM64 wherever a CUDA build of PyTorch exists; no server "
            "process, container or extra port is needed.",
            "BF16 is the checkpoints' native precision; INT8/INT4 are available through bitsandbytes.",
            "Batch throughput is not a goal of a one-page-at-a-time baseline.",
        ],
        "precisions": {"bf16": "native", "fp16": "supported by PyTorch; Gemma-family models are published "
                       "for bf16 and may be numerically unstable in fp16", "fp32": "supported (twice the memory)",
                       "int8": "bitsandbytes 8-bit", "int4": "bitsandbytes 4-bit NF4",
                       "fp8": "not available: needs a separately converted checkpoint"},
    },
    "vllm": {
        "implemented": False,
        "summary": "vLLM OpenAI-compatible server.",
        "why": [
            "Supports these architectures and is faster for many concurrent requests, which this baseline "
            "does not need.",
            "Images pass through an HTTP server and its own preprocessing path, which makes it harder to "
            "verify that the original page reached the model unchanged.",
            "FP8/INT4 serving would use third-party converted checkpoints; the baseline uses official weights only.",
            "ARM64/GB10 support depends on NVIDIA-provided builds and would need separate verification.",
        ],
    },
    "sglang": {
        "implemented": False,
        "summary": "SGLang serving runtime.",
        "why": ["Same trade-offs as vLLM for this use; recommended by the Qwen3.5 card for production "
                "throughput, not needed for a sequential baseline."],
    },
    "llama.cpp / GGUF": {
        "implemented": False,
        "summary": "llama.cpp with GGUF conversions.",
        "why": ["Requires converted (mostly community) GGUF files and separate vision projector files; "
                "these are not the official checkpoints, so they are excluded from the baseline."],
    },
}


def _installed(name: str) -> str | None:
    try:
        return ilm.version(name)
    except ilm.PackageNotFoundError:
        return None


def backend_status() -> dict[str, Any]:
    """Versions and readiness of the implemented backend on this machine."""
    status: dict[str, Any] = {"backend": "transformers", "transformers": _installed("transformers"),
                              "torch": _installed("torch"), "accelerate": _installed("accelerate"),
                              "bitsandbytes": _installed("bitsandbytes"), "cuda_available": False,
                              "problems": []}
    if not status["transformers"]:
        status["problems"].append("transformers is not installed")
    if not status["accelerate"]:
        status["problems"].append("accelerate is not installed (needed for device_map loading)")
    try:
        import torch
        status["cuda_available"] = bool(torch.cuda.is_available())
        status["torch_cuda_build"] = getattr(torch.version, "cuda", None)
        if not status["cuda_available"]:
            status["problems"].append("PyTorch cannot use CUDA on this machine")
    except Exception as exc:
        status["problems"].append(f"PyTorch import failed: {type(exc).__name__}")
    status["ready"] = not status["problems"]
    status["other_backends_installed"] = {k: _installed(k) for k in ("vllm", "sglang")}
    return status


def check_model(spec: ModelSpec) -> dict[str, Any]:
    """Does the installed backend know this model's architecture and processor?"""
    out: dict[str, Any] = {"model_key": spec.key, "backend": spec.backend, "checks": []}
    add = lambda name, ok, detail="": out["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
    if spec.backend != "transformers":
        add("backend_implemented", spec.backend == "mock", f"backend '{spec.backend}'")
        out["compatible"] = spec.backend == "mock"
        return out
    try:
        import transformers
    except Exception as exc:
        add("transformers_importable", False, type(exc).__name__)
        out["compatible"] = False
        return out
    add("transformers_version", not spec.min_transformers or
        _version_tuple(transformers.__version__) >= _version_tuple(spec.min_transformers),
        f"installed {transformers.__version__}, needs >= {spec.min_transformers}")
    add("architecture_class_available", hasattr(transformers, spec.architecture), spec.architecture)
    add("processor_class_available", hasattr(transformers, spec.processor_class), spec.processor_class)
    add("image_input_declared", spec.image_input, "image-text-to-text")
    out["compatible"] = all(c["ok"] for c in out["checks"])
    return out

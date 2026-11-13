"""Shared Hugging Face Transformers implementation.

Subclasses supply what differs between architectures: the model class, the
options of the native image processor, the chat-template arguments (thinking
mode) and how the image processing is described. Loading always uses the local
snapshot directory with `local_files_only=True`, so inference cannot reach the
network or pick up a different checkpoint.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

from PIL import Image

from ..hardware import memory as mem
from ..inference.generation import GenerationConfig
from ..logging_utils import get_logger
from .base import (AdapterError, BaseVLMAdapter, GenerationResult, ImageNotDeliveredError,
                   InsufficientMemoryError, ModelNotAvailableError, UnsupportedPrecisionError)

log = get_logger()


def _version_tuple(v: str) -> tuple[int, ...]:
    out = []
    for part in v.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


class TransformersVLMAdapter(BaseVLMAdapter):
    backend = "transformers"
    MODEL_CLASS = "AutoModelForImageTextToText"
    #: options of the model's own image processor that config.yaml may set
    ALLOWED_IMAGE_OPTIONS: tuple[str, ...] = ()
    #: token that closes a reasoning block, when the model has a thinking mode
    THINK_END_TOKEN: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.processor = None
        self.snapshot_path = None
        self.device_map_used: Any = None
        unknown = set(self.image_options()) - set(self.ALLOWED_IMAGE_OPTIONS)
        if unknown:
            raise AdapterError(f"{self.spec.key}: image_processing options {sorted(unknown)} are not options of "
                               f"this model's native image processor (allowed: {list(self.ALLOWED_IMAGE_OPTIONS)})")

    # ------------------------------------------------------------------ configuration hooks
    def image_options(self) -> dict[str, Any]:
        """Explicitly configured native image-processor options (None = checkpoint default)."""
        return {k: v for k, v in (self.model_cfg.get("image_processing") or {}).items() if v is not None}

    def image_processor_kwargs(self) -> dict[str, Any]:
        """Arguments passed to the image processor through the chat template."""
        return dict(self.image_options())

    def chat_template_kwargs(self, gen: GenerationConfig) -> dict[str, Any]:
        return {}

    def thinking_record(self, gen: GenerationConfig) -> dict[str, Any]:
        if self.spec.supports_thinking:
            return {"supported": True, "enabled": bool(gen.enable_thinking),
                    "checkpoint_default": self.spec.thinking_checkpoint_default}
        return {"supported": False, "enabled": False, "requested": bool(gen.enable_thinking)}

    def describe_image_processing(self, inputs: Any, image: Image.Image) -> dict[str, Any]:
        """Architecture-specific description of what the processor did to the page."""
        return {}

    # ------------------------------------------------------------------ loading
    def _resolve_snapshot(self):
        if self.download_manager is None:
            raise ModelNotAvailableError("No download manager configured")
        status = self.download_manager.local_status(self.spec)
        if status["status"] not in {"complete", "config_only"}:
            raise ModelNotAvailableError(
                f"{self.spec.repo_id} is not available locally (status: {status['status']}). "
                f"Download it first: python -m rxbench models download {self.spec.key}")
        return status

    def _check_backend(self) -> None:
        import transformers
        if self.spec.min_transformers and _version_tuple(transformers.__version__) < _version_tuple(
                self.spec.min_transformers):
            raise AdapterError(f"{self.spec.key} needs transformers >= {self.spec.min_transformers}; "
                               f"installed: {transformers.__version__}")
        if not hasattr(transformers, self.MODEL_CLASS):
            raise AdapterError(f"transformers {transformers.__version__} has no {self.MODEL_CLASS}")

    def load_processor(self) -> None:
        """Load only the processor (tokenizer, chat template, image processor)."""
        from transformers import AutoProcessor
        status = self._resolve_snapshot()
        self.snapshot_path = status["path"]
        self.processor = AutoProcessor.from_pretrained(self.snapshot_path, local_files_only=True)
        actual = type(self.processor).__name__
        if actual != self.spec.processor_class:
            raise AdapterError(f"{self.spec.key}: expected processor {self.spec.processor_class}, got {actual}")

    def _memory_guard(self) -> None:
        if self.exec_cfg.get("skip_memory_guard"):
            return
        import torch
        estimate = self.spec.memory_estimate_gb(self.precision)["total_gb"]
        snap = mem.snapshot()
        if self.exec_cfg.get("unified_memory"):
            available, basis = snap.get("system_available_bytes"), "system available memory (unified)"
        else:
            n = torch.cuda.device_count()
            available = sum(torch.cuda.mem_get_info(i)[0] for i in range(n))
            basis = f"free memory on {n} visible GPU(s)"
        if available is not None and available / 1024**3 < estimate:
            raise InsufficientMemoryError(
                f"{self.spec.display_name} at {self.precision} needs about {estimate:.0f} GB "
                f"but only {available / 1024**3:.1f} GB is available ({basis}). Choose a quantised precision, "
                "free memory, or set execution.skip_memory_guard: true to try anyway.")

    def load_model(self) -> None:
        import torch
        import transformers
        self._check_backend()
        if self.precision not in self.spec.supported_precisions():
            raise UnsupportedPrecisionError(
                f"{self.spec.key} does not list precision '{self.precision}' "
                f"(supported: {self.spec.supported_precisions()})")
        status = self._resolve_snapshot()
        if status["status"] != "complete":
            raise ModelNotAvailableError(f"{self.spec.repo_id}: weights are not downloaded "
                                         f"(local status: {status['status']})")
        if not torch.cuda.is_available() and not self.exec_cfg.get("allow_cpu"):
            raise AdapterError("CUDA is not available to PyTorch on this machine; refusing to load a "
                               f"{self.spec.parameters_total / 1e9:.0f}B-parameter model on the CPU")
        if torch.cuda.is_available():
            self._memory_guard()
        self.snapshot_path = status["path"]

        kwargs: dict[str, Any] = {"local_files_only": True, "low_cpu_mem_usage": True}
        device_map = self.exec_cfg.get("device_map", "auto") if torch.cuda.is_available() else "cpu"
        kwargs["device_map"] = device_map
        float_dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        if self.precision in float_dtypes:
            kwargs["dtype"] = float_dtypes[self.precision]
        elif self.precision in ("int8", "int4"):
            try:
                import bitsandbytes  # noqa: F401
            except Exception as exc:
                raise UnsupportedPrecisionError(
                    f"{self.precision} needs the bitsandbytes package, which is not usable here "
                    f"({type(exc).__name__})") from exc
            if self.precision == "int8":
                kwargs["quantization_config"] = transformers.BitsAndBytesConfig(load_in_8bit=True)
            else:
                kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16)
            kwargs["dtype"] = torch.bfloat16
        else:
            raise UnsupportedPrecisionError(
                f"Precision '{self.precision}' is not available in the Transformers backend. FP8 would need "
                "a separately converted checkpoint, which this baseline does not use.")

        t0 = time.perf_counter()
        self.load_processor()
        model_cls = getattr(transformers, self.MODEL_CLASS)
        self.model = model_cls.from_pretrained(self.snapshot_path, **kwargs)
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - t0

        actual = type(self.model).__name__
        if actual != self.spec.architecture:
            self.unload_model()
            raise AdapterError(f"{self.spec.key}: loaded class {actual} is not the registered architecture "
                               f"{self.spec.architecture}; refusing to continue with a different model")
        self.device_map_used = getattr(self.model, "hf_device_map", None) or {"": str(self.model.device)}
        self.loaded = True
        log.info("model loaded key=%s precision=%s seconds=%.1f", self.spec.key, self.precision, self.load_seconds)

    def unload_model(self) -> None:
        self.model = None
        self.processor = None
        self.loaded = False
        mem.release_torch_memory()

    # ------------------------------------------------------------------ inputs
    def prepare_messages(self, image: Image.Image, prompt_text: str) -> list[dict[str, Any]]:
        """One user turn: the page first, then the complete master prompt. No system prompt."""
        return [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": prompt_text}]}]

    def build_inputs(self, messages: list[dict[str, Any]], gen: GenerationConfig):
        """Apply the model's own chat template and processor. Returns (inputs, info)."""
        if self.processor is None:
            raise AdapterError("Processor is not loaded")
        image = messages[0]["content"][0]["image"]
        template_kwargs = self.chat_template_kwargs(gen)
        image_kwargs = self.image_processor_kwargs()
        call_kwargs: dict[str, Any] = dict(tokenize=True, return_dict=True, return_tensors="pt",
                                           add_generation_prompt=True, **template_kwargs)
        if image_kwargs:
            call_kwargs["images_kwargs"] = image_kwargs
        inputs = self.processor.apply_chat_template(messages, **call_kwargs)

        input_ids = inputs["input_ids"]
        image_token_id = getattr(self.processor, "image_token_id", None)
        n_image_tokens = int((input_ids == image_token_id).sum()) if image_token_id is not None else None
        has_pixels = "pixel_values" in inputs and inputs["pixel_values"].numel() > 0
        if not has_pixels or not n_image_tokens:
            raise ImageNotDeliveredError(
                f"{self.spec.key}: the processed input contains no image (pixel tensor present: {has_pixels}, "
                f"image tokens: {n_image_tokens}). The run is stopped rather than sending text only.")
        rendered = self.processor.decode(input_ids[0], skip_special_tokens=False)
        info = {
            "n_input_tokens": int(input_ids.shape[-1]),
            "n_image_tokens": n_image_tokens,
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "chat_template_arguments": template_kwargs,
            "image_processing": {
                "processor_class": type(self.processor).__name__,
                "image_processor_class": type(self.processor.image_processor).__name__,
                "explicit_arguments": image_kwargs or "none (checkpoint defaults)",
                "input_width": image.width, "input_height": image.height,
                "pixel_values_shape": list(inputs["pixel_values"].shape),
                "image_tokens": n_image_tokens,
                **self.describe_image_processing(inputs, image),
            },
        }
        return inputs, info

    def preflight(self, prepared, prompt, gen: GenerationConfig) -> dict[str, Any]:
        """Processor-only check (no weights): proves the page and prompt reach the model input."""
        if self.processor is None:
            self.load_processor()
        image = self.prepare_image(prepared)
        messages = self.prepare_messages(image, self.prepare_prompt(prompt.text))
        _inputs, info = self.build_inputs(messages, gen)
        prompt_ids = self.processor.tokenizer(prompt.text, add_special_tokens=False)["input_ids"]
        info["prompt_tokens"] = len(prompt_ids)
        info["thinking"] = self.thinking_record(gen)
        info["weights_loaded"] = False
        return info

    # ------------------------------------------------------------------ generation
    def _eos_ids(self) -> set[int]:
        ids: set[int] = set()
        for source in (getattr(self.model, "generation_config", None), self.processor.tokenizer):
            eos = getattr(source, "eos_token_id", None)
            if isinstance(eos, int):
                ids.add(eos)
            elif eos:
                ids.update(int(x) for x in eos)
        return ids

    def generate(self, messages: list[dict[str, Any]], gen: GenerationConfig) -> GenerationResult:
        import torch
        import transformers
        if not self.loaded:
            raise AdapterError("Model is not loaded")
        warnings: list[str] = []
        if gen.enable_thinking and not self.spec.supports_thinking:
            warnings.append("enable_thinking was requested but this checkpoint has no thinking mode; ignored.")

        before = mem.snapshot("before_preprocessing")
        t0 = time.perf_counter()
        inputs, info = self.build_inputs(messages, gen)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        preprocess_seconds = time.perf_counter() - t0
        after = mem.snapshot("after_preprocessing")
        preprocessing_memory = {
            "process_rss_delta_bytes": after.get("process_rss_bytes", 0) - before.get("process_rss_bytes", 0),
            "torch_allocated_delta_bytes": (after.get("torch_allocated_bytes", 0)
                                            - before.get("torch_allocated_bytes", 0)),
            "note": "memory added by image preprocessing and by moving the processed inputs to the device"}

        gen_kwargs = gen.to_transformers_kwargs()
        if "stop_strings" in gen_kwargs:
            gen_kwargs["tokenizer"] = self.processor.tokenizer
        if gen.seed is not None:
            transformers.set_seed(int(gen.seed))
        t1 = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - t1

        new_ids = output[0][info["n_input_tokens"]:]
        decoded = self.decode_output(new_ids, thinking_enabled=bool(gen.enable_thinking and self.spec.supports_thinking))
        n_out = int(new_ids.shape[-1])
        stopped_on_eos = n_out > 0 and int(new_ids[-1]) in self._eos_ids()
        truncated = n_out >= gen.max_new_tokens and not stopped_on_eos
        if truncated:
            warnings.append(f"Output reached max_new_tokens={gen.max_new_tokens} without an end-of-sequence "
                            "token: the response is incomplete. Rerun with a larger limit.")
        recorded_kwargs = {k: v for k, v in gen_kwargs.items() if k != "tokenizer"}
        recorded_kwargs["seed"] = gen.seed
        thinking = self.thinking_record(gen)
        thinking["reasoning_block_found"] = decoded.get("reasoning_text") is not None
        return GenerationResult(
            raw_text=decoded["text"],
            raw_text_with_special_tokens=decoded["text_with_special_tokens"],
            answer_text=decoded.get("answer_text"),
            n_input_tokens=info["n_input_tokens"], n_image_tokens=info["n_image_tokens"], n_output_tokens=n_out,
            stopped_on_eos=stopped_on_eos, truncated=truncated,
            preprocess_seconds=preprocess_seconds, generation_seconds=generation_seconds,
            tokens_per_second=(n_out / generation_seconds if generation_seconds > 0 else None),
            image_processing=info["image_processing"], actual_generation_kwargs=recorded_kwargs,
            thinking={**thinking, "chat_template_arguments": info["chat_template_arguments"]},
            rendered_prompt_sha256=info["rendered_prompt_sha256"],
            preprocessing_memory=preprocessing_memory, warnings=warnings)

    def decode_output(self, output_ids: Any, thinking_enabled: bool = False, **kwargs) -> dict[str, Any]:
        """Decode generated tokens. Nothing is stripped beyond special tokens."""
        tok = self.processor.tokenizer
        out = {"text": tok.decode(output_ids, skip_special_tokens=True),
               "text_with_special_tokens": tok.decode(output_ids, skip_special_tokens=False)}
        if thinking_enabled and self.THINK_END_TOKEN:
            end_id = tok.convert_tokens_to_ids(self.THINK_END_TOKEN)
            positions = (output_ids == end_id).nonzero() if end_id is not None else []
            if len(positions):
                cut = int(positions[0]) + 1
                out["reasoning_text"] = tok.decode(output_ids[:cut], skip_special_tokens=False)
                out["answer_text"] = tok.decode(output_ids[cut:], skip_special_tokens=True)
        return out

    # ------------------------------------------------------------------ metadata
    def collect_runtime_metadata(self) -> dict[str, Any]:
        import torch
        import transformers
        meta: dict[str, Any] = {
            "backend": self.backend,
            "backend_version": f"transformers {transformers.__version__}",
            "torch_version": torch.__version__,
            "model_class": type(self.model).__name__ if self.model is not None else self.MODEL_CLASS,
            "processor_class": type(self.processor).__name__ if self.processor is not None else None,
            "precision": self.precision if self.precision not in ("int8", "int4") else "bf16 compute",
            "quantisation": ({"int8": "bitsandbytes 8-bit", "int4": "bitsandbytes 4-bit NF4 (double quant)"}
                             .get(self.precision, "none")),
            "requested_precision": self.precision,
            "revision": self.snapshot_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if self.snapshot_path else None,
            "load_seconds": self.load_seconds,
        }
        if self.model is not None:
            dm = self.device_map_used or {}
            devices = sorted({str(v) for v in dm.values()})
            meta["device_map"] = {"devices": devices, "modules": len(dm)}
            meta["offloaded_to_cpu_or_disk"] = any(d in ("cpu", "disk") for d in devices) and torch.cuda.is_available()
            meta["model_dtype"] = str(self.model.dtype)
            try:
                meta["checkpoint_generation_defaults"] = self.model.generation_config.to_diff_dict()
            except Exception:
                meta["checkpoint_generation_defaults"] = None
            try:
                meta["memory_footprint_bytes"] = int(self.model.get_memory_footprint())
            except Exception:
                pass
        return meta

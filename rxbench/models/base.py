"""Common interface for every vision-language model adapter.

The experiment protocol talks to adapters only through this interface, so the
user-facing behaviour is identical for every model. Each adapter is free to use
its architecture's own processor, chat template and generation procedure
internally.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from PIL import Image

from ..inference.generation import GenerationConfig
from ..inference.image_loader import PreparedImage
from ..inference.prompt_loader import PromptRecord
from .registry import ModelSpec


class AdapterError(RuntimeError):
    """Base class for adapter failures."""


class ModelNotAvailableError(AdapterError):
    """The checkpoint is not (completely) present on local storage."""


class UnsupportedPrecisionError(AdapterError):
    """The requested precision or quantisation is not supported for this model/backend."""


class InsufficientMemoryError(AdapterError):
    """The model is not expected to fit in the memory currently available."""


class ImageNotDeliveredError(AdapterError):
    """The processed model input contains no image content."""


@dataclass
class GenerationResult:
    raw_text: str                                   # everything the model generated, special tokens removed
    raw_text_with_special_tokens: str | None = None
    answer_text: str | None = None                  # raw_text without a leading reasoning block, if one was split off
    n_input_tokens: int | None = None
    n_image_tokens: int | None = None
    n_output_tokens: int | None = None
    stopped_on_eos: bool | None = None
    truncated: bool = False
    preprocess_seconds: float | None = None
    generation_seconds: float | None = None
    tokens_per_second: float | None = None
    image_processing: dict[str, Any] = field(default_factory=dict)
    actual_generation_kwargs: dict[str, Any] = field(default_factory=dict)
    thinking: dict[str, Any] = field(default_factory=dict)
    rendered_prompt_sha256: str | None = None
    input_pixel_sha256: str | None = None
    prompt_sha256: str | None = None
    preprocessing_memory: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pixel_sha256(image: Image.Image) -> str:
    h = hashlib.sha256()
    h.update(f"{image.width}x{image.height}:{image.mode}:".encode())
    h.update(image.tobytes())
    return h.hexdigest()


class BaseVLMAdapter(ABC):
    backend: str = "abstract"
    is_mock: bool = False

    def __init__(self, spec: ModelSpec, model_cfg: dict[str, Any] | None = None, download_manager=None,
                 exec_cfg: dict[str, Any] | None = None):
        self.spec = spec
        self.model_cfg = model_cfg or {}
        self.download_manager = download_manager
        self.exec_cfg = exec_cfg or {}
        self.precision: str = self.model_cfg.get("precision") or spec.default_precision
        self.loaded = False
        self.load_seconds: float | None = None

    # ---- model lifecycle -------------------------------------------------
    def inspect_model(self) -> dict[str, Any]:
        """Registry metadata plus local availability; does not load weights."""
        info = self.spec.to_dict()
        info["memory_estimate"] = self.spec.memory_estimate_gb(self.precision)
        if self.download_manager is not None:
            info["local"] = self.download_manager.local_status(self.spec)
        return info

    def download_model(self, confirmed: bool = False, **kwargs) -> dict[str, Any]:
        if self.download_manager is None:
            raise AdapterError("No download manager configured")
        return self.download_manager.download(self.spec, confirmed=confirmed, **kwargs)

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def unload_model(self) -> None: ...

    # ---- input preparation ----------------------------------------------
    def prepare_image(self, prepared: PreparedImage) -> Image.Image:
        """Return the page for the model's processor.

        The same decoded RGB page goes to every model. Adapters must not crop,
        enhance, binarise or otherwise edit it; resizing or tiling is left to
        the model's own processor and recorded from there.
        """
        if prepared.image.mode != "RGB":
            raise AdapterError("Prepared image must be RGB")
        return prepared.image

    def prepare_prompt(self, prompt_text: str) -> str:
        """Return the prompt text for the model. It is passed through unchanged."""
        return prompt_text

    @abstractmethod
    def prepare_messages(self, image: Image.Image, prompt_text: str) -> Any: ...

    # ---- inference -------------------------------------------------------
    @abstractmethod
    def generate(self, messages: Any, gen: GenerationConfig) -> GenerationResult: ...

    @abstractmethod
    def decode_output(self, output_ids: Any, **kwargs) -> dict[str, Any]: ...

    @abstractmethod
    def collect_runtime_metadata(self) -> dict[str, Any]: ...

    def run(self, prepared: PreparedImage, prompt: PromptRecord, gen: GenerationConfig) -> GenerationResult:
        """Protocol step: the original page and the exact prompt go in, one response comes out."""
        image = self.prepare_image(prepared)
        text = self.prepare_prompt(prompt.text)
        messages = self.prepare_messages(image, text)
        result = self.generate(messages, gen)
        result.input_pixel_sha256 = pixel_sha256(image)
        result.prompt_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return result

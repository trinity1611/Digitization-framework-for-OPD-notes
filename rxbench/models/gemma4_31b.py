"""Gemma 4 31B IT (google/gemma-4-31B-it).

Official usage (model card): AutoModelForMultimodalLM + AutoProcessor, image
before text, `enable_thinking` passed to the chat template. The vision encoder
keeps the aspect ratio and takes a visual token budget (`max_soft_tokens`) of
70, 140, 280, 560 or 1120.
"""
from __future__ import annotations

from typing import Any

from ..inference.generation import GenerationConfig
from .base import AdapterError
from .transformers_adapter import TransformersVLMAdapter

SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


class Gemma4FamilyAdapter(TransformersVLMAdapter):
    MODEL_CLASS = "AutoModelForMultimodalLM"
    ALLOWED_IMAGE_OPTIONS = ("max_soft_tokens",)
    THINK_END_TOKEN = "<channel|>"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        budget = self.image_options().get("max_soft_tokens")
        if budget is not None and budget not in SUPPORTED_SOFT_TOKENS:
            raise AdapterError(f"{self.spec.key}: max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS}, got {budget}")

    def chat_template_kwargs(self, gen: GenerationConfig) -> dict[str, Any]:
        return {"enable_thinking": bool(gen.enable_thinking)}

    def describe_image_processing(self, inputs: Any, image) -> dict[str, Any]:
        ip = self.processor.image_processor
        default_budget = getattr(ip, "max_soft_tokens", None)
        budget = self.image_options().get("max_soft_tokens", default_budget)
        return {
            "method": "aspect-ratio-preserving resize to fit a visual token budget",
            "visual_token_budget": budget,
            "checkpoint_default_budget": default_budget,
            "patch_size": getattr(ip, "patch_size", None),
            "pooling_kernel_size": getattr(ip, "pooling_kernel_size", None),
            "patches": int(inputs["pixel_values"].shape[1]),
            "aspect_ratio_preserved": True,
            "note": f"The page is encoded as {int(inputs['pixel_values'].shape[1])} patches pooled to at most "
                    f"{budget} image tokens.",
        }


class Gemma4_31BAdapter(Gemma4FamilyAdapter):
    pass

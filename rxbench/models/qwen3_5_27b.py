"""Qwen3.5 27B (Qwen/Qwen3.5-27B): optional natively multimodal model.

It shares the Qwen3-VL processor. Its chat template thinks by default; the
adapter always passes `enable_thinking` explicitly so the setting is never
implicit, and the baseline default is off.
"""
from __future__ import annotations

from typing import Any

from ..inference.generation import GenerationConfig
from .qwen3vl_32b import QwenVLFamilyAdapter


class Qwen3_5_27BAdapter(QwenVLFamilyAdapter):
    MODEL_CLASS = "Qwen3_5ForConditionalGeneration"
    THINK_END_TOKEN = "</think>"

    def chat_template_kwargs(self, gen: GenerationConfig) -> dict[str, Any]:
        return {"enable_thinking": bool(gen.enable_thinking)}

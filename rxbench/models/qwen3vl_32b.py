"""Qwen3-VL 32B Instruct (Qwen/Qwen3-VL-32B-Instruct).

Official usage (model card): Qwen3VLForConditionalGeneration + AutoProcessor,
messages with an image entry followed by text. The vision encoder works at
dynamic resolution: the page is resized to a multiple of 32 pixels per side
within a pixel budget and yields one token per 32x32 pixels.
"""
from __future__ import annotations

from typing import Any

from .transformers_adapter import TransformersVLMAdapter


class QwenVLFamilyAdapter(TransformersVLMAdapter):
    ALLOWED_IMAGE_OPTIONS = ("min_pixels", "max_pixels")

    def image_processor_kwargs(self) -> dict[str, Any]:
        opts = self.image_options()
        if not opts:
            return {}
        # The processor takes the pixel budget as `size`; a bare max_pixels argument is ignored by it.
        default = dict(getattr(self.processor.image_processor, "size", None) or {})
        return {"size": {"shortest_edge": int(opts.get("min_pixels", default.get("shortest_edge", 65536))),
                         "longest_edge": int(opts.get("max_pixels", default.get("longest_edge", 16777216)))}}

    def describe_image_processing(self, inputs: Any, image) -> dict[str, Any]:
        ip = self.processor.image_processor
        size = dict(getattr(ip, "size", None) or {})
        patch = getattr(ip, "patch_size", 16)
        merge = getattr(ip, "merge_size", 2)
        out: dict[str, Any] = {
            "method": "dynamic-resolution resize (aspect ratio approximately preserved)",
            "checkpoint_min_pixels": size.get("shortest_edge"),
            "checkpoint_max_pixels": size.get("longest_edge"),
            "patch_size": patch, "merge_size": merge,
            "aspect_ratio_preserved": True,
        }
        if "image_grid_thw" in inputs:
            _t, h, w = (int(x) for x in inputs["image_grid_thw"][0])
            out["grid_patches_height_width"] = [h, w]
            out["resized_width"] = w * patch
            out["resized_height"] = h * patch
            out["note"] = (f"The {image.width}x{image.height} page is resized to {w * patch}x{h * patch} "
                           f"and encoded as {h * w // (merge * merge)} image tokens.")
        return out


class Qwen3VL32BAdapter(QwenVLFamilyAdapter):
    MODEL_CLASS = "Qwen3VLForConditionalGeneration"

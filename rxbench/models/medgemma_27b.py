"""MedGemma 27B Multimodal IT (google/medgemma-27b-it), Gemma 3 architecture.

Official usage (model card): AutoModelForImageTextToText + AutoProcessor,
bfloat16, chat template with an image entry followed by text.
"""
from __future__ import annotations

from typing import Any

from .transformers_adapter import TransformersVLMAdapter


class Gemma3FamilyAdapter(TransformersVLMAdapter):
    """Gemma 3 vision path: SigLIP encoder, fixed 896x896 view(s), 256 tokens per view."""
    MODEL_CLASS = "AutoModelForImageTextToText"
    ALLOWED_IMAGE_OPTIONS = ("do_pan_and_scan", "pan_and_scan_min_crop_size", "pan_and_scan_max_num_crops",
                             "pan_and_scan_min_ratio_to_activate")

    def image_processor_kwargs(self) -> dict[str, Any]:
        opts = self.image_options()
        if not opts.get("do_pan_and_scan"):
            return {}                      # checkpoint default: a single resized view
        return opts

    def describe_image_processing(self, inputs: Any, image) -> dict[str, Any]:
        ip = self.processor.image_processor
        size = getattr(ip, "size", None)
        size = dict(size) if size is not None else None
        views = int(inputs["pixel_values"].shape[0])
        pan = bool(self.image_options().get("do_pan_and_scan"))
        return {
            "method": "resize to a fixed square view" + (" plus pan-and-scan crops" if pan else ""),
            "native_view_size": size,
            "views": views,
            "tokens_per_view": getattr(self.processor, "image_seq_length", None),
            "pan_and_scan": pan,
            "aspect_ratio_preserved": False,
            "note": (f"The {image.width}x{image.height} page is resized to "
                     f"{size.get('width') if size else '?'}x{size.get('height') if size else '?'} without "
                     "keeping its aspect ratio" + (f"; pan-and-scan added {views - 1} crop(s)." if views > 1 else ".")),
        }


class MedGemma27BAdapter(Gemma3FamilyAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "text" in self.spec.repo_id.rsplit("/", 1)[-1].split("-"):
            raise ValueError(f"{self.spec.repo_id} is a text-only MedGemma variant; the multimodal "
                             "google/medgemma-27b-it checkpoint is required")

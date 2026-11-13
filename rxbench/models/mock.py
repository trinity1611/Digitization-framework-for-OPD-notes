"""Mock adapter: a software test double, NOT a model.

It exists so the storage, engine, parsing, comparison and interface code can be
tested without model weights. Every result it produces is marked `is_mock` in
metadata.json and labelled as mock in the interface and reports. Its text is
canned; it does not read the image and says nothing about any real model.
"""
from __future__ import annotations

import time
from typing import Any

from PIL import Image

from ..inference.generation import GenerationConfig
from .base import AdapterError, BaseVLMAdapter, GenerationResult, pixel_sha256
from .registry import ModelSpec

MOCK_BANNER = "MOCK OUTPUT - software test double, not produced by any model."

_TEMPLATE = """{banner}

## A. Document overview

- Document type: Mock prescription
- Language: English
- Pre-existing printed or pre-filled content present: Yes
- Handwritten content present: Yes
- Number of identifiable handwritten regions: 2
- Overall legibility: Moderate
- Layout observations: mock variant {variant}; input pixels {pixel_hash}
- Any uncertain handwritten/printed distinction: None

## B. Verbatim transcription

T. Mockamol {strength}  1-0-1 x 5 days
Syp Testex 5 ml [UNCERTAIN: "TDS" / "BD"]
[ILLEGIBLE]
Review after 7 days

## C. Structured clinical extraction

### Chief complaints
- Original text: None

### Medications
- Original medicine text: T. Mockamol {strength}
- Possible medicine name, if uncertain:
- Strength: {strength}
- Formulation: T.
- Route:
- Source region: R1
- Evidence level: EXPLICIT

- Original medicine text: Syp Testex 5 ml
- Possible medicine name, if uncertain: Testex
- Strength: 5 ml
- Formulation: Syp
- Route:
- Source region: R1
- Evidence level: CONTEXTUALLY SUPPORTED

### Medication instructions
- Original instruction: 1-0-1 x 5 days
- Dose: 1
- Frequency: 1-0-1
- Timing:
- Food relationship:
- Duration: x 5 days
- Associated medication: T. Mockamol {strength}
- Association evidence: same line
- Source region: R1
- Evidence level: EXPLICIT

### Follow-up / referral
- Original text: Review after 7 days
- Interpreted instruction: Review in one week
- Source region: R2
- Evidence level: EXPLICIT

## D. Uncertainty register

- Visible handwritten fragment: TDS / BD
- Possible readings: "TDS", "BD"
- Reason for uncertainty: mock
- Source location: Middle-left
- Additional evidence that could help: none

## E. Final validation

1. All identifiable handwritten regions were considered. (mock)
"""


def mock_spec(key: str, variant: str = "alpha", **options: Any) -> ModelSpec:
    return ModelSpec(
        key=key, display_name=f"MOCK {variant} (not a model)", repo_id=f"mock/{key}",
        repo_url="about:blank", architecture="MockArchitecture", adapter="rxbench.models.mock:MockAdapter",
        processor_class="MockProcessor", backend="mock", image_input=True, parameters_total=0,
        licence="n/a", access_requirements="none", tier="mock", revision="mock-revision",
        precision_options=["bf16"], default_precision="bf16", is_mock=True,
        notes="variant=" + variant + ";" + ";".join(f"{k}={v}" for k, v in options.items()))


class MockAdapter(BaseVLMAdapter):
    backend = "mock"
    is_mock = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = dict(kv.split("=", 1) for kv in (self.spec.notes or "").split(";") if "=" in kv)
        self.options.update({k: str(v) for k, v in (self.model_cfg.get("mock") or {}).items()})

    def load_model(self) -> None:
        if self.options.get("fail_on_load") == "True":
            raise AdapterError("mock load failure (requested by test)")
        self.loaded = True
        self.load_seconds = 0.0

    def unload_model(self) -> None:
        self.loaded = False

    def prepare_messages(self, image: Image.Image, prompt_text: str) -> dict[str, Any]:
        return {"image": image, "prompt": prompt_text}

    def decode_output(self, output_ids: Any, **kwargs) -> dict[str, Any]:
        return {"text": str(output_ids), "text_with_special_tokens": str(output_ids) + "<mock_eos>"}

    def generate(self, messages: dict[str, Any], gen: GenerationConfig) -> GenerationResult:
        if not self.loaded:
            raise AdapterError("mock model is not loaded")
        if self.options.get("fail_on_generate") == "True":
            raise AdapterError("mock generation failure (requested by test)")
        t0 = time.perf_counter()
        image = messages["image"]
        variant = self.options.get("variant", "alpha")
        text = self.options.get("output_text") or _TEMPLATE.format(
            banner=MOCK_BANNER, variant=variant, pixel_hash=pixel_sha256(image)[:16],
            strength="650" if variant == "alpha" else "500 mg")
        truncated = False
        n_tokens = len(text.split())
        if n_tokens > gen.max_new_tokens:
            text = " ".join(text.split(" ")[: gen.max_new_tokens])
            truncated, n_tokens = True, gen.max_new_tokens
        decoded = self.decode_output(text)
        return GenerationResult(
            raw_text=decoded["text"], raw_text_with_special_tokens=decoded["text_with_special_tokens"],
            n_input_tokens=len(messages["prompt"].split()) + 1, n_image_tokens=1, n_output_tokens=n_tokens,
            stopped_on_eos=not truncated, truncated=truncated, preprocess_seconds=0.0,
            generation_seconds=time.perf_counter() - t0, tokens_per_second=None,
            image_processing={"method": "mock: image not processed", "input_width": image.width,
                              "input_height": image.height, "image_tokens": 1},
            actual_generation_kwargs=gen.to_transformers_kwargs() | {"seed": gen.seed},
            thinking={"supported": False, "enabled": False},
            warnings=["MOCK RUN: not real model inference."]
            + (["Output reached max_new_tokens: the response is incomplete."] if truncated else []))

    def collect_runtime_metadata(self) -> dict[str, Any]:
        return {"backend": "mock", "backend_version": "mock", "model_class": "MockArchitecture",
                "processor_class": "MockProcessor", "precision": "n/a", "quantisation": "n/a",
                "revision": "mock-revision", "load_seconds": self.load_seconds,
                "device_map": None, "checkpoint_generation_defaults": None}

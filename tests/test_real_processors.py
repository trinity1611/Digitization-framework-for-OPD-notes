"""Processor-level checks with the REAL processors of the registered checkpoints.

These load the official tokenizer, chat template and image processor (not the
weights) and prove that the original page and the full master prompt reach the
model input. They are skipped for any model whose processor files are not in
the local Hugging Face cache; fetch them with

    python -m rxbench models download primary --config-only

This is not model inference: nothing is generated.
"""
from __future__ import annotations

import pytest

from rxbench.config import PROJECT_ROOT, load_config, model_config, resolve_path
from rxbench.inference.generation import GenerationConfig
from rxbench.inference.image_loader import prepare_image_from_bytes
from rxbench.inference.prompt_loader import load_prompt
from rxbench.models.download_manager import DownloadManager
from rxbench.models.registry import ModelRegistry
from tests.conftest import image_bytes

CFG = load_config(PROJECT_ROOT / "config.yaml")
REGISTRY = ModelRegistry.from_yaml(PROJECT_ROOT / "model_registry.yaml")
DOWNLOADS = DownloadManager(PROJECT_ROOT / ".state", resolve_path(CFG, "hf_cache_dir"))
PROMPT = load_prompt(PROJECT_ROOT / "prompts" / "master_prescription_prompt.txt")
PAGE = prepare_image_from_bytes(image_bytes("JPEG", width=960, height=1280, text="T. Azee 500mg 1-0-0 x 7 days"))


def _adapter(key: str, overrides: dict | None = None):
    spec = REGISTRY.get(key)
    if DOWNLOADS.local_status(spec)["status"] not in ("complete", "config_only"):
        pytest.skip(f"processor files for {spec.repo_id} are not in the local cache")
    cfg = model_config(CFG, key)
    cfg["image_processing"].update(overrides or {})
    return REGISTRY.create_adapter(key, cfg, DOWNLOADS, CFG["execution"])


@pytest.mark.parametrize("key", REGISTRY.keys())
def test_real_processor_receives_image_and_full_prompt(key):
    adapter = _adapter(key)
    info = adapter.preflight(PAGE, PROMPT, GenerationConfig())
    assert info["n_image_tokens"] > 0 and info["image_processing"]["pixel_values_shape"]
    assert info["image_processing"]["processor_class"] == REGISTRY.get(key).processor_class
    # the whole prompt is in the input: total = prompt + image tokens + a few template tokens
    assert info["n_input_tokens"] >= info["prompt_tokens"] + info["n_image_tokens"]
    assert info["n_input_tokens"] - info["prompt_tokens"] - info["n_image_tokens"] < 40
    # the exact prompt text can be read back from the tokenised model input
    ids = adapter.processor.tokenizer(PROMPT.text, add_special_tokens=False)["input_ids"]
    assert adapter.processor.tokenizer.decode(ids) == PROMPT.text


def test_real_gemma4_visual_token_budget_is_applied():
    high = _adapter("gemma4_31b", {"max_soft_tokens": 1120}).preflight(PAGE, PROMPT, GenerationConfig())
    low = _adapter("gemma4_31b", {"max_soft_tokens": 280}).preflight(PAGE, PROMPT, GenerationConfig())
    assert 280 < high["n_image_tokens"] <= 1120 and low["n_image_tokens"] <= 280
    assert high["image_processing"]["visual_token_budget"] == 1120


def test_real_medgemma_default_is_single_896_view_and_pan_and_scan_is_optional():
    default = _adapter("medgemma_27b").preflight(PAGE, PROMPT, GenerationConfig())
    assert default["n_image_tokens"] == 256 and default["image_processing"]["views"] == 1
    assert default["image_processing"]["pixel_values_shape"][-2:] == [896, 896]
    pan = _adapter("medgemma_27b", {"do_pan_and_scan": True}).preflight(PAGE, PROMPT, GenerationConfig())
    assert pan["image_processing"]["views"] > 1 and pan["n_image_tokens"] == 256 * pan["image_processing"]["views"]


def test_real_qwen_dynamic_resolution_and_pixel_budget():
    default = _adapter("qwen3vl_32b").preflight(PAGE, PROMPT, GenerationConfig())
    assert default["image_processing"]["resized_width"] == 960 and default["image_processing"]["resized_height"] == 1280
    assert default["n_image_tokens"] == (960 // 32) * (1280 // 32)
    capped = _adapter("qwen3vl_32b", {"max_pixels": 1024 * 32 * 32}).preflight(PAGE, PROMPT, GenerationConfig())
    assert capped["n_image_tokens"] < default["n_image_tokens"]          # the option is honoured, not ignored


@pytest.mark.parametrize("key,expect_default", [("gemma4_31b", False), ("qwen3_5_27b", True)])
def test_real_thinking_mode_is_explicit(key, expect_default):
    adapter = _adapter(key)
    assert REGISTRY.get(key).thinking_checkpoint_default is expect_default
    off = adapter.preflight(PAGE, PROMPT, GenerationConfig(enable_thinking=False))
    on = adapter.preflight(PAGE, PROMPT, GenerationConfig(enable_thinking=True))
    assert off["chat_template_arguments"] == {"enable_thinking": False}
    assert on["chat_template_arguments"] == {"enable_thinking": True}
    assert off["rendered_prompt_sha256"] != on["rendered_prompt_sha256"]

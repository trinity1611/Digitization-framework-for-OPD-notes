"""Configuration loading and path resolution."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "paths": {
        "experiments_dir": "experiments",
        "prompt_file": "prompts/master_prescription_prompt.txt",
        "annotations_dir": "annotations",
        "state_dir": ".state",
        "log_dir": "logs",
        "model_registry": "model_registry.yaml",
        "hf_cache_dir": None,
    },
    "execution": {
        "mode": "sequential",
        "allow_simultaneous": False,
        "device_map": "auto",
        "unload_tolerance_gb": 4.0,
        "memory_sample_interval_s": 0.25,
    },
    "generation": {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": None,
        "top_k": None,
        "max_new_tokens": 8192,
        "repetition_penalty": 1.0,
        "seed": 0,
        "stop_sequences": [],
        "enable_thinking": False,
    },
    "image": {"apply_exif_orientation": True, "pdf_render_dpi": 200},
    "models": {},
    "privacy": {"force_offline_inference": True},
}

# Environment overrides for paths (used by the Slurm job scripts).
_ENV_PATH_OVERRIDES = {
    "experiments_dir": "RXBENCH_EXPERIMENTS_DIR",
    "state_dir": "RXBENCH_STATE_DIR",
    "log_dir": "RXBENCH_LOG_DIR",
    "annotations_dir": "RXBENCH_ANNOTATIONS_DIR",
    "hf_cache_dir": "RXBENCH_HF_CACHE_DIR",
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load config.yaml merged over the built-in defaults."""
    cfg_path = Path(path or os.environ.get("RXBENCH_CONFIG") or PROJECT_ROOT / "config.yaml")
    user: dict = {}
    if cfg_path.exists():
        user = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg = _deep_merge(DEFAULTS, user)
    cfg["_config_file"] = str(cfg_path)
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path | None:
    """Resolve a `paths.<key>` entry; relative paths are relative to the project root."""
    env_name = _ENV_PATH_OVERRIDES.get(key)
    value = os.environ.get(env_name) if env_name else None
    if not value:
        value = cfg.get("paths", {}).get(key)
    if value in (None, ""):
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def model_config(cfg: dict[str, Any], model_key: str) -> dict[str, Any]:
    """Per-model settings (precision, native image-processor options)."""
    m = copy.deepcopy(cfg.get("models", {}).get(model_key) or {})
    m.setdefault("image_processing", {})
    return m

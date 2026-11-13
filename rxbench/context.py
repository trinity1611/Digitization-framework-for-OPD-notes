"""Shared application context used by the command line and the Streamlit interface."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, load_config, resolve_path
from .logging_utils import add_file_log
from .models.download_manager import DownloadManager
from .models.mock import mock_spec
from .models.registry import ModelRegistry
from .storage.experiment_manager import ExperimentManager


@dataclass
class AppContext:
    cfg: dict[str, Any]
    registry: ModelRegistry
    downloads: DownloadManager
    manager: ExperimentManager
    prompt_path: Path
    annotations_dir: Path

    def resolve_models(self, selection: str | list[str] | None) -> list[str]:
        """'primary', 'optional', 'all', or comma-separated keys -> list of registered keys."""
        if selection is None:
            selection = "primary"
        parts = selection if isinstance(selection, list) else [s.strip() for s in selection.split(",") if s.strip()]
        keys: list[str] = []
        for p in parts:
            if p in ("primary", "optional"):
                keys += self.registry.keys(tier=p)
            elif p == "all":
                keys += self.registry.keys(include_mock=False)
            else:
                self.registry.get(p)          # raises on unknown keys
                keys.append(p)
        return list(dict.fromkeys(keys))


def mock_enabled() -> bool:
    return os.environ.get("RXBENCH_ENABLE_MOCK") == "1"


def build_context(config_path: str | None = None, enable_mock: bool | None = None) -> AppContext:
    cfg = load_config(config_path)
    registry = ModelRegistry.from_yaml(resolve_path(cfg, "model_registry"))
    if enable_mock if enable_mock is not None else mock_enabled():
        registry.register("mock_alpha", mock_spec("mock_alpha", "alpha"))
        registry.register("mock_beta", mock_spec("mock_beta", "beta"))
    add_file_log(resolve_path(cfg, "log_dir"))
    return AppContext(
        cfg=cfg, registry=registry,
        downloads=DownloadManager(resolve_path(cfg, "state_dir"), resolve_path(cfg, "hf_cache_dir")),
        manager=ExperimentManager(resolve_path(cfg, "experiments_dir")),
        prompt_path=resolve_path(cfg, "prompt_file"),
        annotations_dir=resolve_path(cfg, "annotations_dir") or PROJECT_ROOT / "annotations",
    )

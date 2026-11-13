"""Model registry: metadata for every model the benchmark can run.

Models are declared in model_registry.yaml. A new model needs an entry there
and, only if its architecture needs different handling, a new adapter class.
"""
from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

PRECISIONS = ("bf16", "fp16", "fp32", "fp8", "int8", "int4")
BYTES_PER_PARAM = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}
REQUIRED_FIELDS = ("display_name", "repo_id", "repo_url", "architecture", "adapter", "processor_class",
                   "backend", "image_input", "parameters_total", "precision_options", "licence",
                   "access_requirements")


@dataclass
class ModelSpec:
    key: str
    display_name: str
    repo_id: str
    repo_url: str
    architecture: str
    adapter: str
    processor_class: str
    backend: str
    image_input: bool
    parameters_total: int
    licence: str
    access_requirements: str
    tier: str = "optional"
    docs_url: str | None = None
    revision: str | None = None
    model_type: str | None = None
    parameters_active: int | None = None
    download_size_gb: float | None = None
    precision_options: list[str] = field(default_factory=lambda: ["bf16"])
    quantisation_options: list[str] = field(default_factory=list)
    default_precision: str = "bf16"
    licence_url: str | None = None
    gated: bool = False
    min_transformers: str | None = None
    supports_thinking: bool = False
    thinking_checkpoint_default: bool | None = None
    forbidden_repo_ids: list[str] = field(default_factory=list)
    notes: str | None = None
    is_mock: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_moe(self) -> bool:
        return self.parameters_active is not None

    def supported_precisions(self) -> list[str]:
        return list(self.precision_options) + list(self.quantisation_options)

    def weights_gb(self, precision: str) -> float:
        """Weight storage at a precision, computed from the TOTAL parameter count.

        For a mixture-of-experts model every expert is stored and loaded, so the
        active-parameter count must never be used for sizing.
        """
        factor = BYTES_PER_PARAM[precision]
        if precision in ("int8", "int4"):
            factor *= 1.15   # embeddings, norms and the vision tower stay in higher precision
        return self.parameters_total * factor / 1024**3

    def memory_estimate_gb(self, precision: str | None = None) -> dict[str, Any]:
        precision = precision or self.default_precision
        weights = self.weights_gb(precision)
        # activations, vision features and KV cache for an ~8k-token answer; grows with model size
        runtime = round(min(8.0, 2.0 + 0.12 * self.weights_gb("bf16")), 1)
        return {"precision": precision, "weights_gb": round(weights, 1), "runtime_overhead_gb": runtime,
                "total_gb": round(weights + runtime, 1),
                "basis": "estimate: total parameters x bytes per parameter + fixed runtime allowance; "
                         "measured values are in each experiment's metadata.json"}


def load_adapter_class(path: str):
    module_name, _, class_name = path.partition(":")
    if not class_name:
        raise ValueError(f"Adapter must be 'module:ClassName', got {path!r}")
    return getattr(importlib.import_module(module_name), class_name)


class ModelRegistry:
    def __init__(self, specs: dict[str, ModelSpec] | None = None, source: str | None = None):
        self._specs: dict[str, ModelSpec] = dict(specs or {})
        self.source = source

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ModelRegistry":
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        reg = cls(source=str(path))
        for key, entry in (data.get("models") or {}).items():
            reg.register(key, entry)
        return reg

    def register(self, key: str, entry: dict[str, Any] | ModelSpec) -> ModelSpec:
        if isinstance(entry, ModelSpec):
            spec = entry
        else:
            missing = [f for f in REQUIRED_FIELDS if f not in entry]
            if missing:
                raise ValueError(f"Model '{key}' is missing registry fields: {missing}")
            spec = ModelSpec(key=key, **entry)
        if key in self._specs:
            raise ValueError(f"Model key already registered: {key}")
        if not spec.image_input:
            raise ValueError(f"Model '{key}' has no image input; this benchmark needs multimodal models")
        bad = [p for p in spec.supported_precisions() if p not in PRECISIONS]
        if bad:
            raise ValueError(f"Model '{key}' lists unknown precisions: {bad}")
        if spec.default_precision not in spec.supported_precisions():
            raise ValueError(f"Model '{key}': default precision not among its supported precisions")
        if spec.repo_id in spec.forbidden_repo_ids:
            raise ValueError(f"Model '{key}': repo_id {spec.repo_id} is listed as a forbidden look-alike")
        self._specs[key] = spec
        return spec

    def get(self, key: str) -> ModelSpec:
        if key not in self._specs:
            raise KeyError(f"Unknown model '{key}'. Registered: {', '.join(self._specs) or 'none'}")
        return self._specs[key]

    def keys(self, tier: str | None = None, include_mock: bool = True) -> list[str]:
        return [k for k, s in self._specs.items()
                if (tier is None or s.tier == tier) and (include_mock or not s.is_mock)]

    def all(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def __contains__(self, key: str) -> bool:
        return key in self._specs

    def create_adapter(self, key: str, model_cfg: dict[str, Any], download_manager, exec_cfg: dict[str, Any]):
        spec = self.get(key)
        adapter_cls = load_adapter_class(spec.adapter)
        return adapter_cls(spec=spec, model_cfg=model_cfg, download_manager=download_manager, exec_cfg=exec_cfg)

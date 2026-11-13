"""Generation settings shared by every model."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GenerationConfig:
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    max_new_tokens: int = 8192
    repetition_penalty: float = 1.0
    seed: int | None = 0
    stop_sequences: list[str] = field(default_factory=list)
    enable_thinking: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "GenerationConfig":
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown generation settings: {sorted(unknown)}")
        cfg = cls(**d)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.do_sample and self.temperature <= 0:
            raise ValueError("Sampling needs temperature > 0; use do_sample=false for greedy decoding")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")

    @property
    def is_greedy(self) -> bool:
        return not self.do_sample

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_transformers_kwargs(self) -> dict[str, Any]:
        """Arguments for `model.generate`.

        A checkpoint's generation_config.json usually enables sampling
        (for example temperature 0.7 or 1.0). For greedy decoding the sampling
        parameters are set to None explicitly so those defaults cannot leak in.
        Temperature 0 is expressed as do_sample=False, which is how Transformers
        implements deterministic decoding.
        """
        kw: dict[str, Any] = {"max_new_tokens": int(self.max_new_tokens),
                              "repetition_penalty": float(self.repetition_penalty)}
        if self.do_sample:
            kw.update(do_sample=True, temperature=float(self.temperature))
            if self.top_p is not None:
                kw["top_p"] = float(self.top_p)
            if self.top_k is not None:
                kw["top_k"] = int(self.top_k)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if self.stop_sequences:
            kw["stop_strings"] = list(self.stop_sequences)
        return kw

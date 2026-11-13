"""Environment record attached to every model run."""
from __future__ import annotations

from typing import Any

from ..hardware.detect import detect_all, detect_packages, summary_for_metadata

_cached: dict[str, Any] | None = None


def environment_record(hardware_report: dict[str, Any] | None = None, refresh: bool = False) -> dict[str, Any]:
    """Compact hardware + software description. Detected once per process."""
    global _cached
    if _cached is not None and not refresh and hardware_report is None:
        return _cached
    report = hardware_report or detect_all()
    record = summary_for_metadata(report)
    pk = report.get("packages") or detect_packages()
    record["packages"] = {k: pk.get(k) for k in ("torch", "transformers", "accelerate", "bitsandbytes",
                                                 "huggingface_hub", "safetensors", "tokenizers", "pillow",
                                                 "pypdfium2", "numpy")}
    _cached = record
    return record

"""Export experiments and metadata (local files only)."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

from .experiment_manager import Experiment, ExperimentManager


def experiment_zip_bytes(exp: Experiment) -> bytes:
    """Zip one experiment directory (inputs, prompt, raw outputs, metadata, reports)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(exp.path.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(Path(exp.exp_id) / path.relative_to(exp.path)))
    return buf.getvalue()


def collect_metadata(manager: ExperimentManager, exp_ids: list[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp_id in exp_ids or manager.list_ids():
        exp = manager.get(exp_id)
        for key in manager.model_keys(exp):
            run = manager.load_model_run(exp, key)
            if run["metadata"]:
                rows.append(run["metadata"])
    return rows


def flatten_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """One flat row per model run for the reports and CSV export."""
    mem = meta.get("memory") or {}
    inference = mem.get("inference") or {}
    load = mem.get("load") or {}
    gb = lambda v: None if v is None else round(v / 1024**3, 2)
    return {
        "experiment_id": meta.get("experiment_id"),
        "batch_id": meta.get("batch_id"),
        "model_key": (meta.get("model") or {}).get("key"),
        "model_identifier": (meta.get("model") or {}).get("identifier"),
        "model_revision": (meta.get("model") or {}).get("revision"),
        "is_mock": meta.get("is_mock"),
        "completion_status": meta.get("completion_status"),
        "backend": (meta.get("backend") or {}).get("name"),
        "backend_version": (meta.get("backend") or {}).get("version"),
        "precision": meta.get("precision"),
        "quantisation": meta.get("quantisation"),
        "image_sha256": (meta.get("input") or {}).get("image_sha256"),
        "prompt_sha256": (meta.get("prompt") or {}).get("sha256"),
        "image_tokens": (meta.get("tokens") or {}).get("image"),
        "input_tokens": (meta.get("tokens") or {}).get("input"),
        "output_tokens": (meta.get("tokens") or {}).get("output"),
        "truncated": (meta.get("tokens") or {}).get("truncated"),
        "model_load_seconds": (meta.get("timing") or {}).get("model_load_seconds"),
        "generation_seconds": (meta.get("timing") or {}).get("generation_seconds"),
        "tokens_per_second": (meta.get("timing") or {}).get("tokens_per_second"),
        "peak_memory_gb_load": gb(load.get("primary_peak_bytes")),
        "peak_memory_gb_inference": gb(inference.get("primary_peak_bytes")),
        "peak_memory_basis": inference.get("primary_peak_basis") or load.get("primary_peak_basis"),
        "available_before_inference_gb": gb((inference.get("baseline") or {}).get("system_available_bytes")),
        "memory_released_after_unload": (mem.get("after_unload_check") or {}).get("verified"),
        "structured_parse": ((meta.get("output") or {}).get("parse_status") or {}).get("method"),
        "errors": "; ".join(meta.get("errors") or []),
        "started_at": meta.get("started_at"),
    }


def metadata_csv(rows: list[dict[str, Any]]) -> str:
    flat = [flatten_metadata(r) for r in rows]
    if not flat:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(flat[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(flat)
    return buf.getvalue()


def metadata_json(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, indent=2, ensure_ascii=False, default=str)


def runtime_report_markdown(rows: list[dict[str, Any]], title: str = "Memory and inference-time report") -> str:
    """Per-model summary of memory use and timing, from stored metadata only."""
    flat = [flatten_metadata(r) for r in rows]
    lines = [f"# {title}", ""]
    if not flat:
        return "\n".join(lines + ["No model runs found.", ""])
    if any(r["is_mock"] for r in flat):
        lines += ["> **This report contains MOCK runs.** Mock rows come from a software test double and say "
                  "nothing about any model's memory use, speed or accuracy.", ""]
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in flat:
        by_model.setdefault(r["model_key"], []).append(r)

    def stats(values: list[float | None]) -> str:
        v = [x for x in values if x is not None]
        return "n/a" if not v else f"{sum(v) / len(v):.2f} (min {min(v):.2f}, max {max(v):.2f})"

    lines += ["## Per model", "",
              "| model | mock | runs | complete | truncated | failed | precision | load s | generation s: mean (min, max) "
              "| output tokens: mean (min, max) | tokens/s: mean (min, max) | peak memory GB, load "
              "| peak memory GB, inference: mean (min, max) | released after unload |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, rs in by_model.items():
        ok = sum(r["completion_status"] == "complete" for r in rs)
        trunc = sum(r["completion_status"] == "incomplete_truncated" for r in rs)
        failed = len(rs) - ok - trunc
        loads = sorted({r["model_load_seconds"] for r in rs if r["model_load_seconds"] is not None})
        load_peaks = sorted({r["peak_memory_gb_load"] for r in rs if r["peak_memory_gb_load"] is not None})
        released = {r["memory_released_after_unload"] for r in rs} - {None}
        lines.append("| " + " | ".join([
            key, "yes" if rs[0]["is_mock"] else "no", str(len(rs)), str(ok), str(trunc), str(failed),
            ", ".join(sorted({str(r["quantisation"] or r["precision"]) for r in rs})),
            ", ".join(f"{x:.1f}" for x in loads) or "n/a",
            stats([r["generation_seconds"] for r in rs]),
            stats([r["output_tokens"] for r in rs]),
            stats([r["tokens_per_second"] for r in rs]),
            ", ".join(f"{x:.2f}" for x in load_peaks) or "n/a",
            stats([r["peak_memory_gb_inference"] for r in rs]),
            ", ".join(str(x) for x in sorted(released, key=str)) or "n/a",
        ]) + " |")
    basis = sorted({r["peak_memory_basis"] for r in flat if r["peak_memory_basis"]})
    lines += ["", "Peak memory basis: " + ("; ".join(basis) if basis else "not recorded") + ".", ""]
    lines += ["## Per run", "",
              "| experiment | model | status | image tokens | input tokens | output tokens | generation s | "
              "peak memory GB (inference) | errors |", "|---|---|---|---|---|---|---|---|---|"]
    for r in flat:
        lines.append("| " + " | ".join(str(x) if x is not None else "" for x in [
            r["experiment_id"], r["model_key"], r["completion_status"], r["image_tokens"], r["input_tokens"],
            r["output_tokens"], None if r["generation_seconds"] is None else f"{r['generation_seconds']:.2f}",
            r["peak_memory_gb_inference"], (r["errors"] or "")[:120]]) + " |")
    lines.append("")
    return "\n".join(lines)

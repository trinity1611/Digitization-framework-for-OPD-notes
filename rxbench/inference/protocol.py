"""The zero-shot protocol as one call, shared by the command line and the interface.

1. Store the original document and the exact master prompt (one experiment per page).
2. Run each selected model, sequentially and independently, on the stored page and prompt.
3. Write a comparison of the outputs (differences only; nothing is merged or chosen).
"""
from __future__ import annotations

from typing import Any, Callable

from ..context import AppContext
from ..evaluation.comparison import write_comparison
from ..storage.experiment_manager import Experiment
from . import image_loader
from .engine import InferenceEngine
from .generation import GenerationConfig
from .prompt_loader import PromptRecord, load_prompt


def create_experiments(ctx: AppContext, data: bytes, filename: str, prompt: PromptRecord,
                       pages: str | list[int] | None = None, batch_id: str | None = None) -> list[Experiment]:
    """One experiment per selected page (images have exactly one page)."""
    image_cfg = ctx.cfg["image"]
    kind = image_loader.detect_kind(filename, data)
    if kind == "pdf":
        n = image_loader.pdf_page_count(data)
        indices = pages if isinstance(pages, list) else image_loader.parse_page_selection(pages, n)
    else:
        indices = [None]
    return [ctx.manager.create_experiment(
        source_bytes=data, original_filename=filename, prompt=prompt, page_index=idx,
        apply_exif_orientation=image_cfg["apply_exif_orientation"], pdf_render_dpi=image_cfg["pdf_render_dpi"],
        batch_id=batch_id) for idx in indices]


def run_models(ctx: AppContext, exp_ids: list[str], model_keys: list[str], gen: GenerationConfig,
               batch_id: str | None = None, progress: Callable[[str, dict[str, Any]], None] | None = None,
               skip_completed: bool = False, hardware_report: dict[str, Any] | None = None) -> dict[str, Any]:
    engine = InferenceEngine(ctx.cfg, ctx.registry, ctx.manager, ctx.downloads, hardware_report)
    batch = engine.run(exp_ids, model_keys, gen, batch_id=batch_id, progress=progress,
                       skip_completed=skip_completed)
    for exp_id in exp_ids:
        write_comparison(ctx.manager, ctx.manager.get(exp_id))
    return batch


def run_protocol(ctx: AppContext, files: list[tuple[str, bytes]], model_keys: list[str], gen: GenerationConfig,
                 pages: str | None = None, progress: Callable[[str, dict[str, Any]], None] | None = None,
                 hardware_report: dict[str, Any] | None = None) -> dict[str, Any]:
    prompt = load_prompt(ctx.prompt_path)
    if not prompt.has_printed_text_exclusion:
        raise ValueError("The master prompt does not contain the printed / pre-filled text exclusion rule; "
                         "refusing to start an experiment with it.")
    batch_id = ctx.manager.new_batch_id()
    exp_ids: list[str] = []
    for filename, data in files:
        exp_ids += [e.exp_id for e in create_experiments(ctx, data, filename, prompt, pages, batch_id)]
    return run_models(ctx, exp_ids, model_keys, gen, batch_id=batch_id, progress=progress,
                      hardware_report=hardware_report)

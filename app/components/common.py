"""Helpers shared by the interface pages. The pages contain presentation only; all
experiment logic lives in the `rxbench` package."""
from __future__ import annotations

import io
from typing import Any

import streamlit as st

from rxbench.context import AppContext, build_context
from rxbench.hardware.detect import detect_all
from rxbench.inference.output_parser import CATEGORY_DISPLAY_ORDER, CATEGORY_LABELS
from rxbench.storage.experiment_manager import Experiment

STATUS_TEXT = {
    "complete": "Complete",
    "incomplete_truncated": "INCOMPLETE: output truncated at the token limit",
    "failed_model_load": "FAILED: model could not be loaded",
    "failed_inference": "FAILED: inference error",
    "model_not_available_locally": "NOT RUN: model is not downloaded",
}
INDEPENDENCE_NOTICE = ("Every selected model processes the same original image and the same master prompt "
                       "independently. Models never see each other's output, and outputs are never merged, "
                       "corrected or ranked automatically.")


@st.cache_resource
def get_ctx() -> AppContext:
    return build_context()


def hardware_report(refresh: bool = False) -> dict[str, Any]:
    if refresh or "hardware_report" not in st.session_state:
        ctx = get_ctx()
        from rxbench.config import PROJECT_ROOT
        st.session_state["hardware_report"] = detect_all({
            "project": PROJECT_ROOT, "experiments": ctx.manager.root,
            "model_cache": ctx.downloads.effective_cache_dir()})
    return st.session_state["hardware_report"]


def header(title: str, caption: str | None = None) -> None:
    st.title(title)
    if caption:
        st.caption(caption)


def gb(n: float | int | None, digits: int = 1) -> str:
    return "n/a" if n is None else f"{n / 1024**3:.{digits}f} GB"


def seconds(v: float | None) -> str:
    return "n/a" if v is None else (f"{v:.1f} s" if v < 120 else f"{v / 60:.1f} min")


def ordered_model_keys(ctx: AppContext, keys: list[str]) -> list[str]:
    """Primary models first (registry order), then optional models, then anything else."""
    registry_order = ctx.registry.keys(tier="primary") + ctx.registry.keys(tier="optional")
    return [k for k in registry_order if k in keys] + sorted(k for k in keys if k not in registry_order)


def model_name(ctx: AppContext, key: str) -> str:
    return ctx.registry.get(key).display_name if key in ctx.registry else key


def select_experiment(ctx: AppContext, key: str) -> Experiment | None:
    ids = ctx.manager.list_ids()
    if not ids:
        st.info("No experiments yet. Start one on the **New experiment** page.")
        return None
    default = st.session_state.get("selected_experiment")
    index = ids.index(default) if default in ids else len(ids) - 1

    def label(exp_id: str) -> str:
        rec = ctx.manager.get(exp_id).record
        page = f", page {rec['input']['pdf']['page_number']}" if rec["input"].get("pdf") else ""
        return f"{exp_id}  ·  {rec['input']['original_filename']}{page}  ·  {rec['created_at'][:16].replace('T', ' ')}"

    chosen = st.selectbox("Experiment", ids, index=index, format_func=label, key=key)
    st.session_state["selected_experiment"] = chosen
    return ctx.manager.get(chosen)


def show_document(exp: Experiment, width: int | None = None) -> None:
    from PIL import Image
    image = Image.open(io.BytesIO(exp.model_input_path.read_bytes()))
    inp = exp.record["input"]
    caption = (f"{inp['original_filename']}  ·  {inp['image']['original_width']}×{inp['image']['original_height']} px"
               f"  ·  SHA-256 {inp['original_sha256'][:16]}…")
    if width:
        st.image(image, caption=caption, width=width)
    else:
        st.image(image, caption=caption)


def mock_banner(is_mock: bool) -> None:
    if is_mock:
        st.error("MOCK RUN. This output was produced by a software test double, not by a model. "
                 "It is not evidence of any model's behaviour.")


def status_line(meta: dict[str, Any]) -> None:
    status = meta.get("completion_status")
    text = STATUS_TEXT.get(status, str(status))
    if status == "complete":
        st.success(text)
    elif status == "incomplete_truncated":
        st.warning(text + ". Rerun with a larger output-token limit to obtain the full response.")
    else:
        st.error(text)
    for err in meta.get("errors") or []:
        st.error(err)
    for warning in meta.get("warnings") or []:
        st.warning(warning)


def category_order() -> list[tuple[str, str]]:
    return [(k, CATEGORY_LABELS[k]) for k in CATEGORY_DISPLAY_ORDER]

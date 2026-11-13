"""Screen 4: results, one tab per model. Outputs are never merged."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components import common  # noqa: E402
from rxbench.inference.generation import GenerationConfig  # noqa: E402
from rxbench.inference.protocol import run_models  # noqa: E402


def _memory_rows(meta: dict) -> list[dict]:
    mem = meta.get("memory") or {}
    load, inference = mem.get("load") or {}, mem.get("inference") or {}
    check = mem.get("after_unload_check") or {}
    base = inference.get("baseline") or {}
    after = mem.get("after_unload") or {}
    pre = mem.get("image_preprocessing") or {}
    return [
        {"measure": "Peak during model loading", "value": common.gb(load.get("primary_peak_bytes"), 2)},
        {"measure": "Peak during inference", "value": common.gb(inference.get("primary_peak_bytes"), 2)},
        {"measure": "Basis of the peak figures", "value": inference.get("primary_peak_basis") or load.get("primary_peak_basis")},
        {"measure": "System memory available before inference", "value": common.gb(base.get("system_available_bytes"), 2)},
        {"measure": "Lowest system memory available during inference", "value": common.gb(inference.get("min_system_available_bytes"), 2)},
        {"measure": "GPU memory in use during inference (all processes, peak)", "value": common.gb(inference.get("peak_device_used_bytes_all_processes"), 2)},
        {"measure": "Image preprocessing overhead (process / device)",
         "value": f"{common.gb(pre.get('process_rss_delta_bytes'), 3)} / {common.gb(pre.get('torch_allocated_delta_bytes'), 3)}"},
        {"measure": "System memory available after unloading", "value": common.gb(after.get("system_available_bytes"), 2)},
        {"measure": "Memory released after unloading", "value": {True: "yes", False: "NO", None: "not checked"}[check.get("verified")]},
        {"measure": "Memory model", "value": "unified (GPU shares system memory)" if mem.get("unified_memory") else "discrete GPU memory"},
    ]


def _model_tab(ctx, exp, key: str) -> None:
    run = ctx.manager.load_model_run(exp, key)
    meta = run["metadata"] or {}
    common.mock_banner(meta.get("is_mock", False))
    common.status_line(meta)
    timing, tokens = meta.get("timing") or {}, meta.get("tokens") or {}
    a, b, c, d = st.columns(4)
    a.metric("Generation time", common.seconds(timing.get("generation_seconds")))
    b.metric("Model load time", common.seconds(timing.get("model_load_seconds")))
    c.metric("Output tokens", tokens.get("output") if tokens.get("output") is not None else "n/a")
    d.metric("Peak memory (inference)", common.gb(((meta.get("memory") or {}).get("inference") or {}).get("primary_peak_bytes")))

    structured = run["structured"] or {}
    parse = structured.get("parse_status") or {}
    raw_tab, struct_tab, unc_tab, mem_tab, cfg_tab = st.tabs(
        ["Raw output", "Structured clinical extraction", "Uncertainty register", "Time and memory", "Model configuration"])

    with raw_tab:
        if run["raw_text"] is None:
            st.info("No output was produced.")
        else:
            st.caption(f"Authoritative record, stored exactly as generated  ·  {len(run['raw_text']):,} characters  ·  "
                       f"SHA-256 `{(meta.get('output') or {}).get('raw_output_sha256')}`")
            st.download_button("Download raw_output.txt", run["raw_text"].encode("utf-8"),
                               file_name=f"{exp.exp_id}_{key}_raw_output.txt", key=f"dl_raw_{key}")
            transcription = structured.get("verbatim_transcription")
            if isinstance(transcription, str):
                st.markdown("**Verbatim transcription (section B, as located in the raw output)**")
                st.code(transcription, language=None)
            st.markdown("**Complete raw output**")
            st.code(run["raw_text"], language=None)

    with struct_tab:
        if not parse.get("structured_parse_successful"):
            st.warning("Structured parsing was unsuccessful: the output is neither valid JSON nor in the section "
                       "layout the prompt asks for. The raw output is unchanged and shown in the first tab. "
                       "No model is used to repair it.")
        else:
            st.caption(f"Located by fixed rules ({parse.get('method')}); nothing is corrected or added. "
                       + (f"Sections not found: {', '.join(parse['missing_sections'])}. " if parse.get("missing_sections") else ""))
            if structured.get("json") is not None:
                st.json(structured["json"])
            overview = structured.get("document_overview")
            if overview:
                st.markdown("**Document overview**")
                st.dataframe([{"field": k, "value": v} for k, v in overview.items()], hide_index=True)
            extraction = structured.get("structured_extraction") or {}
            for cat, label in common.category_order():
                items = extraction.get(cat)
                st.markdown(f"**{label}**")
                if items is None:
                    st.caption("Section not found in the output.")
                elif not items:
                    st.caption("No items.")
                else:
                    st.dataframe(items, hide_index=True)

    with unc_tab:
        register = structured.get("uncertainty_register")
        markers = (structured.get("uncertainty_markers") or {}).get("in_transcription")
        if markers is not None:
            st.markdown("**Uncertainty markers in the transcription**")
            st.dataframe([{"marker": k.replace("_", " "), "count": v} for k, v in markers.items()], hide_index=True)
        if not register:
            st.info("No uncertainty register was located in the output.")
        else:
            if register.get("items"):
                st.dataframe(register["items"], hide_index=True)
            st.markdown("**Register as written by the model**")
            st.code(register.get("text") or "", language=None)

    with mem_tab:
        st.dataframe(_memory_rows(meta), hide_index=True)
        st.dataframe([
            {"measure": "Image preprocessing", "value": common.seconds(timing.get("image_preprocessing_seconds"))},
            {"measure": "Generation", "value": common.seconds(timing.get("generation_seconds"))},
            {"measure": "Tokens per second", "value": None if timing.get("tokens_per_second") is None else f"{timing['tokens_per_second']:.1f}"},
            {"measure": "Input tokens (of which image)", "value": f"{tokens.get('input')} ({tokens.get('image')})"},
            {"measure": "Output tokens / limit", "value": f"{tokens.get('output')} / {tokens.get('max_new_tokens')}"},
            {"measure": "Stopped at end of sequence", "value": str(tokens.get("stopped_on_end_of_sequence"))},
        ], hide_index=True)

    with cfg_tab:
        model = meta.get("model") or {}
        st.dataframe([
            {"setting": "Model identifier", "value": model.get("identifier")},
            {"setting": "Revision", "value": model.get("revision")},
            {"setting": "Backend", "value": f"{(meta.get('backend') or {}).get('name')} ({(meta.get('backend') or {}).get('version')})"},
            {"setting": "Precision", "value": str(meta.get("precision"))},
            {"setting": "Quantisation", "value": str(meta.get("quantisation"))},
            {"setting": "Image SHA-256", "value": (meta.get("input") or {}).get("image_sha256")},
            {"setting": "Prompt SHA-256", "value": (meta.get("prompt") or {}).get("sha256")},
        ], hide_index=True)
        st.markdown("**Image processing by the model's own processor**")
        st.json(meta.get("image_processing") or {})
        st.markdown("**Generation settings (requested and actually used)**")
        st.json(meta.get("generation") or {})
        with st.expander("Complete metadata.json"):
            st.json(meta)
        history = ctx.manager.history(exp, key)
        if history:
            st.caption(f"Earlier runs kept in history/: {', '.join(history)}")

    with st.expander("Run this model again on this experiment"):
        current = int(((meta.get("generation") or {}).get("requested") or {}).get("max_new_tokens") or ctx.cfg["generation"]["max_new_tokens"])
        new_limit = st.number_input("Maximum new tokens", 256, 131072, max(current * 2 if meta.get("completion_status") == "incomplete_truncated" else current, 256),
                                    step=1024, key=f"limit_{key}")
        st.caption("The current result is moved to history/ and kept. All other settings stay as they were.")
        if st.button("Rerun", key=f"rerun_{key}"):
            requested = dict((meta.get("generation") or {}).get("requested") or ctx.cfg["generation"])
            requested["max_new_tokens"] = int(new_limit)
            with st.spinner("Running…"):
                run_models(ctx, [exp.exp_id], [key], GenerationConfig.from_dict(requested),
                           hardware_report=common.hardware_report())
            st.rerun()


def render() -> None:
    ctx = common.get_ctx()
    common.header("Results", "One tab per model. Outputs are shown separately and are never merged.")
    exp = common.select_experiment(ctx, "results_exp")
    if exp is None:
        return
    keys = common.ordered_model_keys(ctx, ctx.manager.model_keys(exp))
    left, right = st.columns([1, 2])
    with left:
        common.show_document(exp)
        inp = exp.record["input"]
        st.caption(f"Prompt SHA-256 `{exp.record['prompt']['sha256'][:16]}…`")
        for t in inp["transformations"]:
            st.caption(f"Before the processors: {t['detail']}")
    with right:
        if not keys:
            st.info("No model has been run on this experiment yet.")
            return
        for tab, key in zip(st.tabs([common.model_name(ctx, k) for k in keys]), keys):
            with tab:
                _model_tab(ctx, exp, key)


render()

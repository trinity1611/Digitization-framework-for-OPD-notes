"""Screen 6: experiment history."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components import common  # noqa: E402
from rxbench.storage import export  # noqa: E402


def render() -> None:
    ctx = common.get_ctx()
    common.header("Experiment history", f"Stored locally in `{ctx.manager.root}`.")
    experiments = ctx.manager.list_experiments()
    if not experiments:
        st.info("No experiments yet.")
        return
    rows = []
    for exp in experiments:
        inp = exp.record["input"]
        statuses = []
        for key in common.ordered_model_keys(ctx, ctx.manager.model_keys(exp)):
            meta = ctx.manager.load_model_run(exp, key)["metadata"] or {}
            statuses.append(f"{key}: {meta.get('completion_status')}{' (MOCK)' if meta.get('is_mock') else ''}")
        rows.append({"experiment": exp.exp_id, "created": exp.record["created_at"][:19].replace("T", " "),
                     "document": inp["original_filename"],
                     "page": inp["pdf"]["page_number"] if inp.get("pdf") else "",
                     "image SHA-256": inp["original_sha256"][:16] + "…",
                     "prompt SHA-256": exp.record["prompt"]["sha256"][:12] + "…",
                     "batch": exp.record.get("batch_id"), "models": "; ".join(statuses)})
    st.dataframe(rows, hide_index=True)

    all_rows = export.collect_metadata(ctx.manager)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Export all metadata (CSV)", export.metadata_csv(all_rows) or "no runs", file_name="rxbench_metadata.csv")
    c2.download_button("Export all metadata (JSON)", export.metadata_json(all_rows), file_name="rxbench_metadata.json")
    c3.download_button("Memory and inference-time report", export.runtime_report_markdown(all_rows),
                       file_name="runtime_report.md")

    st.divider()
    exp = common.select_experiment(ctx, "history_exp")
    if exp is None:
        return
    left, right = st.columns([1, 2])
    with left:
        common.show_document(exp)
        st.download_button("Download original document", exp.original_path.read_bytes(),
                           file_name=f"{exp.exp_id}_{exp.original_path.name}")
    with right:
        st.download_button("Download this experiment (zip: input, prompt, raw outputs, metadata, reports)",
                           export.experiment_zip_bytes(exp), file_name=f"{exp.exp_id}.zip", mime="application/zip")
        st.download_button("Export this experiment's metadata (JSON)",
                           export.metadata_json(export.collect_metadata(ctx.manager, [exp.exp_id])),
                           file_name=f"{exp.exp_id}_metadata.json")
        with st.expander("Experiment record"):
            st.json(exp.record)
        for key in common.ordered_model_keys(ctx, ctx.manager.model_keys(exp)):
            run = ctx.manager.load_model_run(exp, key)
            meta = run["metadata"] or {}
            with st.expander(f"Raw output: {common.model_name(ctx, key)} ({meta.get('completion_status')})"):
                common.mock_banner(meta.get("is_mock", False))
                st.code(run["raw_text"] if run["raw_text"] is not None else "no output", language=None)
                earlier = ctx.manager.history(exp, key)
                if earlier:
                    st.caption("Earlier runs kept on disk: " + ", ".join(f"{key}/history/{h}" for h in earlier))

        st.divider()
        st.markdown("**Delete this experiment**")
        st.caption("Removes the stored document, prompt snapshot, every model output and report for this "
                   "experiment from local storage. This cannot be undone.")
        confirm = st.checkbox(f"I want to permanently delete {exp.exp_id}", key=f"confirm_delete_{exp.exp_id}")
        if st.button("Delete experiment", disabled=not confirm):
            ctx.manager.delete(exp.exp_id)
            st.session_state.pop("selected_experiment", None)
            st.rerun()


render()

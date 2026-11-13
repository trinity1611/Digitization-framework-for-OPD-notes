"""Evaluation against human-verified ground truth."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components import common  # noqa: E402
from rxbench.evaluation.evaluate import AnnotationError, evaluate_experiment, render_markdown  # noqa: E402


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def render() -> None:
    ctx = common.get_ctx()
    common.header("Evaluation", "Accuracy is reported only against human-verified ground truth for the same image.")
    exp = common.select_experiment(ctx, "evaluation_exp")
    if exp is None:
        return
    path = ctx.annotations_dir / f"{exp.record['input']['original_sha256']}.json"
    if not path.exists():
        st.info("There is no annotation for this image yet. Create one on the **Annotation** page. Without verified "
                "ground truth no accuracy figure is shown.")
        return
    annotation = json.loads(path.read_text(encoding="utf-8"))
    provisional = False
    if not annotation.get("verified"):
        st.warning("This annotation is not marked as human-verified.")
        provisional = st.checkbox("Show provisional figures anyway (they are not accuracy results)")
        if not provisional:
            return
    try:
        report = evaluate_experiment(ctx.manager, exp, annotation, allow_unverified=provisional)
    except AnnotationError as exc:
        st.error(str(exc))
        return
    if report["ground_truth"]["provisional"]:
        st.error("PROVISIONAL: computed against an unverified annotation.")

    rows = []
    for key in common.ordered_model_keys(ctx, list(report["models"])):
        entry = report["models"][key]
        name = common.model_name(ctx, key) + (" (MOCK)" if entry.get("is_mock") else "")
        if not entry.get("evaluated"):
            rows.append({"model": name, "note": entry.get("reason")})
            continue
        m = entry["metrics"]
        whole = m["transcription"]["whole_handwritten_text"]
        rows.append({
            "model": name + (" (truncated output)" if entry.get("output_truncated") else ""),
            "CER verbatim": _pct(whole["cer_verbatim"]["value"]),
            "WER verbatim": _pct(whole["wer_verbatim"]["value"]),
            "CER normalised": _pct(whole["cer_relaxed_markers_stripped"]["value"]),
            "numeric value errors": _pct(m["transcription"]["numerical_values"]["sequence_error_rate"]),
            "medicine name, exact": _pct(m["medicines"]["exact_medicine_name_match"]),
            "medicine name, normalised": _pct(m["medicines"]["normalised_medicine_name_match"]),
            "category F1, exact": _pct(m["classification"]["micro"]["exact"]["f1"]),
            "printed text transcribed": _pct(m["printed_text_exclusion"]["2_prefilled_text_extracted"]["transcription_rate"]),
            "in-field handwriting kept": _pct(m["printed_text_exclusion"]["3_handwriting_in_printed_fields_retained"]["rate"]),
            "illegible regions flagged": _pct(m["uncertainty"]["illegible_regions"]["correctly_identified_rate"]),
            "wrong and confident": _pct(m["uncertainty"]["readable_regions"]["incorrectly_confident_rate"]),
        })
    st.dataframe(rows, hide_index=True)
    st.caption("Verbatim and normalised figures are separate measurements. Printed-text and uncertainty measures "
               "are reported apart from CER and WER. Definitions: docs/EVALUATION_GUIDE.md.")

    for key in common.ordered_model_keys(ctx, list(report["models"])):
        entry = report["models"][key]
        if not entry.get("evaluated"):
            continue
        with st.expander(f"Details: {common.model_name(ctx, key)}"):
            m = entry["metrics"]
            tabs = st.tabs(["Transcription", "Categories", "Medicines", "Printed-text exclusion", "Uncertainty"])
            with tabs[0]:
                st.json(m["transcription"])
            with tabs[1]:
                st.dataframe([{"category": cat, "reference items": v["reference_items"], "predicted items": v["predicted_items"],
                               "precision (exact)": _pct(v["exact"]["precision"]), "recall (exact)": _pct(v["exact"]["recall"]),
                               "F1 (exact)": _pct(v["exact"]["f1"]), "F1 (normalised)": _pct(v["normalised"]["f1"]),
                               "F1 (fuzzy)": _pct(v["fuzzy"]["f1"]), "exact match": v["exact_match"],
                               "field-level accuracy": _pct(v["field_level_accuracy"])}
                              for cat, v in m["classification"]["per_category"].items()], hide_index=True)
            with tabs[2]:
                st.json({k: v for k, v in m["medicines"].items() if k != "details"})
                st.dataframe([{"reference name": d["reference_name"], "model text": d["predicted_text"],
                               "exact name": d["exact_name_match"], "normalised name": d["normalised_name_match"]}
                              for d in m["medicines"]["details"]], hide_index=True)
            with tabs[3]:
                st.json(m["printed_text_exclusion"])
            with tabs[4]:
                st.json({k: v for k, v in m["uncertainty"].items() if k != "details"})
                st.dataframe(m["uncertainty"]["details"], hide_index=True)
    st.download_button("Download evaluation report (Markdown)", render_markdown(report),
                       file_name=f"{exp.exp_id}_evaluation_report.md")
    st.download_button("Download evaluation report (JSON)", json.dumps(report, indent=2, default=str),
                       file_name=f"{exp.exp_id}_evaluation_report.json")


render()

"""Evaluate one experiment's model outputs against a human-verified annotation."""
from __future__ import annotations

from typing import Any

from ..inference.output_parser import CATEGORY_KEYS, item_text
from ..storage.experiment_manager import Experiment, ExperimentManager, utc_now, write_json, write_text_exact
from . import annotation as ann
from .cer import cer, item_matched_cer
from .field_metrics import category_metrics
from .medicine_metrics import medicine_metrics, predicted_medicines
from .printed_text_metrics import printed_text_metrics
from .uncertainty_metrics import uncertainty_metrics
from .wer import numeric_values, wer


class AnnotationError(ValueError):
    pass


def _predicted_categories(structured: dict[str, Any] | None) -> dict[str, list[str]]:
    extraction = (structured or {}).get("structured_extraction") or {}
    return {k: [item_text(it) for it in extraction.get(k, []) or [] if item_text(it).strip()] for k in CATEGORY_KEYS}


def evaluate_model_output(annotation: dict[str, Any], structured: dict[str, Any] | None) -> dict[str, Any]:
    """All metric groups for one model output. `structured` is the parsed output (may be None)."""
    transcription = (structured or {}).get("verbatim_transcription")
    reference = ann.reference_transcription(annotation)
    result: dict[str, Any] = {"transcription_located": isinstance(transcription, str)}
    hyp = transcription if isinstance(transcription, str) else ""

    # A. Transcription metrics (verbatim first; normalised variants labelled as such)
    result["transcription"] = {
        "whole_handwritten_text": {
            "cer_verbatim": cer(reference, hyp),
            "wer_verbatim": wer(reference, hyp),
            "cer_whitespace_normalised": cer(reference, hyp, "whitespace"),
            "wer_whitespace_normalised": wer(reference, hyp, "whitespace"),
            "cer_relaxed_markers_stripped": cer(reference, hyp, "relaxed", strip_uncertainty_markers=True),
            "wer_relaxed_markers_stripped": wer(reference, hyp, "relaxed", strip_uncertainty_markers=True),
        },
        "numerical_values": numeric_values(reference, hyp),
    }
    ref_meds = annotation.get("medications") or []
    pred_meds = predicted_medicines(structured or {})
    ref_names = [m.get("text_verbatim") or m.get("name_verbatim") or "" for m in ref_meds]
    ref_instr = [m["instruction_verbatim"] for m in ref_meds if (m.get("instruction_verbatim") or "").strip()]
    pred_cats = _predicted_categories(structured)
    ref_cats = ann.reference_categories(annotation)
    med_keys = {"medications", "medication_instructions"}
    other_ref = [t for k in CATEGORY_KEYS if k not in med_keys for t in ref_cats.get(k, [])]
    other_pred = [t for k in CATEGORY_KEYS if k not in med_keys for t in pred_cats.get(k, [])]
    result["transcription"]["medicine_names"] = {
        "cer_verbatim": item_matched_cer(ref_names, [m["text"] for m in pred_meds]),
        "cer_relaxed": item_matched_cer(ref_names, [m["text"] for m in pred_meds], "relaxed")}
    result["transcription"]["dosage_instructions"] = {
        "cer_verbatim": item_matched_cer(ref_instr, pred_cats["medication_instructions"]),
        "cer_relaxed": item_matched_cer(ref_instr, pred_cats["medication_instructions"], "relaxed")}
    result["transcription"]["other_clinical_text"] = {
        "cer_verbatim": item_matched_cer(other_ref, other_pred),
        "cer_relaxed": item_matched_cer(other_ref, other_pred, "relaxed")}

    # B-E
    result["classification"] = category_metrics(ref_cats, pred_cats)
    result["medicines"] = medicine_metrics(ref_meds, structured or {})
    result["printed_text_exclusion"] = printed_text_metrics(annotation, transcription, structured)
    result["uncertainty"] = uncertainty_metrics(annotation, transcription, structured)
    return result


def evaluate_experiment(manager: ExperimentManager, exp: Experiment, annotation: dict[str, Any],
                        allow_unverified: bool = False) -> dict[str, Any]:
    problems = ann.validate(annotation)
    if problems:
        raise AnnotationError("Annotation is not valid:\n- " + "\n- ".join(problems))
    if annotation["image_sha256"] != exp.record["input"]["original_sha256"]:
        raise AnnotationError(f"Annotation is for a different image (sha256 {annotation['image_sha256'][:12]}...) "
                              f"than {exp.exp_id} ({exp.record['input']['original_sha256'][:12]}...)")
    if not annotation.get("verified") and not allow_unverified:
        raise AnnotationError("Annotation is not marked as human-verified (verified: false). Accuracy is only "
                              "reported against verified ground truth. Use allow_unverified for a provisional run.")
    report: dict[str, Any] = {
        "experiment_id": exp.exp_id, "evaluated_at": utc_now(),
        "ground_truth": {"verified": bool(annotation.get("verified")), "verified_by": annotation.get("verified_by"),
                         "annotator": annotation.get("annotator"), "image_sha256": annotation["image_sha256"],
                         "provisional": not annotation.get("verified")},
        "models": {},
    }
    for key in manager.model_keys(exp):
        run = manager.load_model_run(exp, key)
        meta = run["metadata"] or {}
        entry: dict[str, Any] = {"completion_status": meta.get("completion_status"), "is_mock": meta.get("is_mock", False)}
        if run["raw_text"] is None:
            entry["evaluated"] = False
            entry["reason"] = "no output was produced"
        else:
            entry["evaluated"] = True
            entry["structured_parse"] = ((meta.get("output") or {}).get("parse_status") or {}).get("method")
            entry["output_truncated"] = meta.get("completion_status") == "incomplete_truncated"
            entry["metrics"] = evaluate_model_output(annotation, run["structured"])
        report["models"][key] = entry
    out = exp.path / "evaluation"
    write_json(out / "evaluation_report.json", report)
    write_text_exact(out / "evaluation_report.md", render_markdown(report))
    return report


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    gt = report["ground_truth"]
    lines = [f"# Evaluation: {report['experiment_id']}", ""]
    if gt["provisional"]:
        lines += ["> **PROVISIONAL: the annotation is not marked as human-verified. These figures are not "
                  "accuracy results.**", ""]
    else:
        lines += [f"Ground truth verified by {gt['verified_by']}.", ""]
    lines += ["Error rates are relative to the annotated handwritten text. Verbatim and normalised figures are "
              "separate measurements and must not be mixed.", "",
              "| model | CER verbatim | WER verbatim | CER relaxed | numeric error | medicine name exact | "
              "medicine name normalised | category micro-F1 exact | printed text transcribed | "
              "in-field handwriting kept | illegible flagged | wrong and confident |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, entry in report["models"].items():
        name = key + (" (MOCK)" if entry.get("is_mock") else "")
        if not entry.get("evaluated"):
            lines.append(f"| {name} | not evaluated: {entry.get('reason')} |" + " |" * 10)
            continue
        m = entry["metrics"]
        t = m["transcription"]["whole_handwritten_text"]
        lines.append("| " + " | ".join([
            name + (" (truncated)" if entry.get("output_truncated") else ""),
            _pct(t["cer_verbatim"]["value"]), _pct(t["wer_verbatim"]["value"]),
            _pct(t["cer_relaxed_markers_stripped"]["value"]),
            _pct(m["transcription"]["numerical_values"]["sequence_error_rate"]),
            _pct(m["medicines"]["exact_medicine_name_match"]), _pct(m["medicines"]["normalised_medicine_name_match"]),
            _pct(m["classification"]["micro"]["exact"]["f1"]),
            _pct(m["printed_text_exclusion"]["2_prefilled_text_extracted"]["transcription_rate"]),
            _pct(m["printed_text_exclusion"]["3_handwriting_in_printed_fields_retained"]["rate"]),
            _pct(m["uncertainty"]["illegible_regions"]["correctly_identified_rate"]),
            _pct(m["uncertainty"]["readable_regions"]["incorrectly_confident_rate"]),
        ]) + " |")
    lines += ["", "Full per-category, per-medicine, printed-text and uncertainty details are in "
              "`evaluation_report.json`.", ""]
    return "\n".join(lines)

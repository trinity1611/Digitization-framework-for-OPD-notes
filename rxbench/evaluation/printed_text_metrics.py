"""Printed / pre-filled text exclusion metrics (reported separately from CER and WER).

They answer five questions against a human annotation:

1. Were printed headings or labels transcribed as if handwritten?
2. Was pre-filled / printed text extracted into the clinical categories?
3. Was handwriting written inside printed fields kept?
4. Did the model confuse printed and handwritten content?
5. Was printed text used as layout context without being reproduced?
"""
from __future__ import annotations

from typing import Any

from ..inference.output_parser import CATEGORY_KEYS, item_text
from .textdist import contains_tokens, normalise, partial_similarity, strip_markers

HEADING_KINDS = {"heading", "label", "table_header"}


def _norm(text: str) -> str:
    return normalise(strip_markers(text), "relaxed").replace(":", " ").strip()


def _present(needle: str, haystack_norm: str) -> bool:
    n = " ".join(_norm(needle).split())
    return len(n) >= 2 and contains_tokens(" ".join(haystack_norm.split()), n)


def printed_text_metrics(annotation: dict[str, Any], transcription: str | None,
                         structured: dict[str, Any] | None, retention_threshold: float = 0.8) -> dict[str, Any]:
    transcription_norm = _norm(transcription or "")
    extraction = (structured or {}).get("structured_extraction") or {}
    extracted_by_cat = {k: [_norm(item_text(it)) for it in extraction.get(k, []) or []] for k in CATEGORY_KEYS}
    extracted_all = " \n ".join(t for items in extracted_by_cat.values() for t in items)

    # Printed strings that the doctor also wrote by hand cannot be scored as leaks.
    printed = [p for p in annotation.get("printed_text") or [] if not p.get("also_handwritten")]
    headings = [p for p in printed if p.get("kind") in HEADING_KINDS]
    heading_leaks = [p["text"] for p in headings if _present(p["text"], transcription_norm)]
    transcribed_leaks = [p["text"] for p in printed if _present(p["text"], transcription_norm)]
    extracted_leaks = [p["text"] for p in printed if _present(p["text"], extracted_all)]

    in_field = [r for r in annotation.get("regions") or []
                if r.get("origin") == "handwritten" and r.get("in_printed_field")
                and r.get("legibility") != "illegible" and (r.get("text") or "").strip()]
    retained, dropped, layout_ok, layout_scored = [], [], 0, 0
    for r in in_field:
        text_norm = _norm(r["text"])
        kept = partial_similarity(text_norm, transcription_norm) >= retention_threshold
        (retained if kept else dropped).append(r["region_id"])
        label = r.get("printed_label")
        cats = r.get("category")
        cats = [cats] if isinstance(cats, str) else (cats or [])
        if label and cats:
            layout_scored += 1
            in_expected = any(partial_similarity(text_norm, t) >= retention_threshold
                              for c in cats for t in extracted_by_cat.get(c, []))
            label_reproduced = _present(label, transcription_norm)
            layout_ok += bool(kept and in_expected and not label_reproduced)

    overview = (structured or {}).get("document_overview") or {}
    declared = next((v for k, v in overview.items() if "uncertain handwritten/printed" in k.lower()), None)
    uncertain_regions = [r["region_id"] for r in annotation.get("regions") or [] if r.get("origin") == "uncertain"]
    rate = lambda a, b: (a / b) if b else None
    return {
        "reported_separately_from": "CER and WER",
        "1_printed_headings_transcribed": {
            "printed_headings_annotated": len(headings), "transcribed": len(heading_leaks),
            "rate": rate(len(heading_leaks), len(headings)), "items": heading_leaks},
        "2_prefilled_text_extracted": {
            "printed_items_annotated": len(printed),
            "found_in_transcription": len(transcribed_leaks), "found_in_structured_extraction": len(extracted_leaks),
            "transcription_rate": rate(len(transcribed_leaks), len(printed)),
            "extraction_rate": rate(len(extracted_leaks), len(printed)),
            "items_in_transcription": transcribed_leaks, "items_in_extraction": extracted_leaks},
        "3_handwriting_in_printed_fields_retained": {
            "regions_annotated": len(in_field), "retained": len(retained), "dropped": len(dropped),
            "rate": rate(len(retained), len(in_field)), "dropped_regions": dropped,
            "threshold": retention_threshold},
        "4_printed_handwritten_confusion": {
            "printed_treated_as_handwritten": len(set(transcribed_leaks) | set(extracted_leaks)),
            "handwritten_treated_as_printed_or_omitted": len(dropped),
            "regions_annotated_uncertain_origin": len(uncertain_regions),
            "model_statement_on_uncertain_origin": declared},
        "5_layout_context_used_without_reproduction": {
            "regions_scored": layout_scored, "satisfied": layout_ok, "rate": rate(layout_ok, layout_scored),
            "definition": "handwriting inside a labelled printed field is kept, appears in the category that "
                          "the label implies, and the label itself is not reproduced in the transcription"},
        "method": "String containment after case-folding and punctuation removal; printed strings that the "
                  "annotator marked also_handwritten are excluded.",
    }

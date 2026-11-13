"""Ground-truth annotation. A person transcribes from the image; model output is deliberately not shown here."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components import common  # noqa: E402
from rxbench.evaluation import annotation as ann  # noqa: E402
from rxbench.inference.output_parser import CATEGORY_KEYS  # noqa: E402

REGION_COLUMNS = ["region_id", "reading_order", "location", "origin", "text", "legibility", "acceptable_readings",
                  "crossed_out", "category", "in_printed_field", "printed_label"]


def _clean(value):
    """Data-editor cells come back as NaN/None when empty."""
    return None if value is None or (isinstance(value, float) and value != value) else value


def _regions_from_rows(rows: list[dict]) -> list[dict]:
    regions = []
    for n, row in enumerate(rows, start=1):
        row = {k: _clean(v) for k, v in row.items()}
        if not any(row.get(k) for k in ("text", "region_id")):
            continue
        readings = row.get("acceptable_readings") or ""
        regions.append({
            "region_id": row.get("region_id") or f"R{n}", "reading_order": int(row.get("reading_order") or n),
            "location": row.get("location") or "", "bbox": None, "origin": row.get("origin") or "handwritten",
            "text": row.get("text") or "", "legibility": row.get("legibility") or "legible",
            "acceptable_readings": [r.strip() for r in str(readings).split("|") if r.strip()],
            "crossed_out": bool(row.get("crossed_out")), "category": row.get("category") or "other_unclassified",
            "in_printed_field": bool(row.get("in_printed_field")), "printed_label": row.get("printed_label") or None})
    return regions


def render() -> None:
    ctx = common.get_ctx()
    common.header("Ground-truth annotation",
                  "Transcribe from the image only. Model outputs are not shown on this page, and model agreement is "
                  "never a substitute for human verification.")
    exp = common.select_experiment(ctx, "annotation_exp")
    if exp is None:
        return
    sha = exp.record["input"]["original_sha256"]
    path = ctx.annotations_dir / f"{sha}.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else ann.empty_annotation(
        sha, exp.record["input"]["original_filename"])
    st.caption(f"Annotation file: `{path}` ({'exists' if path.exists() else 'not saved yet'}). "
               "The format is documented in docs/EVALUATION_GUIDE.md.")

    left, right = st.columns([1, 1])
    with left:
        common.show_document(exp)
    with right:
        st.markdown("**Text regions** (handwritten regions in reading order; add printed or uncertain-origin "
                    "regions only when their origin matters). Separate alternative readings with `|`.")
        region_rows = [{**{k: r.get(k) for k in REGION_COLUMNS},
                        "acceptable_readings": " | ".join(r.get("acceptable_readings") or []),
                        "category": r["category"] if isinstance(r.get("category"), str) else (r.get("category") or [""])[0]}
                       for r in existing["regions"]] or [{k: None for k in REGION_COLUMNS}]
        regions = st.data_editor(region_rows, num_rows="dynamic", key=f"regions_{sha}", column_config={
            "origin": st.column_config.SelectboxColumn(options=list(ann.ORIGINS)),
            "legibility": st.column_config.SelectboxColumn(options=list(ann.LEGIBILITY)),
            "location": st.column_config.SelectboxColumn(options=list(ann.LOCATIONS)),
            "category": st.column_config.SelectboxColumn(options=CATEGORY_KEYS),
            "crossed_out": st.column_config.CheckboxColumn(), "in_printed_field": st.column_config.CheckboxColumn(),
            "reading_order": st.column_config.NumberColumn(min_value=1, step=1)})

        st.markdown("**Pre-existing printed or pre-filled text** that a model must not transcribe")
        printed = st.data_editor(existing["printed_text"] or [{"text": None, "kind": None, "also_handwritten": False}],
                                 num_rows="dynamic", key=f"printed_{sha}", column_config={
                                     "kind": st.column_config.SelectboxColumn(options=list(ann.PRINTED_KINDS)),
                                     "also_handwritten": st.column_config.CheckboxColumn()})

        st.markdown("**Medicines** (verbatim, as written; leave a field empty when it is not written)")
        medications = st.data_editor(existing["medications"] or [{k: None for k in ann.MEDICATION_FIELDS}],
                                     num_rows="dynamic", key=f"meds_{sha}", column_config={
                                         "name_legibility": st.column_config.SelectboxColumn(options=list(ann.LEGIBILITY))})

    c1, c2, c3 = st.columns(3)
    annotator = c1.text_input("Annotator", existing.get("annotator", ""))
    verified_by = c2.text_input("Verified by", existing.get("verified_by", ""))
    verified_at = c3.text_input("Verified on (date)", existing.get("verified_at", ""))
    verified = st.checkbox("A person has checked every field of this annotation against the image",
                           value=bool(existing.get("verified")))
    notes = st.text_area("Notes", existing.get("notes", ""))

    if st.button("Validate and save", type="primary"):
        annotation = {
            "annotation_version": ann.ANNOTATION_VERSION, "image_sha256": sha,
            "image_filename": exp.record["input"]["original_filename"], "annotator": annotator,
            "verified": bool(verified), "verified_by": verified_by, "verified_at": verified_at, "notes": notes,
            "regions": _regions_from_rows(list(regions)),
            "printed_text": [{"text": _clean(p.get("text")), "kind": _clean(p.get("kind")) or "other",
                              "also_handwritten": bool(_clean(p.get("also_handwritten")))}
                             for p in printed if _clean(p.get("text"))],
            "medications": [{k: _clean(m.get(k)) for k in ann.MEDICATION_FIELDS}
                            for m in medications if _clean(m.get("name_verbatim"))],
            "categories": existing.get("categories") or {},
        }
        problems = ann.validate(annotation)
        if problems:
            st.error("Not saved. Please fix:\n\n" + "\n".join(f"- {p}" for p in problems))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(annotation, indent=2, ensure_ascii=False), encoding="utf-8")
            st.success(f"Saved {path.name} ({len(annotation['regions'])} regions, "
                       f"{len(annotation['medications'])} medicines, verified: {annotation['verified']}).")


render()

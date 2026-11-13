"""Ground-truth annotation format (version 1.0), validation and helpers.

An annotation is written and verified by a person looking at the original
image. It is never generated from, or pre-filled with, a model's output, and
agreement between models is never used in its place.

    {
      "annotation_version": "1.0",
      "image_sha256": "<sha256 of the original file>",
      "image_filename": "0022.jpg",
      "annotator": "initials or id",
      "verified": true,                      # set only after a human has checked every field
      "verified_by": "initials or id",
      "verified_at": "2026-11-20",
      "notes": "",
      "regions": [                           # every text region, handwritten AND printed-but-ambiguous
        {
          "region_id": "R1",
          "reading_order": 1,
          "location": "Middle-left",         # one of the nine labels in the master prompt
          "bbox": [0.08, 0.28, 0.85, 0.36],  # optional, normalised [x_min, y_min, x_max, y_max]
          "origin": "handwritten",           # handwritten | printed | uncertain
          "text": "T. Azee 500mg 1-0-0 x 7 days",   # verbatim, as written
          "legibility": "legible",           # legible | partially_legible | illegible
          "acceptable_readings": [],         # other readings a careful human accepts
          "crossed_out": false,
          "category": "medications",         # one category key, or a list of keys
          "in_printed_field": false,         # handwriting written into a printed field
          "printed_label": null              # that field's printed label, e.g. "Diagnosis:"
        }
      ],
      "printed_text": [                      # pre-existing text the model must NOT transcribe
        {"text": "Rx", "kind": "heading", "also_handwritten": false}
      ],
      "medications": [
        {
          "name_verbatim": "Azee", "name_normalised": "azee", "text_verbatim": "T. Azee 500mg",
          "strength": "500mg", "formulation": "T.", "route": null, "dose": null,
          "frequency": "1-0-0", "duration": "x 7 days", "instruction_verbatim": "1-0-0 x 7 days",
          "name_legibility": "legible"
        }
      ],
      "categories": {}                       # optional explicit items per category; overrides `regions`
    }
"""
from __future__ import annotations

from typing import Any

from ..inference.output_parser import CATEGORY_KEYS

ANNOTATION_VERSION = "1.0"
ORIGINS = ("handwritten", "printed", "uncertain")
LEGIBILITY = ("legible", "partially_legible", "illegible")
PRINTED_KINDS = ("heading", "label", "prefilled", "hospital_info", "instruction", "boilerplate", "table_header", "other")
LOCATIONS = ("Top-left", "Top-centre", "Top-right", "Middle-left", "Middle-centre", "Middle-right",
             "Bottom-left", "Bottom-centre", "Bottom-right")
MEDICATION_FIELDS = ("name_verbatim", "name_normalised", "text_verbatim", "strength", "formulation", "route",
                     "dose", "frequency", "duration", "instruction_verbatim", "name_legibility")


def empty_annotation(image_sha256: str, image_filename: str) -> dict[str, Any]:
    """A blank annotation for a human to fill in. Contains no model output."""
    return {"annotation_version": ANNOTATION_VERSION, "image_sha256": image_sha256,
            "image_filename": image_filename, "annotator": "", "verified": False, "verified_by": "",
            "verified_at": "", "notes": "", "regions": [], "printed_text": [], "medications": [],
            "categories": {}}


def empty_region(n: int) -> dict[str, Any]:
    return {"region_id": f"R{n}", "reading_order": n, "location": "", "bbox": None, "origin": "handwritten",
            "text": "", "legibility": "legible", "acceptable_readings": [], "crossed_out": False,
            "category": "other_unclassified", "in_printed_field": False, "printed_label": None}


def validate(annotation: dict[str, Any]) -> list[str]:
    """Return a list of problems (empty when the annotation is well formed)."""
    problems: list[str] = []
    if annotation.get("annotation_version") != ANNOTATION_VERSION:
        problems.append(f"annotation_version must be '{ANNOTATION_VERSION}'")
    sha = annotation.get("image_sha256") or ""
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        problems.append("image_sha256 must be the 64-character lower-case SHA-256 of the original file")
    if not isinstance(annotation.get("verified"), bool):
        problems.append("verified must be true or false")
    if annotation.get("verified") and not annotation.get("verified_by"):
        problems.append("verified is true but verified_by is empty")
    seen: set[str] = set()
    for n, r in enumerate(annotation.get("regions") or [], start=1):
        tag = f"regions[{n}]"
        rid = r.get("region_id")
        if not rid or rid in seen:
            problems.append(f"{tag}: region_id missing or duplicated")
        seen.add(rid)
        if r.get("origin") not in ORIGINS:
            problems.append(f"{tag}: origin must be one of {ORIGINS}")
        if r.get("legibility") not in LEGIBILITY:
            problems.append(f"{tag}: legibility must be one of {LEGIBILITY}")
        if not isinstance(r.get("reading_order"), int):
            problems.append(f"{tag}: reading_order must be an integer")
        cats = r.get("category")
        for c in ([cats] if isinstance(cats, str) else (cats or [])):
            if c not in CATEGORY_KEYS:
                problems.append(f"{tag}: unknown category '{c}'")
        if r.get("legibility") != "illegible" and not (r.get("text") or "").strip():
            problems.append(f"{tag}: text is empty but the region is not marked illegible")
        bbox = r.get("bbox")
        if bbox is not None:
            ok = (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in bbox)
                  and bbox[0] < bbox[2] and bbox[1] < bbox[3])
            if not ok:
                problems.append(f"{tag}: bbox must be [x_min, y_min, x_max, y_max] within 0..1")
        if r.get("location") and r["location"] not in LOCATIONS:
            problems.append(f"{tag}: location must be one of the nine labels, e.g. 'Middle-left'")
    for n, p in enumerate(annotation.get("printed_text") or [], start=1):
        if not (p.get("text") or "").strip():
            problems.append(f"printed_text[{n}]: text is empty")
        if p.get("kind") not in PRINTED_KINDS:
            problems.append(f"printed_text[{n}]: kind must be one of {PRINTED_KINDS}")
    for n, m in enumerate(annotation.get("medications") or [], start=1):
        if not (m.get("name_verbatim") or "").strip():
            problems.append(f"medications[{n}]: name_verbatim is empty")
        if m.get("name_legibility") and m["name_legibility"] not in LEGIBILITY:
            problems.append(f"medications[{n}]: name_legibility must be one of {LEGIBILITY}")
    for key in (annotation.get("categories") or {}):
        if key not in CATEGORY_KEYS:
            problems.append(f"categories: unknown category '{key}'")
    return problems


def handwritten_regions(annotation: dict[str, Any], include_crossed_out: bool = False) -> list[dict[str, Any]]:
    regions = [r for r in annotation.get("regions") or []
               if r.get("origin") == "handwritten" and (include_crossed_out or not r.get("crossed_out"))]
    return sorted(regions, key=lambda r: r.get("reading_order", 0))


def reference_transcription(annotation: dict[str, Any]) -> str:
    """Handwritten regions in reading order, one per line (crossed-out and illegible text excluded)."""
    return "\n".join(r["text"] for r in handwritten_regions(annotation)
                     if r.get("legibility") != "illegible" and (r.get("text") or "").strip())


def reference_categories(annotation: dict[str, Any]) -> dict[str, list[str]]:
    explicit = annotation.get("categories") or {}
    if any(explicit.values()):
        return {k: list(explicit.get(k, [])) for k in CATEGORY_KEYS}
    out: dict[str, list[str]] = {k: [] for k in CATEGORY_KEYS}
    for r in handwritten_regions(annotation):
        if r.get("legibility") == "illegible" or not (r.get("text") or "").strip():
            continue
        cats = r.get("category")
        for c in ([cats] if isinstance(cats, str) else (cats or [])):
            out[c].append(r["text"])
    return out

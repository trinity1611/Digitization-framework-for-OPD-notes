"""Medicine-specific metrics.

Exact verbatim performance and normalised matching are always reported
separately. "Normalised" here means string normalisation only (case,
punctuation, spacing, and an annotator-supplied normalised name). No medicine
catalogue or brand/generic mapping is used.
"""
from __future__ import annotations

import re
from typing import Any

from ..inference.output_parser import get_field, item_text
from .textdist import contains_tokens, normalise, partial_similarity, similarity, strip_markers

ATTRIBUTES = ("strength", "formulation", "route", "dose", "frequency", "duration")
_FORM_PREFIX = re.compile(r"^(?:tab(?:let)?s?|t|cap(?:sule)?s?|c|syp|syr(?:up)?|inj(?:ection)?|oint(?:ment)?|"
                          r"susp|drops?|gel|cream|lotion|sachet)\b\.?\s*", re.I)
_EMPTY = {"", "-", "none", "n/a", "na", "nil", "not written", "not specified", "not mentioned", "unspecified"}


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip()
    return None if v.lower().rstrip(".") in _EMPTY else v


def _compact(value: str) -> str:
    """Normalised attribute form: relaxed normalisation with spaces removed ('500 mg' == '500mg')."""
    return normalise(value, "relaxed").replace(" ", "")


def name_core(medicine_text: str) -> str:
    """Predicted medicine text without formulation prefix and numeric strength (string rules only)."""
    t = _FORM_PREFIX.sub("", strip_markers(medicine_text).strip())
    t = re.sub(r"\b\d+(?:[.,/]\d+)*\s*(?:mg|mcg|g|gm|ml|iu|%|units?)?\b", " ", t, flags=re.I)
    return " ".join(t.split())


def predicted_medicines(structured: dict[str, Any]) -> list[dict[str, Any]]:
    """Medicines from a model's structured extraction, with their associated instructions."""
    extraction = (structured or {}).get("structured_extraction") or {}
    meds = []
    for item in extraction.get("medications", []) or []:
        text = item_text(item)
        if not _clean(text):
            continue
        meds.append({"text": text, "strength": _clean(get_field(item, "strength")),
                     "formulation": _clean(get_field(item, "formulation")),
                     "route": _clean(get_field(item, "route")),
                     "possible_name": _clean(get_field(item, "possible medicine name")),
                     "evidence_level": _clean(get_field(item, "evidence level")),
                     "dose": None, "frequency": None, "duration": None, "instruction": None})
    for item in extraction.get("medication_instructions", []) or []:
        assoc = _clean(get_field(item, "associated medication"))
        if not assoc or not meds:
            continue
        scores = [similarity(normalise(assoc, "relaxed"), normalise(m["text"], "relaxed")) for m in meds]
        best = max(range(len(meds)), key=lambda k: scores[k])
        if scores[best] < 0.5 and not contains_tokens(normalise(meds[best]["text"], "relaxed"),
                                                      normalise(name_core(assoc), "relaxed")):
            continue
        m = meds[best]
        m["instruction"] = m["instruction"] or _clean(item_text(item))
        for attr in ("dose", "frequency", "duration", "route"):
            m[attr] = m[attr] or _clean(get_field(item, attr))
    return meds


def _attribute_result(reference: str | None, predicted: str | None) -> dict[str, bool] | None:
    ref = _clean(reference)
    if ref is None:
        return None                      # not annotated for this medicine: not scored
    pred = _clean(predicted)
    return {"exact": pred is not None and pred.strip() == ref.strip(),
            "normalised": pred is not None and _compact(pred) == _compact(ref)}


def medicine_metrics(reference_medicines: list[dict[str, Any]], structured: dict[str, Any],
                     pairing_threshold: float = 0.5) -> dict[str, Any]:
    preds = predicted_medicines(structured)
    refs = reference_medicines or []
    # Pair reference and predicted medicines by name similarity (one-to-one, greedy).
    scored = []
    for gi, g in enumerate(refs):
        g_norm = normalise(g.get("name_normalised") or g.get("name_verbatim") or "", "relaxed")
        for pi, p in enumerate(preds):
            p_norm = normalise(p["text"], "relaxed")
            score = 1.0 if contains_tokens(p_norm, g_norm) else max(
                similarity(g_norm, normalise(name_core(p["text"]), "relaxed")), partial_similarity(g_norm, p_norm))
            scored.append((score, gi, pi))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    pair_of: dict[int, int] = {}
    used_p: set[int] = set()
    for score, gi, pi in scored:
        if score < pairing_threshold:
            break
        if gi in pair_of or pi in used_p:
            continue
        pair_of[gi] = pi
        used_p.add(pi)

    details = []
    exact_names = normalised_names = 0
    attr_totals = {a: {"annotated": 0, "exact": 0, "normalised": 0} for a in ATTRIBUTES}
    for gi, g in enumerate(refs):
        p = preds[pair_of[gi]] if gi in pair_of else None
        name_v = (g.get("name_verbatim") or "").strip()
        row: dict[str, Any] = {"reference_name": name_v, "predicted_text": p["text"] if p else None,
                               "paired": p is not None}
        if p:
            plain = strip_markers(p["text"])
            row["exact_name_match"] = bool(name_v) and contains_tokens(plain, name_v)
            candidates = {normalise(name_v, "relaxed")}
            if g.get("name_normalised"):
                candidates.add(normalise(g["name_normalised"], "relaxed"))
            row["normalised_name_match"] = any(c and contains_tokens(normalise(plain, "relaxed"), c)
                                               for c in candidates)
        else:
            row["exact_name_match"] = row["normalised_name_match"] = False
        exact_names += row["exact_name_match"]
        normalised_names += row["normalised_name_match"]
        row["attributes"] = {}
        for attr in ATTRIBUTES:
            res = _attribute_result(g.get(attr), p.get(attr) if p else None)
            row["attributes"][attr] = res
            if res is not None:
                attr_totals[attr]["annotated"] += 1
                attr_totals[attr]["exact"] += res["exact"]
                attr_totals[attr]["normalised"] += res["normalised"]
        details.append(row)

    n = len(refs)
    rate = lambda k: (k / n) if n else None
    return {
        "reference_medicines": n, "predicted_medicines": len(preds), "paired": len(pair_of),
        "missed_medicines": n - len(pair_of), "unpaired_predicted_medicines": len(preds) - len(pair_of),
        "exact_medicine_name_match": rate(exact_names),
        "normalised_medicine_name_match": rate(normalised_names),
        "attribute_accuracy": {
            attr: {"annotated": t["annotated"],
                   "exact": (t["exact"] / t["annotated"]) if t["annotated"] else None,
                   "normalised": (t["normalised"] / t["annotated"]) if t["annotated"] else None}
            for attr, t in attr_totals.items()},
        "definitions": {
            "exact_medicine_name_match": "the annotated verbatim name occurs, as whole tokens and with identical "
                                         "characters, in the model's 'Original medicine text'",
            "normalised_medicine_name_match": "the same test after case-folding and punctuation removal, also "
                                              "accepting the annotator's normalised name",
            "attribute_accuracy": "share of reference medicines with that attribute annotated for which the "
                                  "model's value is identical (exact) or identical after normalisation; "
                                  "medicines the model missed count as wrong",
        },
        "details": details,
    }

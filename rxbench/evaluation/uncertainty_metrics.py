"""Uncertainty evaluation: how honestly does a model report what it could not read?

Reference regions (human annotation, in reading order) are aligned with the
lines of the model's verbatim transcription. For each aligned pair the model
either flagged uncertainty (any of the master prompt's markers) or it did not,
and its reading is either right or wrong. That gives:

* illegible regions correctly identified / guessed / omitted
* legible regions incorrectly marked as uncertain
* incorrectly guessed words and incorrectly confident outputs
* medicine names: correct or wrong, flagged or confident
"""
from __future__ import annotations

from typing import Any

from ..inference.output_parser import ANY_MARKER
from .annotation import handwritten_regions
from .medicine_metrics import medicine_metrics, predicted_medicines
from .textdist import normalise, similarity, strip_markers


def _plain(text: str) -> str:
    return normalise(strip_markers(text), "relaxed")


def _pair_score(region: dict[str, Any], line: str) -> float:
    has_marker = bool(ANY_MARKER.search(line))
    if region.get("legibility") == "illegible" or not (region.get("text") or "").strip():
        # An unreadable region matches a line that is (mostly) an uncertainty marker.
        return 0.6 if has_marker and len(_plain(line)) <= 12 else 0.0
    candidates = [region["text"], *(region.get("acceptable_readings") or [])]
    return max(similarity(_plain(c), _plain(line)) for c in candidates)


def align_regions_to_lines(regions: list[dict[str, Any]], lines: list[str],
                           min_score: float = 0.3) -> list[tuple[int, int | None, float]]:
    """Order-preserving alignment (dynamic programming) of reference regions to transcription lines."""
    n, m = len(regions), len(lines)
    score = [[_pair_score(r, l) for l in lines] for r in regions]
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            take = best[i - 1][j - 1] + (score[i - 1][j - 1] if score[i - 1][j - 1] >= min_score else -1.0)
            best[i][j] = max(best[i - 1][j], best[i][j - 1], take)
    out: list[tuple[int, int | None, float]] = []
    i, j = n, m
    while i > 0:
        if j > 0 and score[i - 1][j - 1] >= min_score and best[i][j] == best[i - 1][j - 1] + score[i - 1][j - 1]:
            out.append((i - 1, j - 1, score[i - 1][j - 1]))
            i, j = i - 1, j - 1
        elif j > 0 and best[i][j] == best[i][j - 1]:
            j -= 1
        else:
            out.append((i - 1, None, 0.0))
            i -= 1
    return list(reversed(out))


def uncertainty_metrics(annotation: dict[str, Any], transcription: str | None,
                        structured: dict[str, Any] | None) -> dict[str, Any]:
    regions = handwritten_regions(annotation, include_crossed_out=False)
    lines = [l for l in (transcription or "").splitlines() if l.strip()]
    alignment = align_regions_to_lines(regions, lines)

    illegible = {"total": 0, "correctly_flagged": 0, "guessed_without_flag": 0, "omitted": 0}
    readable = {"total": 0, "correct_and_confident": 0, "correct_but_flagged": 0, "wrong_but_flagged": 0,
                "wrong_and_confident": 0, "omitted": 0}
    details = []
    for ri, li, score in alignment:
        region = regions[ri]
        line = lines[li] if li is not None else None
        flagged = bool(line and ANY_MARKER.search(line))
        row = {"region_id": region.get("region_id"), "legibility": region.get("legibility"),
               "aligned": line is not None, "flagged": flagged, "alignment_score": round(score, 3)}
        if region.get("legibility") == "illegible":
            illegible["total"] += 1
            if line is None:
                illegible["omitted"] += 1
                row["outcome"] = "omitted"
            elif flagged:
                illegible["correctly_flagged"] += 1
                row["outcome"] = "correctly_flagged"
            else:
                illegible["guessed_without_flag"] += 1
                row["outcome"] = "guessed_without_flag"
        else:
            readable["total"] += 1
            if line is None:
                readable["omitted"] += 1
                row["outcome"] = "omitted"
            else:
                accepted = {_plain(x) for x in [region["text"], *(region.get("acceptable_readings") or [])]}
                correct = _plain(line) in accepted
                key = ("correct_" if correct else "wrong_") + ("but_flagged" if flagged else "and_confident")
                readable[key] += 1
                row["outcome"] = key
        details.append(row)

    fully_legible = [d for d, (ri, _l, _s) in zip(details, alignment) if regions[ri].get("legibility") == "legible"]
    legible_flagged = sum(d["flagged"] for d in fully_legible if d["aligned"])

    # Medicine names: was a wrong reading at least flagged as uncertain?
    med = medicine_metrics(annotation.get("medications") or [], structured or {})
    preds = {p["text"]: p for p in predicted_medicines(structured or {})}
    register_text = ((structured or {}).get("uncertainty_register") or {}).get("text") or ""
    names = {"total": 0, "correct_and_confident": 0, "correct_but_flagged": 0, "wrong_but_flagged": 0,
             "wrong_and_confident": 0, "missed": 0}
    for row in med["details"]:
        names["total"] += 1
        if not row["paired"]:
            names["missed"] += 1
            continue
        p = preds.get(row["predicted_text"], {})
        level = (p.get("evidence_level") or "").upper()
        flagged = (bool(ANY_MARKER.search(row["predicted_text"])) or bool(p.get("possible_name"))
                   or (level != "" and "EXPLICIT" not in level)
                   or (row["reference_name"] and row["reference_name"].casefold() in register_text.casefold()))
        correct = row["normalised_name_match"]
        names[("correct_" if correct else "wrong_") + ("but_flagged" if flagged else "and_confident")] += 1

    rate = lambda a, b: (a / b) if b else None
    return {
        "illegible_regions": {**illegible,
                              "correctly_identified_rate": rate(illegible["correctly_flagged"], illegible["total"])},
        "legible_regions_incorrectly_marked": {
            "legible_regions": len(fully_legible), "marked_uncertain": legible_flagged,
            "rate": rate(legible_flagged, len(fully_legible))},
        "readable_regions": {**readable,
                             "incorrectly_guessed_rate": rate(readable["wrong_and_confident"]
                                                              + readable["wrong_but_flagged"], readable["total"]),
                             "incorrectly_confident_rate": rate(readable["wrong_and_confident"], readable["total"])},
        "medicine_names": {**names,
                           "wrong_and_confident_rate": rate(names["wrong_and_confident"], names["total"]),
                           "flagged_when_wrong_rate": rate(names["wrong_but_flagged"],
                                                           names["wrong_but_flagged"] + names["wrong_and_confident"])},
        "honesty_summary": {
            "errors_total": readable["wrong_and_confident"] + readable["wrong_but_flagged"],
            "errors_flagged": readable["wrong_but_flagged"],
            "share_of_errors_flagged": rate(readable["wrong_but_flagged"],
                                            readable["wrong_and_confident"] + readable["wrong_but_flagged"]),
            "definition": "of the regions the model read wrongly, the share where it had marked uncertainty"},
        "method": "Regions are aligned to transcription lines in reading order; a line counts as flagged when "
                  "it contains any uncertainty marker; a reading is correct when it equals the annotated text "
                  "or an accepted alternative after case-folding and punctuation removal.",
        "details": details,
    }

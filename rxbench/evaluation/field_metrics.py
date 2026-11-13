"""Classification metrics per clinical category: precision, recall, F1, exact match, field-level accuracy.

Each category is evaluated independently against human-verified items. Three
matching levels are reported side by side and never merged:

* exact       identical strings (leading/trailing whitespace ignored)
* normalised  identical after case-folding, whitespace collapsing and punctuation removal
* fuzzy       normalised similarity >= threshold (default 0.8)
"""
from __future__ import annotations

from typing import Any

from ..inference.output_parser import CATEGORY_KEYS
from .textdist import greedy_match, normalise, prf


def match_counts(reference: list[str], predicted: list[str], level: str, threshold: float = 0.8) -> dict[str, Any]:
    if level == "exact":
        ref, hyp, min_sim = [r.strip() for r in reference], [p.strip() for p in predicted], 1.0
    elif level == "normalised":
        ref, hyp, min_sim = [normalise(r, "relaxed") for r in reference], [normalise(p, "relaxed") for p in predicted], 1.0
    elif level == "fuzzy":
        ref, hyp, min_sim = ([normalise(r, "relaxed") for r in reference],
                             [normalise(p, "relaxed") for p in predicted], threshold)
    else:
        raise ValueError(f"Unknown matching level '{level}'")
    pairs = greedy_match(ref, hyp, min_sim)
    tp = len(pairs)
    return {**prf(tp, len(hyp) - tp, len(ref) - tp), "level": level,
            "threshold": threshold if level == "fuzzy" else None}


def category_metrics(reference: dict[str, list[str]], predicted: dict[str, list[str]],
                     fuzzy_threshold: float = 0.8) -> dict[str, Any]:
    per_category: dict[str, Any] = {}
    totals = {lvl: {"tp": 0, "fp": 0, "fn": 0} for lvl in ("exact", "normalised", "fuzzy")}
    for key in CATEGORY_KEYS:
        ref = [x for x in reference.get(key, []) if x.strip()]
        hyp = [x for x in predicted.get(key, []) if x.strip()]
        levels = {lvl: match_counts(ref, hyp, lvl, fuzzy_threshold) for lvl in totals}
        for lvl in totals:
            for k in ("tp", "fp", "fn"):
                totals[lvl][k] += levels[lvl][k]
        per_category[key] = {
            "reference_items": len(ref), "predicted_items": len(hyp), **levels,
            # exact match: the predicted set of items is exactly the reference set
            "exact_match": sorted(x.strip() for x in ref) == sorted(x.strip() for x in hyp),
            "normalised_exact_match": (sorted(normalise(x, "relaxed") for x in ref)
                                       == sorted(normalise(x, "relaxed") for x in hyp)),
            # field-level accuracy: share of reference items reproduced exactly in this category
            "field_level_accuracy": (levels["exact"]["tp"] / len(ref)) if ref else None,
            "field_level_accuracy_normalised": (levels["normalised"]["tp"] / len(ref)) if ref else None,
        }
    evaluated = [k for k in CATEGORY_KEYS if per_category[k]["reference_items"] or per_category[k]["predicted_items"]]
    macro = {}
    for lvl in totals:
        f1s = [per_category[k][lvl]["f1"] for k in evaluated if per_category[k][lvl]["f1"] is not None]
        macro[lvl] = (sum(f1s) / len(f1s)) if f1s else None
    return {"per_category": per_category,
            "micro": {lvl: prf(**totals[lvl]) for lvl in totals},
            "macro_f1": macro,
            "categories_with_content": evaluated,
            "categories_exact_match_rate": (sum(per_category[k]["exact_match"] for k in evaluated) / len(evaluated)
                                            if evaluated else None)}

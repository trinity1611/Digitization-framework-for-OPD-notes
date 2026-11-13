"""Character Error Rate.

    CER = (Substitutions + Deletions + Insertions) / Number of reference characters
"""
from __future__ import annotations

from typing import Any

from .textdist import edit_counts, greedy_match, normalise, strip_markers


def cer(reference: str, hypothesis: str, normalisation: str = "verbatim",
        strip_uncertainty_markers: bool = False) -> dict[str, Any]:
    """CER of `hypothesis` against a human-verified `reference`.

    `normalisation` and `strip_uncertainty_markers` are reported with the result so
    verbatim and normalised figures are never confused.
    """
    if strip_uncertainty_markers:
        hypothesis = strip_markers(hypothesis)
        reference = strip_markers(reference)
    ref = normalise(reference, normalisation)
    hyp = normalise(hypothesis, normalisation)
    counts = edit_counts(ref, hyp)
    n = len(ref)
    return {"metric": "CER", "value": (counts["distance"] / n) if n else None, **counts,
            "reference_length": n, "hypothesis_length": len(hyp), "unit": "characters",
            "normalisation": normalisation, "uncertainty_markers_stripped": strip_uncertainty_markers}


def item_matched_cer(reference_items: list[str], predicted_items: list[str],
                     normalisation: str = "verbatim", min_similarity: float = 0.3) -> dict[str, Any]:
    """Micro-averaged CER over lists of short items (medicine names, dosage instructions, ...).

    Items are paired one-to-one by similarity. Unpaired reference items count as
    deletions of all their characters and unpaired predicted items as insertions.
    """
    ref = [normalise(x, normalisation) for x in reference_items]
    hyp = [normalise(x, normalisation) for x in predicted_items]
    pairs = greedy_match(ref, hyp, min_similarity)
    s = d = i = 0
    for gi, pi, _sim in pairs:
        c = edit_counts(ref[gi], hyp[pi])
        s, d, i = s + c["substitutions"], d + c["deletions"], i + c["insertions"]
    matched_g = {p[0] for p in pairs}
    matched_p = {p[1] for p in pairs}
    unmatched_ref = [x for k, x in enumerate(ref) if k not in matched_g]
    unmatched_hyp = [x for k, x in enumerate(hyp) if k not in matched_p]
    d += sum(len(x) for x in unmatched_ref)
    i += sum(len(x) for x in unmatched_hyp)
    n = sum(len(x) for x in ref)
    return {"metric": "CER (item-matched, micro)", "value": ((s + d + i) / n) if n else None,
            "substitutions": s, "deletions": d, "insertions": i, "reference_length": n,
            "reference_items": len(ref), "predicted_items": len(hyp), "matched_pairs": len(pairs),
            "unmatched_reference_items": len(unmatched_ref), "unmatched_predicted_items": len(unmatched_hyp),
            "normalisation": normalisation}

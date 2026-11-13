"""Word Error Rate.

    WER = (Substitutions + Deletions + Insertions) / Number of reference words

Words are whitespace-separated tokens.
"""
from __future__ import annotations

import re
from typing import Any

from .textdist import edit_counts, normalise, prf, strip_markers

# Numbers as written on prescriptions: 500, 6.8, 130/80, 1-0-1, 1+0+1, 0.5
NUMERIC_TOKEN = re.compile(r"\d+(?:[.,/:+\-]\d+)*")


def wer(reference: str, hypothesis: str, normalisation: str = "verbatim",
        strip_uncertainty_markers: bool = False) -> dict[str, Any]:
    if strip_uncertainty_markers:
        hypothesis = strip_markers(hypothesis)
        reference = strip_markers(reference)
    ref = normalise(reference, normalisation).split()
    hyp = normalise(hypothesis, normalisation).split()
    counts = edit_counts(ref, hyp)
    n = len(ref)
    return {"metric": "WER", "value": (counts["distance"] / n) if n else None, **counts,
            "reference_length": n, "hypothesis_length": len(hyp), "unit": "words",
            "normalisation": normalisation, "uncertainty_markers_stripped": strip_uncertainty_markers}


def numeric_values(reference: str, hypothesis: str) -> dict[str, Any]:
    """Accuracy on numerical values only (doses, strengths, measurements, durations)."""
    ref = NUMERIC_TOKEN.findall(strip_markers(reference))
    hyp = NUMERIC_TOKEN.findall(strip_markers(hypothesis))
    counts = edit_counts(ref, hyp)
    remaining = list(hyp)
    tp = 0
    for token in ref:
        if token in remaining:
            remaining.remove(token)
            tp += 1
    return {"metric": "numeric values",
            "sequence_error_rate": (counts["distance"] / len(ref)) if ref else None, **counts,
            "reference_count": len(ref), "hypothesis_count": len(hyp),
            "bag_of_values": prf(tp, len(hyp) - tp, len(ref) - tp)}

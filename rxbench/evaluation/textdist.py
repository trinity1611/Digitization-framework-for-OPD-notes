"""Edit distance, text normalisation and item matching shared by the metrics.

All of this is plain string arithmetic. No dictionary, medicine catalogue or
model is consulted anywhere in the evaluation code.
"""
from __future__ import annotations

import re
import string
from typing import Any, Sequence

try:                                   # fast path; the pure-Python fallback gives identical counts
    from rapidfuzz.distance import Levenshtein as _RFLev
except Exception:                      # pragma: no cover - exercised only without rapidfuzz
    _RFLev = None

NORMALISATIONS = ("verbatim", "whitespace", "relaxed")

_MARKER_REWRITES = [
    (re.compile(r"\[\s*UNCERTAIN\s*:\s*[\"“']?(.*?)[\"”']?\s*(?:/[^\]]*)?\]", re.I), r"\1"),
    (re.compile(r"\[\s*PARTIALLY LEGIBLE\s*:\s*[\"“']?(.*?)(?:\.\.\.|…)?[\"”']?\s*\]", re.I), r"\1"),
    (re.compile(r"\[\s*OVERWRITTEN\s*:\s*(.*?)\s*\]", re.I), r"\1"),
    (re.compile(r"\[\s*CROSSED OUT\s*:\s*(.*?)\s*\]", re.I), ""),
    (re.compile(r"\[\s*ILLEGIBLE\s*\]", re.I), ""),
    (re.compile(r"\[\s*HANDWRITING UNCLEAR\s*\]", re.I), ""),
]
_PUNCT = str.maketrans({c: " " for c in string.punctuation if c not in "./+-%:"})


def strip_markers(text: str) -> str:
    """Reduce uncertainty markers to plain text: first reading of UNCERTAIN, the readable part of
    PARTIALLY LEGIBLE, the text of OVERWRITTEN; CROSSED OUT / ILLEGIBLE / UNCLEAR are removed."""
    for rx, repl in _MARKER_REWRITES:
        text = rx.sub(repl, text)
    return text


def normalise(text: str, mode: str = "verbatim") -> str:
    """verbatim: unchanged. whitespace: runs of whitespace -> one space. relaxed: whitespace +
    case-folding + punctuation removed (characters that carry dosage meaning are kept: / + - % : and
    the decimal point inside numbers)."""
    if mode == "verbatim":
        return text
    if mode == "whitespace":
        return " ".join(text.split())
    if mode == "relaxed":
        # A full stop is kept only inside a number (6.8); "Tab." and "T." lose theirs.
        text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
        return " ".join(text.casefold().translate(_PUNCT).split())
    raise ValueError(f"Unknown normalisation '{mode}' (choose from {NORMALISATIONS})")


def _edit_counts_python(ref: Sequence, hyp: Sequence) -> tuple[int, int, int]:
    """Levenshtein alignment counts (substitutions, deletions, insertions) by dynamic programming."""
    n, m = len(ref), len(hyp)
    prev = [(j, 0, 0, j) for j in range(m + 1)]          # (cost, S, D, I)
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        ri = ref[i - 1]
        for j in range(1, m + 1):
            if ri == hyp[j - 1]:
                best = prev[j - 1]
            else:
                c, s, d, ins = prev[j - 1]
                best = (c + 1, s + 1, d, ins)
            c, s, d, ins = prev[j]
            if c + 1 < best[0]:
                best = (c + 1, s, d + 1, ins)
            c, s, d, ins = cur[j - 1]
            if c + 1 < best[0]:
                best = (c + 1, s, d, ins + 1)
            cur.append(best)
        prev = cur
    _, s, d, ins = prev[m]
    return s, d, ins


def edit_counts(ref: Sequence, hyp: Sequence) -> dict[str, int]:
    """Substitutions, deletions and insertions turning `ref` into `hyp`."""
    if _RFLev is not None:
        s = d = i = 0
        for op in _RFLev.editops(ref, hyp):
            if op.tag == "replace":
                s += 1
            elif op.tag == "delete":
                d += 1
            else:
                i += 1
    else:
        s, d, i = _edit_counts_python(ref, hyp)
    return {"substitutions": s, "deletions": d, "insertions": i, "distance": s + d + i}


def similarity(a: str, b: str) -> float:
    """1 - normalised edit distance, in [0, 1]."""
    if not a and not b:
        return 1.0
    return 1.0 - edit_counts(a, b)["distance"] / max(len(a), len(b))


def contains_tokens(haystack: str, needle: str) -> bool:
    """True if the tokens of `needle` occur contiguously, as whole tokens, in `haystack`."""
    h, n = haystack.split(), needle.split()
    if not n:
        return False
    return any(h[i:i + len(n)] == n for i in range(len(h) - len(n) + 1))


def partial_similarity(needle: str, haystack: str) -> float:
    """Best similarity between `needle` and any equally long (in tokens, +-1) window of `haystack`."""
    n_tok, h_tok = needle.split(), haystack.split()
    if not n_tok:
        return 0.0
    if len(h_tok) <= len(n_tok):
        return similarity(needle, haystack)
    best = 0.0
    for width in {max(1, len(n_tok) - 1), len(n_tok), len(n_tok) + 1}:
        for i in range(len(h_tok) - width + 1):
            best = max(best, similarity(needle, " ".join(h_tok[i:i + width])))
            if best == 1.0:
                return best
    return best


def greedy_match(gt: list[str], pred: list[str], min_similarity: float = 0.0) -> list[tuple[int, int, float]]:
    """One-to-one pairing of reference and predicted items by descending similarity."""
    scored = sorted(((similarity(g, p), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)),
                    key=lambda t: (-t[0], t[1], t[2]))
    used_g: set[int] = set()
    used_p: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for sim, gi, pi in scored:
        if sim < min_similarity:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi, sim))
    return pairs


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall > 0 else
          (0.0 if precision is not None and recall is not None else None))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

"""Deterministic parsing of a model's raw output into structured_output.json.

The raw output is the authoritative record and is never modified. This module
only *reads* it. No model is used, nothing is repaired, corrected or completed,
and no clinical fact is added: every value in the structured output is a
substring of the raw output.

Two formats are recognised:

1. JSON (whole output, or a fenced ```json block). Parsed as-is.
2. The Markdown layout that the master prompt asks for (sections A to E).
   Sections and "- Field: value" items are located with fixed rules.

If neither applies, structured parsing is reported as unsuccessful and the raw
output remains available for inspection.
"""
from __future__ import annotations

import json
import re
from typing import Any

# Section C categories: canonical key -> heading keywords (lower case).
CATEGORIES: list[tuple[str, str, tuple[str, ...]]] = [
    ("chief_complaints", "Chief complaints", ("chief complaint",)),
    ("clinical_history", "Clinical history", ("clinical history", "history")),
    ("conditions_diagnoses", "Conditions / diagnoses", ("condition", "diagnos")),
    ("clinical_observations", "Clinical observations", ("clinical observation", "examination finding", "observation")),
    ("measurements_vital_signs", "Measurements / vital signs", ("measurement", "vital sign")),
    ("medication_instructions", "Medication instructions", ("medication instruction", "dosage", "instruction")),
    ("medications", "Medications", ("medication", "medicine")),
    ("investigations_tests", "Investigations / tests", ("investigation", "test")),
    ("advice_counselling", "Advice / counselling", ("advice", "counsel")),
    ("follow_up_referral", "Follow-up / referral", ("follow-up", "follow up", "followup", "referral")),
    ("other_unclassified", "Other / unclassified", ("other", "unclassified")),
]
CATEGORY_KEYS = [c[0] for c in CATEGORIES]
CATEGORY_LABELS = {c[0]: c[1] for c in CATEGORIES}
# Display order used by the interface (medications before their instructions).
CATEGORY_DISPLAY_ORDER = [
    "chief_complaints", "clinical_history", "conditions_diagnoses", "clinical_observations",
    "measurements_vital_signs", "medications", "medication_instructions", "investigations_tests",
    "advice_counselling", "follow_up_referral", "other_unclassified",
]

TOP_SECTIONS: list[tuple[str, str, tuple[str, ...]]] = [
    ("document_overview", "A", ("document overview",)),
    ("verbatim_transcription", "B", ("verbatim transcription",)),
    ("structured_extraction", "C", ("structured clinical extraction", "structured extraction")),
    ("uncertainty_register", "D", ("uncertainty register",)),
    ("final_validation", "E", ("final validation",)),
]

UNCERTAINTY_MARKERS = {
    "illegible": re.compile(r"\[\s*ILLEGIBLE\s*\]", re.I),
    "partially_legible": re.compile(r"\[\s*PARTIALLY LEGIBLE\s*:[^\]]*\]", re.I),
    "uncertain": re.compile(r"\[\s*UNCERTAIN\s*:[^\]]*\]", re.I),
    "crossed_out": re.compile(r"\[\s*CROSSED OUT\s*:[^\]]*\]", re.I),
    "overwritten": re.compile(r"\[\s*OVERWRITTEN\s*:[^\]]*\]", re.I),
    "handwriting_unclear": re.compile(r"\[\s*HANDWRITING UNCLEAR\s*\]", re.I),
}
ANY_MARKER = re.compile(r"\[\s*(?:ILLEGIBLE|PARTIALLY LEGIBLE|UNCERTAIN|CROSSED OUT|OVERWRITTEN|"
                        r"HANDWRITING UNCLEAR)\b[^\]]*\]", re.I)

_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s*(.+?)\s*#*\s*$")
_BOLD_HEADING = re.compile(r"^\s{0,3}\*\*(.+?)\*\*\s*:?\s*$")
_FIELD = re.compile(r"^\s*(?:[-*+•]|\d+[.)])?\s*\**\s*([A-Za-z][A-Za-z0-9 ,/()'\-]{1,70}?)\s*\**\s*:\s*\**\s*(.*)$")
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+(.*)$")
_EMPTY_VALUES = {"", "-", "—", "–", "none", "n/a", "na", "nil", "not applicable", "not written",
                 "not specified", "not mentioned", "none identified", "none found", "not present"}
_NO_ITEMS = re.compile(r"^\W*(no(ne)?\b|not (identified|present|found|documented|applicable|mentioned)|"
                       r"n/?a\b|nil\b|—|-$)", re.I)

PARSER_VERSION = 2

_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)                      # Qwen
_GEMMA_THOUGHT = re.compile(r"^\s*<\|channel>thought\n.*?<channel\|>", re.S)        # Gemma 4
# Gemma 3 / MedGemma have no reasoning switch in the chat template, but the model can open a
# reasoning block on its own; it is delimited by these two reserved tokens.
_GEMMA3_THOUGHT = re.compile(r"^\s*<unused94>.*?<unused95>\s*", re.S)


def split_thinking(raw: str) -> tuple[str | None, str]:
    """Separate a leading reasoning block from the answer (for parsing only)."""
    for pattern in (_THINK_BLOCK, _GEMMA_THOUGHT, _GEMMA3_THOUGHT):
        m = pattern.match(raw)
        if m:
            return m.group(0).strip(), raw[m.end():]
    if "</think>" in raw and "<think>" not in raw:  # template opened the block in the prompt
        head, tail = raw.split("</think>", 1)
        return head.strip(), tail
    return None, raw


# --------------------------------------------------------------------------- JSON

def extract_json(text: str) -> tuple[Any, str | None]:
    """Return (parsed, method) if the output is valid JSON; nothing is repaired."""
    stripped = text.strip()
    if not stripped:
        return None, None
    try:
        return json.loads(stripped), "whole_output"
    except ValueError:
        pass
    m = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n?```", stripped, re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1)), "fenced_block"
        except ValueError:
            pass
    return None, None


# --------------------------------------------------------------------------- Markdown

def _clean_heading(text: str) -> str:
    t = re.sub(r"[*_`]", "", text).strip()
    t = re.sub(r"^(?:[A-E]|\d+(?:\.\d+)*)[.):]\s+", "", t)     # "A. ", "6.1 "
    return t.strip(" :").lower()


def _heading_of(line: str) -> tuple[int, str] | None:
    m = _HEADING.match(line)
    if m:
        return len(m.group(1)), m.group(2)
    m = _BOLD_HEADING.match(line)
    if m and len(m.group(1)) <= 60:
        return 7, m.group(1)
    return None


def _match_top(heading: str) -> str | None:
    h = _clean_heading(heading)
    for key, _letter, kws in TOP_SECTIONS:
        if any(k in h for k in kws):
            return key
    return None


def _match_category(heading: str) -> str | None:
    h = _clean_heading(heading)
    for key, _label, kws in CATEGORIES:
        if any(h.startswith(k) or k in h for k in kws):
            return key
    return None


def _split_top_sections(text: str) -> dict[str, str]:
    lines = text.splitlines()
    marks: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        hd = _heading_of(line)
        if hd:
            key = _match_top(hd[1])
            if key and key not in [k for _, k in marks]:
                marks.append((i, key))
    sections: dict[str, str] = {}
    for n, (start, key) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(lines)
        sections[key] = "\n".join(lines[start + 1:end]).strip("\n")
    return sections


def _strip_code_fence(block: str) -> str:
    m = re.fullmatch(r"\s*```[a-zA-Z]*\n(.*?)\n?```\s*", block, re.S)
    return m.group(1) if m else block


def _trim_rule(block: str) -> str:
    """Drop a trailing horizontal rule that separates sections."""
    lines = block.rstrip().splitlines()
    while lines and re.fullmatch(r"\s*(-{3,}|\*{3,}|_{3,})\s*", lines[-1]):
        lines.pop()
    return "\n".join(lines).rstrip()


def _parse_table(lines: list[str]) -> list[dict[str, str]]:
    rows = [l.strip() for l in lines if l.strip().startswith("|")]
    if len(rows) < 2:
        return []
    split = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]
    header = split(rows[0])
    items = []
    for r in rows[1:]:
        cells = split(r)
        if all(re.fullmatch(r":?-{2,}:?", c or "--") for c in cells):
            continue
        items.append({header[i] if i < len(header) else f"column_{i + 1}": c for i, c in enumerate(cells)})
    return items


def _parse_items(block: str) -> tuple[list[dict[str, str]], str | None]:
    """Parse "- Field: value" groups (or a table) into items. Returns (items, note)."""
    lines = [l for l in block.splitlines()]
    content = [l for l in lines if l.strip() and not re.fullmatch(r"\s*(-{3,}|\*{3,}|_{3,})\s*", l)]
    content = [l for l in content if not re.match(r"^\s*for each( item)?\s*:?\s*$", l, re.I)]
    if not content:
        return [], "empty"
    if any(l.strip().startswith("|") for l in content):
        table = _parse_table(content)
        if table:
            return table, None
    items: list[dict[str, str]] = []
    current: dict[str, str] = {}
    last_key: str | None = None
    free_lines: list[str] = []
    for line in content:
        m = _FIELD.match(line)
        is_item_header = bool(re.match(r"^\s*(?:[-*+•]|\d+[.)]|#{1,6})?\s*\**\s*(item|medication|medicine|entry)\s*\d+\b",
                                       line, re.I))
        if is_item_header and not (m and m.group(2).strip()):
            if current:
                items.append(current)
            current, last_key = {}, None
            continue
        if m:
            key = re.sub(r"\s+", " ", m.group(1)).strip()
            value = m.group(2).strip().strip("*").strip()
            norm = key.lower()
            if norm in {k.lower() for k in current}:      # a repeated field starts the next item
                items.append(current)
                current = {}
            current[key] = value
            last_key = key
        elif last_key is not None and (line.startswith((" ", "\t")) and not _BULLET.match(line)):
            current[last_key] = (current[last_key] + "\n" + line.strip()).strip()   # wrapped value
        else:
            b = _BULLET.match(line)
            free_lines.append((b.group(1) if b else line).strip())
    if current:
        items.append(current)
    if not items:
        if free_lines and all(_NO_ITEMS.match(l) for l in free_lines):
            return [], "model stated no items"
        return [{"text": l} for l in free_lines], "unlabelled lines"
    # An item is the model saying "nothing here" when its verbatim field ("Original ...") is empty or
    # "N/A"; for items without such a field, when every value is.
    def _is_placeholder(it: dict[str, str]) -> bool:
        blank = lambda v: v.strip().lower().rstrip(".") in _EMPTY_VALUES
        verbatim = [v for k, v in it.items() if k.lower().startswith("original")]
        return all(blank(v) for v in verbatim) if verbatim else all(blank(v) for v in it.values())

    kept = [it for it in items if not _is_placeholder(it)]
    note = "model stated no items" if items and not kept else None
    return kept, note


def _split_categories(block: str) -> dict[str, str]:
    lines = block.splitlines()
    marks: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        hd = _heading_of(line)
        if hd:
            key = _match_category(hd[1])
            if key and key not in [k for _, k in marks]:
                marks.append((i, key))
    out: dict[str, str] = {}
    for n, (start, key) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(lines)
        out[key] = "\n".join(lines[start + 1:end]).strip("\n")
    return out


def count_markers(text: str) -> dict[str, int]:
    return {name: len(rx.findall(text or "")) for name, rx in UNCERTAINTY_MARKERS.items()}


def item_text(item: dict[str, Any]) -> str:
    """The verbatim field of an extracted item ("Original text", "Original value", ...)."""
    for key, value in item.items():
        if key.lower().startswith("original"):
            return str(value)
    if "text" in item:
        return str(item["text"])
    return str(next(iter(item.values()), ""))


def get_field(item: dict[str, Any], *names: str) -> str | None:
    """Case-insensitive prefix lookup of a field in a parsed item."""
    for name in names:
        n = name.lower()
        for key, value in item.items():
            if key.lower().startswith(n):
                return str(value)
    return None


def parse_markdown(text: str) -> dict[str, Any] | None:
    sections = _split_top_sections(text)
    if not sections:
        return None
    out: dict[str, Any] = {"sections_found": list(sections)}
    if "document_overview" in sections:
        items, _ = _parse_items(sections["document_overview"])
        merged: dict[str, str] = {}
        for it in items:
            merged.update(it)
        out["document_overview"] = merged
    if "verbatim_transcription" in sections:
        out["verbatim_transcription"] = _strip_code_fence(_trim_rule(sections["verbatim_transcription"]))
    if "structured_extraction" in sections:
        cats = _split_categories(sections["structured_extraction"])
        extraction: dict[str, list] = {}
        notes: dict[str, str] = {}
        for key in CATEGORY_KEYS:
            if key in cats:
                items, note = _parse_items(cats[key])
                extraction[key] = items
                if note:
                    notes[key] = note
        out["structured_extraction"] = extraction
        out["categories_found"] = [k for k in CATEGORY_KEYS if k in cats]
        out["category_notes"] = notes
    if "uncertainty_register" in sections:
        block = _trim_rule(sections["uncertainty_register"])
        items, note = _parse_items(block)
        out["uncertainty_register"] = {"items": items, "note": note, "text": block}
    if "final_validation" in sections:
        out["final_validation"] = _trim_rule(sections["final_validation"])
    return out


# --------------------------------------------------------------------------- entry point

def parse_output(raw: str) -> dict[str, Any]:
    """Build the structured record for one raw output. `raw` is not altered."""
    thinking, answer = split_thinking(raw)
    result: dict[str, Any] = {
        "parser": {"name": "rxbench.inference.output_parser", "version": PARSER_VERSION,
                   "note": "Deterministic rule-based parse of the raw output. No model was used and no "
                           "content was corrected, completed or added."},
        "reasoning_block_present": thinking is not None,
        "reasoning_block_chars": len(thinking) if thinking is not None else 0,
    }
    parsed_json, method = extract_json(answer)
    status: dict[str, Any] = {"json_valid": parsed_json is not None, "json_method": method,
                              "markdown_sections_found": [], "missing_sections": [],
                              "missing_categories": [], "method": "none",
                              "structured_parse_successful": False}
    if parsed_json is not None:
        status.update(method="json", structured_parse_successful=True)
        result["json"] = parsed_json
        if isinstance(parsed_json, dict):
            expected = [k for k, _, _ in TOP_SECTIONS]
            status["missing_sections"] = [k for k in expected if k not in parsed_json]
    else:
        md = parse_markdown(answer)
        if md:
            found = md.pop("sections_found")
            status["markdown_sections_found"] = found
            status["missing_sections"] = [k for k, _, _ in TOP_SECTIONS if k not in found]
            status["missing_categories"] = [k for k in CATEGORY_KEYS if k not in md.get("categories_found", [])]
            # Usable for comparison when at least the transcription or the extraction was located.
            status["structured_parse_successful"] = ("verbatim_transcription" in found
                                                     or "structured_extraction" in found)
            status["method"] = "markdown_sections" if status["structured_parse_successful"] else "none"
            md.pop("categories_found", None)
            result.update(md)
    result["parse_status"] = status
    transcription = result.get("verbatim_transcription")
    result["uncertainty_markers"] = {
        "in_transcription": count_markers(transcription) if isinstance(transcription, str) else None,
        "in_whole_output": count_markers(answer),
    }
    return result

"""Load the master prompt exactly as stored, and fingerprint it."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# Phrases that prove the printed / pre-filled text exclusion rule is present.
# Checked case-insensitively; all must be found.
PRINTED_TEXT_RULE_MARKERS = (
    "ignore pre-existing printed or pre-filled text",
    "do not transcribe or clinically classify pre-existing printed or pre-filled text",
    "mark its origin as uncertain",
    "do not ignore handwriting merely because it appears inside a printed field",
)


@dataclass(frozen=True)
class PromptRecord:
    text: str
    raw_bytes: bytes
    sha256: str
    source: str

    @property
    def n_chars(self) -> int:
        return len(self.text)

    @property
    def has_printed_text_exclusion(self) -> bool:
        low = " ".join(self.text.lower().split())
        return all(m in low for m in PRINTED_TEXT_RULE_MARKERS)


def prompt_from_bytes(raw: bytes, source: str = "<memory>") -> PromptRecord:
    """The prompt text is the UTF-8 decoding of the file bytes: no stripping, no rewriting."""
    if not raw.strip():
        raise ValueError(f"Master prompt is empty: {source}")
    text = raw.decode("utf-8")  # raises on invalid UTF-8 rather than guessing
    return PromptRecord(text=text, raw_bytes=raw, sha256=hashlib.sha256(raw).hexdigest(), source=source)


def load_prompt(path: Path | str) -> PromptRecord:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Master prompt file not found: {path}")
    return prompt_from_bytes(path.read_bytes(), source=str(path))

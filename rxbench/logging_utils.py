"""Application logging.

Privacy rule: log lines carry experiment IDs, model keys, hashes, counts and
timings only. Never log prompt text, model output, file names supplied by the
user, or any other document content. Use `short_hash` when a reference to a
document is needed.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOGGER_NAME = "rxbench"
_configured_file: Path | None = None


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def add_file_log(log_dir: Path | None) -> Path | None:
    """Attach a local log file (idempotent). Returns the file path."""
    global _configured_file
    if log_dir is None:
        return None
    log_dir = Path(log_dir)
    target = log_dir / "rxbench.log"
    if _configured_file == target:
        return target
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(target, encoding="utf-8")
    except OSError:
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger = get_logger()
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            h.close()
    logger.addHandler(handler)
    _configured_file = target
    return target


def short_hash(sha256_hex: str | None) -> str:
    return (sha256_hex or "unknown")[:12]

"""Local experiment storage: one directory per experiment (one document page).

    experiments/
        EXP_0001/
            experiment.json
            input/original_prescription.<ext>      exact uploaded bytes
            input/page_0003.png                    only for PDFs: the rendered page
            prompt/master_prompt_snapshot.txt      exact prompt bytes
            prompt/prompt_hash.txt
            <model_key>/raw_output.txt             authoritative record
            <model_key>/raw_output_with_special_tokens.txt
            <model_key>/structured_output.json
            <model_key>/metadata.json
            <model_key>/history/<timestamp>/...    earlier runs, never deleted by a rerun
            comparison/comparison_report.{json,md}
            evaluation/...
        _batches/BATCH_*.json

Everything stays on the local file system.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..inference import image_loader
from ..inference.image_loader import PreparedImage
from ..inference.prompt_loader import PromptRecord, prompt_from_bytes

EXP_PATTERN = re.compile(r"^EXP_(\d{4,})$")
RESERVED_DIRS = {"input", "prompt", "comparison", "evaluation"}
RAW_FILE = "raw_output.txt"
RAW_SPECIAL_FILE = "raw_output_with_special_tokens.txt"
STRUCTURED_FILE = "structured_output.json"
METADATA_FILE = "metadata.json"


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def write_text_exact(path: Path, text: str) -> None:
    """Write text without newline translation so the stored bytes equal the UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def read_text_exact(path: Path) -> str:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class Experiment:
    exp_id: str
    path: Path
    record: dict[str, Any]

    @property
    def model_input_path(self) -> Path:
        return self.path / "input" / self.record["input"]["model_input_file"]

    @property
    def original_path(self) -> Path:
        return self.path / "input" / self.record["input"]["stored_original_file"]


class ExperimentManager:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ creation
    def _allocate(self) -> tuple[str, Path]:
        existing = [int(m.group(1)) for p in self.root.iterdir() if (m := EXP_PATTERN.match(p.name))]
        n = max(existing, default=0) + 1
        while True:
            exp_id = f"EXP_{n:04d}"
            path = self.root / exp_id
            try:
                path.mkdir(exist_ok=False)
                return exp_id, path
            except FileExistsError:   # another process took the number
                n += 1

    def create_experiment(self, *, source_bytes: bytes, original_filename: str, prompt: PromptRecord,
                          page_index: int | None = None, apply_exif_orientation: bool = True,
                          pdf_render_dpi: int = 200, batch_id: str | None = None,
                          notes: str | None = None) -> Experiment:
        """Store the original document and the prompt snapshot for one page."""
        kind = image_loader.detect_kind(original_filename, source_bytes)
        ext = "." + original_filename.rsplit(".", 1)[-1].lower()
        exp_id, path = self._allocate()
        (path / "input").mkdir()
        (path / "prompt").mkdir()
        source_sha = image_loader.sha256_hex(source_bytes)
        transformations: list[dict[str, Any]] = []

        if kind == "pdf":
            n_pages = image_loader.pdf_page_count(source_bytes)
            page_index = 0 if page_index is None else page_index
            stored_original = "original_document.pdf"
            (path / "input" / stored_original).write_bytes(source_bytes)
            rendered = image_loader.render_pdf_page(source_bytes, page_index, pdf_render_dpi)
            page_bytes = image_loader.png_bytes(rendered)
            model_input_file = f"page_{page_index + 1:04d}.png"
            (path / "input" / model_input_file).write_bytes(page_bytes)
            model_input_sha = image_loader.sha256_hex(page_bytes)
            transformations.append({"operation": "pdf_page_render",
                                    "detail": f"page {page_index + 1} of {n_pages} rendered with pypdfium2 at "
                                              f"{pdf_render_dpi} dpi to lossless PNG"})
            pdf_info = {"page_index": page_index, "page_number": page_index + 1, "page_count": n_pages,
                        "render_dpi": pdf_render_dpi}
        else:
            stored_original = f"original_prescription{ext}"
            (path / "input" / stored_original).write_bytes(source_bytes)
            model_input_file = stored_original
            model_input_sha = source_sha
            pdf_info = None

        prepared = image_loader.prepare_image_from_bytes((path / "input" / model_input_file).read_bytes(),
                                                         apply_exif_orientation)
        (path / "prompt" / "master_prompt_snapshot.txt").write_bytes(prompt.raw_bytes)
        write_text_exact(path / "prompt" / "prompt_hash.txt", prompt.sha256 + "\n")

        record = {
            "experiment_id": exp_id,
            "created_at": utc_now(),
            "batch_id": batch_id,
            "notes": notes,
            "input": {
                "original_filename": original_filename,
                "kind": kind,
                "stored_original_file": stored_original,
                "original_sha256": source_sha,
                "original_size_bytes": len(source_bytes),
                "model_input_file": model_input_file,
                "model_input_file_sha256": model_input_sha,
                "pdf": pdf_info,
                "apply_exif_orientation": apply_exif_orientation,
                "image": prepared.describe(),
                "transformations": transformations + prepared.transformations,
            },
            "prompt": {"sha256": prompt.sha256, "n_chars": prompt.n_chars, "source": prompt.source,
                       "has_printed_text_exclusion": prompt.has_printed_text_exclusion},
            "protocol": {"zero_shot": True, "ocr": False, "retrieval": False, "medicine_catalogue": False,
                         "model_to_model_communication": False, "automatic_correction": False,
                         "consensus_output": False},
        }
        write_json(path / "experiment.json", record)
        return Experiment(exp_id, path, record)

    # ------------------------------------------------------------------ reading
    def exists(self, exp_id: str) -> bool:
        return bool(EXP_PATTERN.match(exp_id)) and (self.root / exp_id / "experiment.json").exists()

    def get(self, exp_id: str) -> Experiment:
        if not EXP_PATTERN.match(exp_id):
            raise ValueError(f"Invalid experiment id: {exp_id!r}")
        path = self.root / exp_id
        if not (path / "experiment.json").exists():
            raise FileNotFoundError(f"Experiment not found: {exp_id}")
        return Experiment(exp_id, path, read_json(path / "experiment.json"))

    def list_ids(self) -> list[str]:
        ids = [p.name for p in self.root.iterdir()
               if EXP_PATTERN.match(p.name) and (p / "experiment.json").exists()]
        return sorted(ids, key=lambda s: int(EXP_PATTERN.match(s).group(1)))

    def list_experiments(self) -> list[Experiment]:
        return [self.get(i) for i in self.list_ids()]

    def load_prompt(self, exp: Experiment) -> PromptRecord:
        """Read the prompt snapshot and check it against the recorded hash."""
        raw = (exp.path / "prompt" / "master_prompt_snapshot.txt").read_bytes()
        record = prompt_from_bytes(raw, source=f"{exp.exp_id}/prompt/master_prompt_snapshot.txt")
        if record.sha256 != exp.record["prompt"]["sha256"]:
            raise ValueError(f"{exp.exp_id}: prompt snapshot does not match its recorded hash")
        return record

    def load_prepared_image(self, exp: Experiment) -> PreparedImage:
        """Decode the stored page, verifying that the stored bytes are unchanged."""
        data = exp.model_input_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != exp.record["input"]["model_input_file_sha256"]:
            raise ValueError(f"{exp.exp_id}: stored input file does not match its recorded hash")
        return image_loader.prepare_image_from_bytes(data, exp.record["input"]["apply_exif_orientation"])

    # ------------------------------------------------------------------ model runs
    def model_dir(self, exp: Experiment, model_key: str) -> Path:
        if model_key in RESERVED_DIRS or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", model_key):
            raise ValueError(f"Invalid model key: {model_key!r}")
        return exp.path / model_key

    def model_keys(self, exp: Experiment) -> list[str]:
        return sorted(p.name for p in exp.path.iterdir()
                      if p.is_dir() and p.name not in RESERVED_DIRS and (p / METADATA_FILE).exists())

    def _archive_existing(self, mdir: Path) -> str | None:
        files = [mdir / n for n in (RAW_FILE, RAW_SPECIAL_FILE, STRUCTURED_FILE, METADATA_FILE)
                 if (mdir / n).exists()]
        if not files:
            return None
        stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S%f")
        dest = mdir / "history" / stamp
        dest.mkdir(parents=True)
        for f in files:
            shutil.move(str(f), str(dest / f.name))
        return stamp

    def save_model_run(self, exp: Experiment, model_key: str, *, metadata: dict[str, Any],
                       raw_text: str | None = None, raw_text_with_special_tokens: str | None = None,
                       structured: dict[str, Any] | None = None) -> Path:
        """Write one model's result. A previous result is moved to history/, not overwritten."""
        mdir = self.model_dir(exp, model_key)
        mdir.mkdir(exist_ok=True)
        archived = self._archive_existing(mdir)
        if archived:
            metadata = {**metadata, "previous_run_archived_as": f"history/{archived}"}
        if raw_text is not None:
            write_text_exact(mdir / RAW_FILE, raw_text)
        if raw_text_with_special_tokens is not None:
            write_text_exact(mdir / RAW_SPECIAL_FILE, raw_text_with_special_tokens)
        if structured is not None:
            write_json(mdir / STRUCTURED_FILE, structured)
        write_json(mdir / METADATA_FILE, metadata)
        return mdir

    def update_metadata(self, exp: Experiment, model_key: str, patch: dict[str, Any]) -> None:
        path = self.model_dir(exp, model_key) / METADATA_FILE
        if path.exists():
            meta = read_json(path)
            meta.update(patch)
            write_json(path, meta)

    def load_model_run(self, exp: Experiment, model_key: str) -> dict[str, Any]:
        mdir = self.model_dir(exp, model_key)
        out: dict[str, Any] = {"model_key": model_key, "dir": mdir, "metadata": None, "raw_text": None,
                               "raw_text_with_special_tokens": None, "structured": None}
        if (mdir / METADATA_FILE).exists():
            out["metadata"] = read_json(mdir / METADATA_FILE)
        if (mdir / RAW_FILE).exists():
            out["raw_text"] = read_text_exact(mdir / RAW_FILE)
        if (mdir / RAW_SPECIAL_FILE).exists():
            out["raw_text_with_special_tokens"] = read_text_exact(mdir / RAW_SPECIAL_FILE)
        if (mdir / STRUCTURED_FILE).exists():
            out["structured"] = read_json(mdir / STRUCTURED_FILE)
        return out

    def history(self, exp: Experiment, model_key: str) -> list[str]:
        h = self.model_dir(exp, model_key) / "history"
        return sorted(p.name for p in h.iterdir() if p.is_dir()) if h.exists() else []

    # ------------------------------------------------------------------ batches
    def new_batch_id(self) -> str:
        return "BATCH_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_batch(self, batch_id: str, record: dict[str, Any]) -> Path:
        path = self.root / "_batches" / f"{batch_id}.json"
        write_json(path, record)
        return path

    def list_batches(self) -> list[dict[str, Any]]:
        d = self.root / "_batches"
        return [read_json(p) for p in sorted(d.glob("BATCH_*.json"))] if d.exists() else []

    # ------------------------------------------------------------------ deletion
    def delete(self, exp_id: str) -> None:
        """Delete one experiment directory. Refuses anything that is not an experiment under the root."""
        exp = self.get(exp_id)
        resolved = exp.path.resolve()
        if resolved.parent != self.root.resolve() or not EXP_PATTERN.match(resolved.name):
            raise ValueError(f"Refusing to delete {resolved}")
        shutil.rmtree(resolved)

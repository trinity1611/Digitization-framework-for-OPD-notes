"""Model download, local availability and verification.

Rules enforced here:
* only the registered repository and pinned revision are ever downloaded;
* a full download needs explicit confirmation (`confirmed=True`);
* disk space is checked first;
* interrupted downloads resume (huggingface_hub keeps partial files);
* the resolved commit hash is stored;
* an already complete snapshot is never downloaded again.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

from ..logging_utils import get_logger
from ..storage.experiment_manager import read_json, utc_now, write_json
from .registry import ModelSpec

log = get_logger()

# Small files needed to build the processor and chat template (no weights).
CONFIG_PATTERNS = ["*.json", "*.jinja", "*.txt", "*.model", "tokenizer*", "LICENSE*", "README.md"]


class ConfirmationRequired(RuntimeError):
    """Raised when a full download is requested without explicit confirmation."""

    def __init__(self, plan: dict[str, Any]):
        super().__init__("Explicit confirmation is required before downloading model weights")
        self.plan = plan


class DownloadError(RuntimeError):
    pass


def _hub():
    import huggingface_hub
    import huggingface_hub.constants  # noqa: F401  (submodule is lazily exposed)
    return huggingface_hub


def _hub_cache() -> str:
    from huggingface_hub import constants
    return str(constants.HF_HUB_CACHE)


class DownloadManager:
    def __init__(self, state_dir: Path | str, cache_dir: Path | str | None = None):
        self.state_dir = Path(state_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.state_file = self.state_dir / "models_state.json"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ state
    def _read_state(self) -> dict[str, Any]:
        try:
            return read_json(self.state_file)
        except (OSError, ValueError):
            return {}

    def get_state(self, key: str) -> dict[str, Any]:
        return self._read_state().get(key, {})

    def update_state(self, key: str, **fields: Any) -> None:
        with self._lock:
            state = self._read_state()
            state.setdefault(key, {}).update(fields)
            try:
                write_json(self.state_file, state)
            except OSError as exc:   # read-only project directory must not break inference
                log.warning("could not write model state file: %s", type(exc).__name__)

    # ------------------------------------------------------------------ locations
    def effective_cache_dir(self) -> Path:
        if self.cache_dir:
            return self.cache_dir
        return Path(_hub_cache())

    def repo_cache_dir(self, spec: ModelSpec) -> Path:
        return self.effective_cache_dir() / ("models--" + spec.repo_id.replace("/", "--"))

    def local_snapshot_path(self, spec: ModelSpec) -> Path | None:
        """Path of the pinned revision's snapshot if present locally (no network)."""
        hub = _hub()
        try:
            path = hub.snapshot_download(spec.repo_id, revision=spec.revision, cache_dir=self.cache_dir,
                                         local_files_only=True)
            return Path(path)
        except Exception:
            return None

    @staticmethod
    def inspect_snapshot(path: Path) -> dict[str, Any]:
        """Classify a snapshot directory: complete / config_only / partial."""
        info: dict[str, Any] = {"path": str(path), "revision": path.name, "has_config": (path / "config.json").exists(),
                                "weight_files_expected": [], "weight_files_missing": [], "weights_bytes": 0}
        index = path / "model.safetensors.index.json"
        if index.exists():
            try:
                shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
            except (ValueError, KeyError):
                shards = []
            info["weight_files_expected"] = shards
        elif (path / "model.safetensors").exists():
            info["weight_files_expected"] = ["model.safetensors"]
        for name in info["weight_files_expected"]:
            f = path / name
            if f.exists():
                info["weights_bytes"] += f.stat().st_size
            else:
                info["weight_files_missing"].append(name)
        has_tokenizer = any((path / n).exists() for n in ("tokenizer.json", "tokenizer.model", "vocab.json"))
        info["has_tokenizer"] = has_tokenizer
        if info["has_config"] and info["weight_files_expected"] and not info["weight_files_missing"]:
            info["status"] = "complete"
        elif info["has_config"] and has_tokenizer and len(info["weight_files_missing"]) == len(
                info["weight_files_expected"]):
            info["status"] = "config_only"
        elif not info["has_config"] and info["weights_bytes"] == 0:
            info["status"] = "not_downloaded"      # only incidental files such as a cached README
        else:
            info["status"] = "partial"
        return info

    def local_status(self, spec: ModelSpec) -> dict[str, Any]:
        path = self.local_snapshot_path(spec)
        state = self.get_state(spec.key)
        if path is None:
            partial = self.downloaded_bytes(spec) > 50 * 1024**2      # weight data present, snapshot incomplete
            return {"status": "partial" if partial else "not_downloaded", "path": None, "revision": None,
                    "state": state}
        info = self.inspect_snapshot(path)
        info["state"] = state
        if spec.revision and len(spec.revision) == 40 and info["revision"] != spec.revision:
            info["status"] = "revision_mismatch"
        return info

    def is_available(self, spec: ModelSpec) -> bool:
        return self.local_status(spec)["status"] == "complete"

    def downloaded_bytes(self, spec: ModelSpec) -> int:
        """Bytes present for this repository, including incomplete files (for progress display)."""
        blobs = self.repo_cache_dir(spec) / "blobs"
        total = 0
        if blobs.exists():
            for f in blobs.iterdir():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total

    # ------------------------------------------------------------------ remote
    def remote_info(self, spec: ModelSpec) -> dict[str, Any]:
        """Repository facts from the Hub (network). Never raises; errors are reported."""
        hub = _hub()
        try:
            info = hub.HfApi().model_info(spec.repo_id, revision=spec.revision, files_metadata=True)
            files = [{"name": s.rfilename, "size": s.size or 0,
                      "sha256": (s.lfs.sha256 if getattr(s, "lfs", None) else None)} for s in info.siblings or []]
            return {"ok": True, "sha": info.sha, "gated": info.gated, "files": files,
                    "total_bytes": sum(f["size"] for f in files),
                    "pipeline_tag": info.pipeline_tag, "last_modified": str(info.last_modified)}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    def access_status(self, spec: ModelSpec) -> dict[str, Any]:
        """Whether the current Hugging Face credentials can download this repository."""
        hub = _hub()
        out: dict[str, Any] = {"token_present": bool(hub.get_token()), "gated": spec.gated}
        try:
            hub.HfApi().auth_check(spec.repo_id)
            out.update(ok=True, detail="Access confirmed.")
        except Exception as exc:
            name = type(exc).__name__
            if name == "GatedRepoError":
                detail = (f"Access not granted. Open {spec.repo_url}, accept the licence terms, then "
                          "authenticate with `hf auth login`.")
            elif name == "RepositoryNotFoundError":
                detail = "Repository not found or not visible with the current credentials."
            else:
                detail = f"Could not check access ({name}). The machine may be offline."
            out.update(ok=False, detail=detail, error=name)
        return out

    # ------------------------------------------------------------------ download
    def plan(self, spec: ModelSpec, remote: dict[str, Any] | None = None) -> dict[str, Any]:
        remote = remote if remote is not None else self.remote_info(spec)
        size = remote["total_bytes"] if remote.get("ok") else int((spec.download_size_gb or 0) * 1e9)
        cache = self.effective_cache_dir()
        probe = cache
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        free = shutil.disk_usage(probe).free
        present = self.downloaded_bytes(spec)
        remaining = max(0, size - present)
        return {
            "model_key": spec.key, "display_name": spec.display_name, "repo_id": spec.repo_id,
            "repo_url": spec.repo_url, "revision": spec.revision, "licence": spec.licence,
            "licence_url": spec.licence_url, "gated": spec.gated,
            "access_requirements": spec.access_requirements,
            "download_bytes": size, "already_present_bytes": present, "remaining_bytes": remaining,
            "size_source": "Hugging Face Hub" if remote.get("ok") else "registry estimate (Hub not reachable)",
            "cache_dir": str(cache), "free_bytes": free,
            "enough_space": free >= remaining * 1.02 + 2 * 1024**3,
            "local_status": self.local_status(spec)["status"],
            "remote_error": None if remote.get("ok") else remote.get("error"),
        }

    def download(self, spec: ModelSpec, confirmed: bool = False, config_only: bool = False,
                 max_workers: int = 4, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Download the pinned revision. Weights need `confirmed=True`.

        `config_only=True` fetches just the processor/tokenizer/config files
        (tens of MB) for a processor-level preflight; it needs no confirmation.
        """
        hub = _hub()
        status = self.local_status(spec)
        if status["status"] == "complete" or (config_only and status["status"] == "config_only"):
            return {"downloaded": False, "reason": "already present", **status}
        if spec.repo_id in spec.forbidden_repo_ids:
            raise DownloadError(f"{spec.repo_id} is a forbidden look-alike for '{spec.key}'")
        plan = self.plan(spec) if not config_only else None
        if not config_only:
            if not confirmed:
                raise ConfirmationRequired(plan)
            if not plan["enough_space"]:
                raise DownloadError(
                    f"Not enough disk space in {plan['cache_dir']}: {plan['remaining_bytes'] / 1e9:.1f} GB "
                    f"still to download, {plan['free_bytes'] / 1e9:.1f} GB free")
        if progress:
            progress(f"downloading {spec.repo_id}@{(spec.revision or 'main')[:12]}"
                     + (" (config files only)" if config_only else ""))
        self.update_state(spec.key, download_status="in_progress", download_started_at=utc_now())
        try:
            path = hub.snapshot_download(spec.repo_id, revision=spec.revision, cache_dir=self.cache_dir,
                                         allow_patterns=CONFIG_PATTERNS if config_only else None,
                                         max_workers=max_workers)
        except Exception as exc:
            name = type(exc).__name__
            self.update_state(spec.key, download_status="failed", download_error=name)
            hint = ""
            if name == "GatedRepoError":
                hint = f" Accept the licence at {spec.repo_url} and run `hf auth login`."
            elif name in {"LocalEntryNotFoundError", "OfflineModeIsEnabled"}:
                hint = " The Hub is not reachable (offline mode or no network)."
            raise DownloadError(f"Download of {spec.repo_id} failed: {name}: {str(exc)[:300]}.{hint} "
                                "Re-run the same command to resume.") from exc
        info = self.inspect_snapshot(Path(path))
        self.update_state(spec.key, download_status=info["status"], revision=info["revision"],
                          download_finished_at=utc_now(), download_error=None, snapshot_path=str(path))
        log.info("download finished model=%s revision=%s status=%s", spec.key, info["revision"][:12], info["status"])
        return {"downloaded": True, **info}

    # ------------------------------------------------------------------ verification
    def verify(self, spec: ModelSpec, check_remote_sizes: bool = False, deep_hash: bool = False) -> dict[str, Any]:
        """Verify the local snapshot without loading it onto a GPU.

        Checks: all weight shards present; every shard opens as a valid
        safetensors file; config and processor load. Optionally compares file
        sizes (and SHA-256 with `deep_hash`) with the Hub's records.
        """
        result: dict[str, Any] = {"model_key": spec.key, "checked_at": utc_now(), "checks": [], "ok": False}
        add = lambda name, ok, detail="": result["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        path = self.local_snapshot_path(spec)
        if path is None:
            add("snapshot_present", False, "pinned revision not found in the local cache")
            self.update_state(spec.key, verification=result)
            return result
        info = self.inspect_snapshot(path)
        result["revision"] = info["revision"]
        add("snapshot_present", True, str(path))
        add("revision_matches_registry", not (spec.revision and len(spec.revision) == 40) or
            info["revision"] == spec.revision, info["revision"])
        add("all_weight_files_present", info["status"] == "complete",
            f"{len(info['weight_files_expected']) - len(info['weight_files_missing'])}/"
            f"{len(info['weight_files_expected'])} shards; missing: {info['weight_files_missing'][:3]}")
        if info["status"] == "complete":
            try:
                from safetensors import safe_open
                tensors = 0
                for name in info["weight_files_expected"]:
                    with safe_open(str(path / name), framework="pt") as f:
                        tensors += len(list(f.keys()))
                add("safetensors_headers_valid", True, f"{tensors} tensors in {len(info['weight_files_expected'])} files")
            except Exception as exc:
                add("safetensors_headers_valid", False, f"{type(exc).__name__}: {str(exc)[:200]}")
        try:
            cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
            arch = (cfg.get("architectures") or [None])[0]
            add("architecture_matches_registry", arch == spec.architecture, f"config.json: {arch}")
        except Exception as exc:
            add("architecture_matches_registry", False, f"{type(exc).__name__}")
        try:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(str(path), local_files_only=True)
            add("processor_loads", type(proc).__name__ == spec.processor_class, type(proc).__name__)
        except Exception as exc:
            add("processor_loads", False, f"{type(exc).__name__}: {str(exc)[:200]}")
        if check_remote_sizes or deep_hash:
            remote = self.remote_info(spec)
            if remote.get("ok"):
                bad = [f["name"] for f in remote["files"]
                       if (path / f["name"]).exists() and (path / f["name"]).stat().st_size != f["size"]]
                add("file_sizes_match_hub", not bad, f"mismatched: {bad[:3]}" if bad else "all sizes match")
                if deep_hash:
                    import hashlib
                    wrong = []
                    for f in remote["files"]:
                        if f["sha256"] and (path / f["name"]).exists():
                            h = hashlib.sha256()
                            with open(path / f["name"], "rb") as fh:
                                for chunk in iter(lambda: fh.read(8 << 20), b""):
                                    h.update(chunk)
                            if h.hexdigest() != f["sha256"]:
                                wrong.append(f["name"])
                    add("sha256_match_hub", not wrong, f"mismatched: {wrong[:3]}" if wrong else "all hashes match")
            else:
                add("file_sizes_match_hub", False, f"Hub not reachable: {remote.get('error')}")
        result["ok"] = all(c["ok"] for c in result["checks"])
        result["note"] = ("File-level verification only. Whether the model loads and generates is recorded "
                          "separately under load status after a real run.")
        self.update_state(spec.key, verification=result, revision=info["revision"])
        return result

    def record_load(self, spec: ModelSpec, ok: bool, detail: dict[str, Any]) -> None:
        self.update_state(spec.key, last_load={"ok": ok, "at": utc_now(), **detail})

    def manual_download_command(self, spec: ModelSpec) -> str:
        rev = f" --revision {spec.revision}" if spec.revision else ""
        cache = f' --cache-dir "{self.cache_dir}"' if self.cache_dir else ""
        return f"hf download {spec.repo_id}{rev}{cache}"


def hf_environment() -> dict[str, Any]:
    """Relevant Hugging Face environment settings (token value is never returned)."""
    hub = _hub()
    return {"HF_HOME": os.environ.get("HF_HOME"), "HF_HUB_CACHE": _hub_cache(),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"), "token_present": bool(hub.get_token()),
            "huggingface_hub_version": hub.__version__}

"""Offline verification.

`NetworkGuard` makes any attempt to open a non-loopback connection (or resolve
a host name) inside this process raise an error and be recorded. Running the
application's inference path under the guard therefore tests two things at
once: that inference works without the network, and that it does not even try
to use it.

The guard isolates this Python process. It is in addition to, not a
replacement for, running on a machine whose network is actually disconnected;
the report states which situation applied.
"""
from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

from .inference.generation import GenerationConfig
from .inference.prompt_loader import load_prompt
from .storage.experiment_manager import utc_now

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


class OfflineViolation(OSError):
    """Raised when code tries to use the network while the guard is active."""


def _is_local(address: Any) -> bool:
    if isinstance(address, (str, bytes)):          # AF_UNIX path
        return True
    if isinstance(address, tuple) and address:
        host = str(address[0])
        return host in _LOOPBACK_HOSTS or host.startswith("127.")
    return False


class NetworkGuard:
    """Context manager that blocks and records outbound network use in this process."""

    def __init__(self):
        self.attempts: list[str] = []
        self._orig: dict[str, Any] = {}

    def __enter__(self) -> "NetworkGuard":
        self._orig = {"connect": socket.socket.connect, "connect_ex": socket.socket.connect_ex,
                      "getaddrinfo": socket.getaddrinfo, "create_connection": socket.create_connection}
        guard = self

        def connect(sock, address, *a, **k):
            if _is_local(address):
                return guard._orig["connect"](sock, address, *a, **k)
            guard.attempts.append(f"connect {address!r}")
            raise OfflineViolation(f"network access blocked by offline guard: {address!r}")

        def connect_ex(sock, address, *a, **k):
            if _is_local(address):
                return guard._orig["connect_ex"](sock, address, *a, **k)
            guard.attempts.append(f"connect_ex {address!r}")
            raise OfflineViolation(f"network access blocked by offline guard: {address!r}")

        def getaddrinfo(host, *a, **k):
            if host is None or str(host) in _LOOPBACK_HOSTS or str(host).startswith("127."):
                return guard._orig["getaddrinfo"](host, *a, **k)
            guard.attempts.append(f"resolve {host!r}")
            raise OfflineViolation(f"name resolution blocked by offline guard: {host!r}")

        def create_connection(address, *a, **k):
            if _is_local(address):
                return guard._orig["create_connection"](address, *a, **k)
            guard.attempts.append(f"create_connection {address!r}")
            raise OfflineViolation(f"network access blocked by offline guard: {address!r}")

        socket.socket.connect = connect
        socket.socket.connect_ex = connect_ex
        socket.getaddrinfo = getaddrinfo
        socket.create_connection = create_connection
        return self

    def __exit__(self, *exc) -> None:
        socket.socket.connect = self._orig["connect"]
        socket.socket.connect_ex = self._orig["connect_ex"]
        socket.getaddrinfo = self._orig["getaddrinfo"]
        socket.create_connection = self._orig["create_connection"]


def probe_network(host: str = "huggingface.co", port: int = 443, timeout: float = 3.0) -> dict[str, Any]:
    """Is the outside network reachable right now? Opens and closes one TCP connection; sends no data."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return {"reachable": True, "target": f"{host}:{port}"}
    except OSError as exc:
        return {"reachable": False, "target": f"{host}:{port}", "detail": type(exc).__name__}


def run_offline_check(*, cfg: dict[str, Any], registry, download_manager, manager, prompt_path: Path,
                      model_keys: list[str], image_path: Path | None = None, inference: bool = False,
                      probe: bool = True, hardware_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the local pipeline under the network guard and report what was actually verified."""
    report: dict[str, Any] = {"checked_at": utc_now(), "models": {}, "steps": []}
    step = lambda name, ok, detail="": report["steps"].append({"step": name, "ok": bool(ok), "detail": detail})
    report["network_probe"] = probe_network() if probe else {"reachable": None, "detail": "probe skipped"}
    reachable = report["network_probe"]["reachable"]
    report["isolation"] = ("machine network unreachable AND process-level guard active" if reachable is False else
                           "process-level guard only (the machine itself still has network access)"
                           if reachable else "process-level guard (machine network state not probed)")

    with NetworkGuard() as guard:
        try:
            prompt = load_prompt(prompt_path)
            step("load_master_prompt", True, f"sha256 {prompt.sha256[:12]}")
            step("printed_text_exclusion_rule_present", prompt.has_printed_text_exclusion)
        except Exception as exc:
            step("load_master_prompt", False, f"{type(exc).__name__}: {exc}")
            prompt = None

        exp = None
        if image_path is not None and prompt is not None:
            try:
                exp = manager.create_experiment(
                    source_bytes=Path(image_path).read_bytes(), original_filename=Path(image_path).name,
                    prompt=prompt, notes="offline verification run",
                    apply_exif_orientation=cfg["image"]["apply_exif_orientation"],
                    pdf_render_dpi=cfg["image"]["pdf_render_dpi"])
                step("store_document_locally", True, exp.exp_id)
            except Exception as exc:
                step("store_document_locally", False, f"{type(exc).__name__}: {exc}")

        gen = GenerationConfig.from_dict(cfg.get("generation"))
        from .config import model_config
        from .inference.engine import InferenceEngine
        for key in model_keys:
            spec = registry.get(key)
            entry: dict[str, Any] = {"is_mock": spec.is_mock, "inference_offline_verified": False}
            before = len(guard.attempts)
            try:
                status = ({"status": "complete"} if spec.is_mock else download_manager.local_status(spec))
                entry["local_status"] = status["status"]
                if spec.is_mock:
                    entry["processor_offline"] = "not applicable (mock)"
                elif status["status"] in ("complete", "config_only") and exp is not None:
                    adapter = registry.create_adapter(key, model_config(cfg, key), download_manager,
                                                      cfg.get("execution", {}))
                    info = adapter.preflight(manager.load_prepared_image(exp), prompt, gen)
                    entry["processor_offline"] = (f"ok: {info['n_image_tokens']} image tokens, "
                                                  f"{info['n_input_tokens']} input tokens")
                else:
                    entry["processor_offline"] = "not tested (files not present locally or no image given)"
                if inference and exp is not None and status["status"] == "complete":
                    engine = InferenceEngine(cfg, registry, manager, download_manager, hardware_report)
                    engine.run([exp.exp_id], [key], gen)
                    run = manager.load_model_run(exp, key)
                    done = (run["metadata"] or {}).get("completion_status")
                    entry["inference_status"] = done
                    entry["inference_offline_verified"] = (done in ("complete", "incomplete_truncated")
                                                           and len(guard.attempts) == before)
                    if (run["metadata"] or {}).get("errors"):
                        entry["inference_errors"] = run["metadata"]["errors"]
                elif inference:
                    entry["inference_status"] = "not run (weights not downloaded or no image given)"
                else:
                    entry["inference_status"] = "not requested"
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            entry["network_attempts_during_model"] = guard.attempts[before:]
            report["models"][key] = entry
        report["network_attempts"] = list(guard.attempts)

    report["no_network_use_attempted"] = not report["network_attempts"]
    real = {k: v for k, v in report["models"].items() if not v["is_mock"]}
    report["real_models_inference_verified_offline"] = [k for k, v in real.items() if v["inference_offline_verified"]]
    report["real_models_not_verified"] = [k for k, v in real.items() if not v["inference_offline_verified"]]
    report["offline_experiment_id"] = exp.exp_id if exp else None
    report["conclusion"] = (
        "Offline inference verified for: " + ", ".join(report["real_models_inference_verified_offline"])
        if report["real_models_inference_verified_offline"] else
        "Offline inference has NOT been verified for any real model in this run "
        "(no model weights were run under the guard).")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Offline verification report", "", f"Checked at {report['checked_at']}.", "",
             f"- Isolation: {report['isolation']}.",
             f"- Network probe: {report['network_probe']}.",
             f"- Network use attempted by the application: "
             f"{'none' if report['no_network_use_attempted'] else report['network_attempts']}.",
             f"- **{report['conclusion']}**", "", "| step | result | detail |", "|---|---|---|"]
    for s in report["steps"]:
        lines.append(f"| {s['step']} | {'ok' if s['ok'] else 'FAILED'} | {s['detail']} |")
    lines += ["", "| model | mock | local files | processor offline | inference | offline inference verified |",
              "|---|---|---|---|---|---|"]
    for key, m in report["models"].items():
        lines.append(f"| {key} | {'yes' if m['is_mock'] else 'no'} | {m.get('local_status')} | "
                     f"{m.get('processor_offline')} | {m.get('inference_status', m.get('error'))} | "
                     f"{'yes' if m['inference_offline_verified'] else 'no'} |")
    lines.append("")
    return "\n".join(lines)

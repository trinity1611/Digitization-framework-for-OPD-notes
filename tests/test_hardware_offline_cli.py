import json
import re
import socket

import pytest

from rxbench import cli, offline
from rxbench.config import PROJECT_ROOT
from rxbench.hardware import detect, memory
from rxbench.hardware.diagnostics import render_markdown

SRC = PROJECT_ROOT / "rxbench"
APP = PROJECT_ROOT / "app"


# ---------------------------------------------------------------- hardware

def test_detect_all_on_this_machine():
    report = detect.detect_all({"project": PROJECT_ROOT})
    for key in ("python", "os", "cpu", "system_memory", "storage", "nvidia_smi", "torch", "gb10",
                "nvidia_software", "scheduler", "packages", "compatibility"):
        assert key in report
    assert report["cpu"]["architecture_family"] in ("x86_64", "arm64")
    assert report["storage"]["project"]["free_bytes"] > 0
    md = render_markdown(report, "test")
    assert "## GB10 identification" in md and "## System memory" in md and "## Compatibility assessment" in md
    summary = detect.summary_for_metadata(report)
    for key in ("gpu_name", "cpu_architecture", "python_version", "torch_version", "cuda_version",
                "nvidia_driver_version", "unified_memory", "is_gb10"):
        assert key in summary


def test_nvidia_smi_not_supported_values():
    assert detect._num("[N/A]") is None and detect._num("[Not Supported]") is None and detect._num("") is None
    assert detect._num(" 8188 ") == 8188.0


def test_gb10_is_identified_from_evidence_not_assumed():
    arm = {"architecture": "aarch64", "architecture_family": "arm64"}
    x86 = {"architecture": "x86_64", "architecture_family": "x86_64"}
    spark = {"gpus": [{"name": "NVIDIA GB10", "memory_total_mib": None, "memory_reporting_supported": False}]}
    h200 = {"gpus": [{"name": "NVIDIA H200", "memory_total_mib": 143771.0, "memory_reporting_supported": True}]}
    gb10 = detect.detect_gb10({}, arm, spark, {"devices": [{"name": "NVIDIA GB10"}]})
    assert gb10["is_gb10"] and gb10["unified_memory"] and gb10["confidence"] == "high"
    assert any("GB10" in e for e in gb10["evidence"])
    discrete = detect.detect_gb10({}, x86, h200, {"devices": [{"name": "NVIDIA H200"}]})
    assert not discrete["is_gb10"] and not discrete["unified_memory"]
    assert not detect.detect_gb10({}, arm, {"gpus": []}, {})["is_gb10"]     # ARM alone is not GB10


def test_memory_monitor_and_unload_check():
    mon = memory.MemoryMonitor(0.05, unified_memory=True).start()
    blob = bytearray(40 * 1024 * 1024)
    blob[::4096] = b"x" * len(blob[::4096])
    summary = mon.stop()
    del blob
    assert summary["samples"] >= 1 and summary["peak_process_rss_bytes"] > 0
    assert "unified" in summary["primary_peak_basis"]
    assert memory.MemoryMonitor(0.05, unified_memory=False).start().stop()["primary_peak_basis"]
    before = {"system_available_bytes": 100 * 2**30}
    assert memory.released_after_unload(before, {"system_available_bytes": 99 * 2**30}, 4 * 2**30, True)["verified"]
    assert not memory.released_after_unload(before, {"system_available_bytes": 40 * 2**30}, 4 * 2**30, True)["verified"]
    assert memory.released_after_unload({"torch_allocated_bytes": 0}, {"torch_allocated_bytes": 60 * 2**30},
                                        4 * 2**30, False)["verified"] is False


# ---------------------------------------------------------------- no external services in the code

FORBIDDEN_IMPORTS = ["pytesseract", "easyocr", "paddleocr", "doctr", "keras_ocr", "rapidocr", "surya", "tesserocr",
                     "cv2", "layoutparser", "openai", "anthropic", "google.generativeai", "cohere", "boto3",
                     "requests", "httpx", "aiohttp", "urllib.request", "langchain", "llama_index", "faiss", "chromadb",
                     "rxnorm", "drugbank", "sentry_sdk", "wandb", "mlflow"]


@pytest.mark.parametrize("folder", [SRC, APP])
def test_no_ocr_api_retrieval_or_telemetry_libraries_are_imported(folder):
    pattern = re.compile(r"^\s*(?:import|from)\s+(" + "|".join(re.escape(m) for m in FORBIDDEN_IMPORTS) + r")\b", re.M)
    offenders = {str(p.relative_to(PROJECT_ROOT)): pattern.findall(p.read_text(encoding="utf-8"))
                 for p in folder.rglob("*.py")}
    assert {k: v for k, v in offenders.items() if v} == {}


def test_inference_loads_only_from_local_snapshots():
    text = (SRC / "models" / "transformers_adapter.py").read_text(encoding="utf-8")
    assert text.count("local_files_only=True") + text.count('"local_files_only": True') >= 2
    assert "from_pretrained(self.spec.repo_id" not in text          # never by Hub name
    engine = (SRC / "inference" / "engine.py").read_text(encoding="utf-8")
    assert "download(" not in engine                                 # the engine never downloads


# ---------------------------------------------------------------- offline guard

def test_network_guard_blocks_and_records_then_restores():
    original = socket.socket.connect
    with offline.NetworkGuard() as guard:
        with pytest.raises(offline.OfflineViolation):
            socket.create_connection(("192.0.2.1", 443), timeout=1)
        with pytest.raises(offline.OfflineViolation):
            socket.getaddrinfo("huggingface.co", 443)
        s = socket.socket()
        with pytest.raises(offline.OfflineViolation):
            s.connect(("192.0.2.1", 80))
        s.close()
        assert len(guard.attempts) == 3
    assert socket.socket.connect is original


def test_network_guard_allows_loopback():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    with offline.NetworkGuard() as guard:
        client = socket.create_connection(server.getsockname(), timeout=2)
        client.close()
    server.close()
    assert guard.attempts == []


def test_offline_check_with_mock_does_not_claim_real_models(ctx, tmp_path, png_bytes):
    image = tmp_path / "page.png"
    image.write_bytes(png_bytes)
    report = offline.run_offline_check(cfg=ctx.cfg, registry=ctx.registry, download_manager=ctx.downloads,
                                       manager=ctx.manager, prompt_path=ctx.prompt_path, model_keys=["mock_alpha"],
                                       image_path=image, inference=True, probe=False)
    assert report["no_network_use_attempted"]
    assert report["models"]["mock_alpha"]["inference_status"] == "complete"
    assert report["real_models_inference_verified_offline"] == []
    assert "NOT been verified for any real model" in report["conclusion"]
    assert all(s["ok"] for s in report["steps"])
    assert "Offline verification report" in offline.render_markdown(report)


def test_offline_check_reports_missing_real_weights_as_not_verified(ctx, tmp_path, png_bytes, monkeypatch):
    image = tmp_path / "page.png"
    image.write_bytes(png_bytes)
    monkeypatch.setattr(ctx.downloads, "local_status", lambda spec: {"status": "not_downloaded", "path": None})
    report = offline.run_offline_check(cfg=ctx.cfg, registry=ctx.registry, download_manager=ctx.downloads,
                                       manager=ctx.manager, prompt_path=ctx.prompt_path, model_keys=["gemma4_31b"],
                                       image_path=image, inference=True, probe=False)
    assert report["real_models_not_verified"] == ["gemma4_31b"]
    assert report["models"]["gemma4_31b"]["inference_offline_verified"] is False


# ---------------------------------------------------------------- command line

def test_cli_mock_run_report_and_export(ctx, tmp_path, png_bytes, capsys):
    image = tmp_path / "page.png"
    image.write_bytes(png_bytes)
    assert cli.main(["run", "--mock", "--input", str(image), "--max-new-tokens", "4096"]) == 0
    assert cli.main(["list"]) == 0
    assert "EXP_0001" in capsys.readouterr().out
    assert cli.main(["report", "--out", str(tmp_path / "report.md")]) == 0
    assert "MOCK" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert cli.main(["export", "metadata", "--out", str(tmp_path / "meta.json")]) == 0
    rows = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert {r["model"]["key"] for r in rows} == {"mock_alpha", "mock_beta"}
    assert rows[0]["generation"]["requested"]["max_new_tokens"] == 4096
    assert cli.main(["annotation-template", "EXP_0001"]) == 0
    assert cli.main(["evaluate", "EXP_0001"]) == 1               # blank, unverified annotation is refused
    assert cli.main(["delete", "EXP_0001"]) == 2 and cli.main(["delete", "EXP_0001", "--yes"]) == 0


def test_cli_download_is_not_confirmed_implicitly(ctx, monkeypatch, capsys):
    from rxbench.models.download_manager import DownloadManager
    monkeypatch.setattr(DownloadManager, "plan", lambda self, spec, remote=None: {
        "download_bytes": 6e10, "size_source": "test", "already_present_bytes": 0, "cache_dir": "x",
        "free_bytes": 1e12, "local_status": "not_downloaded", "enough_space": True, "remaining_bytes": 6e10})
    monkeypatch.setattr(DownloadManager, "download", lambda *a, **k: pytest.fail("download started without confirmation"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert cli.main(["models", "download", "gemma4_31b"]) == 2
    assert "pass --yes" in capsys.readouterr().out

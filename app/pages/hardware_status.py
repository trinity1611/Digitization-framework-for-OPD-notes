"""Screen 1: hardware and environment."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.components import common  # noqa: E402
from rxbench.hardware.diagnostics import render_markdown  # noqa: E402
from rxbench.models import backend_manager  # noqa: E402


def render() -> None:
    common.header("Hardware and environment",
                  "Detected on this machine now. Nothing is assumed from a product specification.")
    if st.button("Detect again"):
        common.hardware_report(refresh=True)
    report = common.hardware_report()
    gb10, cpu, mem, torch_i = report["gb10"], report["cpu"], report["system_memory"], report["torch"]
    devices = torch_i.get("devices") or []
    smi_gpus = report["nvidia_smi"].get("gpus") or []
    gpu = devices[0]["name"] if devices else (smi_gpus[0]["name"] if smi_gpus else "none detected")

    a, b, c, d = st.columns(4)
    a.metric("GPU", gpu)
    b.metric("NVIDIA GB10", "yes" if gb10["is_gb10"] else "no")
    c.metric("CPU architecture", cpu.get("architecture"))
    d.metric("Unified memory", "yes" if gb10["unified_memory"] else "no")
    a, b, c, d = st.columns(4)
    a.metric("System memory", common.gb(mem.get("total_bytes")))
    b.metric("Available now", common.gb(mem.get("available_bytes")))
    gpu_total = devices[0]["mem_get_info_total_bytes"] if devices else None
    gpu_free = devices[0]["mem_get_info_free_bytes"] if devices else None
    c.metric("GPU memory (PyTorch)", common.gb(gpu_total))
    d.metric("GPU memory free", common.gb(gpu_free))
    st.info(gb10["note"])

    st.subheader("GB10 identification")
    for ev in gb10["evidence"] or ["No GB10 evidence found (GPU name, DMI product, device tree, /etc/dgx-release)."]:
        st.write("- " + ev)

    st.subheader("Storage")
    st.dataframe([{"location": k, "path": v.get("path"), "total": common.gb(v.get("total_bytes")),
                   "free": common.gb(v.get("free_bytes"))} for k, v in report["storage"].items()],
                 hide_index=True)

    st.subheader("Software")
    smi = report["nvidia_smi"]
    st.dataframe([
        {"item": "Operating system", "value": report["os"].get("distribution") or report["os"].get("platform")},
        {"item": "Kernel", "value": report["os"].get("release")},
        {"item": "Python", "value": report["python"]["version"]},
        {"item": "PyTorch", "value": torch_i.get("version") or "not installed"},
        {"item": "CUDA (PyTorch build)", "value": torch_i.get("cuda_build")},
        {"item": "CUDA usable", "value": str(torch_i.get("cuda_available"))},
        {"item": "NVIDIA driver", "value": smi_gpus[0]["driver_version"] if smi_gpus else smi.get("driver_version")},
        {"item": "CPU", "value": f"{cpu.get('model')} ({cpu.get('physical_cores')} cores / {cpu.get('logical_cores')} threads)"},
        *({"item": name, "value": version or "not installed"} for name, version in report["packages"].items()),
    ], hide_index=True)

    st.subheader("Inference backend")
    status = backend_manager.backend_status()
    if status["ready"]:
        st.success(f"Transformers backend ready: transformers {status['transformers']}, torch {status['torch']}, "
                   f"CUDA build {status.get('torch_cuda_build')}.")
    else:
        st.error("Transformers backend is not ready: " + "; ".join(status["problems"]))
    for name, info in backend_manager.BACKENDS.items():
        with st.expander(f"{name}: {'used' if info['implemented'] else 'assessed, not used'}"):
            st.write(info["summary"])
            for reason in info["why"]:
                st.write("- " + reason)

    st.subheader("Compatibility")
    for note in report["compatibility"]:
        {"ok": st.success, "info": st.info}.get(note["level"], st.warning)(note["text"])

    markdown = render_markdown(report)
    st.download_button("Download hardware report (Markdown)", markdown, file_name="HARDWARE_REPORT.md",
                       mime="text/markdown")
    with st.expander("Full report"):
        st.markdown(markdown)


render()

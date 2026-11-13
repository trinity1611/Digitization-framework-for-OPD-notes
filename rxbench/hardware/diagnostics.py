"""Render the hardware discovery result as docs/HARDWARE_REPORT.md."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .detect import detect_all


def _gb(n: float | int | None) -> str:
    return "not reported" if n is None else f"{n / 1024**3:.1f} GB"


def _row(*cells: Any) -> str:
    return "| " + " | ".join("" if c is None else str(c) for c in cells) + " |"


def render_markdown(report: dict[str, Any], role_note: str | None = None) -> str:
    os_i, cpu, mem = report["os"], report["cpu"], report["system_memory"]
    smi, torch_i, gb10 = report["nvidia_smi"], report["torch"], report["gb10"]
    lines: list[str] = ["# Hardware report", ""]
    lines.append(f"Generated {report['generated_at']} by `python -m rxbench hardware-report` on host "
                 f"`{os_i.get('hostname')}`. Every value below was read from this machine at that time; "
                 "nothing is assumed from a product specification.")
    lines.append("")
    if role_note:
        lines += [f"> {role_note}", ""]

    lines += ["## Summary", "", _row("item", "detected"), _row("---", "---")]
    gpus = smi.get("gpus") or []
    devices = torch_i.get("devices") or []
    gpu_names = ", ".join(sorted({g["name"] for g in gpus} | {d["name"] for d in devices})) or "none detected"
    lines += [
        _row("GPU(s)", gpu_names),
        _row("NVIDIA GB10", f"{'yes' if gb10['is_gb10'] else 'no'} ({gb10['confidence']})"),
        _row("Unified memory", "yes" if gb10["unified_memory"] else "no"),
        _row("CPU architecture", f"{cpu.get('architecture')} ({cpu.get('architecture_family')})"),
        _row("CPU", cpu.get("model")),
        _row("CPU cores", f"{cpu.get('physical_cores')} physical / {cpu.get('logical_cores')} logical"
             + (f"; {cpu['cores_available_to_process']} available to this process"
                if cpu.get("cores_available_to_process") else "")),
        _row("System RAM", f"{_gb(mem.get('total_bytes'))} total, {_gb(mem.get('available_bytes'))} available"),
        _row("Operating system", os_i.get("distribution") or os_i.get("platform")),
        _row("Kernel", os_i.get("release")),
        _row("Python", report["python"]["version"]),
        _row("PyTorch", torch_i.get("version") if torch_i.get("installed") else "not installed"),
        _row("CUDA (PyTorch build)", torch_i.get("cuda_build")),
        _row("CUDA usable from PyTorch", torch_i.get("cuda_available")),
        _row("NVIDIA driver", (gpus[0]["driver_version"] if gpus else smi.get("driver_version"))),
        _row("CUDA (driver)", smi.get("driver_cuda_version")),
    ]
    lines.append("")

    lines += ["## GB10 identification", ""]
    lines.append(f"- GB10 detected: **{'yes' if gb10['is_gb10'] else 'no'}** (confidence: {gb10['confidence']}).")
    lines.append(f"- Unified memory: **{'yes' if gb10['unified_memory'] else 'no'}**. {gb10['note']}")
    for ev in gb10["evidence"] or ["No GB10 evidence found (GPU name, DMI product, device tree, /etc/dgx-release)."]:
        lines.append(f"- Evidence: {ev}")
    lines.append("")

    lines += ["## GPU memory", ""]
    if gpus:
        lines += [_row("index", "name", "total (nvidia-smi)", "used", "free", "compute capability"),
                  _row("---", "---", "---", "---", "---", "---")]
        for g in gpus:
            fmt = lambda v: "not reported" if v is None else f"{v / 1024:.1f} GB"
            lines.append(_row(g["index"], g["name"], fmt(g["memory_total_mib"]), fmt(g["memory_used_mib"]),
                              fmt(g["memory_free_mib"]), g["compute_capability"]))
    else:
        lines.append(f"`nvidia-smi` returned no GPUs ({smi.get('note') or 'no output'}).")
    lines.append("")
    if devices:
        lines += [_row("index", "name (PyTorch)", "total_memory", "mem_get_info free / total"),
                  _row("---", "---", "---", "---")]
        for d in devices:
            lines.append(_row(d["index"], d["name"], _gb(d["total_memory_bytes"]),
                              f"{_gb(d['mem_get_info_free_bytes'])} / {_gb(d['mem_get_info_total_bytes'])}"))
        lines.append("")
    lines.append("`nvidia-smi` is not used alone: on unified memory it may print `[N/A]` for memory, and "
                 "the usable budget is the system's available memory shown below.")
    lines.append("")

    lines += ["## System memory", ""]
    lines.append(f"- Total {_gb(mem.get('total_bytes'))}, available {_gb(mem.get('available_bytes'))}, "
                 f"swap {_gb(mem.get('swap_total_bytes'))}.")
    if mem.get("cgroup_limit_bytes"):
        lines.append(f"- Control-group limit for this process (for example a Slurm `--mem` request): "
                     f"{_gb(mem['cgroup_limit_bytes'])}.")
    if mem.get("proc_meminfo"):
        lines.append("- /proc/meminfo: " + ", ".join(f"{k} {v}" for k, v in mem["proc_meminfo"].items()))
    if mem.get("free_h"):
        lines += ["", "```", mem["free_h"], "```"]
    lines.append("")

    lines += ["## Storage", "", _row("location", "path", "total", "free"), _row("---", "---", "---", "---")]
    for label, s in report["storage"].items():
        if "error" in s:
            lines.append(_row(label, s["path"], s["error"], ""))
        else:
            lines.append(_row(label, s["path"], _gb(s["total_bytes"]), _gb(s["free_bytes"])))
    lines.append("")

    lines += ["## CPU", ""]
    for k, v in (cpu.get("lscpu") or {}).items():
        if k != "Flags":
            lines.append(f"- {k}: {v}")
    if not cpu.get("lscpu"):
        lines.append(f"- {cpu.get('model')} ({cpu.get('architecture')}); `lscpu` not available on this system.")
    lines.append("")

    lines += ["## NVIDIA software", ""]
    nv = report["nvidia_software"]
    shown = False
    for k in ("kernel_module", "nvcc", "nvidia_container_cli", "nvidia_ctk", "docker", "nvtop", "nvitop"):
        if nv.get(k):
            lines.append(f"- {k}: {str(nv[k]).splitlines()[0]}")
            shown = True
    if nv.get("cuda_toolkit_file"):
        lines.append(f"- CUDA toolkit file {nv['cuda_toolkit_file']['path']} present.")
        shown = True
    if nv.get("dpkg_nvidia_packages"):
        lines.append(f"- {len(nv['dpkg_nvidia_packages'])} NVIDIA/CUDA system packages installed, including: "
                     + ", ".join(nv["dpkg_nvidia_packages"][:12]))
        shown = True
    for k, v in (nv.get("env") or {}).items():
        lines.append(f"- ${k} = `{v}`")
        shown = True
    if not shown:
        lines.append("- No NVIDIA toolkit components were found on PATH (driver information is in the summary).")
    lines.append("")

    sched = report["scheduler"]
    lines += ["## Scheduler", ""]
    if sched["slurm_job"]:
        lines.append("- Running inside a Slurm job: " + ", ".join(f"{k}={v}" for k, v in sched["env"].items()))
    else:
        lines.append("- Not inside a Slurm job." + (" Slurm client commands are present."
                                                   if sched["slurm_client_present"] else ""))
    lines.append("")

    lines += ["## Python packages", "", _row("package", "version"), _row("---", "---")]
    for name, ver in report["packages"].items():
        lines.append(_row(name, ver or "not installed"))
    lines.append("")

    lines += ["## Compatibility assessment", ""]
    for note in report["compatibility"]:
        lines.append(f"- **{note['level']}**: {note['text']}")
    lines.append("")

    if smi.get("raw"):
        lines += ["## Raw nvidia-smi output", "", "```", smi["raw"], "```", ""]
    if os_i.get("uname"):
        lines += ["## uname -a", "", "```", os_i["uname"], "```", ""]
    return "\n".join(lines)


def write_report(out_path: Path, paths: dict[str, Any] | None = None,
                 role_note: str | None = None) -> tuple[dict[str, Any], str]:
    report = detect_all(paths)
    md = render_markdown(report, role_note)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8", newline="\n")
    return report, md


if __name__ == "__main__":
    # Stand-alone entry point for the very first inspection of a machine. It needs only the Python
    # standard library (psutil and torch are used when present), so it runs before any environment exists:
    #     python -m rxbench.hardware.diagnostics --out HARDWARE_REPORT.md
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Hardware discovery (standard library only)")
    parser.add_argument("--out", default="HARDWARE_REPORT.md")
    parser.add_argument("--json")
    parser.add_argument("--role-note")
    parser.add_argument("--path", action="append", default=[], metavar="LABEL=PATH",
                        help="file system to report capacity for (repeatable)")
    cli_args = parser.parse_args()
    probe_paths = {"working_directory": Path.cwd()}
    probe_paths.update(dict(item.split("=", 1) for item in cli_args.path))
    full_report, text = write_report(Path(cli_args.out), probe_paths, cli_args.role_note)
    if cli_args.json:
        Path(cli_args.json).write_text(json.dumps(full_report, indent=2, default=str), encoding="utf-8")
    print(text)

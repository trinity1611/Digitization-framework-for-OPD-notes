"""Hardware and software environment discovery.

Nothing here assumes a particular machine. The target may be an NVIDIA GB10
system (DGX Spark class: ARM64, unified memory), a Slurm cluster node with
discrete data-centre GPUs, or a development laptop. Every probe is optional and
failure of one probe never stops the others. Torch is imported lazily so the
report can be produced before the ML environment exists.
"""
from __future__ import annotations

import datetime as _dt
import importlib
import importlib.metadata as _ilm
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PACKAGES_OF_INTEREST = [
    "torch", "torchvision", "transformers", "accelerate", "bitsandbytes", "huggingface_hub",
    "hf_xet", "safetensors", "tokenizers", "pillow", "pypdfium2", "numpy", "psutil", "PyYAML",
    "streamlit", "pandas", "rapidfuzz", "pytest", "vllm", "sglang", "llama_cpp_python",
]


def _run(cmd: list[str], timeout: float = 20.0) -> dict[str, Any]:
    """Run a diagnostic command; never raises."""
    exe = shutil.which(cmd[0])
    if exe is None:
        return {"available": False, "ok": False, "stdout": "", "note": f"{cmd[0]} not found"}
    try:
        p = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout,
                           errors="replace", stdin=subprocess.DEVNULL)
        return {"available": True, "ok": p.returncode == 0, "stdout": (p.stdout or "").strip(),
                "stderr": (p.stderr or "").strip()[:500], "returncode": p.returncode}
    except subprocess.TimeoutExpired:
        return {"available": True, "ok": False, "stdout": "", "note": "timed out"}
    except OSError as exc:
        return {"available": True, "ok": False, "stdout": "", "note": f"{type(exc).__name__}: {exc}"}


def _read_text(path: str, limit: int = 2000) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").replace("\x00", "").strip()[:limit]
    except OSError:
        return None


def _num(value: str) -> float | None:
    """nvidia-smi prints [N/A] / [Not Supported] where a value is unavailable (typical on unified memory)."""
    v = (value or "").strip()
    if not v or v.startswith("[") or v.lower() in {"n/a", "not supported"}:
        return None
    try:
        return float(v)
    except ValueError:
        return None


# --------------------------------------------------------------------------- OS / CPU

def detect_os() -> dict[str, Any]:
    info: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "kernel": platform.release() if platform.system() == "Linux" else None,
    }
    os_release = _read_text("/etc/os-release")
    if os_release:
        fields = dict(line.split("=", 1) for line in os_release.splitlines() if "=" in line)
        info["distribution"] = fields.get("PRETTY_NAME", "").strip('"') or None
    uname = _run(["uname", "-a"])
    if uname["ok"]:
        info["uname"] = uname["stdout"]
    return info


def detect_cpu() -> dict[str, Any]:
    machine = platform.machine().lower()
    arch_family = ("arm64" if machine in {"aarch64", "arm64"} else
                   "x86_64" if machine in {"x86_64", "amd64"} else machine)
    info: dict[str, Any] = {"architecture": platform.machine(), "architecture_family": arch_family,
                            "logical_cores": os.cpu_count(), "model": platform.processor() or None}
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"] = psutil.cpu_count(logical=True)
    except Exception:  # psutil missing or unsupported platform
        pass
    lscpu = _run(["lscpu"])
    if lscpu["ok"]:
        fields = {}
        for line in lscpu["stdout"].splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fields[k.strip()] = v.strip()
        info["lscpu"] = {k: fields[k] for k in (
            "Architecture", "CPU(s)", "Model name", "Vendor ID", "Thread(s) per core",
            "Core(s) per socket", "Core(s) per cluster", "Socket(s)", "CPU max MHz", "Flags") if k in fields}
        if "Flags" in info["lscpu"]:
            info["lscpu"]["Flags"] = info["lscpu"]["Flags"][:300]
        info["model"] = fields.get("Model name") or info["model"]
    # Affinity shows what a Slurm job was actually given.
    if hasattr(os, "sched_getaffinity"):
        try:
            info["cores_available_to_process"] = len(os.sched_getaffinity(0))
        except OSError:
            pass
    return info


# --------------------------------------------------------------------------- memory / storage

def detect_system_memory() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        info.update({"total_bytes": vm.total, "available_bytes": vm.available, "used_bytes": vm.used,
                     "swap_total_bytes": sm.total, "swap_used_bytes": sm.used})
    except Exception:
        pass
    meminfo = _read_text("/proc/meminfo", limit=6000)
    if meminfo:
        wanted = {"MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapTotal", "SwapFree",
                  "HugePages_Total", "Hugepagesize", "CmaTotal", "CmaFree"}
        info["proc_meminfo"] = {k.strip(): v.strip() for k, v in
                                (line.split(":", 1) for line in meminfo.splitlines() if ":" in line)
                                if k.strip() in wanted}
        # Without psutil (bare Python before the environment exists) take the totals from /proc/meminfo.
        def _kb(key: str) -> int | None:
            raw = info["proc_meminfo"].get(key, "").split()
            return int(raw[0]) * 1024 if raw and raw[0].isdigit() else None
        info.setdefault("total_bytes", _kb("MemTotal"))
        info.setdefault("available_bytes", _kb("MemAvailable"))
        info.setdefault("swap_total_bytes", _kb("SwapTotal"))
    free = _run(["free", "-h"])
    if free["ok"]:
        info["free_h"] = free["stdout"]
    # cgroup limit: a Slurm --mem request caps the job below the node total.
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        raw = _read_text(path)
        if raw and raw.isdigit() and int(raw) < 1 << 60:
            info["cgroup_limit_bytes"] = int(raw)
            break
    return info


def detect_storage(paths: dict[str, Path | str | None]) -> dict[str, Any]:
    """Capacity of the file systems holding the given paths (no directory sweeps)."""
    out: dict[str, Any] = {}
    for label, p in paths.items():
        if p is None:
            continue
        probe = Path(p)
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            usage = shutil.disk_usage(probe)
            out[label] = {"path": str(p), "probed": str(probe), "total_bytes": usage.total,
                          "used_bytes": usage.used, "free_bytes": usage.free}
        except OSError as exc:
            out[label] = {"path": str(p), "error": f"{type(exc).__name__}: {exc}"}
    return out


# --------------------------------------------------------------------------- GPU / NVIDIA

def detect_nvidia_smi() -> dict[str, Any]:
    q = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version,"
              "compute_cap,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"])
    info: dict[str, Any] = {"available": q.get("available", False), "ok": q.get("ok", False), "gpus": []}
    if not q.get("available"):
        info["note"] = q.get("note")
        return info
    if q["ok"]:
        for line in q["stdout"].splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 9:
                continue
            info["gpus"].append({
                "index": parts[0], "name": parts[1],
                "memory_total_mib": _num(parts[2]), "memory_used_mib": _num(parts[3]),
                "memory_free_mib": _num(parts[4]), "driver_version": parts[5],
                "compute_capability": parts[6], "utilization_percent": _num(parts[7]),
                "temperature_c": _num(parts[8]),
                "memory_reporting_supported": _num(parts[2]) is not None,
            })
    else:
        info["note"] = q.get("stderr") or q.get("note")
    raw = _run(["nvidia-smi"])
    if raw.get("ok"):
        info["raw"] = raw["stdout"][:4000]
        # Header formats differ between driver generations ("Driver Version"/"CUDA Version" and
        # "KMD Version"/"CUDA UMD Version").
        m = re.search(r"CUDA(?: UMD)? Version:\s*([0-9.]+)", raw["stdout"])
        if m:
            info["driver_cuda_version"] = m.group(1)
        m = re.search(r"(?:Driver|KMD) Version:\s*([0-9.]+)", raw["stdout"])
        if m:
            info["driver_version"] = m.group(1)
    return info


def detect_torch() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return {"installed": False, "note": f"{type(exc).__name__}: {exc}"}
    info: dict[str, Any] = {"installed": True, "version": torch.__version__,
                            "cuda_build": getattr(torch.version, "cuda", None), "cuda_available": False}
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        if info["cuda_available"]:
            devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_b, total_b = torch.cuda.mem_get_info(i)
                devices.append({"index": i, "name": props.name, "total_memory_bytes": props.total_memory,
                                "mem_get_info_free_bytes": free_b, "mem_get_info_total_bytes": total_b,
                                "capability": ".".join(map(str, torch.cuda.get_device_capability(i)))})
            info["devices"] = devices
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception as exc:
        info["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return info


def detect_gb10(os_info: dict, cpu: dict, smi: dict, torch_info: dict) -> dict[str, Any]:
    """Heuristic identification of an NVIDIA GB10 system. Evidence is listed, not assumed."""
    evidence: list[str] = []
    names = [g.get("name", "") for g in smi.get("gpus", [])] + \
            [d.get("name", "") for d in torch_info.get("devices", []) or []]
    gpu_named_gb10 = any("gb10" in n.lower() for n in names)
    if gpu_named_gb10:
        evidence.append("GPU name contains 'GB10': " + ", ".join(sorted(set(n for n in names if n))))
    product = _read_text("/sys/class/dmi/id/product_name") or ""
    vendor = _read_text("/sys/class/dmi/id/sys_vendor") or ""
    board = _read_text("/sys/class/dmi/id/board_name") or ""
    dt_model = _read_text("/proc/device-tree/model") or ""
    for label, value in (("DMI product", product), ("DMI vendor", vendor), ("DMI board", board),
                         ("device-tree model", dt_model)):
        if value:
            evidence.append(f"{label}: {value}")
    dgx_release = _read_text("/etc/dgx-release")
    if dgx_release:
        evidence.append("/etc/dgx-release present")
    product_blob = " ".join([product, board, dt_model]).lower()
    product_says_spark = "spark" in product_blob or "gb10" in product_blob
    is_arm = cpu.get("architecture_family") == "arm64"
    smi_mem_unsupported = any(not g.get("memory_reporting_supported", True) for g in smi.get("gpus", []))
    if smi_mem_unsupported:
        evidence.append("nvidia-smi does not report GPU memory totals (typical of unified-memory systems)")
    is_gb10 = bool(gpu_named_gb10 or (product_says_spark and is_arm))
    unified = bool(is_gb10 or (smi_mem_unsupported and is_arm))
    return {
        "is_gb10": is_gb10,
        "is_dgx_spark": bool(is_gb10 and "spark" in product_blob),
        "unified_memory": unified,
        "confidence": "high" if gpu_named_gb10 else ("medium" if is_gb10 else "not detected"),
        "evidence": evidence,
        "dgx_release": dgx_release,
        "dmi_product_name": product or None,
        "note": ("On unified memory the GPU shares system RAM: use MemAvailable (free/psutil) as the "
                 "memory budget, not nvidia-smi." if unified else
                 "Discrete GPU memory: nvidia-smi and torch.cuda.mem_get_info describe the GPU; "
                 "system RAM is separate."),
    }


def detect_nvidia_software() -> dict[str, Any]:
    info: dict[str, Any] = {}
    nvcc = _run(["nvcc", "--version"])
    if nvcc.get("ok"):
        info["nvcc"] = nvcc["stdout"].splitlines()[-1] if nvcc["stdout"] else None
    for path in ("/usr/local/cuda/version.json", "/usr/local/cuda/version.txt"):
        txt = _read_text(path, 600)
        if txt:
            info["cuda_toolkit_file"] = {"path": path, "content": txt[:600]}
            break
    for name, cmd in (("nvidia_container_cli", ["nvidia-container-cli", "--version"]),
                      ("nvidia_ctk", ["nvidia-ctk", "--version"]),
                      ("docker", ["docker", "--version"]),
                      ("nvtop", ["nvtop", "--version"]),
                      ("nvitop", ["nvitop", "--version"])):
        r = _run(cmd, timeout=10)
        if r.get("available"):
            info[name] = (r["stdout"].splitlines()[0] if r.get("stdout") else r.get("note") or "present")
    dpkg = _run(["dpkg-query", "-W", "-f=${Package} ${Version}\\n"], timeout=30)
    if dpkg.get("ok"):
        keep = [l for l in dpkg["stdout"].splitlines()
                if any(k in l.lower() for k in ("nvidia", "cuda", "tensorrt", "dgx", "cudnn"))]
        info["dpkg_nvidia_packages"] = keep[:80]
    kmod = _read_text("/proc/driver/nvidia/version", 400)
    if kmod:
        info["kernel_module"] = kmod
    info["env"] = {k: os.environ.get(k) for k in ("CUDA_HOME", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")
                   if os.environ.get(k)}
    if "LD_LIBRARY_PATH" in info["env"]:
        info["env"]["LD_LIBRARY_PATH"] = info["env"]["LD_LIBRARY_PATH"][:400]
    return info


def detect_scheduler() -> dict[str, Any]:
    keys = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURMD_NODENAME", "SLURM_JOB_GPUS",
            "SLURM_GPUS_ON_NODE", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_JOB_NODELIST")
    env = {k: os.environ[k] for k in keys if k in os.environ}
    return {"slurm_job": bool(env.get("SLURM_JOB_ID")), "env": env,
            "slurm_client_present": shutil.which("sbatch") is not None}


def detect_packages() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGES_OF_INTEREST:
        try:
            out[name] = _ilm.version(name)
        except _ilm.PackageNotFoundError:
            out[name] = None
    return out


# --------------------------------------------------------------------------- aggregate

def assess_compatibility(cpu: dict, torch_info: dict, packages: dict, gb10: dict) -> list[dict[str, str]]:
    """Plain statements about whether this machine can run the benchmark."""
    notes: list[dict[str, str]] = []
    fam = cpu.get("architecture_family")
    if fam in {"x86_64", "arm64"}:
        notes.append({"level": "ok", "text": f"CPU architecture {cpu.get('architecture')} ({fam}) is supported by "
                      "PyTorch, Transformers and the pure-Python parts of this application."})
    else:
        notes.append({"level": "warning", "text": f"CPU architecture {cpu.get('architecture')} is untested."})
    if fam == "arm64":
        notes.append({"level": "info", "text": "ARM64: install the NVIDIA-supported CUDA build of PyTorch for "
                      "aarch64; x86-only wheels and binaries will not work. bitsandbytes (INT8/INT4) must be "
                      "verified on this machine before it is relied on."})
    if not torch_info.get("installed"):
        notes.append({"level": "warning", "text": "PyTorch is not installed in this Python environment."})
    elif not torch_info.get("cuda_available"):
        notes.append({"level": "warning", "text": "PyTorch cannot use CUDA here; 27B-33B models cannot be run."})
    else:
        notes.append({"level": "ok", "text": f"PyTorch {torch_info.get('version')} sees CUDA "
                      f"(build {torch_info.get('cuda_build')})."})
    if not packages.get("transformers"):
        notes.append({"level": "warning", "text": "Transformers is not installed."})
    if gb10.get("unified_memory"):
        notes.append({"level": "info", "text": "Unified memory detected: model weights, vision activations and the "
                      "rest of the system share one pool."})
    return notes


def detect_all(paths: dict[str, Path | str | None] | None = None) -> dict[str, Any]:
    os_info = detect_os()
    cpu = detect_cpu()
    smi = detect_nvidia_smi()
    torch_info = detect_torch()
    gb10 = detect_gb10(os_info, cpu, smi, torch_info)
    packages = detect_packages()
    report = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                   "executable": sys.executable},
        "os": os_info,
        "cpu": cpu,
        "system_memory": detect_system_memory(),
        "storage": detect_storage(paths or {"working_directory": Path.cwd()}),
        "nvidia_smi": smi,
        "torch": torch_info,
        "gb10": gb10,
        "nvidia_software": detect_nvidia_software(),
        "scheduler": detect_scheduler(),
        "packages": packages,
    }
    report["compatibility"] = assess_compatibility(cpu, torch_info, packages, gb10)
    return report


def summary_for_metadata(report: dict[str, Any]) -> dict[str, Any]:
    """Compact hardware record stored in every metadata.json."""
    smi_gpus = report.get("nvidia_smi", {}).get("gpus", [])
    torch_info = report.get("torch", {})
    devices = torch_info.get("devices") or []
    return {
        "gpu_name": (devices[0]["name"] if devices else (smi_gpus[0]["name"] if smi_gpus else None)),
        "gpu_count_visible": len(devices) if devices else len(smi_gpus),
        "gpu_compute_capability": devices[0]["capability"] if devices else None,
        "gpu_total_memory_bytes": devices[0]["total_memory_bytes"] if devices else None,
        "is_gb10": report.get("gb10", {}).get("is_gb10"),
        "unified_memory": report.get("gb10", {}).get("unified_memory"),
        "cpu_architecture": report.get("cpu", {}).get("architecture"),
        "cpu_model": report.get("cpu", {}).get("model"),
        "system_memory_total_bytes": report.get("system_memory", {}).get("total_bytes"),
        "os": report.get("os", {}).get("distribution") or report.get("os", {}).get("platform"),
        "kernel": report.get("os", {}).get("release"),
        "python_version": report.get("python", {}).get("version"),
        "torch_version": torch_info.get("version"),
        "cuda_version": torch_info.get("cuda_build"),
        "nvidia_driver_version": (smi_gpus[0].get("driver_version") if smi_gpus
                                  else report.get("nvidia_smi", {}).get("driver_version")),
        "slurm_job_id": report.get("scheduler", {}).get("env", {}).get("SLURM_JOB_ID"),
        "node": report.get("os", {}).get("hostname"),
    }

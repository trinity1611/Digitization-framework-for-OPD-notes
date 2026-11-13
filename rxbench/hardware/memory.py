"""Memory measurement that is honest about unified versus discrete GPU memory.

On a discrete GPU, model weights live in GPU memory and `torch.cuda` /
`mem_get_info` describe them. On a unified-memory machine (NVIDIA GB10) the GPU
allocates from system RAM, so the authoritative figure is the drop in the
operating system's *available* memory; `nvidia-smi` may not report a total at
all. Every snapshot therefore records both views, and `primary_peak_bytes`
says which one was used.
"""
from __future__ import annotations

import gc
import os
import threading
import time
from typing import Any


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def snapshot(label: str | None = None) -> dict[str, Any]:
    snap: dict[str, Any] = {"label": label, "time": time.time()}
    try:
        import psutil
        vm = psutil.virtual_memory()
        snap["system_total_bytes"] = vm.total
        snap["system_available_bytes"] = vm.available
        snap["process_rss_bytes"] = psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        pass
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            snap["cuda_free_bytes"] = free_b
            snap["cuda_total_bytes"] = total_b
            snap["cuda_used_bytes"] = total_b - free_b  # all processes on the device
            snap["torch_allocated_bytes"] = torch.cuda.memory_allocated()
            snap["torch_reserved_bytes"] = torch.cuda.memory_reserved()
            snap["torch_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
        except Exception as exc:  # driver hiccup must not break an experiment
            snap["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return snap


def reset_torch_peak() -> None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def release_torch_memory() -> None:
    """Free Python garbage and cached CUDA blocks after a model is unloaded."""
    gc.collect()
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


class MemoryMonitor:
    """Background sampler that tracks peaks between `start()` and `stop()`."""

    def __init__(self, interval_s: float = 0.25, unified_memory: bool = False):
        self.interval_s = max(0.05, float(interval_s))
        self.unified_memory = unified_memory
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.baseline: dict[str, Any] = {}
        self._reset_peaks()

    def _reset_peaks(self) -> None:
        self.samples = 0
        self.peak_process_rss = 0
        self.min_system_available: int | None = None
        self.peak_cuda_used = 0
        self.peak_torch_reserved = 0

    def _sample(self) -> None:
        s = snapshot()
        with self._lock:
            self.samples += 1
            self.peak_process_rss = max(self.peak_process_rss, s.get("process_rss_bytes", 0))
            avail = s.get("system_available_bytes")
            if avail is not None:
                self.min_system_available = avail if self.min_system_available is None else min(
                    self.min_system_available, avail)
            self.peak_cuda_used = max(self.peak_cuda_used, s.get("cuda_used_bytes", 0))
            self.peak_torch_reserved = max(self.peak_torch_reserved, s.get("torch_reserved_bytes", 0))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def start(self) -> "MemoryMonitor":
        self.baseline = snapshot("baseline")
        self._reset_peaks()
        reset_torch_peak()
        self._sample()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="rxbench-memory", daemon=True)
        self._thread.start()
        return self

    def window(self) -> "MemoryMonitor":
        """Start a new measurement window without restarting the thread."""
        with self._lock:
            self._reset_peaks()
        self.baseline = snapshot("baseline")
        reset_torch_peak()
        self._sample()
        return self

    def summary(self) -> dict[str, Any]:
        self._sample()
        end = snapshot("end")
        with self._lock:
            base_avail = self.baseline.get("system_available_bytes")
            sys_drop = (base_avail - self.min_system_available
                        if base_avail is not None and self.min_system_available is not None else None)
            torch_peak = end.get("torch_max_allocated_bytes")
            if self.unified_memory:
                primary, basis = sys_drop, "largest drop in system available memory (unified memory)"
            elif torch_peak is not None:
                primary, basis = torch_peak, "torch.cuda.max_memory_allocated (discrete GPU memory)"
            else:
                primary, basis = self.peak_process_rss, "process resident set size (no CUDA device)"
            return {
                "unified_memory": self.unified_memory,
                "primary_peak_bytes": primary,
                "primary_peak_basis": basis,
                "peak_torch_allocated_bytes": torch_peak,
                "peak_torch_reserved_bytes": self.peak_torch_reserved or None,
                "peak_device_used_bytes_all_processes": self.peak_cuda_used or None,
                "peak_process_rss_bytes": self.peak_process_rss or None,
                "min_system_available_bytes": self.min_system_available,
                "max_system_available_drop_bytes": sys_drop,
                "baseline": self.baseline,
                "end": end,
                "samples": self.samples,
                "sample_interval_s": self.interval_s,
            }

    def stop(self) -> dict[str, Any]:
        out = self.summary()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return out


def released_after_unload(before_load: dict[str, Any], after_unload: dict[str, Any],
                          tolerance_bytes: int, unified_memory: bool) -> dict[str, Any]:
    """Check that unloading returned memory to (about) the pre-load level."""
    if not unified_memory and "torch_allocated_bytes" in after_unload:
        residual = after_unload.get("torch_allocated_bytes", 0) - before_load.get("torch_allocated_bytes", 0)
        basis = "torch allocated bytes"
    elif "system_available_bytes" in after_unload and "system_available_bytes" in before_load:
        residual = before_load["system_available_bytes"] - after_unload["system_available_bytes"]
        basis = "system available bytes"
    else:
        return {"verified": None, "basis": "no memory information available"}
    return {"verified": residual <= tolerance_bytes, "residual_bytes": residual,
            "tolerance_bytes": tolerance_bytes, "basis": basis}

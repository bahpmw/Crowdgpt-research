"""Accelerator helpers: pick a device and read its memory on CUDA, ROCm or Apple MPS.

ROCm reports ``device.type == "cuda"``, so the CUDA branch covers both. MPS has no
peak-memory counter; ``driver_allocated_memory`` (live tensors plus the allocator's
cache) is the closest analogue of CUDA's ``max_memory_reserved`` when read right
after the work, because the cache keeps the high-water mark until it is emptied.
"""

from __future__ import annotations

import platform
import subprocess
from contextlib import nullcontext

import torch


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def default_device() -> str:
    """``cuda`` (incl. ROCm), then ``mps``, then ``cpu``."""
    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def resolve_device(name: str) -> torch.device:
    """``auto`` resolves to :func:`default_device`; anything else is passed to torch."""
    return torch.device(default_device() if name == "auto" else name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        torch.mps.empty_cache()


def peak_memory_gb(device: torch.device) -> float | None:
    """Reserved accelerator memory in GiB, or ``None`` on CPU."""
    if device.type == "cuda":
        return round(torch.cuda.max_memory_reserved(device) / 1024 ** 3, 3)
    if device.type == "mps":
        return round(torch.mps.driver_allocated_memory() / 1024 ** 3, 3)
    return None


def device_name(device: torch.device) -> str | None:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            chip = ""
        return chip or f"Apple {platform.machine()}"
    return None


def autocast(device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """bf16 autocast on an accelerator, a no-op on CPU."""
    if device.type in ("cuda", "mps"):
        return torch.autocast(device.type, dtype=dtype)
    return nullcontext()

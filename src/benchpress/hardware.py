"""Apple Silicon hardware detection and memory bandwidth estimation."""

from __future__ import annotations

import platform
import subprocess
import re


def get_chip_name() -> str:
    if platform.system() != "Darwin":
        return platform.processor() or "unknown"
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True, timeout=5
        )
        m = re.search(r"Chip:\s+(.+)", out)
        return m.group(1).strip() if m else "Apple Silicon"
    except Exception:
        return "Apple Silicon"


def get_memory_gb() -> float:
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True, timeout=5
        )
        m = re.search(r"Memory:\s+([\d.]+)\s*GB", out)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def get_process_memory_mb() -> float:
    """RSS memory of the current process in MB."""
    try:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def hardware_summary() -> dict[str, str | float]:
    return {
        "chip": get_chip_name(),
        "memory_gb": get_memory_gb(),
        "os": f"{platform.system()} {platform.mac_ver()[0]}",
        "python": platform.python_version(),
    }

"""nvidia-smi helpers. GPU index follows ``CUDA_VISIBLE_DEVICES``.

``nvidia-smi --id=0`` is the first *physical* card. Torch ``cuda:0`` is the
first *visible* card. Multi-GPU boxes must use the same id or ``--codec auto``
plans VRAM for the wrong device.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

__all__ = ["device_id", "executable", "query_mib", "total_mib", "used_mib"]


def device_id() -> str:
    """Selector for ``nvidia-smi --id=``: index, UUID, or PCI bus id."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not vis:
        return "0"
    first = vis.split(",")[0].strip()
    if first in ("", "-1"):
        return "0"
    return first


def executable() -> str | None:
    hit = shutil.which("nvidia-smi")
    if hit:
        return hit
    if os.name != "nt":
        return None
    system = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    program = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    for candidate in (
        system / "System32" / "nvidia-smi.exe",
        program / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def query_mib(field: str) -> int | None:
    exe = executable()
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [
                exe,
                f"--id={device_id()}",
                f"--query-gpu={field}",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def total_mib() -> int | None:
    value = query_mib("memory.total")
    if value is None or value <= 0:
        return None
    return value


def used_mib() -> int | None:
    """Occupied MiB. ``0`` is a real empty-card reading, not a miss."""
    return query_mib("memory.used")

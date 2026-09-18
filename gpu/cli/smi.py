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

__all__ = [
    "compute_cap",
    "device_id",
    "executable",
    "gpu_name",
    "query_field",
    "query_mib",
    "total_mib",
    "used_mib",
    "visible_gpu_looks_sm120",
]


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


def query_field(field: str) -> str | None:
    """One nvidia-smi CSV cell for the visible GPU, or None."""
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
        line = out.stdout.strip().splitlines()[0].strip()
    except IndexError:
        return None
    if not line or line.lower() in {"[n/a]", "n/a"}:
        return None
    return line


def query_mib(field: str) -> int | None:
    raw = query_field(field)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def gpu_name() -> str | None:
    return query_field("name")


def compute_cap() -> tuple[int, int] | None:
    """Driver-reported compute capability, e.g. ``(12, 0)`` on a 5070 Ti."""
    raw = query_field("compute_cap")
    if raw is None:
        return None
    parts = raw.replace(",", ".").split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    return int(parts[0]), int(parts[1])


def visible_gpu_looks_sm120() -> bool:
    """nvidia-smi only. Used by setup before torch is installed."""
    from gpu.arch_family import looks_sm120

    return looks_sm120(capability=compute_cap(), device_name=gpu_name())


def total_mib() -> int | None:
    value = query_mib("memory.total")
    if value is None or value <= 0:
        return None
    return value


def used_mib() -> int | None:
    """Occupied MiB. ``0`` is a real empty-card reading, not a miss."""
    return query_mib("memory.used")

"""Locate ``nvcc`` after a CUDA Toolkit install that did not refresh PATH.

Torch wheels ship the CUDA *runtime*. Compiling ``gpu/nf4`` still needs the
toolkit compiler: ``PATH``, ``CUDA_HOME`` / ``CUDA_PATH``, or the default
Windows layout under ``NVIDIA GPU Computing Toolkit\\CUDA\\v*``.
Prefer 12.4 to match the cu124 index.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = ["find_nvcc", "inject_cuda_env"]

_NVCC = "nvcc.exe" if os.name == "nt" else "nvcc"


def _version_parts(folder: str) -> tuple[int, ...]:
    raw = folder.lstrip("vV")
    out: list[int] = []
    for piece in raw.split("."):
        if piece.isdigit():
            out.append(int(piece))
    return tuple(out)


def _nvcc_rank(nvcc: Path) -> tuple:
    """Lower is better. Torch cu124 → CUDA 12.4 first, then newer 12.x."""
    name = nvcc.parent.parent.name
    parts = _version_parts(name)
    major, minor = (parts + (0, 0))[:2]
    match_cu124 = 0 if (major, minor) == (12, 4) else 1
    return (match_cu124, -major, -minor)


def _toolkit_nvccs() -> list[Path]:
    found: list[Path] = []
    if os.name == "nt":
        pf = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        base = pf / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if base.is_dir():
            for root in base.glob("v*"):
                hit = root / "bin" / _NVCC
                if hit.is_file():
                    found.append(hit)
    else:
        roots = [
            Path("/usr/local/cuda"),
            Path("/opt/cuda"),
            Path("/usr/lib/nvidia-cuda-toolkit"),
            *sorted(Path("/usr/local").glob("cuda-12*")),
            *sorted(Path("/usr/lib").glob("cuda*")),
        ]
        for root in roots:
            hit = root / "bin" / _NVCC
            if hit.is_file():
                found.append(hit)
    found.sort(key=_nvcc_rank)
    return found


def find_nvcc() -> str | None:
    """PATH, then ``CUDA_HOME`` / ``CUDA_PATH``, then the default toolkit tree."""
    which = shutil.which("nvcc")
    if which:
        return which
    for env in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env)
        if not root:
            continue
        candidate = Path(root) / "bin" / _NVCC
        if candidate.is_file():
            return str(candidate)
    hits = _toolkit_nvccs()
    return str(hits[0]) if hits else None


def inject_cuda_env() -> str | None:
    """Put the toolkit ``bin`` on PATH and set ``CUDA_HOME`` for this process."""
    nvcc = find_nvcc()
    if nvcc is None:
        return None
    bindir = Path(nvcc).resolve().parent
    root = bindir.parent
    os.environ["CUDA_HOME"] = str(root)
    os.environ["CUDA_PATH"] = str(root)
    path = os.environ.get("PATH", "")
    if str(bindir) not in path.split(os.pathsep):
        os.environ["PATH"] = str(bindir) + os.pathsep + path
    return nvcc

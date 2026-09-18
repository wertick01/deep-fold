"""Locate ``nvcc`` after a CUDA Toolkit install that did not refresh PATH.

Torch wheels ship the CUDA *runtime*. Compiling ``gpu/nf4`` still needs the
toolkit compiler: ``PATH``, ``CUDA_HOME`` / ``CUDA_PATH``, or the default
Windows layout under ``NVIDIA GPU Computing Toolkit\\CUDA\\v*``.
Prefer 12.4 to match the cu124 index. Native ``sm_120`` cubin needs 12.8+.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

__all__ = [
    "PREFER_SM120_NVCC_ENV",
    "find_nvcc",
    "inject_cuda_env",
    "nvcc_supports_sm120",
    "nvcc_version",
    "prefer_sm120_nvcc",
]

# First toolkit that can emit native GeForce Blackwell cubin (sm_120).
_SM120_NVCC = (12, 8)
PREFER_SM120_NVCC_ENV = "DEEPFOLD_PREFER_SM120_NVCC"

_NVCC = "nvcc.exe" if os.name == "nt" else "nvcc"


def prefer_sm120_nvcc() -> bool:
    """Neighbor 5070 Ti / SM120 wants CUDA 12.8+. The 3080 lab stays on 12.4."""
    raw = os.environ.get(PREFER_SM120_NVCC_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "sm120"}


def _version_parts(folder: str) -> tuple[int, ...]:
    raw = folder.lstrip("vV")
    out: list[int] = []
    for piece in raw.split("."):
        if piece.isdigit():
            out.append(int(piece))
    return tuple(out)


def _nvcc_rank(nvcc: Path, *, prefer_sm120: bool = False) -> tuple:
    """Lower is better.

    3080 lab: CUDA 12.4 first (matches torch cu124). SM120 neighbor: 12.8+
    first so ``nvcc`` can emit ``sm_120`` instead of picking 12.4 from a
    dual-toolkit box.
    """
    name = nvcc.parent.parent.name
    parts = _version_parts(name)
    major, minor = (parts + (0, 0))[:2]
    if prefer_sm120:
        can_sm120 = 0 if (major, minor) >= _SM120_NVCC else 1
        return (can_sm120, -major, -minor)
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
    found.sort(key=lambda p: _nvcc_rank(p, prefer_sm120=prefer_sm120_nvcc()))
    return found


def _nvcc_candidates() -> list[Path]:
    found: list[Path] = []
    which = shutil.which("nvcc")
    if which:
        found.append(Path(which))
    for env in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env)
        if not root:
            continue
        candidate = Path(root) / "bin" / _NVCC
        if candidate.is_file():
            found.append(candidate)
    found.extend(_toolkit_nvccs())
    uniq: list[Path] = []
    seen: set[str] = set()
    for hit in found:
        try:
            key = str(hit.resolve())
        except OSError:
            key = str(hit)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(hit)
    return uniq


def find_nvcc(*, prefer_sm120: bool | None = None) -> str | None:
    """PATH, then ``CUDA_HOME`` / ``CUDA_PATH``, then the default toolkit tree.

    On SM120 (``DEEPFOLD_PREFER_SM120_NVCC=1``) a 12.8+ toolkit wins even
    when PATH still points at 12.4.
    """
    want = prefer_sm120_nvcc() if prefer_sm120 is None else prefer_sm120
    which = shutil.which("nvcc")
    if which and not want:
        return which
    candidates = _nvcc_candidates()
    if not candidates:
        return None
    if want:
        ranked = sorted(
            candidates, key=lambda p: _nvcc_rank(p, prefer_sm120=True)
        )
        for hit in ranked:
            folder = hit.resolve().parent.parent.name if hit.exists() else hit.parent.parent.name
            parts = _version_parts(folder)
            major, minor = (parts + (0, 0))[:2]
            if (major, minor) >= _SM120_NVCC:
                return str(hit)
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


_NVCC_VERSION: tuple[int, int] | None | bool = False


def nvcc_version(nvcc: str | None = None) -> tuple[int, int] | None:
    """``(major, minor)`` of this process's nvcc, or ``None`` if missing."""
    global _NVCC_VERSION
    path = nvcc or find_nvcc()
    if path is None:
        return None
    if nvcc is None and _NVCC_VERSION is not False:
        return _NVCC_VERSION  # type: ignore[return-value]
    folder = Path(path).resolve().parent.parent.name
    parts = _version_parts(folder)
    parsed: tuple[int, int] | None
    if len(parts) >= 2:
        parsed = (parts[0], parts[1])
    else:
        parsed = _nvcc_version_from_binary(path)
    if nvcc is None:
        _NVCC_VERSION = parsed
    return parsed


def _nvcc_version_from_binary(path: str) -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stdout + out.stderr
    hit = re.search(r"release\s+(\d+)\.(\d+)", text)
    if hit is None:
        return None
    return int(hit.group(1)), int(hit.group(2))


def nvcc_supports_sm120(nvcc: str | None = None) -> bool:
    """CUDA 12.8+ can emit ``sm_120``. 12.4 (the 3080 lab) cannot."""
    ver = nvcc_version(nvcc)
    if ver is None:
        return False
    return ver >= _SM120_NVCC


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

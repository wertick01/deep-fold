"""Locate in-tree CUDA extension binaries.

Windows ships ``.pyd``, Linux ``.so``. Doctor already globs both; the
NF4/VQ wrappers used to look only for ``.pyd``, so a Linux build sitting
next to the ``.cu`` was invisible at generate time.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = ["find_ext", "have_host_compiler"]

_NATIVE = ".pyd" if os.name == "nt" else ".so"
_OTHER = ".so" if os.name == "nt" else ".pyd"


def find_ext(directory: str | Path, stem: str) -> list[Path]:
    """``stem*.pyd`` / ``stem*.so`` in ``directory``. Native suffix first."""
    root = Path(directory)
    hits = sorted(root.glob(f"{stem}*{_NATIVE}")) + sorted(root.glob(f"{stem}*{_OTHER}"))
    return hits


def have_host_compiler() -> bool:
    """Can we JIT? MSVC on Windows; ``g++``/``c++`` on POSIX. No vcvars on Linux."""
    if os.name == "nt":
        from gpu.win_toolchain import inject_msvc_env, which_cl

        return bool(inject_msvc_env() or which_cl())
    return bool(shutil.which("g++") or shutil.which("c++"))

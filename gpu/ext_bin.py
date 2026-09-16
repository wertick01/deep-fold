"""Locate in-tree CUDA extension binaries.

Windows ships ``.pyd``, Linux ``.so``. Doctor already globs both; the
NF4/VQ wrappers used to look only for ``.pyd``, so a Linux build sitting
next to the ``.cu`` was invisible at generate time.

``find_ext`` is the generate path: native suffix and this interpreter's ABI.
A Windows ``.pyd`` in a Linux checkout (or ``cpython-311`` next to Python 3.12)
must not be imported. ``list_ext`` is the unfiltered glob for doctor.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

__all__ = [
    "abi_tag",
    "find_ext",
    "have_host_compiler",
    "is_native",
    "list_ext",
    "matches_abi",
]

_NATIVE = ".pyd" if os.name == "nt" else ".so"
_OTHER = ".so" if os.name == "nt" else ".pyd"


def abi_tag() -> str:
    """Setuptools-style tag: ``cp311``, ``cp312``."""
    return f"cp{sys.version_info[0]}{sys.version_info[1]}"


def matches_abi(filename: str, tag: str | None = None) -> bool:
    """True if ``filename`` is untagged or built for this CPython.

    Windows: ``chr_nf4_ext.cp311-win_amd64.pyd``.
    Linux: ``chr_nf4_ext.cpython-311-x86_64-linux-gnu.so`` (no ``cp311`` substring).
    """
    tag = tag or abi_tag()
    name = Path(filename).name
    posix = f"cpython-{tag[2:]}" if tag.startswith("cp") else tag
    windows_tagged = "cp3" in name and "cpython" not in name
    posix_tagged = "cpython-3" in name
    if not windows_tagged and not posix_tagged:
        return True
    return tag in name or posix in name


def is_native(path: Path | str) -> bool:
    return Path(path).suffix.lower() == _NATIVE


def list_ext(directory: str | Path, stem: str) -> list[Path]:
    """Every ``stem*.pyd`` / ``stem*.so``. Native suffix first."""
    root = Path(directory)
    return sorted(root.glob(f"{stem}*{_NATIVE}")) + sorted(
        root.glob(f"{stem}*{_OTHER}")
    )


def find_ext(directory: str | Path, stem: str) -> list[Path]:
    """Importable artifacts: this OS suffix and this interpreter ABI."""
    return [p for p in list_ext(directory, stem) if is_native(p) and matches_abi(p.name)]


def have_host_compiler() -> bool:
    """Can we JIT? MSVC on Windows; ``g++``/``c++`` on POSIX. No vcvars on Linux."""
    if os.name == "nt":
        from gpu.win_toolchain import inject_msvc_env, which_cl

        return bool(inject_msvc_env() or which_cl())
    return bool(shutil.which("g++") or shutil.which("c++"))

"""MSVC + ninja for CUDA JIT on Windows.

Jupyter kernels and Cursor terminals do not inherit ``vcvars64.bat``.
``torch.utils.cpp_extension`` then dies on ``where cl`` before it even
compiles. This module dumps the VS x64 environment into ``os.environ`` of
the current process so a rebuild can happen in-place.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_VCVARS = Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    r"\VC\Auxiliary\Build\vcvars64.bat"
)
_NINJA_DIR = Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    r"\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
)
_KEEP = {
    "PATH",
    "INCLUDE",
    "LIB",
    "LIBPATH",
    "WINDOWSSDKDIR",
    "WINDOWSSDKVERBINPATH",
    "VCINSTALLDIR",
    "VCTOOLSINSTALLDIR",
    "VCTOOLSVERSION",
    "UNIVERSALCRTSDKDIR",
    "UCRTVERSION",
}

_injected = False


def which_cl() -> str | None:
    return shutil.which("cl")


def ensure_ninja_on_path() -> None:
    ninja = _NINJA_DIR / "ninja.exe"
    if ninja.is_file() and str(_NINJA_DIR) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(_NINJA_DIR) + os.pathsep + os.environ.get("PATH", "")


def _vcvars_path() -> Path | None:
    if _VCVARS.is_file():
        return _VCVARS
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / (
        r"Microsoft Visual Studio\Installer\vswhere.exe"
    )
    if not vswhere.is_file():
        return None
    try:
        out = subprocess.check_output(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-find",
                r"**\vcvars64.bat",
            ],
            text=True,
            errors="replace",
        )
    except subprocess.CalledProcessError:
        return None
    for line in out.splitlines():
        p = Path(line.strip())
        if p.is_file():
            return p
    return None


def inject_msvc_env() -> bool:
    """Return True if ``cl.exe`` is visible after this call."""
    global _injected
    ensure_ninja_on_path()
    if which_cl():
        _injected = True
        return True
    if _injected:
        return bool(which_cl())

    vcvars = _vcvars_path()
    if vcvars is None:
        return False

    # cmd.exe quoting: one string so ``&& set`` runs *after* vcvars.
    cmd = f'"{vcvars}" && set'
    try:
        out = subprocess.check_output(cmd, shell=True, text=True, errors="replace")
    except subprocess.CalledProcessError:
        return False

    for line in out.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.upper() in _KEEP:
            os.environ[key] = value

    ensure_ninja_on_path()
    _injected = True
    return bool(which_cl())

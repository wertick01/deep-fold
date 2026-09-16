"""Ensure the NF4 CUDA extension exists on a neighbor box.

If ``gpu/nf4/chr_nf4_ext*.pyd`` (or ``.so``) is already importable, do nothing.
Otherwise look for MSVC ``cl.exe`` / ``g++`` and ``nvcc``. On Windows, missing
tools are installed with winget (VS 2022 Build Tools C++ workload, CUDA 12.4).
Then ``python gpu/nf4/setup.py build_ext --inplace``. Linux compiles when the
compilers are already there; it does not ``sudo apt``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .paths import REPO

__all__ = ["KernelBuildError", "ensure_kernel", "main"]

SETUP_PY = REPO / "gpu" / "nf4" / "setup.py"


class KernelBuildError(RuntimeError):
    """Missing compilers, winget failed, or ``build_ext`` failed."""


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def _run(cmd: list[str], *, cwd: str | None = None) -> int:
    """Spy seam. Tests replace this."""
    return int(subprocess.call(cmd, cwd=cwd))


def _winget() -> str | None:
    """Spy seam."""
    hit = shutil.which("winget")
    if hit:
        return hit
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidate = Path(local) / "Microsoft" / "WindowsApps" / "winget.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def _existing_ext() -> tuple[Path | None, bool, bool]:
    """Spy seam. ``(path, stale, abi_ok)`` from doctor."""
    from .doctor import _nf4_artifact

    return _nf4_artifact()


def _find_host_cc() -> str | None:
    """Spy seam. Injects vcvars on Windows."""
    if os.name == "nt":
        from gpu.win_toolchain import inject_msvc_env, which_cl

        inject_msvc_env()
        return which_cl()
    return shutil.which("g++") or shutil.which("c++")


def _find_nvcc() -> str | None:
    """Spy seam."""
    from gpu.cuda_env import find_nvcc

    return find_nvcc()


def _winget_install(args: list[str]) -> int:
    """Install from the community ``winget`` source only.

    Without ``--source winget``, App Installer also queries ``msstore``. A
    broken Store certificate (``0x8a15005e``) aborts the whole search even
    when the package is already listed under ``winget``.
    """
    winget = _winget()
    if winget is None:
        raise KernelBuildError(
            "winget not found. Install VS 2022 Build Tools (C++ workload) and "
            "CUDA Toolkit 12.4 by hand, or install App Installer from Microsoft Store."
        )
    common = [
        "--source",
        "winget",
        "--accept-package-agreements",
        "--accept-source-agreements",
    ]
    cmd = [winget, "install", *args, *common, "--disable-interactivity"]
    code = _run(cmd)
    if code != 0:
        cmd = [winget, "install", *args, *common]
        code = _run(cmd)
    return code


def _install_vs() -> int:
    """Spy seam. Visual Studio 2022 Build Tools + C++ workload."""
    print(
        "Installing Visual Studio 2022 Build Tools (C++). Several GB; "
        "UAC / Administrator may be required.",
        flush=True,
    )
    return _winget_install(
        [
            "--id",
            "Microsoft.VisualStudio.2022.BuildTools",
            "--exact",
            "--override",
            "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools "
            "--includeRecommended --norestart",
        ]
    )


def _install_cuda() -> int:
    """Spy seam. CUDA Toolkit 12.4 so nvcc matches torch cu124."""
    print(
        "Installing NVIDIA CUDA Toolkit 12.4 (nvcc). Several GB; "
        "UAC / Administrator may be required. Driver is not installed.",
        flush=True,
    )
    last = 1
    for extra in (
        ["--id", "Nvidia.CUDA", "--version", "12.4.1", "--exact"],
        ["--id", "Nvidia.CUDA", "--version", "12.4", "--exact"],
        ["--id", "Nvidia.CUDA", "--exact"],
    ):
        last = _winget_install(extra)
        from gpu.cuda_env import inject_cuda_env

        inject_cuda_env()
        if _find_nvcc():
            return 0
    return last


def _compile() -> int:
    """Spy seam. Inherits PATH after vcvars + CUDA inject."""
    if os.name == "nt":
        from gpu.win_toolchain import ensure_msvccompiler_attr, inject_msvc_env

        inject_msvc_env()
        ensure_msvccompiler_attr()
    from gpu.cuda_env import inject_cuda_env

    inject_cuda_env()
    print("building gpu/nf4 (first compile ~1 min)...", flush=True)
    return _run(
        [sys.executable, str(SETUP_PY), "build_ext", "--inplace"],
        cwd=str(SETUP_PY.parent),
    )


def _ensure_windows_tools(*, install: bool) -> None:
    from gpu.win_toolchain import reset_msvc_injection

    cc = _find_host_cc()
    if cc is None:
        if not install:
            raise KernelBuildError(
                "cl.exe not found. Install Visual Studio 2022 Build Tools (C++), "
                "or re-run without --no-install-tools."
            )
        code = _install_vs()
        reset_msvc_injection()
        cc = _find_host_cc()
        if cc is None:
            raise KernelBuildError(
                "cl.exe still missing after Build Tools install "
                f"(winget exit {code}). Run scripts/setup.ps1 from an "
                "Administrator PowerShell, or install the C++ workload by hand."
            )
    else:
        print(f"host compiler: {cc}", flush=True)

    from gpu.cuda_env import inject_cuda_env

    inject_cuda_env()
    nvcc = _find_nvcc()
    if nvcc is None:
        if not install:
            raise KernelBuildError(
                "nvcc not found. Install CUDA Toolkit 12.4, or re-run without "
                "--no-install-tools."
            )
        code = _install_cuda()
        inject_cuda_env()
        nvcc = _find_nvcc()
        if nvcc is None:
            raise KernelBuildError(
                "nvcc still missing after CUDA Toolkit install "
                f"(winget exit {code}). Check CUDA_HOME / CUDA_PATH, or install "
                "CUDA 12.4 from NVIDIA (matches torch cu124)."
            )
    else:
        print(f"nvcc: {nvcc}", flush=True)


def _ensure_posix_tools() -> None:
    cc = _find_host_cc()
    nvcc = _find_nvcc()
    if cc and nvcc:
        print(f"host compiler: {cc}", flush=True)
        print(f"nvcc: {nvcc}", flush=True)
        return
    missing = []
    if cc is None:
        missing.append("g++")
    if nvcc is None:
        missing.append("nvcc")
    raise KernelBuildError(
        "missing " + " and ".join(missing) + ". On Linux install a C++ compiler "
        "and CUDA 12.x toolkit (nvcc), e.g. g++ plus the NVIDIA CUDA 12.4 "
        "toolkit (or nvidia-cuda-toolkit). This script does not sudo apt."
    )


def ensure_kernel(*, install: bool = True) -> Path:
    """Return the in-tree extension, installing tools and compiling if needed."""
    path, stale, abi_ok = _existing_ext()
    if path is not None and abi_ok and not stale:
        print(f"nf4 kernel: {path}", flush=True)
        return path

    if sys.platform == "darwin":
        raise KernelBuildError(
            "macOS has no CUDA NF4 kernel. compress still works; generate does not."
        )

    if os.name == "nt":
        _ensure_windows_tools(install=install)
    else:
        _ensure_posix_tools()

    code = _compile()
    if code != 0:
        raise KernelBuildError(
            f"gpu/nf4 setup.py build_ext failed (exit {code}). "
            "Need cl.exe + nvcc on Windows, or g++ + nvcc on Linux."
        )

    path, stale, abi_ok = _existing_ext()
    if path is None or not abi_ok:
        raise KernelBuildError(
            "build_ext reported success but no importable chr_nf4_ext was found "
            f"under {SETUP_PY.parent}."
        )
    print(f"nf4 kernel: {path}", flush=True)
    return path


def main(argv: list[str] | None = None) -> int:
    del argv
    install = "--no-install-tools" not in sys.argv[1:]
    try:
        path = ensure_kernel(install=install)
    except KernelBuildError as exc:
        _err(f"kernel_build: {exc}")
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

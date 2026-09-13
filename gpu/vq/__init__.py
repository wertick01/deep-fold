"""Python wrapper around ``chr_vq_gemm``.

``vq_gemm(index, book, x, M, K, K_pad) -> y`` with ``y`` BF16 ``[M, N]``.
Caller owns every tensor; the kernel does not allocate.

Compile (Windows, sm_86), from a VS x64 prompt or after vcvars64.bat:

    C:\\Users\\Professional\\anaconda3\\envs\\torch-gpu\\python.exe gpu/vq/setup.py build_ext --inplace

    Or just import this package / run ``gpu/vq/verify.py`` and let JIT build.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

_DIR = Path(__file__).resolve().parent
_REPO = _DIR.parents[1]
_INCLUDE = _REPO / "gpu" / "include"
_KERNEL_SOURCES = (
    _DIR / "bindings.cpp",
    _DIR / "vq_gemm.cu",
)
_SOURCES = _KERNEL_SOURCES + (_INCLUDE / "chr_gpu.h",)
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.win_toolchain import inject_msvc_env, which_cl  # noqa: E402

_ext: Any = None


def _sources_newer_than(artifact: Path, sources=_SOURCES) -> bool:
    if not artifact.is_file():
        return True
    t = artifact.stat().st_mtime
    return any(s.is_file() and s.stat().st_mtime > t for s in sources)


def _inplace_pyd() -> Path | None:
    matches = sorted(_DIR.glob("chr_vq_ext*.pyd"))
    return matches[0] if matches else None


def _import_pyd(directory: Path):
    matches = sorted(directory.glob("chr_vq_ext*.pyd"))
    if not matches:
        return None
    sys.path.insert(0, str(directory))
    import importlib

    if "chr_vq_ext" in sys.modules:
        del sys.modules["chr_vq_ext"]
    return importlib.import_module("chr_vq_ext")


def _try_import_pyd(directory: Path):
    matches = sorted(directory.glob("chr_vq_ext*.pyd"))
    if not matches or _sources_newer_than(matches[0]):
        return None
    return _import_pyd(directory)


def _jit_load():
    from torch.utils.cpp_extension import load

    cxx_flags = ["/O2"] if os.name == "nt" else ["-O3"]
    return load(
        name="chr_vq_ext",
        sources=[str(_DIR / "bindings.cpp"), str(_DIR / "vq_gemm.cu")],
        extra_include_paths=[str(_INCLUDE)],
        extra_cflags=cxx_flags,
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_86,code=sm_86",
            "--expt-relaxed-constexpr",
            "-lineinfo",
        ],
        verbose=True,
    )


def _load_ext():
    global _ext
    if _ext is not None:
        return _ext

    pyd = _inplace_pyd()
    if pyd is not None and not _sources_newer_than(pyd, _KERNEL_SOURCES):
        compiled = _import_pyd(_DIR)
        if compiled is not None:
            _ext = compiled
            return _ext

    from torch.utils.cpp_extension import _get_build_directory

    cached = _try_import_pyd(Path(_get_build_directory("chr_vq_ext", False)))
    if cached is not None:
        _ext = cached
        return _ext

    import warnings

    if inject_msvc_env() or which_cl():
        try:
            _ext = _jit_load()
            return _ext
        except Exception as exc:
            if pyd is None:
                raise
            warnings.warn(
                f"JIT rebuild of chr_vq_ext failed ({type(exc).__name__}: {exc}). "
                f"Loading {pyd.name}.",
                stacklevel=2,
            )

    if pyd is not None:
        warnings.warn(
            f"{pyd.name} is older than gpu/vq sources or chr_gpu.h. "
            "Loading the existing binary.",
            stacklevel=2,
        )
        compiled = _import_pyd(_DIR)
        if compiled is not None:
            _ext = compiled
            return _ext

    raise RuntimeError(
        "chr_vq_ext: no compiler (cl.exe) and no .pyd. "
        r'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
        r'\VC\Auxiliary\Build\vcvars64.bat" then re-run, or setup.py build_ext --inplace.'
    )


def vq_gemm(
    index: torch.Tensor,
    book: torch.Tensor,
    x: torch.Tensor,
    M: int,
    K: int,
    K_pad: int,
) -> torch.Tensor:
    """Fused VQ 2x8 dequant-MMA. ``x`` is BF16 ``[K, N]`` or ``[K]`` (N must be 1)."""
    n = 1 if x.dim() == 1 else int(x.size(-1))
    if n != 1:
        raise RuntimeError("chr_vq_gemm: N != 1 (prefill not implemented)")
    return _load_ext().vq_gemm(index, book, x, int(M), int(K), int(K_pad))

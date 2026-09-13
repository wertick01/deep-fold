"""Python wrapper around ``chr_nf4_gemm``.

``nf4_gemm(packed, scale, x, M, K, K_pad) -> y`` with ``y`` BF16 ``[M, N]``.
Caller owns every tensor; the kernel does not allocate.

Compile (Windows, sm_86), from a VS x64 prompt or after vcvars64.bat:

    C:\\Users\\Professional\\anaconda3\\envs\\torch-gpu\\python.exe gpu/nf4/setup.py build_ext --inplace

Or just import this package / run ``gpu/nf4/verify.py`` and let JIT build.
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
    _DIR / "nf4_gemm.cu",
)
_SOURCES = _KERNEL_SOURCES + (_INCLUDE / "chr_gpu.h",)
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.ext_bin import find_ext, have_host_compiler  # noqa: E402

_ext: Any = None


def _inplace_pyd() -> Path | None:
    matches = find_ext(_DIR, "chr_nf4_ext")
    return matches[0] if matches else None


def _sources_newer_than(artifact: Path, sources=_SOURCES) -> bool:
    if not artifact.is_file():
        return True
    t = artifact.stat().st_mtime
    return any(s.is_file() and s.stat().st_mtime > t for s in sources)


def _import_pyd(directory: Path):
    matches = find_ext(directory, "chr_nf4_ext")
    if not matches:
        return None
    sys.path.insert(0, str(directory))
    import importlib

    if "chr_nf4_ext" in sys.modules:
        del sys.modules["chr_nf4_ext"]
    return importlib.import_module("chr_nf4_ext")


def _try_import_pyd(directory: Path):
    matches = find_ext(directory, "chr_nf4_ext")
    if not matches or _sources_newer_than(matches[0]):
        return None
    return _import_pyd(directory)


def _jit_load():
    from torch.utils.cpp_extension import load

    cxx_flags = ["/O2"] if os.name == "nt" else ["-O3"]
    return load(
        name="chr_nf4_ext",
        sources=[str(_DIR / "bindings.cpp"), str(_DIR / "nf4_gemm.cu")],
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
    # Header-only comment edits must not block Jupyter: rebuild only if .cu/bindings moved.
    if pyd is not None and not _sources_newer_than(pyd, _KERNEL_SOURCES):
        compiled = _import_pyd(_DIR)
        if compiled is not None:
            _ext = compiled
            return _ext

    from torch.utils.cpp_extension import _get_build_directory

    cached = _try_import_pyd(Path(_get_build_directory("chr_nf4_ext", False)))
    if cached is not None:
        _ext = cached
        return _ext

    import warnings

    if have_host_compiler():
        try:
            _ext = _jit_load()
            return _ext
        except Exception as exc:
            if pyd is None:
                raise
            warnings.warn(
                f"JIT rebuild of chr_nf4_ext failed ({type(exc).__name__}: {exc}). "
                f"Loading {pyd.name}.",
                stacklevel=2,
            )

    if pyd is not None:
        warnings.warn(
            f"{pyd.name} is older than gpu/nf4 sources or chr_gpu.h. "
            "Loading the existing binary. Rebuild from a VS x64 prompt if numbers look wrong.",
            stacklevel=2,
        )
        compiled = _import_pyd(_DIR)
        if compiled is not None:
            _ext = compiled
            return _ext

    if os.name == "nt":
        raise RuntimeError(
            "chr_nf4_ext: no compiler (cl.exe) and no .pyd/.so. "
            r'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
            r'\VC\Auxiliary\Build\vcvars64.bat" then re-run, or pip/setup.py build_ext --inplace.'
        )
    raise RuntimeError(
        "chr_nf4_ext: no compiler (nvcc/g++) and no .pyd/.so. "
        "python gpu/nf4/setup.py build_ext --inplace"
    )


def nf4_gemm(
    packed: torch.Tensor,
    scale: torch.Tensor,
    x: torch.Tensor,
    M: int,
    K: int,
    K_pad: int,
) -> torch.Tensor:
    """Fused NF4 dequant-MMA. ``x`` is BF16 ``[K, N]`` or ``[K]`` (N in 1..16).

    ``N>16`` is a host problem: this wrapper raises; ``nf4_linear`` chunks.
    """
    n = 1 if x.dim() == 1 else int(x.size(-1))
    if n < 1 or n > 16:
        raise RuntimeError(
            f"chr_nf4_gemm: N={n} not in 1..16 (host must chunk N>16)"
        )
    return _load_ext().nf4_gemm(packed, scale, x, int(M), int(K), int(K_pad))


def nf4_plan(
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    N: int = 1,  # noqa: N803
    *,
    K_pad: int | None = None,  # noqa: N803
    have_ws: bool = True,
) -> dict:
    """The grid ``chr_nf4_gemm`` would launch. No device memory, no launch.

    Mirrored in pure Python by :mod:`gpu.nf4.plan` so the occupancy claim can be
    asserted without a GPU; ``gpu/nf4/test_plan.py`` checks the two agree.
    """
    from .plan import k_pad as _k_pad

    kp = _k_pad(K) if K_pad is None else int(K_pad)
    return _load_ext().nf4_plan(int(M), int(K), int(kp), int(N), bool(have_ws))


def nf4_set_tuning(path: int = 0, split_k: int = 0, one_wave: int = 0) -> None:
    """Override the planner for one process (microbench / A-B).

    Default (``path=0``, ``split_k=0``) is the occupancy fix: BM=64 decode tile
    plus split-K. ``path`` 1 = wave-2 BM=128, 2 = BM=64. ``split_k`` 0 = auto,
    >0 forced. ``one_wave`` 0 = 70 SMs on this card. Also ``CHR_NF4_PATH`` /
    ``CHR_NF4_SPLIT_K`` / ``CHR_NF4_ONE_WAVE`` before import.
    """
    _load_ext().nf4_set_tuning(int(path), int(split_k), int(one_wave))

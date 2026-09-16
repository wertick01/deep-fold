"""CPU fused NF4 GEMV (AVX2 + persistent std::thread pool). No CUDA, no GGUF.

``nf4_gemm(packed, scale, x, M, K, K_pad) -> y`` float32 ``[M, N]``.
Never materializes ``W_hat``. Call from ``gpu.host.cpu_linear``; TokenLoop
stays Python for RMS / RoPE / SDPA.

    C:\\Users\\Professional\\anaconda3\\envs\\torch-gpu\\python.exe gpu/cpu/setup.py build_ext --inplace
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

_DIR = Path(__file__).resolve().parent
_REPO = _DIR.parents[1]
_KERNEL_SOURCES = (
    _DIR / "bindings.cpp",
    _DIR / "nf4_gemv.cpp",
    _DIR / "nf4_gemv.h",
)
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.ext_bin import find_ext, have_host_compiler  # noqa: E402

_ext: Any = None
_load_error: str | None = None


def _inplace_pyd() -> Path | None:
    matches = find_ext(_DIR, "chr_nf4_cpu_ext")
    return matches[0] if matches else None


def _sources_newer_than(artifact: Path, sources=_KERNEL_SOURCES) -> bool:
    if not artifact.is_file():
        return True
    t = artifact.stat().st_mtime
    return any(s.is_file() and s.stat().st_mtime > t for s in sources)


def _import_pyd(directory: Path):
    matches = find_ext(directory, "chr_nf4_cpu_ext")
    if not matches:
        return None
    sys.path.insert(0, str(directory))
    import importlib

    if "chr_nf4_cpu_ext" in sys.modules:
        del sys.modules["chr_nf4_cpu_ext"]
    return importlib.import_module("chr_nf4_cpu_ext")


def _try_import_pyd(directory: Path):
    matches = find_ext(directory, "chr_nf4_cpu_ext")
    if not matches or _sources_newer_than(matches[0]):
        return None
    return _import_pyd(directory)


def _cxx_flags() -> list[str]:
    if os.name == "nt":
        return ["/O2", "/arch:AVX2", "/std:c++17", "/DCHR_NF4_FORCE_AVX2"]
    return ["-O3", "-mavx2", "-mfma", "-mf16c", "-std=c++17", "-DCHR_NF4_FORCE_AVX2"]


def _jit_load():
    from torch.utils.cpp_extension import load

    return load(
        name="chr_nf4_cpu_ext",
        sources=[str(_DIR / "bindings.cpp"), str(_DIR / "nf4_gemv.cpp")],
        extra_cflags=_cxx_flags(),
        verbose=True,
    )


def _load_ext():
    global _ext, _load_error
    if _ext is not None:
        return _ext
    if _load_error is not None:
        raise RuntimeError(_load_error)

    pyd = _inplace_pyd()
    if pyd is not None and not _sources_newer_than(pyd):
        compiled = _import_pyd(_DIR)
        if compiled is not None:
            _ext = compiled
            return _ext

    from torch.utils.cpp_extension import _get_build_directory

    cached = _try_import_pyd(Path(_get_build_directory("chr_nf4_cpu_ext", False)))
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
                _load_error = f"chr_nf4_cpu_ext JIT failed: {type(exc).__name__}: {exc}"
                raise RuntimeError(_load_error) from exc
            warnings.warn(
                f"JIT rebuild of chr_nf4_cpu_ext failed ({type(exc).__name__}: {exc}). "
                f"Loading {pyd.name}.",
                stacklevel=2,
            )

    if pyd is not None:
        compiled = _import_pyd(_DIR)
        if compiled is not None:
            _ext = compiled
            return _ext

    _load_error = (
        "chr_nf4_cpu_ext: no compiler and no .pyd/.so. "
        "python gpu/cpu/setup.py build_ext --inplace"
    )
    raise RuntimeError(_load_error)


def available() -> bool:
    """True when the AVX2 (or scalar C) extension can run. Does not JIT if a .pyd exists."""
    try:
        _load_ext()
        return True
    except Exception:
        return False


def isa() -> str:
    """``avx2`` or ``scalar``. Raises if the extension is missing."""
    return str(_load_ext().isa())


def nf4_gemm(
    packed: torch.Tensor,
    scale: torch.Tensor,
    x: torch.Tensor,
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    K_pad: int,  # noqa: N803
) -> torch.Tensor:
    """Fused CPU NF4 GEMV. ``x`` is ``[K, N]`` or ``[K]``. ``y`` is float32 ``[M, N]``."""
    n = 1 if x.dim() == 1 else int(x.size(-1))
    if n < 1 or n > 32:
        raise RuntimeError(f"chr_nf4_cpu: N={n} not in 1..32")
    return _load_ext().nf4_gemm(packed, scale, x, int(M), int(K), int(K_pad))


def i4c_gemm(
    packed: torch.Tensor,
    scale: torch.Tensor,
    x: torch.Tensor,
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    K_pad: int,  # noqa: N803
) -> torch.Tensor:
    """Fused CPU i4c GEMV. ``x`` is ``[K, 1]`` or ``[K]``. ``y`` is float32 ``[M, 1]``."""
    n = 1 if x.dim() == 1 else int(x.size(-1))
    if n != 1:
        raise RuntimeError(f"chr_i4c_cpu: N={n}, decode is N=1")
    return _load_ext().i4c_gemm(packed, scale, x, int(M), int(K), int(K_pad))

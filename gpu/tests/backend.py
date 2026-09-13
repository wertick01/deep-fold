"""Import shims for the two things the gate judges but does not own.

    gpu/chr0  (agent 1) -- materialize_nf4(path, name, device) -> ChrMatrix
    gpu/nf4   (agent 2) -- NF4 GEMM: packed + scale + x -> y

Neither exists yet at the time this gate was written, so both are probed at
runtime and reported as BLOCKED rather than crashing the run. The field names of
ChrMatrix are frozen by docs/spec/gpu-abi.md; the fallback below uses exactly
them, so a toy matrix built here is indistinguishable from a loaded one.

Kernel entry points understood by the adapter (agent 2 shipped the first one):

    gpu.nf4.nf4_gemm(packed, scale, x, M, K, K_pad) -> y   # blobs style
    gpu.nf4.gemm(w, x)                              -> y   # ChrMatrix style
    gpu.nf4.chr_nf4_gemm(w, x, y, N, stream=0)             # in-place, rc == 0

In every style x is BF16 [K, N] and y is BF16 [M, N] (docs/spec/stitch-gpu.md).
Override with DEEPFOLD_NF4_GEMM="module.path:attr" if the name ever changes.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

GEMM_CANDIDATE_MODULES = ("gpu.nf4", "gpu.nf4.gemm", "gpu.nf4.nf4_gemm", "gpu.nf4.api")
GEMM_CANDIDATE_ATTRS = ("gemm", "nf4_gemm", "chr_nf4_gemm", "nf4_linear", "linear", "matmul")
LOADER_CANDIDATE_MODULES = ("gpu.chr0", "gpu.chr0.loader", "gpu.chr0.chr0")


@dataclass
class ChrMatrixFallback:
    """Same field names as the frozen ChrMatrix ABI (docs/spec/gpu-abi.md)."""

    name: str
    M: int
    K: int
    K_pad: int
    packed: Any     # torch.Tensor uint8 [M, K_pad/2], cuda
    scale: Any      # torch.Tensor float16 [M, K_pad/64], cuda


@dataclass
class Probe:
    ok: bool
    obj: Any
    detail: str


def ensure_build_tools_on_path() -> None:
    """Put the interpreter's Scripts/Library\\bin ahead of PATH.

    gpu/nf4 JIT-compiles through torch.utils.cpp_extension, which shells out to
    `ninja`. In this env ninja lives in <prefix>\\Scripts, so the gate works
    under `conda activate torch-gpu` but not when python.exe is called by its
    absolute path -- and the failure surfaces as a random "Ninja is required" on
    whichever check happens to touch the kernel first.
    """
    prefix = os.path.dirname(os.path.abspath(sys.executable))
    extra = [prefix, os.path.join(prefix, "Scripts"), os.path.join(prefix, "Library", "bin")]
    path = os.environ.get("PATH", "")
    have = {p.lower() for p in path.split(os.pathsep)}
    missing = [p for p in extra if os.path.isdir(p) and p.lower() not in have]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing + [path])


def probe_loader() -> Probe:
    errors = []
    for modname in LOADER_CANDIDATE_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:
            errors.append(f"{modname}: {type(exc).__name__}: {exc}")
            continue
        fn = getattr(mod, "materialize_nf4", None)
        if callable(fn):
            return Probe(True, fn, f"{modname}.materialize_nf4")
        errors.append(f"{modname}: no materialize_nf4")
    return Probe(False, None, "; ".join(errors))


def _iter_gemm_candidates() -> list[tuple[str, Callable[..., Any]]]:
    found: list[tuple[str, Callable[..., Any]]] = []
    override = os.environ.get("DEEPFOLD_NF4_GEMM")
    if override:
        modname, _, attr = override.partition(":")
        mod = importlib.import_module(modname)
        found.append((override, getattr(mod, attr)))
        return found
    for modname in GEMM_CANDIDATE_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for attr in GEMM_CANDIDATE_ATTRS:
            fn = getattr(mod, attr, None)
            if callable(fn):
                found.append((f"{modname}.{attr}", fn))
    return found


def probe_gemm() -> Probe:
    ensure_build_tools_on_path()
    try:
        cands = _iter_gemm_candidates()
    except Exception as exc:
        return Probe(False, None, f"DEEPFOLD_NF4_GEMM override failed: {type(exc).__name__}: {exc}")
    if not cands:
        return Probe(
            False,
            None,
            "no NF4 gemm found; tried "
            + ", ".join(f"{m}.{{{'|'.join(GEMM_CANDIDATE_ATTRS)}}}" for m in GEMM_CANDIDATE_MODULES),
        )
    name, fn = cands[0]
    return Probe(True, GemmAdapter(name, fn), name)


class GemmAdapter:
    """Normalizes agent 2's entry point to `y_bf16 = adapter(w, x_bf16, n)`."""

    def __init__(self, name: str, fn: Callable[..., Any]):
        self.name = name
        self.fn = fn
        self.style: str | None = None

    def _guess_style(self) -> str:
        try:
            params = [
                p
                for p in inspect.signature(self.fn).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
        except (TypeError, ValueError):
            return "unknown"
        names = [p.name.lower() for p in params]
        if names[:3] == ["packed", "scale", "x"]:
            return "blobs"
        if len(params) <= 2:
            return "returns_y"
        if any(n in ("y", "out", "output") for n in names):
            return "inplace_y"
        if len(params) == 3 and names[2] in ("n", "batch", "n_cols", "ncols"):
            return "returns_y_n"
        return "inplace_y"

    def __call__(self, w: Any, x: Any, n: int = 1) -> Any:
        import torch

        if self.style is None:
            self.style = self._guess_style()
        attempts = (
            [self.style] if self.style != "unknown" else ["blobs", "returns_y", "returns_y_n", "inplace_y"]
        )
        last: Exception | None = None
        for style in attempts:
            try:
                if style == "blobs":
                    y = self.fn(w.packed, w.scale, x, int(w.M), int(w.K), int(w.K_pad))
                elif style == "returns_y":
                    y = self.fn(w, x)
                elif style == "returns_y_n":
                    y = self.fn(w, x, n)
                else:
                    y = torch.empty((int(w.M), n), dtype=torch.bfloat16, device=x.device)
                    rc = self.fn(w, x, y, n)
                    if isinstance(rc, int) and rc != 0:
                        raise RuntimeError(f"{self.name} returned {rc}")
                self.style = style
                return self._check(w, y, n)
            except TypeError as exc:
                last = exc
                continue
        raise TypeError(f"cannot call {self.name} with any known signature: {last}")

    @staticmethod
    def _check(w: Any, y: Any, n: int) -> Any:
        import torch

        if not isinstance(y, torch.Tensor):
            raise TypeError(f"gemm returned {type(y).__name__}, want torch.Tensor bf16 [M, N]")
        if y.dtype != torch.bfloat16:
            raise TypeError(f"y dtype {y.dtype} != bfloat16 (stitch-gpu.md: y is BF16 [M, N])")
        if y.numel() != int(w.M) * n:
            raise ValueError(f"y numel {y.numel()} != M*N = {int(w.M) * n}")
        return y.reshape(int(w.M), n)


def loaded_kernel_binary() -> tuple[str | None, list[str]]:
    """(path of the loaded extension .pyd/.so, kernel source paths).

    Needed because torch's JIT keeps an in-process version cache: if a rebuild
    fails (no MSVC on PATH, say), a second import can silently fall back to the
    previously built binary. A gate that certifies a stale binary certifies
    nothing, so oracle_gate checks these mtimes (K1).
    """
    path = None
    for name, mod in list(sys.modules.items()):
        if name.split(".")[-1] == "chr_nf4_ext":
            path = getattr(mod, "__file__", None)
            break
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    sources = [
        os.path.join(repo, "gpu", "nf4", "bindings.cpp"),
        os.path.join(repo, "gpu", "nf4", "nf4_gemm.cu"),
        os.path.join(repo, "gpu", "include", "chr_gpu.h"),
    ]
    return path, [s for s in sources if os.path.isfile(s)]


def preferred_matrix_cls() -> Any:
    """Agent 1's ChrMatrix when importable, so toys look exactly like real loads."""
    try:
        import importlib

        return getattr(importlib.import_module("gpu.chr0"), "ChrMatrix")
    except Exception:
        return ChrMatrixFallback


def make_matrix(
    name: str,
    packed_np: Any,
    scale_np: Any,
    m: int,
    k: int,
    device: str = "cuda",
    matrix_cls: Any = None,
) -> Any:
    """Build a device ChrMatrix from host blobs, without touching any .chr."""
    import numpy as np
    import torch

    k_pad = 64 * ((k + 63) // 64)
    packed = torch.from_numpy(np.ascontiguousarray(packed_np)).to(device)
    scale = torch.from_numpy(np.ascontiguousarray(scale_np.view(np.float16))).to(device)
    cls = matrix_cls or preferred_matrix_cls()
    try:
        return cls(name=name, M=m, K=k, K_pad=k_pad, packed=packed, scale=scale)
    except TypeError:
        return ChrMatrixFallback(name=name, M=m, K=k, K_pad=k_pad, packed=packed, scale=scale)

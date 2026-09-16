"""CPU i4c GEMV: affine INT4 packed from BF16. No CUDA, no GGUF.

    from gpu.host.i4c import pack_i4c_torch, i4c_gemm_cpu
"""

from __future__ import annotations

import numpy as np
import torch

from gpu.host.cpu_linear import LIVE_MAX_N, ensure_cpu_threads
from gpu.tests.i4c_oracle import GROUP, decode_i4c, k_pad_i4c, pack_i4c

__all__ = [
    "GROUP",
    "i4c_gemm_cpu",
    "i4c_linear_cpu",
    "k_pad_i4c",
    "pack_i4c_from_nf4",
    "pack_i4c_torch",
]


def pack_i4c_torch(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """BF16/FP32 ``[M, K]`` CPU → packed uint8, fp16 scale, K_pad."""
    if weight.device.type == "cuda":
        raise RuntimeError("pack_i4c_torch: weight must be CPU")
    w = weight.detach().to(torch.float32).contiguous().numpy()
    packed, scale = pack_i4c(w)
    return (
        torch.from_numpy(packed),
        torch.from_numpy(np.asarray(scale, dtype=np.float16)),
        k_pad_i4c(int(weight.shape[1])),
    )


def pack_i4c_from_nf4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    *,
    row_chunk: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Sidecar i4c from an NF4 CPU table. Not BF16; not written into ``.chr``.

    Live hybrid loads NF4 from the file, decodes row chunks, then packs i4c
    and drops the NF4 bytes. Quality is NF4∘i4c, not BF16∘i4c.
    """
    from .embedding import dequant_nf4_rows

    if packed.device.type == "cuda" or scale.device.type == "cuda":
        raise RuntimeError("pack_i4c_from_nf4: packed/scale must be CPU")
    m = int(M)
    k = int(K)
    chunk = max(1, int(row_chunk))
    parts_p: list[torch.Tensor] = []
    parts_s: list[torch.Tensor] = []
    k_pad = 0
    for lo in range(0, m, chunk):
        hi = min(m, lo + chunk)
        ids = torch.arange(lo, hi, dtype=torch.long)
        w = dequant_nf4_rows(packed, scale, ids, k, dtype=torch.float32)
        p, s = pack_i4c(w.numpy())
        k_pad = k_pad_i4c(k)
        parts_p.append(torch.from_numpy(np.ascontiguousarray(p)))
        parts_s.append(torch.from_numpy(np.ascontiguousarray(s, dtype=np.float16)))
        del w
    out_p = torch.cat(parts_p, dim=0)
    out_s = torch.cat(parts_s, dim=0)
    return out_p, out_s, int(k_pad)


def _i4c_gemm_python(
    packed: torch.Tensor,
    scale: torch.Tensor,
    x: torch.Tensor,
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype,
    row_chunk: int,
) -> torch.Tensor:
    x32 = x.reshape(k, n).to(torch.float32)
    y = torch.empty((m, n), dtype=torch.float32)
    chunk = max(1, int(row_chunk))
    for lo in range(0, m, chunk):
        hi = min(m, lo + chunk)
        w = torch.from_numpy(
            decode_i4c(
                packed[lo:hi].numpy(),
                scale[lo:hi].numpy(),
                hi - lo,
                k,
            )
        )
        y[lo:hi] = w @ x32
    return y.to(dtype)


def i4c_gemm_cpu(
    packed: torch.Tensor,
    scale: torch.Tensor,
    x: torch.Tensor,
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    K_pad: int,  # noqa: N803
    *,
    dtype: torch.dtype = torch.bfloat16,
    impl: str = "auto",
    row_chunk: int = 256,
) -> torch.Tensor:
    """``y[M, N] = dequant_i4c @ x[K, N]``. Fused decode is N=1; N>1 is python."""
    if impl not in ("auto", "avx2", "python"):
        raise ValueError(f"impl={impl!r}")
    ensure_cpu_threads()
    if packed.device.type == "cuda" or scale.device.type == "cuda" or x.device.type == "cuda":
        raise RuntimeError("i4c_gemm_cpu: packed/scale/x must be CPU")
    m = int(M)
    k = int(K)
    n = 1 if x.dim() == 1 else int(x.shape[-1])
    if n < 1:
        raise ValueError(f"i4c_gemm_cpu: empty N from x{tuple(x.shape)}")
    if n > LIVE_MAX_N:
        raise ValueError(f"i4c_gemm_cpu: N={n} > LIVE_MAX_N={LIVE_MAX_N}")
    if impl == "avx2" and n != 1:
        raise ValueError(f"i4c_gemm_cpu: AVX2 decode is N=1, got N={n}")
    if impl != "python" and n == 1:
        from gpu.cpu import available as avx2_available
        from gpu.cpu import i4c_gemm as avx2_gemm

        if impl == "avx2" or (impl == "auto" and avx2_available()):
            y32 = avx2_gemm(packed, scale, x.reshape(k, 1), int(M), int(K), int(K_pad))
            return y32.to(dtype)
    return _i4c_gemm_python(packed, scale, x, m, k, n, dtype, row_chunk)


def i4c_linear_cpu(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    K_pad: int,  # noqa: N803
    bias: torch.Tensor | None = None,
    *,
    row_chunk: int = 256,
) -> torch.Tensor:
    """HuggingFace ``[..., K]`` in, ``[..., M]`` out. Same layout dance as NF4."""
    if x.device.type == "cuda":
        raise RuntimeError(
            "i4c_linear_cpu: activation is CUDA; refuse silent H2D (bounce first)"
        )
    if x.shape[-1] != K:
        raise ValueError(f"x has {x.shape[-1]} features, this linear takes K={K}")
    lead = tuple(x.shape[:-1])
    n = 1
    for d in lead:
        n *= int(d)
    if n < 1:
        raise ValueError(f"i4c_linear_cpu: empty leading dims x{tuple(x.shape)}")

    if n == 1:
        xk = x.reshape(K, 1)
        y = i4c_gemm_cpu(packed, scale, xk, M, K, K_pad, row_chunk=row_chunk)
        out = y.view(M)
        if bias is not None:
            out = out + bias
        return out.view(*lead, M)

    x2 = x.contiguous().view(n, K)
    if n <= LIVE_MAX_N:
        xk = x2.transpose(0, 1).contiguous()
        y = i4c_gemm_cpu(packed, scale, xk, M, K, K_pad, row_chunk=row_chunk)
    else:
        cols = []
        for start in range(0, n, LIVE_MAX_N):
            sl = x2[start : start + LIVE_MAX_N]
            xk = sl.transpose(0, 1).contiguous()
            cols.append(
                i4c_gemm_cpu(packed, scale, xk, M, K, K_pad, row_chunk=row_chunk)
            )
        y = torch.cat(cols, dim=1)
    out = y.transpose(0, 1).contiguous()
    if bias is not None:
        out = out + bias
    return out.view(*lead, M)

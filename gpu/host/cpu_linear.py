"""CPU NF4 GEMM: row-chunked dequant + float32 matmul. No CUDA extension.

``N=1`` is a matvec. ``N=2..32`` is a thin GEMM with the same decode. A 32B
``gate_proj`` ``[27648, 5120]`` never materializes a 540 MiB float32 ``W_hat``.

GIL: torch CPU GEMM can release it; the Python chunk loop does not. Default
threads are ``min(16, cpu_count)`` (Ollama used 16 AVX2 threads on the 5950X).
Honor ``OMP_NUM_THREADS`` / ``torch.set_num_threads`` if already set below 16.
v1 is serial with the GPU prefix: join, then this GEMM, then the 10 KiB bounce.
"""

from __future__ import annotations

import os

import torch

from .embedding import dequant_nf4_rows

__all__ = [
    "GROUP_SIZE",
    "LIVE_MAX_N",
    "cpu_thread_count",
    "ensure_cpu_threads",
    "nf4_gemm_cpu",
    "nf4_linear_cpu",
]

GROUP_SIZE = 64
# Same cap as gpu.nf4.plan.LIVE_MAX_N. Do not import gpu.nf4 (CUDA extension).
LIVE_MAX_N = 32

_THREADS_READY = False


def cpu_thread_count() -> int:
    """Workers for the CPU suffix. Cap 16 so we do not fight a live CUDA prefix."""
    env = os.environ.get("OMP_NUM_THREADS") or os.environ.get("MKL_NUM_THREADS")
    if env:
        try:
            n = int(env)
            if n >= 1:
                return min(16, n)
        except ValueError:
            pass
    return min(16, os.cpu_count() or 1)


def ensure_cpu_threads() -> int:
    """Set ``torch.set_num_threads`` once. Does not oversubscribe 32 HT workers."""
    global _THREADS_READY
    n = cpu_thread_count()
    if not _THREADS_READY:
        torch.set_num_threads(n)
        _THREADS_READY = True
    return n


def nf4_gemm_cpu(
    packed: torch.Tensor,
    scale: torch.Tensor,
    x: torch.Tensor,
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    K_pad: int,  # noqa: N803
    *,
    row_chunk: int = 256,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``y = dequant_nf4(packed, scale) @ x`` on CPU.

    ``packed``: uint8 ``[M, K_pad/2]`` CPU, pageable.
    ``scale``: fp16 ``[M, n_groups]`` CPU.
    ``x``: bf16/fp32 ``[K, N]`` CPU, ``N`` in 1..32 (TokenLoop chunks above that).
    ``y``: ``dtype`` ``[M, N]`` (product path is bf16).
    """
    del K_pad  # packed shape is the source of truth; K_pad is the seat's value
    ensure_cpu_threads()
    if packed.device.type == "cuda" or scale.device.type == "cuda" or x.device.type == "cuda":
        raise RuntimeError(
            "nf4_gemm_cpu: packed/scale/x must be CPU; refuse silent H2D"
        )
    if packed.dtype is not torch.uint8 or packed.dim() != 2:
        raise TypeError(
            f"packed must be uint8 [M, K_pad/2], got {packed.dtype} {tuple(packed.shape)}"
        )
    if scale.dtype is not torch.float16:
        raise TypeError(f"scale must be float16, got {scale.dtype}")
    m = int(M)
    k = int(K)
    if int(packed.shape[0]) != m:
        raise ValueError(f"packed M {int(packed.shape[0])} != M={m}")
    if x.shape[0] != k:
        raise ValueError(f"x is [K={int(x.shape[0])}, N], this gemm takes K={k}")
    n = 1 if x.dim() == 1 else int(x.shape[-1])
    if n < 1:
        raise ValueError(f"nf4_gemm_cpu: empty N from x{tuple(x.shape)}")
    chunk = max(1, int(row_chunk))
    x32 = x.reshape(k, n).to(torch.float32)
    y = torch.empty((m, n), dtype=torch.float32)
    ids_dev = packed.device
    for lo in range(0, m, chunk):
        hi = min(lo + chunk, m)
        ids = torch.arange(lo, hi, device=ids_dev, dtype=torch.long)
        w = dequant_nf4_rows(packed, scale, ids, k, dtype=torch.float32)
        y[lo:hi] = w @ x32
    return y.to(dtype)


def nf4_linear_cpu(
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
    """HuggingFace ``[..., K]`` in, ``[..., M]`` out. Same layout dance as ``nf4_linear``."""
    if x.device.type == "cuda":
        raise RuntimeError(
            "nf4_linear_cpu: activation is CUDA; refuse silent H2D (bounce first)"
        )
    if x.shape[-1] != K:
        raise ValueError(f"x has {x.shape[-1]} features, this linear takes K={K}")
    lead = tuple(x.shape[:-1])
    n = 1
    for d in lead:
        n *= int(d)
    if n < 1:
        raise ValueError(f"nf4_linear_cpu: empty leading dims x{tuple(x.shape)}")

    if n == 1:
        xk = x.reshape(K, 1)
        y = nf4_gemm_cpu(packed, scale, xk, M, K, K_pad, row_chunk=row_chunk)
        out = y.view(M)
        if bias is not None:
            out = out + bias
        return out.view(*lead, M)

    x2 = x.contiguous().view(n, K)
    if n <= LIVE_MAX_N:
        xk = x2.transpose(0, 1).contiguous()
        y = nf4_gemm_cpu(packed, scale, xk, M, K, K_pad, row_chunk=row_chunk)
    else:
        cols = []
        for start in range(0, n, LIVE_MAX_N):
            sl = x2[start : start + LIVE_MAX_N]
            xk = sl.transpose(0, 1).contiguous()
            cols.append(
                nf4_gemm_cpu(packed, scale, xk, M, K, K_pad, row_chunk=row_chunk)
            )
        y = torch.cat(cols, dim=1)
    out = y.transpose(0, 1).contiguous()
    if bias is not None:
        out = out + bias
    return out.view(*lead, M)


def _bench_one(name: str, m: int, k: int, n: int, repeats: int = 3) -> float:
    """Wall ms for one shape. Prints ms only; does not invent tok/s."""
    import time

    from gpu.tests.nf4_oracle import k_pad, toy_nf4

    kp = k_pad(k)
    packed_np, scale_np = toy_nf4(m, k, seed=0, pad_garbage=False)
    packed = torch.from_numpy(packed_np)
    scale = torch.from_numpy(scale_np)
    x = torch.randn(k, n, dtype=torch.bfloat16)
    nf4_gemm_cpu(packed, scale, x, m, k, kp)  # warmup
    best = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        nf4_gemm_cpu(packed, scale, x, m, k, kp)
        ms = (time.perf_counter() - t0) * 1000.0
        best = ms if best is None else min(best, ms)
    print(f"{name} M={m} K={k} N={n}  {best:.2f} ms", flush=True)
    return float(best)


def main(argv: list[str] | None = None) -> int:
    """``python -m gpu.host.cpu_linear`` toy check; ``--bench`` prints 32B-shape ms."""
    import argparse

    p = argparse.ArgumentParser(prog="python -m gpu.host.cpu_linear")
    p.add_argument(
        "--bench",
        action="store_true",
        help="32B q_proj and down_proj, N=1 and N=32; print wall ms, no tok/s",
    )
    args = p.parse_args(argv)
    ensure_cpu_threads()
    print(f"cpu threads={torch.get_num_threads()} (cap 16; OMP_NUM_THREADS honored)", flush=True)
    if not args.bench:
        _bench_one("toy", 64, 64, 1)
        _bench_one("toy", 64, 64, 32)
        return 0
    # Qwen2.5-32B NF4 seats. Allocates toy packed, not a .chr.
    _bench_one("q_proj", 5120, 5120, 1)
    _bench_one("q_proj", 5120, 5120, 32)
    _bench_one("down_proj", 5120, 27648, 1)
    _bench_one("down_proj", 5120, 27648, 32)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

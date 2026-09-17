"""N=1 microbench: CUDA-core GEMV vs tensor-core GEMM. Random NF4, no .chr.

Median of several timed windows. Does not call empty_cache between kernels.

    python gpu/nf4/bench_gemv.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.nf4.numerics import QWEN25_3B_LINEARS, encode_nf4_chunked  # noqa: E402
from gpu.tests.nf4_oracle import k_pad  # noqa: E402
from gpu.tests.skips import cuda_reason  # noqa: E402

# Qwen2.5-32B-Instruct: hidden 5120, intermediate 27648, GQA 40/8 x 128.
QWEN25_32B_LINEARS: tuple[tuple[str, int, int], ...] = (
    ("32b_q", 5120, 5120),
    ("32b_k", 1024, 5120),
    ("32b_gate", 27648, 5120),
    ("32b_down", 5120, 27648),
)


def _window(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _median_us(fn, *, repeats: int = 5, warmup: int = 20, iters: int = 40) -> tuple[float, float, float]:
    samples = [_window(fn, warmup, iters) * 1000.0 for _ in range(repeats)]
    return statistics.median(samples), min(samples), max(samples)


def main() -> int:
    reason = cuda_reason()
    if reason is not None:
        print(f"SKIP {reason}")
        return 0
    from gpu.nf4 import nf4_gemm, nf4_gemv

    extra_3b = (("lm_head_rep", 16384, 2048), ("lm_head", 151936, 2048))
    catalogs = (
        ("3B", QWEN25_3B_LINEARS + extra_3b),
        ("32B-rep", QWEN25_32B_LINEARS),
    )
    print(
        f"{'kind':<12} {'M':>6} {'K':>6} {'mma_us':>10} {'gemv_us':>10} "
        f"{'gemv/mma':>8} {'GB/s':>8} {'gemv_lo':>8} {'gemv_hi':>8}"
    )
    rng = np.random.default_rng(0)
    for _label, rows in catalogs:
        for name, m, k in rows:
            w = rng.standard_normal((m, k), dtype=np.float32) * np.float32(0.05)
            x = rng.standard_normal((k, 1), dtype=np.float32)
            packed, scale = encode_nf4_chunked(w)
            pk = torch.from_numpy(packed).cuda()
            sc = torch.from_numpy(scale).cuda()
            xt = torch.from_numpy(x).to(device="cuda", dtype=torch.bfloat16)
            kp = k_pad(k)
            # Warm both paths before timing either, then alternate windows.
            for _ in range(5):
                nf4_gemm(pk, sc, xt, m, k, kp)
                nf4_gemv(pk, sc, xt, m, k, kp)
            torch.cuda.synchronize()
            mma, _, _ = _median_us(lambda: nf4_gemm(pk, sc, xt, m, k, kp))
            gemv, glo, ghi = _median_us(lambda: nf4_gemv(pk, sc, xt, m, k, kp))
            nbytes = int(pk.nbytes) + int(sc.nbytes) + int(xt.nbytes)
            gbps = nbytes / (gemv * 1e-6) / 1e9 if gemv > 0 else 0.0
            print(
                f"{name:<12} {m:6d} {k:6d} {mma:10.1f} {gemv:10.1f} "
                f"{gemv / mma:8.2f} {gbps:8.1f} {glo:8.1f} {ghi:8.1f}"
            )
            del w, packed, scale
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

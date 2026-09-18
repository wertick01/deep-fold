"""One Decode V2 GEMV/SwiGLU shape for Nsight Compute. L2-rotated. No .chr.

    python -m gpu.nf4.bench_gemv_ncu --kind down_proj --iters 8 --reps 1
    python -m gpu.nf4.bench_gemv_ncu --kind swiglu --iters 8 --reps 1
    python -m gpu.nf4.bench_gemv_ncu --kind o_proj --iters 8 --reps 1

``gpu.nf4.bench`` is tensor-core GEMM. This file is CUDA-core ``nf4_gemv`` /
``nf4_swiglu`` at 3B decode shapes. Weights rotate past L2 the same way as
the GEMM bench so a loop is not timed resident.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.nf4.bench import QWEN25_3B, Shape, make_weights  # noqa: E402
from gpu.nf4.plan import k_pad  # noqa: E402
from gpu.tests.skips import cuda_reason  # noqa: E402

KINDS = ("o_proj", "down_proj", "swiglu")


def _shape(name: str) -> Shape:
    for shape in QWEN25_3B:
        if shape.name == name:
            return shape
    raise KeyError(name)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gpu.nf4.bench_gemv_ncu")
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--l2-bytes", type=int, default=24 << 20)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reason = cuda_reason()
    if reason is not None:
        print(f"SKIP {reason}")
        return 0
    from gpu.nf4 import nf4_gemv, nf4_swiglu

    if args.kind == "swiglu":
        shape = _shape("gate_proj")
        gates = make_weights(shape, args.l2_bytes)
        ups = make_weights(shape, args.l2_bytes)
        copies = min(len(gates), len(ups))
        kp = k_pad(shape.K)
        xs = [torch.randn(shape.K, dtype=torch.bfloat16, device="cuda") for _ in range(copies)]
        ys = [torch.empty(shape.M, dtype=torch.bfloat16, device="cuda") for _ in range(copies)]

        def once(i: int) -> None:
            g_pk, g_sc = gates[i]
            u_pk, u_sc = ups[i]
            nf4_swiglu(g_pk, g_sc, u_pk, u_sc, xs[i], shape.M, shape.K, kp, ys[i])

        print(
            f"bench kind=swiglu M={shape.M} K={shape.K} copies={copies} "
            f"iters={args.iters} reps={args.reps}",
            flush=True,
        )
    else:
        shape = _shape(args.kind)
        weights = make_weights(shape, args.l2_bytes)
        copies = len(weights)
        kp = k_pad(shape.K)
        xs = [torch.randn(shape.K, dtype=torch.bfloat16, device="cuda") for _ in range(copies)]
        ys = [torch.empty(shape.M, dtype=torch.bfloat16, device="cuda") for _ in range(copies)]

        def once(i: int) -> None:
            packed, scale = weights[i]
            nf4_gemv(packed, scale, xs[i], shape.M, shape.K, kp, ys[i])

        print(
            f"bench kind={args.kind} M={shape.M} K={shape.K} copies={copies} "
            f"iters={args.iters} reps={args.reps}",
            flush=True,
        )

    for i in range(min(4, args.iters)):
        once(i % copies)
    torch.cuda.synchronize()
    nvtx = torch.cuda.nvtx
    nvtx.range_push("chr_gemv")
    for _ in range(max(1, args.reps)):
        for i in range(args.iters):
            once(i % copies)
        torch.cuda.synchronize()
    nvtx.range_pop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

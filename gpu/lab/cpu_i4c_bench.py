"""i4c vs NF4 CPU microbench, Qwen2.5-32B shapes, N=1. Not tok/s.

    python -m gpu.lab.cpu_i4c_bench --n1
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from gpu.host.cpu_linear import cpu_thread_count, ensure_cpu_threads, nf4_gemm_cpu
from gpu.host.i4c import i4c_gemm_cpu, pack_i4c_torch
from gpu.lab.script import RUNS_DIR
from gpu.tests.nf4_oracle import k_pad, toy_nf4

SHAPES = (
    ("q_proj", 5120, 5120),
    ("k_proj", 1024, 5120),
    ("v_proj", 1024, 5120),
    ("o_proj", 5120, 5120),
    ("gate_proj", 27648, 5120),
    ("up_proj", 27648, 5120),
    ("down_proj", 5120, 27648),
)
Q4K_LAYER_MS_BUDGET = 6.0


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m gpu.lab.cpu_i4c_bench")
    p.add_argument("--out", default="")
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args(argv)
    dest = Path(args.out) if args.out else Path(RUNS_DIR) / (
        "cpu-i4c-bench-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    dest.mkdir(parents=True, exist_ok=True)
    threads = ensure_cpu_threads()
    rows = []
    layer_i4c = 0.0
    layer_nf4 = 0.0
    for name, m, k in SHAPES:
        w = torch.randn(m, k, generator=torch.Generator().manual_seed(1))
        packed_i4, scale_i4, kp_i4 = pack_i4c_torch(w)
        x = torch.randn(k, 1)
        i4c_gemm_cpu(packed_i4, scale_i4, x, m, k, kp_i4, impl="avx2")
        samples = []
        for _ in range(int(args.reps)):
            t0 = time.perf_counter()
            i4c_gemm_cpu(packed_i4, scale_i4, x, m, k, kp_i4, impl="avx2")
            samples.append((time.perf_counter() - t0) * 1000.0)
        i4_ms = _median(samples)
        packed_nf, scale_nf = toy_nf4(m, k, seed=1, pad_garbage=False)
        packed_nf_t = torch.from_numpy(packed_nf)
        scale_nf_t = torch.from_numpy(scale_nf)
        kp_nf = k_pad(k)
        nf4_gemm_cpu(packed_nf_t, scale_nf_t, x, m, k, kp_nf, impl="avx2")
        nf_samples = []
        for _ in range(int(args.reps)):
            t0 = time.perf_counter()
            nf4_gemm_cpu(packed_nf_t, scale_nf_t, x, m, k, kp_nf, impl="avx2")
            nf_samples.append((time.perf_counter() - t0) * 1000.0)
        nf_ms = _median(nf_samples)
        layer_i4c += i4_ms
        layer_nf4 += nf_ms
        packed_bytes = m * (kp_i4 // 2)
        gbs = (packed_bytes / 1e6) / max(i4_ms, 1e-9)
        row = {
            "name": name,
            "M": m,
            "K": k,
            "i4c_ms": i4_ms,
            "nf4_ms": nf_ms,
            "i4c_packed_GBs": gbs,
        }
        rows.append(row)
        print(
            f"{name} i4c={i4_ms:.2f} ms  nf4={nf_ms:.2f} ms  packed={gbs:.1f} GB/s",
            flush=True,
        )
    payload = {
        "threads": threads,
        "cpu_thread_count": cpu_thread_count(),
        "rows": rows,
        "layer_ms_i4c": layer_i4c,
        "layer_ms_nf4": layer_nf4,
        "q4k_layer_ms_budget": Q4K_LAYER_MS_BUDGET,
    }
    print(
        f"layer i4c={layer_i4c:.1f} ms  nf4={layer_nf4:.1f} ms  "
        f"Q4_K budget={Q4K_LAYER_MS_BUDGET:.1f} ms",
        flush=True,
    )
    (dest / "bench.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

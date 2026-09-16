"""CPU NF4 microbench for 32B q_proj / down_proj. No TokenLoop, no CUDA.

    python -m gpu.lab.cpu_nf4_bench

Writes ms into --out (default under DEEPFOLD_RUNS). Not a tok/s.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from gpu.host.cpu_linear import (
    cpu_thread_count,
    ensure_cpu_threads,
    nf4_cpu_backend,
    nf4_gemm_cpu,
)
from gpu.lab.script import RUNS_DIR
from gpu.tests.nf4_oracle import k_pad, toy_nf4

# Qwen2.5-32B: hidden=5120, kv=1024, mlp=27648. N=1 decode.
SHAPES = (
    ("q_proj", 5120, 5120),
    ("k_proj", 1024, 5120),
    ("v_proj", 1024, 5120),
    ("o_proj", 5120, 5120),
    ("gate_proj", 27648, 5120),
    ("up_proj", 27648, 5120),
    ("down_proj", 5120, 27648),
)
Q4K_LAYER_MS_BUDGET = 6.0  # implied Ollama 32-CPU-layer split vs overflow 2.49


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def bench_one(
    m: int, k: int, n: int, reps: int, impl: str
) -> dict[str, float | int | str]:
    packed_np, scale_np = toy_nf4(m, k, seed=1, pad_garbage=False)
    packed = torch.from_numpy(packed_np)
    scale = torch.from_numpy(scale_np)
    x = torch.randn(k, n, dtype=torch.bfloat16)
    kp = k_pad(k)
    nf4_gemm_cpu(packed, scale, x, m, k, kp, impl=impl)  # warmup
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        nf4_gemm_cpu(packed, scale, x, m, k, kp, impl=impl)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "M": m,
        "K": k,
        "N": n,
        "impl": impl,
        "warmup_then_reps": reps,
        "ms_median": _median(samples),
        "ms_min": min(samples),
        "ms_max": max(samples),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m gpu.lab.cpu_nf4_bench")
    p.add_argument("--out", default="")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--n1", action="store_true", help="N=1 only (skip N=32 and python)")
    args = p.parse_args(argv)
    dest = Path(args.out) if args.out else Path(RUNS_DIR) / (
        "cpu-nf4-bench-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    dest.mkdir(parents=True, exist_ok=True)
    threads = ensure_cpu_threads()
    backend = nf4_cpu_backend()
    impls = ["avx2"] if backend in ("avx2", "scalar") else []
    if not args.n1:
        impls.append("python")
    ns = (1,) if args.n1 else (1, 32)
    rows = []
    for impl in impls:
        for name, m, k in SHAPES:
            for n in ns:
                row = {"name": name, **bench_one(m, k, n, int(args.reps), impl)}
                packed_bytes = m * (k_pad(k) // 2)
                gbs = (packed_bytes / 1e6) / max(row["ms_median"], 1e-9)
                row["packed_GBs"] = gbs
                rows.append(row)
                print(
                    f"{impl} {name} N={n} median={row['ms_median']:.1f} ms "
                    f"min={row['ms_min']:.1f} max={row['ms_max']:.1f} "
                    f"packed={gbs:.2f} GB/s",
                    flush=True,
                )
    layer_ms = sum(
        float(r["ms_median"])
        for r in rows
        if r.get("impl") == "avx2" and int(r.get("N", 0)) == 1
    )
    payload = {
        "threads": threads,
        "cpu_thread_count": cpu_thread_count(),
        "backend": backend,
        "rows": rows,
        "layer_ms_n1_avx2": layer_ms,
        "q4k_layer_ms_budget": Q4K_LAYER_MS_BUDGET,
        "note": "Not tok/s. Layer sum vs implied Ollama Q4_K ~6 ms/layer.",
    }
    print(
        f"layer N=1 avx2 sum={layer_ms:.1f} ms  Q4_K budget={Q4K_LAYER_MS_BUDGET:.1f} ms",
        flush=True,
    )
    (dest / "bench.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {dest} threads={threads}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

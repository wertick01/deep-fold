"""GEMM microbench for ``chr_nf4_gemm`` at the shapes a decode step actually runs.

    C:\\Users\\Professional\\anaconda3\\envs\\torch-gpu\\python.exe -m gpu.nf4.bench
    python -m gpu.nf4.bench --plan-only        # no GPU needed

``docs/tz/wave9-review.md`` §3: wave 2 launched ``grid = (ceil(M / 128), 1, 1)``
with ``split_k = 1``, so a Qwen2.5-3B decode token ran 16 CTAs on ``q``/``o``
and **2** on the GQA ``k``/``v`` against 70 SMs. This file measures the two
grids side by side on the same card, in one process, through
``nf4_set_tuning``:

``wave2``
    ``path=classic, split_k=1`` -- byte-for-byte the shipped wave-2 launch.
``wave9``
    ``path=auto`` -- the BM=64 tile plus ``grid.y = split_k`` for small ``M``.

**What is and is not measured.** This times *our* kernel only. There is no
Marlin, AWQ, bitsandbytes or llama.cpp number here and none is implied: the
comparison is our old grid against our current one.

``ms/token`` is the **serial** sum over the 253 NF4 matmuls of a 3B decode step.
It is not a ceiling and not a prediction in either direction: it excludes norms,
RoPE, SDPA, the KV write and Python, and it also ignores that
:class:`gpu.loop.graph.GemmGroup` forks ``q``/``k``/``v`` and ``gate``/``up``
onto separate streams (445 us back to back vs 162 us overlapped, measured there),
so a live token spends less GEMM time than this sum. The number worth reading is
the **ratio** between modes at identical overlap, i.e. none.

**Why the weight buffers rotate.** A 3B ``q_proj`` holds 2 MiB of packed
nibbles and this card has 6 MiB of L2, so timing one buffer in a loop measures
an L2-resident GEMM that decode never sees -- a real token walks 1.5 GiB of
distinct weights. Each shape therefore allocates enough copies to exceed
``--l2-bytes`` and the loop rotates through them.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.nf4.plan import CLASSIC, SMALL, ONE_WAVE, plan  # noqa: E402

__all__ = ["Shape", "QWEN25_3B", "bench_shape", "main"]

#: Wave-2 launch (``path=classic``) and the current planner, as nf4_set_tuning args.
MODES = {
    "wave2": {"path": CLASSIC + 1, "split_k": 1},  # 1 = force classic decode tile
    "wave9": {"path": 0, "split_k": 0},            # 0 = auto
    "small1": {"path": SMALL + 1, "split_k": 1},   # 2 = small tile, no split-K
    "smallk": {"path": SMALL + 1, "split_k": 0},   # small tile + auto split-K
}

#: ``nf4_set_tuning`` path code -> the planner's path constant, for the label.
_FORCED_PATH = {CLASSIC + 1: CLASSIC, SMALL + 1: SMALL, 0: None}


@dataclass(frozen=True)
class Shape:
    """One NF4 matrix of a decode step. ``per_token`` is how often it runs."""

    name: str
    M: int  # noqa: N815 - out_features
    K: int  # noqa: N815 - in_features
    per_token: int

    @property
    def weight_bytes(self) -> int:
        from gpu.nf4.plan import k_pad

        kp = k_pad(self.K)
        return self.M * kp // 2 + self.M * (kp // 64) * 2


# Qwen2.5-3B-Instruct: 36 layers, hidden 2048, intermediate 11008, 16 heads /
# 2 KV heads x 128 (config.json in C:\dev\models\Qwen2.5-3B-Instruct). These are
# the matrices behind the 253 GEMM launches per token that gpu/loop/graph.py
# groups into 145 groups.
QWEN25_3B = (
    Shape("q_proj", 2048, 2048, 36),
    Shape("k_proj", 256, 2048, 36),
    Shape("v_proj", 256, 2048, 36),
    Shape("o_proj", 2048, 2048, 36),
    Shape("gate_proj", 11008, 2048, 36),
    Shape("up_proj", 11008, 2048, 36),
    Shape("down_proj", 2048, 11008, 36),
    Shape("lm_head", 151936, 2048, 1),
)

#: The five *distinct* (M, K) pairs in the list above -- ``o_proj`` is shaped
#: like ``q_proj``, ``v_proj`` like ``k_proj``, ``up_proj`` like ``gate_proj``.
#: Timing these five is enough to price a whole token, which is what
#: :func:`_token_ms` does, so ``--shapes core`` is not a partial answer.
CORE = ("q_proj", "k_proj", "gate_proj", "down_proj", "lm_head")


def _token_ms(times: dict[tuple[int, int], float]) -> tuple[float, list[str]]:
    """(ms of GEMM per decode token, names with no timing for their shape).

    ``times`` is keyed by ``(M, K)`` at ``N == 1``. Every one of the 253 NF4
    matmuls in a 3B decode step is priced from its own shape, so a run that
    timed only the five distinct shapes still gets the full token.
    """
    total = 0.0
    missing = []
    for s in QWEN25_3B:
        us = times.get((s.M, s.K))
        if us is None:
            missing.append(s.name)
            continue
        total += us * s.per_token / 1000.0
    return total, missing


def _plan_table(shapes, ns, one_wave: int) -> None:
    print(f"launch plan (one_wave={one_wave} SMs, target 2 CTAs/SM)")
    head = f"{'matrix':10s} {'M':>7s} {'K':>6s} {'N':>3s} "
    head += f"{'wave2 grid':>12s} {'ctas':>6s} | {'wave9 grid':>12s} {'ctas':>6s} {'tile':>8s} {'ws KiB':>7s}"
    print(head)
    print("-" * len(head))
    for s in shapes:
        for n in ns:
            if n > 16:
                continue
            old = plan(s.M, s.K, n, have_ws=False, force_path=CLASSIC, one_wave=one_wave)
            new = plan(s.M, s.K, n, have_ws=True, one_wave=one_wave)
            print(
                f"{s.name:10s} {s.M:7d} {s.K:6d} {n:3d} "
                f"{f'({old.grid_x},{old.grid_y})':>12s} {old.ctas:6d} | "
                f"{f'({new.grid_x},{new.grid_y})':>12s} {new.ctas:6d} "
                f"{f'{new.bm}x{new.bk}':>8s} {new.ws_floats * 4 / 1024:7.1f}"
            )


def make_weights(shape: Shape, l2_bytes: int = 24 << 20, device: str = "cuda"):
    """Independent packed/scale pairs whose total exceeds ``l2_bytes``.

    Values only have to be legal nibbles and non-degenerate scales -- this file
    times the launch, gpu/nf4/verify.py is what judges the numbers.
    """
    import torch

    from gpu.nf4.plan import k_pad

    kp = k_pad(shape.K)
    copies = max(1, -(-l2_bytes // max(1, shape.weight_bytes)))
    g = torch.Generator(device=device)
    g.manual_seed(7 + shape.M)
    out = []
    for _ in range(copies):
        packed = torch.randint(
            0, 256, (shape.M, kp // 2), dtype=torch.int32, device=device, generator=g
        ).to(torch.uint8)
        scale = (
            0.04 + 0.20 * torch.rand(shape.M, kp // 64, device=device, generator=g)
        ).to(torch.float16)
        out.append((packed, scale))
    return out


def bench_shape(
    shape: Shape,
    n: int,
    mode: str,
    weights,
    *,
    iters: int = 64,
    reps: int = 5,
    one_wave: int = 0,
) -> float:
    """Median-of-``reps`` microseconds for one ``chr_nf4_gemm`` launch.

    ``n > 16`` is chunked into 16-column calls exactly as
    ``gpu.host.linear.nf4_linear`` does, so the number stays comparable.
    """
    import torch

    from gpu.nf4 import nf4_gemm, nf4_set_tuning
    from gpu.nf4.plan import k_pad

    nf4_set_tuning(one_wave=one_wave, **MODES[mode])

    kp = k_pad(shape.K)
    copies = len(weights)
    chunks = [min(16, n - c) for c in range(0, n, 16)]
    xs = [
        torch.randn(shape.K, c, dtype=torch.bfloat16, device="cuda") for c in chunks
    ]

    def once(w_idx: int) -> None:
        packed, scale = weights[w_idx]
        for x in xs:
            nf4_gemm(packed, scale, x, shape.M, shape.K, kp)

    for i in range(min(16, iters)):
        once(i % copies)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(reps):
        start.record()
        for i in range(iters):
            once(i % copies)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)  # us per launch
    return statistics.median(samples)


def _run(shapes, ns, modes, args) -> None:
    import torch

    dev = torch.cuda.get_device_properties(0)
    one_wave = args.one_wave or dev.multi_processor_count
    l2 = getattr(dev, "L2_cache_size", 0)
    print(f"device {dev.name}, {dev.multi_processor_count} SMs, "
          f"L2 {l2 / (1 << 20):.0f} MiB" if l2 else
          f"device {dev.name}, {dev.multi_processor_count} SMs")
    print(f"iters={args.iters} reps={args.reps} rotating weight copies to exceed "
          f"{args.l2_bytes / (1 << 20):.0f} MiB (L2 must not hold the weights)\n")

    _plan_table(shapes, ns, one_wave)
    print()

    decode_us: dict[str, dict[tuple[int, int], float]] = {m: {} for m in modes}
    head = f"{'matrix':10s} {'M':>7s} {'N':>3s} "
    for m in modes:
        head += f"{m + ' us':>12s} {m + ' GB/s':>12s} {m + ' ctas':>11s}"
    head += f"  {'speedup':>8s}"
    print(head)
    print("-" * len(head))

    for s in shapes:
        weights = make_weights(s, args.l2_bytes)
        for n in ns:
            row = f"{s.name:10s} {s.M:7d} {n:3d} "
            times = {}
            for m in modes:
                us = bench_shape(
                    s,
                    n,
                    m,
                    weights,
                    iters=args.iters,
                    reps=args.reps,
                    one_wave=one_wave,
                )
                times[m] = us
                if n == 1:
                    decode_us[m][(s.M, s.K)] = us
                gbs = s.weight_bytes / (us * 1e-6) / 1e9
                p = plan(
                    s.M,
                    s.K,
                    min(n, 16),
                    have_ws=MODES[m]["split_k"] != 1,
                    force_path=_FORCED_PATH[MODES[m]["path"]],
                    force_split=MODES[m]["split_k"],
                    one_wave=one_wave,
                )
                row += f"{us:12.1f} {gbs:12.1f} {f'({p.grid_x},{p.grid_y})':>11s}"
            if "wave2" in times and "wave9" in times and times["wave9"] > 0:
                row += f"  {times['wave2'] / times['wave9']:7.2f}x"
            print(row)
        del weights
        torch.cuda.empty_cache()

    if 1 not in ns:
        return
    print()
    print("tok-equivalent, decode N=1: 36 layers x (q,k,v,o,gate,up,down) + lm_head")
    print("Serial GEMM time only: no norm/RoPE/SDPA/KV/Python, and no stream")
    print("overlap, which GemmGroup does have -- so a live token is faster than")
    print("this sum. Read the ratio, not the tok/s. Our kernel vs our kernel.")
    per_token_ms = {}
    for m in modes:
        ms, missing = _token_ms(decode_us[m])
        if missing:
            print(f"  {m:7s} incomplete, no timing for {missing}")
            continue
        per_token_ms[m] = ms
        print(f"  {m:7s} {ms:8.2f} ms/token of serial GEMM")
    ref = per_token_ms.get("wave2")
    if ref:
        for m, ms in per_token_ms.items():
            if m != "wave2":
                print(f"  {m} vs wave2: {ref / ms:.2f}x less serial GEMM time")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", default="1,2,4,8,16",
                    help="sequence columns to time (32 is chunked 2x16)")
    ap.add_argument("--shapes", default="core", choices=("core", "all"),
                    help="core = the five distinct 3B matrices, all = 8 with duplicates")
    ap.add_argument("--modes", default="wave2,wave9",
                    help=f"comma list from {sorted(MODES)}")
    ap.add_argument("--iters", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--l2-bytes", type=int, default=24 << 20,
                    help="weight working set per shape; must exceed L2")
    ap.add_argument("--one-wave", type=int, default=0,
                    help="0 = the device's SM count")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the launch table and exit (no GPU)")
    args = ap.parse_args(argv)

    ns = [int(v) for v in args.n.split(",") if v.strip()]
    shapes = QWEN25_3B if args.shapes == "all" else [
        s for s in QWEN25_3B if s.name in CORE
    ]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        print(f"unknown mode(s) {bad}; pick from {sorted(MODES)}")
        return 2

    if args.plan_only:
        _plan_table(shapes, ns, args.one_wave or ONE_WAVE)
        return 0

    try:
        import torch
    except ImportError:
        print("SKIP: torch not importable; --plan-only still works")
        _plan_table(shapes, ns, args.one_wave or ONE_WAVE)
        return 0
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device. Launch plan only:\n")
        _plan_table(shapes, ns, args.one_wave or ONE_WAVE)
        return 0

    _run(shapes, ns, modes, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Gates after H2-accel: T_verify(k) vs T_step, and one HOST-down CPU NF4 GEMV.

    python -m gpu.lab.h2_gates --plan-only
    python -m gpu.lab.h2_gates --max-seq 512 --verify-tokens 32

Does not change product defaults. Writes $DEEPFOLD_RUNS/h2-gates-<stamp>/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from gpu.host.embedding import GROUP_SIZE, NF4_LEVELS, dequant_nf4_rows
from gpu.host.host_image import unpack_arena
from gpu.lab.h2_metrics import CHR_32B, MODEL_32B, dump_json, h2d_ms
from gpu.lab.script import MESSAGES, RUNS_DIR, chat_text
from gpu.nf4.plan import LIVE_MAX_N

__all__ = [
    "nf4_gemv_cpu",
    "main",
]

_LUT_F32 = torch.tensor(NF4_LEVELS, dtype=torch.float32)
_K_CHUNK = 512  # 8 NF4 groups; keeps a [M, 512] scratch, not [M, K]


def nf4_gemv_cpu(
    packed: torch.Tensor,
    scale: torch.Tensor,
    x: torch.Tensor,
    *,
    k: int | None = None,
) -> torch.Tensor:
    """N=1 NF4 GEMV on CPU. No dense ``[M, K]`` table.

    ``packed`` is uint8 ``[M, K_pad/2]``, ``scale`` fp16 ``[M, n_groups]``,
    ``x`` is ``[K]`` (or ``[K_pad]``) on CPU. Output float32 ``[M]``.
    """
    if packed.device.type != "cpu" or scale.device.type != "cpu":
        raise ValueError("nf4_gemv_cpu expects CPU packed/scale")
    packed = packed.contiguous()
    scale = scale.contiguous()
    m = int(packed.shape[0])
    k_pad = int(packed.shape[1]) * 2
    n_groups = k_pad // GROUP_SIZE
    if int(scale.shape[0]) != m or int(scale.shape[1]) != n_groups:
        raise ValueError(
            f"scale {tuple(scale.shape)} vs packed M={m} groups={n_groups}"
        )
    k_use = int(k) if k is not None else k_pad
    xv = x.reshape(-1).to(dtype=torch.float32, device="cpu")
    if int(xv.numel()) < k_use:
        raise ValueError(f"x has {int(xv.numel())} < K={k_use}")
    if int(xv.numel()) < k_pad:
        xp = torch.zeros(k_pad, dtype=torch.float32)
        xp[: int(xv.numel())] = xv
        xv = xp
    lut = _LUT_F32
    y = torch.zeros(m, dtype=torch.float32)
    scale_f = scale.to(torch.float32)
    for k0 in range(0, k_pad, _K_CHUNK):
        k1 = min(k0 + _K_CHUNK, k_pad)
        b0, b1 = k0 // 2, k1 // 2
        g0, g1 = k0 // GROUP_SIZE, k1 // GROUP_SIZE
        rows = packed[:, b0:b1]
        nib = torch.stack((rows & 0x0F, rows >> 4), dim=-1).reshape(m, k1 - k0).long()
        w = lut[nib]
        w = w.view(m, g1 - g0, GROUP_SIZE) * scale_f[:, g0:g1, None]
        y.add_(w.reshape(m, k1 - k0).matmul(xv[k0:k1]))
    return y


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gpu.lab.h2_gates")
    p.add_argument("--model", default=MODEL_32B)
    p.add_argument("--chr", dest="chr_path", default=CHR_32B)
    p.add_argument("--out", default="")
    p.add_argument("--max-seq", type=int, default=512)
    p.add_argument("--verify-tokens", type=int, default=32)
    p.add_argument("--k", default="1,2,4,8")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--no-graphs", dest="graphs", action="store_false")
    p.add_argument("--cpu-repeats", type=int, default=8)
    p.add_argument("--gpu-repeats", type=int, default=8)
    return p


def parse_k(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        n = int(part.strip())
        if n < 1 or n > int(LIVE_MAX_N):
            raise argparse.ArgumentTypeError(f"k={n} not in 1..{LIVE_MAX_N}")
        if n not in out:
            out.append(n)
    return out


def _out_dir(explicit: str) -> Path:
    if explicit:
        dest = Path(explicit)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = Path(RUNS_DIR) / f"h2-gates-{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_cpu_down(gemm) -> dict[str, Any]:
    """One HOST overflow matrix: CPU GEMV vs oracle, no GPU required for CPU half."""
    img = gemm.host_image
    packed, scale = unpack_arena(img.arena.cpu(), gemm.M, gemm.K, k_pad_=gemm.K_pad)
    torch.manual_seed(0)
    x = torch.randn(gemm.K, dtype=torch.float32)
    y = nf4_gemv_cpu(packed, scale, x, k=gemm.K)
    ids = torch.arange(gemm.M, dtype=torch.long)
    w = dequant_nf4_rows(packed, scale, ids, gemm.K, dtype=torch.float32)
    y_ref = w.matmul(x)
    err = float((y - y_ref).abs().max())
    rel = err / max(float(y_ref.abs().max()), 1e-6)
    for _ in range(2):
        nf4_gemv_cpu(packed, scale, x, k=gemm.K)
    nrep = 8
    t0 = time.perf_counter()
    for _ in range(nrep):
        nf4_gemv_cpu(packed, scale, x, k=gemm.K)
    cpu_ms = (time.perf_counter() - t0) * 1000.0 / nrep
    nbytes = int(img.nbytes)
    return {
        "name": gemm.name,
        "M": gemm.M,
        "K": gemm.K,
        "nbytes": nbytes,
        "nbytes_mib": nbytes / (1024 * 1024),
        "cpu_ms": cpu_ms,
        "cpu_threads": int(torch.get_num_threads()),
        "max_abs_vs_dequant": err,
        "rel_vs_dequant": rel,
        "h2d_floor_ms": h2d_ms(nbytes),
        "x_bytes": gemm.K * 2,
        "y_bytes": gemm.M * 2,
    }


def bench_gpu_down(loop, gemm, cpu_row: dict[str, Any], repeats: int) -> dict[str, Any]:
    from gpu.nf4 import nf4_gemm

    ring = loop._ring
    if ring is None:
        cpu_row["gpu_ms"] = None
        cpu_row["notes"] = "no CopyRing"
        return cpu_row
    x = torch.randn(gemm.K, 1, dtype=torch.bfloat16, device=loop.device)
    _sync()
    for _ in range(2):
        ring.arm([gemm])
        ring.prefetch()
        packed, scale = ring.bind_for_gemm(gemm)
        _ = nf4_gemm(packed, scale, x, gemm.M, gemm.K, gemm.K_pad)
        ring.record_gemm(gemm)
        _sync()
    walls = []
    copies = []
    for _ in range(int(repeats)):
        c0 = int(ring.total_copies)
        ring.arm([gemm])
        ring.prefetch()
        _sync()
        t0 = time.perf_counter()
        packed, scale = ring.bind_for_gemm(gemm)
        y = nf4_gemm(packed, scale, x, gemm.M, gemm.K, gemm.K_pad)
        ring.record_gemm(gemm)
        _sync()
        walls.append((time.perf_counter() - t0) * 1000.0)
        copies.append(int(ring.total_copies) - c0)
        del y
    cpu_row["gpu_ms"] = _mean(walls)
    cpu_row["gpu_ms_walls"] = walls
    cpu_row["gpu_copies"] = copies
    return cpu_row


def _pick_host_down(loop):
    for g in loop._host_tape:
        if g.home == "host" and "down" in g.name:
            return g
    for g in loop._host_tape:
        if g.home == "host":
            return g
    return None


def measure_tk(loop, prompt_ids: torch.Tensor, greedy: list[int], ks: list[int]) -> dict[str, Any]:
    from gpu.loop.speculate import measure_verify, verify_block, verify_stats

    out: dict[str, Any] = {"step": None, "forward_n1": None, "k": []}
    loop.reset()
    logits = loop.prefill(prompt_ids)
    step_walls = []
    _sync()
    token = int(logits.argmax())
    for _ in greedy:
        _sync()
        t0 = time.perf_counter()
        logits = loop.step(token)
        _sync()
        step_walls.append(time.perf_counter() - t0)
        token = int(logits.argmax())
    out["step"] = {
        "n": len(step_walls),
        "walls_ms": [w * 1000.0 for w in step_walls],
        "mean_ms": _mean([w * 1000.0 for w in step_walls[1:]]) or _mean(
            [w * 1000.0 for w in step_walls]
        ),
        "first_ms": step_walls[0] * 1000.0 if step_walls else None,
    }

    loop.reset()
    loop.prefill(prompt_ids)
    start = int(loop.kv.seq_len)
    ids1 = torch.tensor([greedy[0]], dtype=torch.long, device=loop.device)
    _sync()
    t0 = time.perf_counter()
    loop.forward(ids1, start, all_positions=True)
    _sync()
    first_fwd = (time.perf_counter() - t0) * 1000.0
    _sync()
    t0 = time.perf_counter()
    loop.forward(
        torch.tensor([greedy[1]], dtype=torch.long, device=loop.device),
        int(loop.kv.seq_len),
        all_positions=True,
    )
    _sync()
    out["forward_n1"] = {
        "first_ms": first_fwd,
        "second_ms": (time.perf_counter() - t0) * 1000.0,
    }

    for k in ks:
        loop.reset()
        loop.prefill(prompt_ids)
        warm = greedy[:k]
        verify_block(loop, warm, int(loop.kv.seq_len))
        _sync()
        loop.reset()
        loop.prefill(prompt_ids)
        measured = measure_verify(loop, greedy, k)
        walls_ms = [w * 1000.0 for w in measured["walls"]]
        rest = walls_ms[1:] if len(walls_ms) > 1 else walls_ms
        copies = measured.get("h2d_copies")
        nbytes = measured.get("h2d_bytes")
        stats = verify_stats(measured)
        row = {
            "k": k,
            "n_tokens": measured["n_tokens"],
            "n_blocks": measured["n_blocks"],
            "block_widths": measured.get("block_widths"),
            "walls_ms": walls_ms,
            "first_ms": walls_ms[0] if walls_ms else None,
            "mean_rest_ms": _mean(rest),
            "ms_per_token": stats["ms_per_token"],
            "measured_width": stats["measured_width"],
            "full_k": stats["full_k"],
            "h2d_copies": copies,
            "h2d_bytes": nbytes,
            "copies_per_block_rest": _mean([float(c) for c in copies[1:]])
            if copies and len(copies) > 1
            else (_mean([float(c) for c in copies]) if copies else None),
            "greedy_match": measured["greedy_match"],
        }
        out["k"].append(row)
        print(
            f"  k={k} first={row['first_ms']:.0f}ms rest_mean={row['mean_rest_ms']:.0f}ms "
            f"copies/block={row['copies_per_block_rest']} "
            f"ms/tok={row['ms_per_token']:.0f} full_k={row['full_k']} "
            f"match={row['greedy_match']}",
            flush=True,
        )
    return out


def run_live(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    from gpu.host import load_model
    from gpu.lab.h2_metrics import auto_max_resident_bytes
    from gpu.loop import TokenLoop
    from transformers import AutoTokenizer

    ks = parse_k(args.k)
    cap, cap_info = auto_max_resident_bytes(args.model, args.chr_path, args.max_seq)
    if cap is None:
        raise RuntimeError(f"expected overflow cap: {cap_info}")
    print(f"load cap_mib={cap / (1024**2):.1f} {cap_info.get('reason', '')[:80]}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    model, report = load_model(
        args.model,
        args.chr_path,
        strict=True,
        max_resident_bytes=int(cap),
        residency_policy="D",
    )
    loop = TokenLoop(model, max_seq=int(args.max_seq), overlap=True)
    print(
        f"loaded streamed={report.streamed} host_tape={len(loop._host_tape)} "
        f"prefill_chunk={loop.prefill_chunk}",
        flush=True,
    )
    loop.warmup(prompt=8, tokens=8)
    if args.graphs:
        loop.capture_graphs()
        print(f"graphs={loop.graph_mode}", flush=True)

    packed = chat_text(tokenizer, MESSAGES[0])
    prompt_ids = tokenizer(packed, return_tensors="pt", add_special_tokens=False).input_ids[0]
    print("generate greedy prefix (ignore EOS so k=8 is actually width 8)...", flush=True)
    gen = loop.generate(prompt_ids, int(args.verify_tokens), stop=())
    greedy = list(gen.tokens)
    print(
        f"greedy n={len(greedy)} prefill_ms={gen.prefill_ms:.0f} "
        f"decode_tok_s={gen.decode_tok_s:.3f}",
        flush=True,
    )

    print("T_k after reset+prefill (warmup N=k discarded)...", flush=True)
    tk = measure_tk(loop, prompt_ids.to(loop.device), greedy, ks)

    gemm = _pick_host_down(loop)
    cpu_row: dict[str, Any]
    if gemm is None:
        cpu_row = {"notes": "no HOST gemm on tape"}
    else:
        print(f"CPU GEMV {gemm.name} M={gemm.M} K={gemm.K}...", flush=True)
        cpu_row = bench_cpu_down(gemm)
        cpu_row = bench_gpu_down(loop, gemm, cpu_row, int(args.gpu_repeats))
        print(
            f"  cpu={cpu_row.get('cpu_ms'):.2f}ms gpu={cpu_row.get('gpu_ms')}ms "
            f"floor={cpu_row.get('h2d_floor_ms'):.2f}ms err={cpu_row.get('max_abs_vs_dequant'):.4g}",
            flush=True,
        )

    t1 = (tk.get("step") or {}).get("mean_ms")
    t8_row = next(
        (
            r
            for r in tk.get("k") or []
            if r.get("k") == 8 and r.get("full_k")
        ),
        None,
    )
    t8 = t8_row.get("mean_rest_ms") if t8_row else None
    ratio = (t8 / t1) if t1 and t8 else None
    cpu_ms = cpu_row.get("cpu_ms") if isinstance(cpu_row, dict) else None
    gate_spec = "blocked" if (ratio is None or ratio > 1.5) else "pass"
    gate_cpu = "blocked" if (cpu_ms is None or cpu_ms > 3.0) else "pass"
    payload = {
        "schema": "deepfold.h2_gates.v1",
        "model": args.model,
        "chr": args.chr_path,
        "max_seq": int(args.max_seq),
        "verify_tokens": int(args.verify_tokens),
        "cap": cap_info,
        "n_host": int(report.streamed),
        "graph_mode": loop.graph_mode,
        "greedy_n": len(greedy),
        "generate_tok_s": gen.decode_tok_s,
        "tk": tk,
        "cpu_gemv": cpu_row,
        "t_step_ms": t1,
        "t8_over_t1": ratio,
        "gate_spec": gate_spec,
        "gate_cpu_gemv": gate_cpu,
    }
    dump_json(out / "gates.json", payload)
    lines = [
        f"schema deepfold.h2_gates.v1  graphs={loop.graph_mode} n_host={report.streamed}",
        f"generate tok/s={gen.decode_tok_s:.3f} greedy_n={len(greedy)}",
        f"T_step mean_rest={t1:.1f} ms" if t1 else "T_step n/a",
        f"T_8/T_1={ratio:.2f}  spec_gate={gate_spec}" if ratio else f"spec_gate={gate_spec}",
        f"CPU GEMV {cpu_row.get('name')} {cpu_ms:.2f} ms  gpu={cpu_row.get('gpu_ms')}  "
        f"floor={cpu_row.get('h2d_floor_ms'):.2f}  gate={gate_cpu}"
        if cpu_ms is not None
        else f"CPU GEMV n/a gate={gate_cpu}",
    ]
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = _parser().parse_args(argv)
    out = _out_dir(args.out)
    if args.plan_only:
        print(f"plan-only out={out} (no CUDA)", flush=True)
        dump_json(
            out / "gates.json",
            {"schema": "deepfold.h2_gates.v1", "plan_only": True, "k": args.k},
        )
        return 0
    run_live(args, out)
    print(f"artifacts: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

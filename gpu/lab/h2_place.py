"""CPU comparison of H2 overflow WHO (which MLP stays HOST). No CUDA, no 32B load.

    python gpu/lab/h2_place.py
    python gpu/lab/h2_place.py --print
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.host.residency import (  # noqa: E402
    DEFAULT_POLICY,
    MIB,
    PIN_KINDS,
    POLICIES,
    ResidencyPlan,
    WeightDesc,
    descs_from_qwen,
    overflow_resident_cap,
    pin_nbytes,
    plan_residency,
    summarize_residency,
)
from gpu.lab.h2_metrics import H2D_CALIB_MIB, H2D_CALIB_MS, h2d_ms  # noqa: E402

__all__ = [
    "PRINT_ORDER",
    "QWEN_32B",
    "compare_policies",
    "format_span",
    "main",
    "overlap_story",
    "placement_row",
]

# Same shapes as gpu/host/test_residency.py (config.json only, no weights).
QWEN_32B = dict(
    hidden_size=5120,
    intermediate_size=27648,
    num_hidden_layers=64,
    num_attention_heads=40,
    num_key_value_heads=8,
    vocab_size=152064,
    tie_word_embeddings=False,
    model_type="qwen2",
)

PRINT_ORDER = (
    "D",
    "downs_only",
    "interleaved_down",
    "pairs_first",
    "all_mlp",
)


def format_span(layers: Sequence[int]) -> str:
    """Compress sorted layer ids: ``none``, ``7``, ``48..63``, ``0..3,8..11``."""
    xs = sorted({int(x) for x in layers})
    if not xs:
        return "none"
    ranges: list[tuple[int, int]] = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
            continue
        ranges.append((start, prev))
        start = prev = x
    ranges.append((start, prev))
    parts = [f"{a}..{b}" if a != b else str(a) for a, b in ranges]
    return ",".join(parts)


def overlap_story(descs: Sequence[WeightDesc], plan: ResidencyPlan) -> str:
    """One line: copy during resident qkv vs idle prefix then a tail burst."""
    summary = summarize_residency(descs, plan)
    downs = set(summary["host_layers"]["down"])
    gates = set(summary["host_layers"]["gate"])
    ups = set(summary["host_layers"]["up"])
    n_layers = max((int(d.layer) for d in descs if d.layer is not None), default=-1) + 1
    all_layers = set(range(n_layers))

    if not downs and not gates and not ups:
        return "no MLP overflow; copy engine idle."
    if downs == gates == ups == all_layers:
        return (
            "every layer streams gate+up+down (3 HOST GEMMs); copy cannot hide "
            "behind resident FFN; serial burst each layer after qkv."
        )
    if downs == all_layers and gates == ups:
        if not gates:
            return (
                "copy during resident qkv/attn/gate+up every layer (all downs "
                "HOST); depth-1 prefetch stays busy; no tail pair burst."
            )
        return (
            f"copy during resident qkv/attn every layer (all {n_layers} downs); "
            f"L{format_span(sorted(gates))} also HOST gate+up (3-matrix tail burst)."
        )
    if not downs and gates == ups and gates:
        prefix = sorted(all_layers - gates)
        gspan = format_span(sorted(gates))
        if prefix:
            return (
                f"L{format_span(prefix)} all-resident (depth-1 prefetch sits on "
                f"first HOST gate, then CE waits); L{gspan} HOST gate+up after "
                "attn, downs DEVICE."
            )
        return (
            f"HOST gate+up L{gspan}, all downs DEVICE; copy after attn, not "
            "during qkv."
        )
    if downs and not gates and not ups:
        dspan = format_span(sorted(downs))
        idle = sorted(all_layers - downs)
        if idle and min(downs) > min(idle):
            return (
                f"HOST downs L{dspan} only; prefix all-resident so CE idle, then "
                "tail-down copies (worse spread than all-down D)."
            )
        if idle and max(downs) < max(idle):
            return (
                f"HOST downs L{dspan} (head-first eviction); tape is still "
                "L-index consume order, not eviction order; tail layers idle."
            )
        return f"HOST downs L{dspan}; copy during qkv on those layers only."
    return (
        f"HOST down {format_span(sorted(downs))}; "
        f"gate+up {format_span(sorted(gates))}."
    )


def placement_row(
    policy: str,
    descs: Sequence[WeightDesc],
    plan: ResidencyPlan,
) -> dict:
    summary = summarize_residency(descs, plan)
    kinds = summary["kinds"]
    host_layers = summary["host_layers"]
    n_down = int(kinds.get("down", {}).get("host", {}).get("n") or 0)
    n_gate = int(kinds.get("gate", {}).get("host", {}).get("n") or 0)
    n_up = int(kinds.get("up", {}).get("host", {}).get("n") or 0)
    streamed = int(plan.streamed_bytes)
    return {
        "policy": policy,
        "n_streamed": int(len(plan.streamed)),
        "streamed_mib": streamed / MIB,
        "n_host_down": n_down,
        "n_host_gate": n_gate,
        "n_host_up": n_up,
        "down_span": format_span(host_layers["down"]),
        "gate_span": format_span(host_layers["gate"]),
        "up_span": format_span(host_layers["up"]),
        "h2d_ms": h2d_ms(streamed),
        "overlap": overlap_story(descs, plan),
        "streamed_bytes": streamed,
        "resident_bytes": int(plan.resident_bytes),
        "slot_nbytes": int(plan.slot_nbytes),
    }


def compare_policies(
    descs: Sequence[WeightDesc],
    cap: int,
    policies: Sequence[str] = PRINT_ORDER,
) -> list[dict]:
    rows = []
    for name in policies:
        if name not in POLICIES:
            raise ValueError(f"unknown policy {name!r}")
        plan = plan_residency(descs, cap, policy=name)
        rows.append(placement_row(name, descs, plan))
    return rows


def _recommend(rows: Sequence[dict]) -> str:
    """Keep D unless another policy is strictly fewer HOST GEMMs at same bytes."""
    by_name = {r["policy"]: r for r in rows}
    seed = by_name[DEFAULT_POLICY]
    for r in rows:
        if r["policy"] == DEFAULT_POLICY:
            continue
        fewer_gemms = r["n_streamed"] < seed["n_streamed"]
        same_or_less_bytes = r["streamed_bytes"] <= seed["streamed_bytes"]
        if fewer_gemms and same_or_less_bytes:
            return (
                f"switch default to {r['policy']}: fewer HOST GEMMs "
                f"({r['n_streamed']} vs {seed['n_streamed']}) at "
                f"{'same' if r['streamed_bytes'] == seed['streamed_bytes'] else 'fewer'} "
                "HOST bytes."
            )
    return (
        "keep D: same or fewer HOST bytes than every named extra, and the only "
        "WHO that streams a down on every layer so depth-1 copy stays busy "
        "during resident qkv/attn. pairs_first matches bytes but parks the "
        "prefetch on the first tail gate. all_mlp doubles the tape."
    )


def _fmt_table(rows: Sequence[dict]) -> str:
    headers = (
        "policy",
        "n",
        "MiB",
        "down",
        "gate",
        "up",
        "downs",
        "gate+up",
        "H2D ms",
    )
    body = []
    for r in rows:
        body.append(
            (
                r["policy"],
                str(r["n_streamed"]),
                f"{r['streamed_mib']:.1f}",
                str(r["n_host_down"]),
                str(r["n_host_gate"]),
                str(r["n_host_up"]),
                r["down_span"],
                r["gate_span"],
                f"{r['h2d_ms']:.1f}",
            )
        )
    widths = [len(h) for h in headers]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    out = [line(headers), line(tuple("-" * w for w in widths))]
    out.extend(line(row) for row in body)
    return "\n".join(out)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python gpu/lab/h2_place.py")
    p.add_argument(
        "--print",
        action="store_true",
        dest="do_print",
        default=True,
        help="print the comparison table (default)",
    )
    p.add_argument(
        "--vram-mib",
        type=int,
        default=12288,
        help="VRAM budget for overflow_resident_cap (default 12288)",
    )
    p.add_argument(
        "--max-seq",
        type=int,
        default=2048,
        help="KV length for overflow_resident_cap (default 2048)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    descs = descs_from_qwen(QWEN_32B)
    cap = overflow_resident_cap(int(args.vram_mib), int(args.max_seq), descs)
    pin = pin_nbytes(descs)
    rows = compare_policies(descs, cap)
    pin_plan = plan_residency(descs, cap, policy="D")
    pin_ok = all(
        d.name in pin_plan.resident for d in descs if d.kind in PIN_KINDS
    )

    print("H2-8 MLP placement vs seed D (Qwen2.5-32B shapes, CPU)")
    print(
        f"cap overflow_resident_cap({args.vram_mib}, {args.max_seq}) = "
        f"{cap} bytes ({cap / MIB:.2f} MiB); pin = {pin / MIB:.2f} MiB"
    )
    print(
        f"H2D {H2D_CALIB_MIB:.0f} MiB / {H2D_CALIB_MS:.2f} ms; "
        "unit = whole packed+scale; pin set DEVICE; gate+up not split"
    )
    print(f"pin set DEVICE on D: {pin_ok}")
    print()
    if args.do_print:
        print(_fmt_table(rows))
        print()
        for r in rows:
            print(f"{r['policy']}: {r['overlap']}")
        print()
    print("recommend:", _recommend(rows))

    # Eviction order vs consume-tape: only visible when some downs stay DEVICE.
    pair = 2 * next(d.nbytes for d in descs if d.kind == "gate")
    down_n = next(d.nbytes for d in descs if d.kind == "down")
    partial_cap = pin + 64 * pair + 32 * down_n
    d_partial = plan_residency(descs, partial_cap, policy="D")
    i_partial = plan_residency(descs, partial_cap, policy="interleaved_down")
    d_sum = summarize_residency(descs, d_partial)
    i_sum = summarize_residency(descs, i_partial)
    print()
    print("eviction order vs tape (toy cap: 32 downs HOST, all pairs DEVICE)")
    print(
        f"  D                 HOST downs {format_span(d_sum['host_layers']['down'])}"
        f"  tape[0]={d_partial.streamed[0] if d_partial.streamed else 'empty'}"
    )
    print(
        f"  interleaved_down  HOST downs {format_span(i_sum['host_layers']['down'])}"
        f"  tape[0]={i_partial.streamed[0] if i_partial.streamed else 'empty'}"
    )
    print(
        "  tape is TokenLoop consume order (increasing layer index), "
        "never tail-first eviction order. On the live overflow cap every "
        "down is HOST, so D and interleaved_down are the same WHO and tape."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

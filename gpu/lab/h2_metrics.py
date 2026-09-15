"""H2 overflow telemetry: PCIe calibration, snapshots, data-path prose.

Stdlib + optional torch/nvidia-smi. No generate. The 32B runner is
``gpu.lab.h2_trace``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from gpu.cli.paths import models_root
from gpu.host.host_image import cpu_is_pinned
from gpu.host.residency import MIB

__all__ = [
    "H2D_CALIB_MIB",
    "H2D_CALIB_MS",
    "MODEL_32B",
    "CHR_32B",
    "auto_max_resident_bytes",
    "canary_overflow_bytes",
    "dump_json",
    "expected_h2d_bytes",
    "h2d_ms",
    "host_pin_snapshot",
    "pcie_gb_s",
    "ram_snapshot",
    "render_data_path",
    "smi_snapshot",
    "torch_snapshot",
]

# docs/plan-h2-ring.md: pinned H2D 256 MiB / 10.31 ms on this 3080.
H2D_CALIB_MIB = 256.0
H2D_CALIB_MS = 10.31

MODEL_32B = os.environ.get(
    "DEEPFOLD_MODEL_32B", str(models_root() / "Qwen2.5-32B-Instruct")
)
CHR_32B = os.environ.get("DEEPFOLD_CHR_32B", str(models_root() / "qwen25-32b.nf4.chr"))

_SMI_FIELDS = (
    "name",
    "driver_version",
    "compute_mode",
    "display_active",
    "display_mode",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "utilization.memory",
    "clocks.sm",
    "clocks.mem",
    "pcie.link.gen.current",
    "pcie.link.width.current",
    "pstate",
    "power.draw",
    "temperature.gpu",
)


def h2d_ms(nbytes: int | float) -> float:
    """Wall of one pinned copy at the 256 MiB / 10.31 ms calibration."""
    return (float(nbytes) / MIB) * H2D_CALIB_MS / H2D_CALIB_MIB


def pcie_gb_s() -> float:
    """Calibration as GiB/s. Plan writes GB/s: 256 MiB / 10.31 ms ≈ 24.3."""
    return (H2D_CALIB_MIB * MIB) / (H2D_CALIB_MS / 1000.0) / (1024**3)


def expected_h2d_bytes(*, streamed_bytes: int, forwards: int) -> int:
    """One full overflow tape per TokenLoop forward (prefill chunk or decode)."""
    return int(streamed_bytes) * int(forwards)


def dump_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def auto_max_resident_bytes(
    model_dir: str | Path,
    chr_path: str | Path,
    max_seq: int,
    *,
    vram_mib: int | None = None,
) -> tuple[int | None, dict]:
    """Same rule as ``deepfold run``: cap only when ``decide().overflow``."""
    from gpu.cli.codec import CodecFitError, decide, detect_vram_mib, load_config
    from gpu.host.residency import overflow_cap_from_chr

    vram = int(vram_mib if vram_mib is not None else detect_vram_mib())
    info: dict[str, Any] = {
        "vram_mib": vram,
        "max_seq": int(max_seq),
        "overflow": False,
        "cap_bytes": None,
        "cap_mib": None,
        "codec": None,
        "reason": "",
    }
    try:
        decision = decide(load_config(model_dir), vram, requested="auto")
    except (CodecFitError, OSError, KeyError, TypeError, ValueError) as exc:
        info["reason"] = f"{type(exc).__name__}: {exc}"
        return None, info
    info["reason"] = decision.reason
    info["overflow"] = bool(decision.overflow)
    info["codec"] = decision.codec
    if not decision.overflow:
        return None, info
    cap = overflow_cap_from_chr(str(chr_path), vram, int(max_seq))
    info["cap_bytes"] = cap
    info["cap_mib"] = cap / MIB
    return cap, info


def canary_overflow_bytes(chr_path: str | Path) -> int:
    """3B-style fake cap: keep pin kinds + 4 resident gate/up pairs.

    Packed NF4 still fits the card; this forces policy D overflow without
    changing decide(). Same formula as ``gpu.host.test_slots`` GPU canary.
    """
    from gpu.chr0 import load_header
    from gpu.host.residency import descs_from_header, pin_nbytes

    descs = descs_from_header(load_header(str(chr_path)))
    pin = pin_nbytes(descs)
    gate_n = next((int(d.nbytes) for d in descs if d.kind == "gate"), None)
    if gate_n is None:
        raise ValueError(f"{chr_path}: no gate tensor for canary overflow cap")
    return int(pin + 4 * (2 * gate_n))


def ram_snapshot() -> dict[str, Any]:
    """Windows physical RAM + pagefile. Empty dict fields on failure."""
    out: dict[str, Any] = {}
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        if not ok:
            out["error"] = "GlobalMemoryStatusEx failed"
            return out
        out["ram_total_mib"] = stat.ullTotalPhys / MIB
        out["ram_avail_mib"] = stat.ullAvailPhys / MIB
        out["ram_load_pct"] = int(stat.dwMemoryLoad)
        out["pagefile_total_mib"] = stat.ullTotalPageFile / MIB
        out["pagefile_avail_mib"] = stat.ullAvailPageFile / MIB
    except Exception as exc:  # noqa: BLE001 -- dump what we can
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def smi_snapshot() -> dict[str, Any]:
    """One nvidia-smi identity + clocks + PCIe + memory row."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                f"--query-gpu={','.join(_SMI_FIELDS)}",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    cells = [c.strip() for c in line.split(",")]
    if len(cells) != len(_SMI_FIELDS):
        return {"error": f"expected {len(_SMI_FIELDS)} cells, got {len(cells)}", "raw": line}
    return dict(zip(_SMI_FIELDS, cells))


def torch_snapshot() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    if not torch.cuda.is_available():
        return {"cuda": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "device_name": props.name,
        "total_memory_mib": props.total_memory / MIB,
        "major": int(props.major),
        "minor": int(props.minor),
        "multi_processor_count": int(props.multi_processor_count),
        "allocated_mib": torch.cuda.memory_allocated() / MIB,
        "reserved_mib": torch.cuda.memory_reserved() / MIB,
        "max_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
        "max_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
    }


def host_pin_snapshot(model) -> dict[str, Any]:
    """Pinned HostImage arenas currently attached to modules."""
    n_img = 0
    bytes_total = 0
    bytes_pinned = 0
    for mod in model.modules():
        img = getattr(mod, "host_image", None)
        if img is None:
            continue
        n_img += 1
        n = int(img.nbytes)
        bytes_total += n
        arena = img.arena
        if cpu_is_pinned(arena):
            bytes_pinned += n
    return {
        "host_images": n_img,
        "host_bytes": bytes_total,
        "host_mib": bytes_total / MIB,
        "pinned_bytes": bytes_pinned,
        "pinned_mib": bytes_pinned / MIB,
        "all_pinned": n_img == 0 or bytes_pinned == bytes_total,
    }


def system_snapshot(*, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "smi": smi_snapshot(),
        "ram": ram_snapshot(),
        "torch": torch_snapshot(),
    }


def _mib(n: int | float | None) -> str:
    if n is None:
        return "?"
    return f"{float(n) / MIB:.1f} MiB"


def _kind_line(kinds: Mapping[str, Mapping[str, Mapping[str, int]]], kind: str) -> str:
    slot = kinds.get(kind) or {}
    dev = slot.get("device") or {"n": 0, "bytes": 0}
    host = slot.get("host") or {"n": 0, "bytes": 0}
    return (
        f"{kind}: DEVICE {dev['n']} ({_mib(dev['bytes'])}), "
        f"HOST {host['n']} ({_mib(host['bytes'])})"
    )


def render_data_path(snap: Mapping[str, Any]) -> str:
    """Human dump of pin → copy_stream → slot → chr_nf4_gemm on this machine."""
    plan = snap.get("residency") or {}
    kinds = plan.get("kinds") or {}
    cap = snap.get("cap") or {}
    loop = snap.get("loop") or {}
    pin = snap.get("pin") or {}
    messages: Sequence[Mapping[str, Any]] = snap.get("messages") or ()
    streamed = int(plan.get("streamed_bytes") or 0)
    slot_n = int(plan.get("slot_nbytes") or 0)
    tape_ms = h2d_ms(streamed)
    slot_ms = h2d_ms(slot_n)
    host_layers = plan.get("host_layers") or {}
    downs = host_layers.get("down") or []
    gates = host_layers.get("gate") or []
    ring = (loop.get("ring") or {}) if isinstance(loop, Mapping) else {}
    n_slots = int(ring.get("n_slots") or snap.get("n_slots") or 2)
    max_ahead = int(ring.get("max_ahead") or snap.get("max_ahead") or max(1, n_slots - 1))
    n_cs = int(
        ring.get("n_copy_streams")
        or snap.get("n_copy_streams")
        or (2 if n_slots >= 3 else 1)
    )
    stream_word = "stream" if n_cs == 1 else "streams"

    lines = [
        "# H2 data path (this run)",
        "",
        "Pinned host image of overflow NF4 (packed‖scale, one arena per matrix).",
        f"{n_slots} static device slots. {n_cs} copy {stream_word}. "
        f"Prefetch depth {max_ahead}. One H2D engine.",
        "Dequant only in registers/smem of `chr_nf4_gemm`. No dense `[M,K]` in HBM.",
        "",
        "## Machine",
        "",
        f"- PCIe calibration: {H2D_CALIB_MIB:.0f} MiB / {H2D_CALIB_MS:.2f} ms "
        f"= {pcie_gb_s():.1f} GB/s (GiB/s) pinned H2D",
        f"- formula: `t_ms = size_MiB × {H2D_CALIB_MS} / {H2D_CALIB_MIB:.0f}`",
        f"- decide(): overflow={cap.get('overflow')} codec={cap.get('codec')} "
        f"vram={cap.get('vram_mib')} MiB",
        f"- cap: {cap.get('cap_mib')} MiB resident packed "
        f"({cap.get('reason', '')})",
        "",
        "## Residency (policy D)",
        "",
        f"- streamed matrices: {plan.get('n_streamed')} "
        f"({_mib(streamed)})",
        f"- resident packed (plan): {_mib(plan.get('resident_bytes'))}",
        f"- slot (worst overflow matrix): {_mib(slot_n)} → {slot_ms:.2f} ms H2D",
        f"- full overflow tape H2D: {_mib(streamed)} → **{tape_ms:.1f} ms** "
        "if serial and copy-bound",
        f"- {_kind_line(kinds, 'down')}",
        f"- {_kind_line(kinds, 'gate')}",
        f"- {_kind_line(kinds, 'up')}",
        f"- {_kind_line(kinds, 'q')} / k / v / o stay DEVICE",
        f"- embed / lm_head: {_kind_line(kinds, 'embed')}; {_kind_line(kinds, 'lm_head')}",
        f"- HOST down layers: { _span(downs) }",
        f"- HOST gate+up layers: { _span(gates) }",
        "",
        "## Load",
        "",
        f"- HostImage arenas: {pin.get('host_images')} "
        f"({_mib(pin.get('host_bytes'))}), pinned={pin.get('all_pinned')} "
        f"({_mib(pin.get('pinned_bytes'))})",
        f"- report.device_mib (HBM weights, not 16601): {snap.get('device_mib')}",
        f"- nvidia-smi after load: {snap.get('smi_after_load')}",
        f"- torch allocated after load: {snap.get('torch_after_load')}",
        "",
        "## Token path (one forward)",
        "",
        "1. `CopyRing.arm(host_tape)` — HOST GEMMs in consume order.",
        "2. `prefetch()` — H2D of tape[0] into slot 0 on `copy_stream` "
        "(overlaps embed).",
        "3. Each layer: DEVICE q/k/v (CUDA graph if captured) → RoPE/SDPA → "
        "DEVICE o → gate/up (DEVICE graph or HOST slot) → down (usually HOST).",
        "4. HOST GEMM: compute.wait(e_copy) → CPU join current copy (WDDM) → "
        "prefetch up to max_ahead → chr_nf4_gemm(slot views) → record e_gemm. "
        "Each layer also kicks prefetch before qkv / before MLP (no-op if already "
        "ahead). Do not join after queueing the next H2D: WDDM drained the copy "
        "stream (0.01 tok/s).",
        "5. Prefill: this whole tape once **per chunk** "
        f"(LIVE_MAX_N={loop.get('prefill_chunk')}), not per column.",
        "6. Decode N=1: same tape once per token.",
        "",
        f"- groups: {loop.get('n_groups')} "
        f"(graphed {loop.get('n_graphed_groups')}, eager {loop.get('n_eager_groups')})",
        f"- GEMMs: DEVICE {loop.get('n_device_gemms')}, HOST {loop.get('n_host_gemms')}",
        f"- graph_mode={loop.get('graph_mode')} error={loop.get('graph_error')}",
        f"- KV: {loop.get('kv_mib')} MiB at max_seq={loop.get('max_seq')}",
        "",
        "## Measured generate",
        "",
    ]
    if not messages:
        lines.append("No generate yet.")
    for i, msg in enumerate(messages, start=1):
        exp = expected_h2d_bytes(
            streamed_bytes=streamed, forwards=int(msg.get("h2d_forwards") or 0)
        )
        got = int(msg.get("h2d_bytes") or 0)
        decode_ms = float(msg.get("decode_ms") or 0)
        steps = int(msg.get("decode_steps") or 0)
        wall_tok = decode_ms / steps if steps else 0.0
        copy_tok = tape_ms
        lines.extend(
            [
                f"### message {i}",
                "",
                f"- prompt_len={msg.get('prompt_len')} prefill={msg.get('prefill_ms')} ms "
                f"({msg.get('prefill_chunks', '?')} chunks)",
                f"- decode {msg.get('decode_tok_s')} tok/s over {steps} steps "
                f"({decode_ms:.0f} ms, {wall_tok:.1f} ms/tok)",
                f"- H2D {got / MIB:.1f} MiB in {msg.get('h2d_copies')} copies, "
                f"{msg.get('h2d_forwards')} forwards "
                f"(expect {exp / MIB:.1f} MiB = tape × forwards)",
                f"- CUDA copy events: {msg.get('h2d_copy_ms')} ms "
                "(None/0 if timing off)",
                f"- serial copy floor ≈ {copy_tok:.1f} ms/tok; "
                f"wall {wall_tok:.1f} ms/tok. "
                + (
                    "Copy-bound, little overlap."
                    if wall_tok and wall_tok >= 0.9 * copy_tok
                    else "Wall below serial copy floor → overlap with resident GEMM is working."
                    if wall_tok
                    else ""
                ),
                f"- quality={msg.get('quality')} needle={msg.get('needles')}",
                f"- smi during/after: {msg.get('smi_used_mib')}",
                "",
            ]
        )
    notes = snap.get("notes") or []
    if notes:
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {n}" for n in notes)
        lines.append("")
    return "\n".join(lines) + "\n"


def _span(layers: Sequence[int]) -> str:
    if not layers:
        return "(none)"
    xs = sorted(int(x) for x in layers)
    return f"{xs[0]}..{xs[-1]} (n={len(xs)})"

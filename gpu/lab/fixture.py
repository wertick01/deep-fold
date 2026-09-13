"""A tiny synthetic run, so the CSV writers and the figure can be tested.

This exists for two reasons: ``python -m gpu.lab.run --dry-plot`` must produce
the four CSVs and ``lab.html`` on a machine with no CUDA, and ``test_lab.py``
must never load the 3B.

The numbers are **shaped** like a real session (that is what makes the plot code
exercised) but they are not a measurement. Every ``summary.notes`` says
``FIXTURE``, and :func:`gpu.lab.comparison_figure` stamps the figure with
"SYNTHETIC FIXTURE DATA" when it sees that. Nothing in here may be quoted as a
result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .bundle import LabBundle, mean
from .script import MESSAGES, TURNS_NOTE, quality_ok

__all__ = ["fixture_bundle"]


@dataclass(frozen=True)
class _Turn:
    t_send: float
    t_first: float
    t_done: float
    new_tokens: int
    prompt_tokens: int
    response: str


@dataclass(frozen=True)
class _Plan:
    """One made-up session, in session-local seconds."""

    codec: str
    vram_idle: float
    vram_loaded: float
    vram_active: float
    torch_loaded: float
    weight_mib: float
    kv_mib: float | None
    power_idle: float
    power_active: float
    load: tuple[float, float]
    warmup: tuple[float, float]
    turns: tuple[_Turn, ...]
    unload: tuple[float, float]
    stop: float
    notes: str


_RESPONSES_OK = (
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "323",
)

_PROMPT_TOKENS = (35, 29, 41)

_BF16 = _Plan(
    codec="bf16",
    vram_idle=920.0,
    vram_loaded=7850.0,
    vram_active=8030.0,
    torch_loaded=5886.0,
    weight_mib=5886.0,
    kv_mib=None,
    power_idle=42.0,
    power_active=228.0,
    load=(0.6, 11.8),
    warmup=(12.1, 13.4),
    turns=(
        _Turn(15.00, 15.42, 16.30, 26, _PROMPT_TOKENS[0], _RESPONSES_OK[0]),
        _Turn(17.50, 17.85, 18.35, 15, _PROMPT_TOKENS[1], _RESPONSES_OK[1]),
        _Turn(19.60, 19.98, 20.30, 10, _PROMPT_TOKENS[2], _RESPONSES_OK[2]),
    ),
    unload=(21.40, 22.60),
    stop=23.40,
    notes=(
        "FIXTURE (synthetic, not a measurement); "
        "HuggingFace from_pretrained + generate, dense bf16 GEMM; "
        f"{TURNS_NOTE}; kv_mib empty: the HF cache is transient"
    ),
)

_NF4 = _Plan(
    codec="nf4",
    vram_idle=920.0,
    vram_loaded=2712.0,
    vram_active=2840.0,
    torch_loaded=1794.0,
    weight_mib=1636.0,
    kv_mib=128.0,
    power_idle=42.0,
    power_active=186.0,
    load=(0.6, 7.9),
    warmup=(8.1, 10.2),
    turns=(
        _Turn(11.50, 12.05, 14.35, 26, _PROMPT_TOKENS[0], _RESPONSES_OK[0]),
        _Turn(15.50, 15.92, 17.20, 15, _PROMPT_TOKENS[1], _RESPONSES_OK[1]),
        _Turn(18.20, 18.60, 19.40, 10, _PROMPT_TOKENS[2], _RESPONSES_OK[2]),
    ),
    unload=(20.50, 21.40),
    stop=22.20,
    notes=(
        "FIXTURE (synthetic, not a measurement); "
        "gpu.host.load_model + gpu.loop.TokenLoop, graph=linears, max_seq=512; "
        f"{TURNS_NOTE}"
    ),
)


def _ramp(t: float, t0: float, t1: float, v0: float, v1: float) -> float:
    if t <= t0:
        return v0
    if t >= t1:
        return v1
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


def _wobble(t: float, amplitude: float, freq: float = 7.1) -> float:
    """Deterministic jitter -- a flat line would not exercise the plot."""
    return amplitude * math.sin(t * freq)


def _turn_of(plan: _Plan, t: float) -> int | None:
    for index, turn in enumerate(plan.turns, start=1):
        if turn.t_send <= t <= turn.t_done:
            return index
    return None


def _timeline(plan: _Plan, interval: float) -> list[dict[str, Any]]:
    load0, load1 = plan.load
    warm0, warm1 = plan.warmup
    unload0, unload1 = plan.unload
    rows: list[dict[str, Any]] = []
    steps = int(plan.stop / interval) + 1
    for step in range(steps + 1):
        t = min(round(step * interval, 4), plan.stop)
        turn = _turn_of(plan, t)
        busy = turn is not None or warm0 <= t <= warm1
        loading = load0 <= t <= load1

        if t < load0:
            vram = plan.vram_idle
        elif loading:
            vram = _ramp(t, load0, load1, plan.vram_idle, plan.vram_loaded)
        elif t < unload0:
            vram = plan.vram_active if busy else plan.vram_loaded
        else:
            vram = _ramp(t, unload0, unload1, plan.vram_loaded, plan.vram_idle)

        if loading:
            util, util_mem = 38.0, 62.0
        elif busy:
            util, util_mem = 94.0, 71.0
        elif t < unload0 and t > load1:
            util, util_mem = 8.0, 4.0
        else:
            util, util_mem = 1.0, 0.0

        power = plan.power_active if (busy or loading) else plan.power_idle
        rows.append(
            {
                "t_s": t,
                "codec": plan.codec,
                "used_mib": round(vram + _wobble(t, 6.0), 1),
                "total_mib": 12288.0,
                "util_gpu": max(0.0, min(100.0, round(util + _wobble(t, 4.0, 3.3), 1))),
                "util_mem": max(0.0, min(100.0, round(util_mem + _wobble(t, 3.0, 5.7), 1))),
                "power_w": round(power + _wobble(t, 9.0, 2.9), 1),
                "temp_c": round(44.0 + util * 0.22 + _wobble(t, 0.6, 1.3), 1),
                "clock_sm_mhz": 1905.0 if (busy or loading) else 210.0,
                "clock_mem_mhz": 9501.0,
                "torch_alloc_mib": round(
                    _ramp(t, load0, load1, 0.0, plan.torch_loaded)
                    if t <= load1
                    else (0.0 if t >= unload1 else plan.torch_loaded + (24.0 if busy else 0.0)),
                    1,
                ),
                "torch_reserved_mib": round(
                    _ramp(t, load0, load1, 0.0, plan.torch_loaded + 96.0)
                    if t <= load1
                    else (0.0 if t >= unload1 else plan.torch_loaded + 96.0),
                    1,
                ),
                "torch_max_alloc_mib": round(
                    _ramp(t, load0, load1, 0.0, plan.torch_loaded + 24.0), 1
                ),
                "message_id": "" if turn is None else str(turn),
            }
        )
    return rows


def _session(plan: _Plan, interval: float) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    timeline = _timeline(plan, interval)
    events: list[dict[str, Any]] = [
        {
            "t_s": 0.0,
            "codec": plan.codec,
            "event": "start",
            "message_id": "",
            "detail": f"codec={plan.codec}, interval={interval:.2f}s (fixture)",
        },
        {
            "t_s": plan.load[0],
            "codec": plan.codec,
            "event": "load_start",
            "message_id": "",
            "detail": "fixture",
        },
        {
            "t_s": plan.load[1],
            "codec": plan.codec,
            "event": "load_end",
            "message_id": "",
            "detail": f"load_s={plan.load[1] - plan.load[0]:.1f}",
        },
        {
            "t_s": plan.warmup[0],
            "codec": plan.codec,
            "event": "warmup_start",
            "message_id": "",
            "detail": "not in any reported number",
        },
        {
            "t_s": plan.warmup[1],
            "codec": plan.codec,
            "event": "warmup_end",
            "message_id": "",
            "detail": f"warmup_ms={(plan.warmup[1] - plan.warmup[0]) * 1000:.0f}",
        },
    ]
    messages: list[dict[str, Any]] = []

    for index, turn in enumerate(plan.turns, start=1):
        prompt = MESSAGES[(index - 1) % len(MESSAGES)]
        prefill_ms = (turn.t_first - turn.t_send) * 1000.0
        decode_ms = (turn.t_done - turn.t_first) * 1000.0
        decode_steps = max(0, turn.new_tokens - 1)
        row = {
            "codec": plan.codec,
            "message_id": index,
            "prompt": prompt,
            "response": turn.response,
            "prompt_tokens": turn.prompt_tokens,
            "new_tokens": turn.new_tokens,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "decode_tok_s": decode_steps / (decode_ms / 1000.0) if decode_ms > 0 else 0.0,
            "stop_reason": "eos",
            "quality_ok": quality_ok(index, turn.response),
        }
        messages.append(row)
        events.extend(
            [
                {
                    "t_s": turn.t_send,
                    "codec": plan.codec,
                    "event": "msg_send",
                    "message_id": str(index),
                    "detail": prompt,
                },
                {
                    "t_s": turn.t_first,
                    "codec": plan.codec,
                    "event": "first_token",
                    "message_id": str(index),
                    "detail": f"ttft={prefill_ms:.0f} ms",
                },
                {
                    "t_s": turn.t_done,
                    "codec": plan.codec,
                    "event": "msg_done",
                    "message_id": str(index),
                    "detail": (
                        f"{row['new_tokens']} new tokens, ttft={prefill_ms:.0f} ms, "
                        f"{row['decode_tok_s']:.1f} tok/s, stop=eos, "
                        f"quality_ok={str(bool(row['quality_ok'])).lower()}"
                    ),
                },
            ]
        )

    events.extend(
        [
            {
                "t_s": plan.unload[0],
                "codec": plan.codec,
                "event": "unload_start",
                "message_id": "",
                "detail": "del model, gc, empty_cache",
            },
            {
                "t_s": plan.unload[1],
                "codec": plan.codec,
                "event": "unload_end",
                "message_id": "",
                "detail": f"smi_after_unload={plan.vram_idle:.0f} MiB",
            },
            {
                "t_s": plan.stop,
                "codec": plan.codec,
                "event": "stop",
                "message_id": "",
                "detail": f"peak_smi={plan.vram_active:.0f} MiB",
            },
        ]
    )
    events.sort(key=lambda event: event["t_s"])

    peak = max(row["used_mib"] for row in timeline)
    summary = {
        "codec": plan.codec,
        "load_s": plan.load[1] - plan.load[0],
        "vram_before_mib": plan.vram_idle,
        "vram_after_load_smi_mib": plan.vram_loaded,
        "vram_after_load_torch_mib": plan.torch_loaded,
        "vram_peak_smi_mib": peak,
        "weight_mib": plan.weight_mib,
        "kv_mib": plan.kv_mib,
        "mean_ttft_ms": mean([row["prefill_ms"] for row in messages]),
        "mean_decode_tok_s": mean([row["decode_tok_s"] for row in messages]),
        "n_messages": len(messages),
        "quality_all_ok": all(bool(row["quality_ok"]) for row in messages),
        "notes": plan.notes,
    }
    return timeline, events, messages, summary


def fixture_bundle(*, interval_s: float = 0.25, codec: str = "both") -> LabBundle:
    """A complete, schema-valid bundle with both codecs and no GPU involved."""
    plans = [plan for plan in (_BF16, _NF4) if codec in ("both", plan.codec)]
    if not plans:
        raise ValueError(f"codec={codec!r}; expected 'both', 'bf16' or 'nf4'")
    timeline: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for plan in plans:
        rows, marks, replies, totals = _session(plan, interval_s)
        timeline += rows
        events += marks
        messages += replies
        summary.append(totals)
    return LabBundle.of(
        timeline=timeline,
        events=events,
        messages=messages,
        summary=summary,
        source="fixture",
    )

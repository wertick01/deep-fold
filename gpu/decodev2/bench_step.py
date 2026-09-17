"""Decode V2 step A/B: MMA vs GEMV on a medium synthetic model. No .chr.

    python gpu/decodev2/bench_step.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.decodev2.graph import capture_greedy  # noqa: E402
from gpu.decodev2.linear import DeviceWeights, set_linear_backend  # noqa: E402
from gpu.decodev2.plan import MEDIUM_LLAMA  # noqa: E402
from gpu.decodev2.runner import consume_prompt  # noqa: E402
from gpu.decodev2.state import DecodeState  # noqa: E402
from gpu.decodev2.step import greedy_decode  # noqa: E402
from gpu.decodev2.synth import build  # noqa: E402
from gpu.tests.skips import cuda_reason  # noqa: E402

PROMPT = [4, 5, 6, 7]
STEPS = 8


def _session(backend: str):
    set_linear_backend(backend)
    spec = MEDIUM_LLAMA
    model = build(spec, 0)
    weights = DeviceWeights.from_synth(model, "cuda", torch.bfloat16)
    state = DecodeState.allocate(spec, weights.embed, dtype=torch.bfloat16)
    return state, weights


def _arm(state, weights) -> None:
    state.reset()
    consume_prompt(state, weights, PROMPT)
    state.token.copy_(state.next_token)


def _median_decode_ms(arm, body, repeats: int = 5) -> float:
    samples = []
    for _ in range(repeats):
        arm()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        body()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / STEPS)
    return statistics.median(samples)


def _run_backend(backend: str) -> tuple[float, float]:
    state, weights = _session(backend)

    def arm():
        _arm(state, weights)

    def eager_body():
        for _ in range(STEPS):
            greedy_decode(state, weights)

    arm()
    for _ in range(2):
        eager_body()
        arm()
    eager_ms = _median_decode_ms(arm, eager_body)
    arm()
    captured = capture_greedy(state, weights, warmup=2)

    def graph_body():
        for _ in range(STEPS):
            captured.replay()

    graph_ms = _median_decode_ms(arm, graph_body)
    set_linear_backend("mma")
    return eager_ms, graph_ms


def main() -> int:
    reason = cuda_reason()
    if reason is not None:
        print(f"SKIP {reason}")
        return 0
    print("medium-llama 4L hidden=512  median ms / decode step (prompt excluded)")
    print(f"{'backend':<8} {'eager_ms':>10} {'graph_ms':>10}")
    for backend in ("mma", "gemv"):
        eager, graphed = _run_backend(backend)
        print(f"{backend:<8} {eager:10.3f} {graphed:10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CUDA graph of greedy_decode vs eager. No checkpoints.

    python gpu/decodev2/test_graph.py
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.decodev2.graph import capture_greedy  # noqa: E402
from gpu.decodev2.linear import DeviceWeights, set_linear_backend  # noqa: E402
from gpu.decodev2.plan import MEDIUM_LLAMA, TINY_INTERNLM, TINY_LLAMA  # noqa: E402
from gpu.decodev2.runner import generate  # noqa: E402
from gpu.decodev2.state import DecodeState  # noqa: E402
from gpu.decodev2.synth import build  # noqa: E402
from gpu.tests.skips import Skip, cuda_reason  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []
PROMPT = [4, 5, 6]


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def _cuda_pair():
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    spec = TINY_LLAMA
    model = build(spec, 0)
    weights = DeviceWeights.from_synth(model, "cuda", torch.bfloat16)
    state = DecodeState.allocate(spec, weights.embed, dtype=torch.bfloat16)
    return model, state, weights


def test_graph_internlm() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    spec = TINY_INTERNLM
    model = build(spec, 1)
    weights = DeviceWeights.from_synth(model, "cuda", torch.bfloat16)
    state = DecodeState.allocate(spec, weights.embed, dtype=torch.bfloat16)
    captured = capture_greedy(state, weights, warmup=2)
    state.reset()
    eager = generate(state, weights, PROMPT, 8)
    state.reset()
    graphed = generate(state, weights, PROMPT, 8, step=captured)
    assert graphed == eager, f"internlm graph {graphed} != eager {eager}"


def test_graph_matches_eager() -> None:
    _, state, weights = _cuda_pair()
    captured = capture_greedy(state, weights, warmup=2)
    state.reset()
    eager = generate(state, weights, PROMPT, 8)
    state.reset()
    graphed = generate(state, weights, PROMPT, 8, step=captured)
    assert graphed == eager, f"graph {graphed} != eager {eager}"
    assert len(eager) >= 4, eager


def test_graph_gemv() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    set_linear_backend("gemv")
    try:
        spec = TINY_LLAMA
        from gpu.decodev2.synth import build

        model = build(spec, 0)
        weights = DeviceWeights.from_synth(model, "cuda", torch.bfloat16)
        state = DecodeState.allocate(spec, weights.embed, dtype=torch.bfloat16)
        captured = capture_greedy(state, weights, warmup=2)
        state.reset()
        eager = generate(state, weights, PROMPT, 8)
        state.reset()
        graphed = generate(state, weights, PROMPT, 8, step=captured)
    finally:
        set_linear_backend("mma")
    assert graphed == eager, f"gemv graph {graphed} != eager {eager}"
    assert len(eager) >= 4, eager


def test_graph_medium() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    spec = MEDIUM_LLAMA
    model = build(spec, 0)
    weights = DeviceWeights.from_synth(model, "cuda", torch.bfloat16)
    state = DecodeState.allocate(spec, weights.embed, dtype=torch.bfloat16)
    captured = capture_greedy(state, weights, warmup=2)
    state.reset()
    eager = generate(state, weights, PROMPT, 8)
    state.reset()
    graphed = generate(state, weights, PROMPT, 8, step=captured)
    assert graphed == eager, f"medium graph {graphed} != eager {eager}"
    assert len(eager) >= 4, eager


def test_graph_position_and_reset() -> None:
    _, state, weights = _cuda_pair()
    captured = capture_greedy(state, weights, warmup=2)
    state.reset()
    ids = generate(state, weights, PROMPT, 8, step=captured)
    want_pos = len(PROMPT) + max(0, len(ids) - 1)
    assert int(state.position.item()) == want_pos, (ids, int(state.position.item()), want_pos)
    assert int(state.valid_len.item()) == int(state.position.item())
    state.reset()
    ids2 = generate(state, weights, PROMPT, 8, step=captured)
    assert ids2 == ids


def test_graph_no_alloc_growth() -> None:
    _, state, weights = _cuda_pair()
    captured = capture_greedy(state, weights, warmup=2)
    state.reset()
    generate(state, weights, PROMPT, 4, step=captured)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    state.reset()
    generate(state, weights, PROMPT, 8, step=captured)
    torch.cuda.synchronize()
    grew = torch.cuda.memory_allocated() - base
    assert grew == 0, f"allocated grew by {grew}"


TESTS = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("test_") and callable(value)
]


def main_runner() -> int:
    print(f"gpu/decodev2 graph, {len(TESTS)} tests\n")
    for fn in TESTS:
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                fn()
        except Skip as exc:
            SKIPPED.append((fn.__name__, str(exc)))
            print(f"  SKIP  {fn.__name__}  -- {exc}")
        except AssertionError as exc:
            check(fn.__name__, False, str(exc).splitlines()[0] if str(exc) else "assert")
        except Exception as exc:  # noqa: BLE001
            check(fn.__name__, False, f"{type(exc).__name__}: {exc}")
        else:
            check(fn.__name__, True)

    failed = [name for name, ok, _ in CHECKS if not ok]
    tail = f", {len(SKIPPED)} skipped" if SKIPPED else ""
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed{tail}")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main_runner())

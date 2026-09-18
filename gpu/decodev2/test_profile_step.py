"""profiled_greedy matches greedy_decode. No live 3B plate.

    python gpu/decodev2/test_profile_step.py
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

from gpu.decodev2.linear import DeviceWeights, set_linear_backend  # noqa: E402
from gpu.decodev2.plan import TINY_INTERNLM, TINY_LLAMA  # noqa: E402
from gpu.decodev2.profile_step import (  # noqa: E402
    CATEGORIES,
    GpuSpans,
    NullSpans,
    capture_profiled,
    profiled_greedy,
    reconstruct_kernels,
    spans_per_step,
)
from gpu.decodev2.runner import generate  # noqa: E402
from gpu.decodev2.state import DecodeState  # noqa: E402
from gpu.decodev2.step import greedy_decode  # noqa: E402
from gpu.decodev2.synth import build  # noqa: E402
from gpu.tests.skips import Skip, cuda_reason  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []
PROMPT = [4, 5, 6]


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def _session(spec, seed, device, dtype):
    model = build(spec, seed)
    weights = DeviceWeights.from_synth(model, device, dtype)
    state = DecodeState.allocate(spec, weights.embed, device=device, dtype=dtype)
    return state, weights


def _ids(state, weights, n_new: int, step) -> list[int]:
    state.reset()
    return generate(state, weights, PROMPT, n_new, step=step)


def test_spans_per_step() -> None:
    assert spans_per_step(2) == 1 + 14 + 3
    assert spans_per_step(36) == 1 + 7 * 36 + 3
    assert set(CATEGORIES) == {
        "embed",
        "rms",
        "qkv",
        "attn",
        "o_proj",
        "swiglu",
        "down",
        "lm_head",
        "commit",
    }


def test_profiled_matches_greedy_cpu() -> None:
    state, weights = _session(TINY_LLAMA, 0, "cpu", torch.float32)
    eager = _ids(state, weights, 8, greedy_decode)
    spans = NullSpans()

    def step(st, wts):
        profiled_greedy(st, wts, spans)

    got = _ids(state, weights, 8, step)
    assert got == eager, f"{got} != {eager}"


def test_profiled_internlm_cpu() -> None:
    state, weights = _session(TINY_INTERNLM, 1, "cpu", torch.float32)
    eager = _ids(state, weights, 8, greedy_decode)
    spans = NullSpans()

    def step(st, wts):
        profiled_greedy(st, wts, spans)

    got = _ids(state, weights, 8, step)
    assert got == eager, f"{got} != {eager}"


def test_gpu_spans_match_and_busy() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    set_linear_backend("gemv")
    try:
        state, weights = _session(TINY_LLAMA, 0, "cuda", torch.bfloat16)
        eager = _ids(state, weights, 8, greedy_decode)
        spans = GpuSpans(spans_per_step(state.spec.n_layers))

        def step(st, wts):
            spans.reset()
            profiled_greedy(st, wts, spans)

        got = _ids(state, weights, 8, step)
        totals = spans.totals_ms()
        spans2 = GpuSpans(spans_per_step(state.spec.n_layers))
        graphed = capture_profiled(state, weights, spans2, warmup=2)
        got_g = _ids(state, weights, 8, graphed)
        recon = reconstruct_kernels(state, weights)
    finally:
        set_linear_backend("mma")
    assert got == eager, f"{got} != {eager}"
    assert got_g == eager, f"instrumented graph {got_g} != {eager}"
    busy = sum(totals[key] for key in CATEGORIES)
    assert busy > 0, totals
    for key in CATEGORIES:
        assert key in totals, key
        assert totals[key] >= 0, (key, totals[key])
    assert recon["reconstructed_ms"] > 0, recon
    assert recon["scaled_ms"]["lm_head"] > 0, recon


TESTS = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("test_") and callable(value)
]


def main_runner() -> int:
    print(f"gpu/decodev2 profile_step, {len(TESTS)} tests\n")
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

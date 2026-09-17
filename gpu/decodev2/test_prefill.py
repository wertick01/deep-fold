"""Chunked MMA prefill vs N=1 teacher-force and the CPU oracle.

    python gpu/decodev2/test_prefill.py
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
from gpu.decodev2.oracle import greedy_ids  # noqa: E402
from gpu.decodev2.plan import MEDIUM_LLAMA, TINY_INTERNLM, TINY_LLAMA  # noqa: E402
from gpu.decodev2.prefill import prefill_chunk_width  # noqa: E402
from gpu.decodev2.runner import consume_prompt, generate  # noqa: E402
from gpu.decodev2.state import DecodeState  # noqa: E402
from gpu.decodev2.step import greedy_decode, teacher_force_token  # noqa: E402
from gpu.decodev2.synth import build  # noqa: E402
from gpu.tests.skips import Skip, cuda_reason  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []
PROMPT = [1, 3, 5, 7]
LONG_PROMPT = [4, 5, 6, 7, 1, 2, 3, 4]


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def _session(spec, model, device, dtype):
    weights = DeviceWeights.from_synth(model, device, dtype)
    state = DecodeState.allocate(spec, weights.embed, device=device, dtype=dtype)
    return state, weights


def _teacher_greedy(state, weights, prompt, n_new):
    for tok in prompt:
        teacher_force_token(state, weights, tok)
    out = [int(state.next_token.item())]
    hit = (state.next_token == state.eos_id).to(dtype=state.finished.dtype)
    state.finished.bitwise_or_(hit)
    if int(state.finished.item()) == 1:
        return out[:n_new]
    state.token.copy_(state.next_token)
    for _ in range(int(n_new) - 1):
        greedy_decode(state, weights)
        out.append(int(state.token.item()))
        if int(state.finished.item()) == 1:
            break
    return out


def test_cpu_empty_prompt() -> None:
    spec = TINY_LLAMA
    model = build(spec, 0)
    state, weights = _session(spec, model, "cpu", torch.float32)
    try:
        consume_prompt(state, weights, [])
    except ValueError as exc:
        assert "non-empty" in str(exc)
        return
    raise AssertionError("empty prompt must raise")


def test_cpu_chunk_width() -> None:
    spec = TINY_LLAMA
    model = build(spec, 0)
    state, _ = _session(spec, model, "cpu", torch.float32)
    assert prefill_chunk_width(state) == spec.max_seq
    assert prefill_chunk_width(state, 1) == 1
    assert prefill_chunk_width(state, 99) == spec.max_seq


def test_cpu_llama_chunked_matches_teacher() -> None:
    spec = TINY_LLAMA
    model = build(spec, 0)
    chunked, w = _session(spec, model, "cpu", torch.float32)
    one, w1 = _session(spec, model, "cpu", torch.float32)
    got = generate(chunked, w, PROMPT, 8, chunk=2)
    want = _teacher_greedy(one, w1, PROMPT, 8)
    oracle = greedy_ids(model, PROMPT, 8)
    assert got == want == oracle, f"chunked {got} teacher {want} oracle {oracle}"
    assert int(chunked.position.item()) == int(one.position.item())
    assert int(chunked.valid_len.item()) == len(PROMPT) + max(0, len(got) - 1)


def test_cpu_suffix_prefill_matches_full() -> None:
    spec = TINY_LLAMA
    model = build(spec, 0)
    full, wf = _session(spec, model, "cpu", torch.float32)
    part, wp = _session(spec, model, "cpu", torch.float32)
    consume_prompt(full, wf, LONG_PROMPT, chunk=2)
    consume_prompt(part, wp, LONG_PROMPT[:3], chunk=2)
    consume_prompt(part, wp, LONG_PROMPT[3:], start=3, chunk=2)
    assert int(full.next_token.item()) == int(part.next_token.item())
    assert int(full.valid_len.item()) == int(part.valid_len.item()) == len(LONG_PROMPT)
    spec = TINY_LLAMA
    model = build(spec, 0)
    state, weights = _session(spec, model, "cpu", torch.float32)
    got = generate(state, weights, LONG_PROMPT, 8, chunk=8)
    want = greedy_ids(model, LONG_PROMPT, 8)
    assert got == want, f"{got} != {want}"
    assert int(state.valid_len.item()) == len(LONG_PROMPT) + max(0, len(got) - 1)


def test_cpu_internlm_chunked() -> None:
    spec = TINY_INTERNLM
    model = build(spec, 1)
    state, weights = _session(spec, model, "cpu", torch.float32)
    got = generate(state, weights, PROMPT, 8, chunk=3)
    want = greedy_ids(model, PROMPT, 8)
    assert got == want, f"{got} != {want}"


def test_cpu_chunk1_equals_chunk4() -> None:
    spec = TINY_LLAMA
    model = build(spec, 0)
    a, wa = _session(spec, model, "cpu", torch.float32)
    b, wb = _session(spec, model, "cpu", torch.float32)
    ids_a = generate(a, wa, LONG_PROMPT, 8, chunk=1)
    ids_b = generate(b, wb, LONG_PROMPT, 8, chunk=4)
    assert ids_a == ids_b, f"chunk1 {ids_a} != chunk4 {ids_b}"


def test_cuda_chunked_matches_oracle() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    spec = TINY_LLAMA
    model = build(spec, 0)
    set_linear_backend("gemv")
    try:
        state, weights = _session(spec, model, "cuda", torch.bfloat16)
        got = generate(state, weights, LONG_PROMPT, 8, chunk=4)
    finally:
        set_linear_backend("mma")
    want = greedy_ids(model, LONG_PROMPT, 8)
    assert got == want, f"cuda chunked {got} != oracle {want}"


def test_cuda_internlm_chunked() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    spec = TINY_INTERNLM
    model = build(spec, 1)
    state, weights = _session(spec, model, "cuda", torch.bfloat16)
    got = generate(state, weights, PROMPT, 8, chunk=2)
    want = greedy_ids(model, PROMPT, 8)
    assert got == want, f"cuda internlm {got} != oracle {want}"


def test_cuda_medium_chunked_gemv() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    spec = MEDIUM_LLAMA
    model = build(spec, 0)
    set_linear_backend("gemv")
    try:
        state, weights = _session(spec, model, "cuda", torch.bfloat16)
        got = generate(state, weights, LONG_PROMPT, 8, chunk=8)
    finally:
        set_linear_backend("mma")
    want = greedy_ids(model, LONG_PROMPT, 8)
    assert got == want, f"medium chunked {got} != oracle {want}"


def test_cuda_chunk_widths_agree() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    spec = TINY_LLAMA
    model = build(spec, 0)
    set_linear_backend("gemv")
    try:
        a, wa = _session(spec, model, "cuda", torch.bfloat16)
        b, wb = _session(spec, model, "cuda", torch.bfloat16)
        ids_a = generate(a, wa, LONG_PROMPT, 8, chunk=1)
        ids_b = generate(b, wb, LONG_PROMPT, 8, chunk=4)
    finally:
        set_linear_backend("mma")
    assert ids_a == ids_b, f"cuda chunk1 {ids_a} != chunk4 {ids_b}"


TESTS = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("test_") and callable(value)
]


def main_runner() -> int:
    print(f"gpu/decodev2 prefill, {len(TESTS)} tests\n")
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

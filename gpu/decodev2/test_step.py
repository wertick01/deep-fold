"""Eager Decode V2 step vs CPU oracle.

    python gpu/decodev2/test_step.py
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
from gpu.decodev2.oracle import greedy_ids, teacher_logits  # noqa: E402
from gpu.decodev2.plan import MEDIUM_LLAMA, TINY_INTERNLM, TINY_LLAMA  # noqa: E402
from gpu.decodev2.state import DecodeState  # noqa: E402
from gpu.decodev2.step import greedy_decode, teacher_force_token  # noqa: E402
from gpu.decodev2.synth import build  # noqa: E402
from gpu.tests.skips import Skip, cuda_reason  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []
PROMPT = [1, 3, 5, 7]
LONG_PROMPT = [4, 5, 6]


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def _session(spec, model, device, dtype):
    weights = DeviceWeights.from_synth(model, device, dtype)
    state = DecodeState.allocate(spec, weights.embed, device=device, dtype=dtype)
    return state, weights


def _teacher_ids(state, weights, prompt):
    ids = []
    for tok in prompt:
        teacher_force_token(state, weights, tok)
        ids.append(int(state.next_token.item()))
    return ids


def _greedy(state, weights, prompt, n_new):
    for tok in prompt:
        teacher_force_token(state, weights, tok)
    out = []
    first = int(state.next_token.item())
    out.append(first)
    hit = (state.next_token == state.eos_id).to(dtype=state.finished.dtype)
    state.finished.bitwise_or_(hit)
    if int(state.finished.item()) == 1:
        return out[:n_new]
    state.token.copy_(state.next_token)
    for _ in range(int(n_new) - 1):
        greedy_decode(state, weights)
        tok = int(state.token.item())
        out.append(tok)
        if int(state.finished.item()) == 1:
            break
    return out


def test_cpu_llama_teacher() -> None:
    spec = TINY_LLAMA
    model = build(spec, 0)
    state, weights = _session(spec, model, "cpu", torch.float32)
    got = _teacher_ids(state, weights, PROMPT)
    want = teacher_logits(model, PROMPT).argmax(axis=-1).tolist()
    assert got == want, f"{got} != {want}"


def test_cpu_llama_greedy() -> None:
    spec = TINY_LLAMA
    model = build(spec, 0)
    state, weights = _session(spec, model, "cpu", torch.float32)
    got = _greedy(state, weights, PROMPT, 8)
    want = greedy_ids(model, PROMPT, 8)
    assert got == want, f"{got} != {want}"


def test_cpu_llama_greedy_long() -> None:
    spec = TINY_LLAMA
    model = build(spec, 0)
    state, weights = _session(spec, model, "cpu", torch.float32)
    got = _greedy(state, weights, LONG_PROMPT, 8)
    want = greedy_ids(model, LONG_PROMPT, 8)
    assert len(want) >= 4, want
    assert got == want, f"{got} != {want}"


def test_cpu_internlm_greedy() -> None:
    spec = TINY_INTERNLM
    model = build(spec, 1)
    state, weights = _session(spec, model, "cpu", torch.float32)
    got = _greedy(state, weights, PROMPT, 8)
    want = greedy_ids(model, PROMPT, 8)
    assert got == want, f"{got} != {want}"


def test_cpu_dirty_tail() -> None:
    spec = TINY_LLAMA
    model = build(spec, 0)
    clean_s, w = _session(spec, model, "cpu", torch.float32)
    dirty_s, w2 = _session(spec, model, "cpu", torch.float32)
    teacher_force_token(clean_s, w, 1)
    teacher_force_token(dirty_s, w2, 1)
    dirty_s.kv.k[:, -1].fill_(1.0e4)
    dirty_s.kv.v[:, -1].fill_(1.0e4)
    teacher_force_token(clean_s, w, 3)
    teacher_force_token(dirty_s, w2, 3)
    assert int(clean_s.next_token.item()) == int(dirty_s.next_token.item())


def test_cuda_internlm_greedy() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    spec = TINY_INTERNLM
    model = build(spec, 1)
    state, weights = _session(spec, model, "cuda", torch.bfloat16)
    prompt = [4, 5, 6]
    got = _greedy(state, weights, prompt, 8)
    want = greedy_ids(model, prompt, 8)
    assert got == want, f"cuda internlm {got} != oracle {want}"


def test_cuda_llama_greedy() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    spec = TINY_LLAMA
    model = build(spec, 0)
    state, weights = _session(spec, model, "cuda", torch.bfloat16)
    got = _greedy(state, weights, LONG_PROMPT, 8)
    want = greedy_ids(model, LONG_PROMPT, 8)
    assert len(want) >= 4, want
    assert got == want, f"cuda {got} != oracle {want}"


def test_cuda_medium_greedy_gemv() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    spec = MEDIUM_LLAMA
    model = build(spec, 0)
    set_linear_backend("gemv")
    try:
        state, weights = _session(spec, model, "cuda", torch.bfloat16)
        got = _greedy(state, weights, LONG_PROMPT, 8)
    finally:
        set_linear_backend("mma")
    want = greedy_ids(model, LONG_PROMPT, 8)
    assert got == want, f"medium gemv {got} != oracle {want}"


def test_cuda_llama_greedy_gemv() -> None:
    reason = cuda_reason()
    if reason is not None:
        raise Skip(reason)
    spec = TINY_LLAMA
    model = build(spec, 0)
    set_linear_backend("gemv")
    try:
        state, weights = _session(spec, model, "cuda", torch.bfloat16)
        got = _greedy(state, weights, LONG_PROMPT, 8)
    finally:
        set_linear_backend("mma")
    want = greedy_ids(model, LONG_PROMPT, 8)
    assert got == want, f"gemv {got} != oracle {want}"


TESTS = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("test_") and callable(value)
]


def main_runner() -> int:
    print(f"gpu/decodev2 step, {len(TESTS)} tests\n")
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

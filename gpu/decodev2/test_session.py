"""CLI-shaped Decode V2 session. CPU only — no GPU plate.

    python gpu/decodev2/test_session.py
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.decodev2.linear import DeviceWeights  # noqa: E402
from gpu.decodev2.plan import TINY_INTERNLM, TINY_LLAMA  # noqa: E402
from gpu.decodev2.runner import generate as runner_generate  # noqa: E402
from gpu.decodev2.session import DecodeV2Loop, decodev2_refused, pick_executor  # noqa: E402
from gpu.decodev2.state import DecodeState  # noqa: E402
from gpu.decodev2.synth import build  # noqa: E402
from gpu.tests.skips import Skip  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []
PROMPT = [1, 3, 5, 7]


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def _loop(spec, seed: int) -> DecodeV2Loop:
    model = build(spec, seed)
    weights = DeviceWeights.from_synth(model, "cpu", torch.float32)
    state = DecodeState.allocate(spec, weights.embed, device="cpu", dtype=torch.float32)
    return DecodeV2Loop(state, weights)


def test_refuse_overflow_and_vq() -> None:
    plan = SimpleNamespace(family="llama_swiglu")
    assert decodev2_refused(SimpleNamespace(overflow=False, codec="nf4"), plan) is None
    why = decodev2_refused(SimpleNamespace(overflow=True, codec="nf4"))
    assert why is not None and why.startswith("decodev2:")
    assert "CopyRing" in why
    why = decodev2_refused(SimpleNamespace(overflow=False, codec="vq"))
    assert why is not None and "VQ" in why
    why = decodev2_refused(
        SimpleNamespace(overflow=False, codec="nf4"),
        SimpleNamespace(family="nope"),
    )
    assert why is not None and "family" in why


def test_pick_executor_auto() -> None:
    plan = SimpleNamespace(family="llama_swiglu")
    nf4 = SimpleNamespace(overflow=False, codec="nf4")
    overflow = SimpleNamespace(overflow=True, codec="nf4")
    vq = SimpleNamespace(overflow=False, codec="vq")
    internlm = SimpleNamespace(family="internlm_gqa")
    assert pick_executor("auto", nf4, plan) == ("decodev2", None)
    assert pick_executor("auto", nf4, internlm) == ("decodev2", None)
    chosen, why = pick_executor("auto", overflow, plan)
    assert chosen == "tokenloop" and why is not None and "CopyRing" in why
    chosen, why = pick_executor("auto", vq, plan)
    assert chosen == "tokenloop" and why is not None and "VQ" in why
    assert pick_executor("tokenloop", nf4, plan) == ("tokenloop", None)
    chosen, why = pick_executor("decodev2", overflow, plan)
    assert chosen == "decodev2" and why is not None
    chosen, why = pick_executor("auto", nf4, SimpleNamespace(family="nope"))
    assert chosen == "tokenloop" and why is not None


def test_cpu_generate_matches_runner() -> None:
    spec = TINY_LLAMA
    loop = _loop(spec, 0)
    other = _loop(spec, 0)
    ids = torch.tensor(PROMPT, dtype=torch.long)
    out = loop.generate(ids, 8, stop=(spec.eos_id,))
    want = runner_generate(other.state, other.weights, PROMPT, 8)
    assert out.tokens == want, f"{out.tokens} != {want}"
    assert out.prompt_len == len(PROMPT)
    assert out.decode_steps == max(0, len(out.tokens) - 1)
    if spec.eos_id in out.tokens:
        assert out.stop_token == spec.eos_id
    else:
        assert out.stop_token is None


def test_cpu_internlm_generate() -> None:
    spec = TINY_INTERNLM
    loop = _loop(spec, 1)
    other = _loop(spec, 1)
    ids = torch.tensor(PROMPT, dtype=torch.long)
    out = loop.generate(ids, 8, stop=(spec.eos_id,))
    want = runner_generate(other.state, other.weights, PROMPT, 8)
    assert out.tokens == want, f"{out.tokens} != {want}"


def test_cpu_stop_and_interrupt() -> None:
    loop = _loop(TINY_LLAMA, 0)
    ids = torch.tensor(PROMPT, dtype=torch.long)
    seen: list[int] = []

    def on_token(tid: int) -> None:
        seen.append(int(tid))

    def should_stop() -> bool:
        return len(seen) >= 2

    out = loop.generate(ids, 8, on_token=on_token, should_stop=should_stop)
    assert out.interrupted is True
    assert out.tokens == seen
    assert len(out.tokens) == 2

    loop.reset()
    first = loop.generate(ids, 8).tokens[0]
    stopped = loop.generate(ids, 8, stop=(first,))
    assert stopped.tokens == [first]
    assert stopped.stop_token == first
    assert stopped.decode_steps == 0


def test_cpu_session_suffix_and_decode() -> None:
    spec = TINY_LLAMA
    full = _loop(spec, 0)
    part = _loop(spec, 0)
    ids = torch.tensor(PROMPT, dtype=torch.long)
    extra = torch.tensor([2, 4], dtype=torch.long)
    whole = torch.cat([ids, extra])
    full.prefill_from(whole, 0)
    part.prefill_from(ids, 0)
    part.prefill_from(extra, len(PROMPT))
    assert int(full.state.next_token.item()) == int(part.state.next_token.item())
    assert part.kv.seq_len == len(PROMPT) + 2
    got = part.decode_from_logits(
        part.state.arena.logits[0],
        8,
        prompt_len=len(PROMPT) + 2,
        stop=(spec.eos_id,),
    )
    want = full.decode_from_logits(
        full.state.arena.logits[0],
        8,
        prompt_len=len(PROMPT) + 2,
        stop=(spec.eos_id,),
    )
    assert got.tokens == want.tokens
    if got.tokens:
        part.seal_last(got.tokens[-1])
        assert part.kv.seq_len == len(PROMPT) + 2 + len(got.tokens)


def test_cpu_warmup_reset_no_speculate() -> None:
    loop = _loop(TINY_LLAMA, 0)
    ms = loop.warmup(prompt=4, tokens=2)
    assert ms >= 0.0
    assert int(loop.state.valid_len.item()) == 0
    assert loop.capture_graphs() == "off"
    assert loop.graph_error == "cpu"
    ids = torch.tensor(PROMPT, dtype=torch.long)
    try:
        loop.generate(ids, 4, speculate=2, draft="lookup")
    except ValueError as exc:
        assert "speculate" in str(exc)
        return
    raise AssertionError("decodev2 must refuse speculation")


def test_weight_bytes_and_kv_mib() -> None:
    loop = _loop(TINY_LLAMA, 0)
    assert loop.weight_bytes > 0
    assert loop.kv.mib > 0
    assert int(loop.kv.seq_len) == 0


def test_from_model_needs_plan() -> None:
    try:
        DecodeV2Loop.from_model(SimpleNamespace(config=None), max_seq=8)
    except RuntimeError as exc:
        assert str(exc).startswith("decodev2:")
        return
    raise AssertionError("missing plan must refuse")


TESTS = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("test_") and callable(value)
]


def main_runner() -> int:
    print(f"gpu/decodev2 session, {len(TESTS)} tests\n")
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
        except Exception as exc:  # pragma: no cover
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

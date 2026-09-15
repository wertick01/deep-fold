"""CPU tests for Accel-2 speculative verify (no GPU, no second model).

    python gpu/loop/test_speculate.py
"""

from __future__ import annotations

import inspect
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.loop.generate import TokenLoop  # noqa: E402
from gpu.loop.speculate import (  # noqa: E402
    accept_greedy,
    lookup_draft,
    measure_verify,
    verify_block,
)

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def _peak(vocab: int, idx: int, rows: int = 1) -> torch.Tensor:
    y = torch.zeros(rows, vocab)
    y[:, int(idx) % vocab] = 1.0
    return y if rows > 1 else y[0]


def _silence_cuda_sync():
    orig = torch.cuda.synchronize
    torch.cuda.synchronize = lambda *args, **kwargs: None
    return orig


class _SeqBox:
    """``seq_len`` with a history so rewind tests can see verify then shrink."""

    def __init__(self) -> None:
        self._seq_len = 0
        self.hist: list[int] = []

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @seq_len.setter
    def seq_len(self, value: int) -> None:
        self._seq_len = int(value)
        self.hist.append(int(value))


class FakeLoop:
    """Position-greedy mock: leftover after seq_len=P+i predicts greedy[i]."""

    generate = TokenLoop.generate

    def __init__(
        self,
        greedy: list[int],
        *,
        prompt_len: int = 4,
        vocab: int = 64,
        max_seq: int = 128,
        ring: bool = False,
    ) -> None:
        self.greedy = list(greedy)
        self.prompt_len = int(prompt_len)
        self.vocab = int(vocab)
        self.device = torch.device("cpu")
        self.max_seq = int(max_seq)
        self.prefill_chunk = 32
        self.graph_mode = "off"
        self.kv = _SeqBox()
        self._ring = SimpleNamespace(total_bytes=0, total_copies=0) if ring else None
        self.forwards: list[tuple[tuple[int, ...], int, bool]] = []

    def reset(self) -> None:
        self.kv.seq_len = 0

    def _capture_decode_glue(self) -> None:
        return None

    def _logits_after(self, pos: int) -> torch.Tensor:
        """Logits after consuming the token at ``pos`` (next slot is pos+1)."""
        dec = (pos + 1) - self.prompt_len
        y = torch.zeros(self.vocab)
        if 0 <= dec < len(self.greedy):
            y[self.greedy[dec]] = 1.0
        return y

    def forward(self, ids, start_pos: int, *, all_positions: bool = False, logits: bool = True):
        ids = ids.reshape(-1).to(dtype=torch.long)
        n = int(ids.numel())
        self.forwards.append((tuple(int(x) for x in ids.tolist()), int(start_pos), bool(all_positions)))
        rows = torch.stack([self._logits_after(int(start_pos) + i) for i in range(n)], dim=0)
        self.kv.seq_len = int(start_pos) + n
        if self._ring is not None:
            self._ring.total_bytes += 100 * n
            self._ring.total_copies += 1
        if all_positions:
            return rows
        return rows[-1]

    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        n = int(ids.reshape(-1).numel())
        self.kv.seq_len = n
        y = torch.zeros(self.vocab)
        if self.greedy:
            y[self.greedy[0]] = 1.0
        return y

    def step(self, token_id: int) -> torch.Tensor:
        tok = torch.tensor([int(token_id)], dtype=torch.long)
        return self.forward(tok, self.kv.seq_len).reshape(-1)


# --------------------------------------------------------------------------- #
# verify_block
# --------------------------------------------------------------------------- #


def test_verify_block_k33_raises_two_tapes() -> None:
    loop = FakeLoop([1, 2, 3])
    try:
        verify_block(loop, list(range(33)), 0)
    except ValueError as exc:
        msg = str(exc).lower()
        assert "two tapes" in msg, msg
        assert "33" in str(exc)
    else:
        raise AssertionError("k=33 must raise")
    assert loop.forwards == [], "k>32 must not call forward"


def test_verify_block_all_positions_shape() -> None:
    loop = FakeLoop([9, 8, 7, 6], prompt_len=2)
    loop.kv.seq_len = 2
    ids = torch.tensor([9, 8, 7], dtype=torch.long)
    y = verify_block(loop, ids, 2)
    assert tuple(y.shape) == (3, loop.vocab), tuple(y.shape)
    assert loop.forwards == [((9, 8, 7), 2, True)]
    assert loop.kv.seq_len == 5
    assert int(y[0].argmax()) == 8
    assert int(y[1].argmax()) == 7


def test_verify_block_k1() -> None:
    loop = FakeLoop([4, 5], prompt_len=1)
    y = verify_block(loop, [4], 1)
    assert tuple(y.shape) == (1, loop.vocab)


# --------------------------------------------------------------------------- #
# accept_greedy
# --------------------------------------------------------------------------- #


def test_accept_greedy_prefix_then_mismatch() -> None:
    vocab = 16
    leftover = _peak(vocab, 5)
    draft = torch.tensor([5, 7, 9, 1], dtype=torch.long)
    block = torch.zeros(4, vocab)
    block[0, 7] = 1.0
    block[1, 9] = 1.0
    block[2, 0] = 1.0  # mismatch: draft[3] is 1
    block[3, 3] = 1.0
    n_acc, nxt, emitted = accept_greedy(leftover, draft, block)
    assert n_acc == 3, n_acc
    assert [int(x) for x in emitted.tolist()] == [5, 7, 9, 0]
    assert int(nxt.argmax()) == 0
    assert torch.equal(nxt, block[2].reshape(-1))


def test_accept_greedy_all_match() -> None:
    vocab = 8
    leftover = _peak(vocab, 1)
    draft = torch.tensor([1, 2, 3], dtype=torch.long)
    block = torch.zeros(3, vocab)
    block[0, 2] = 1.0
    block[1, 3] = 1.0
    block[2, 4] = 1.0
    n_acc, nxt, emitted = accept_greedy(leftover, draft, block)
    assert n_acc == 3
    assert [int(x) for x in emitted.tolist()] == [1, 2, 3]
    assert int(nxt.argmax()) == 4


def test_accept_greedy_reject_first() -> None:
    vocab = 8
    leftover = _peak(vocab, 4)
    draft = torch.tensor([0, 1], dtype=torch.long)
    block = torch.zeros(2, vocab)
    block[0, 1] = 1.0
    block[1, 2] = 1.0
    n_acc, nxt, emitted = accept_greedy(leftover, draft, block)
    assert n_acc == 0
    assert [int(x) for x in emitted.tolist()] == [4]
    assert int(nxt.argmax()) == 4


# --------------------------------------------------------------------------- #
# lookup_draft
# --------------------------------------------------------------------------- #


def test_lookup_draft_trigram() -> None:
    # last two (10, 20) match the start; copy what followed.
    ids = [10, 20, 30, 40, 10, 20]
    d = lookup_draft(ids, 4)
    assert [int(x) for x in d.tolist()] == [30, 40, 10, 20]


def test_lookup_draft_bigram_fallback() -> None:
    ids = [7, 8, 9, 7]
    d = lookup_draft(ids, 3)
    assert [int(x) for x in d.tolist()] == [8, 9, 7]


def test_lookup_draft_no_match_repeats_last() -> None:
    d = lookup_draft([1, 2, 3], 3)
    assert [int(x) for x in d.tolist()] == [3, 3, 3]
    z = lookup_draft([], 2)
    assert [int(x) for x in z.tolist()] == [0, 0]


def test_lookup_draft_deterministic() -> None:
    ids = torch.tensor([4, 5, 6, 4, 5], dtype=torch.long)
    a = lookup_draft(ids, 3)
    b = lookup_draft(ids, 3)
    assert torch.equal(a, b)
    assert [int(x) for x in a.tolist()] == [6, 4, 5]


# --------------------------------------------------------------------------- #
# measure_verify
# --------------------------------------------------------------------------- #


def test_measure_verify_match_and_h2d() -> None:
    greedy = [11, 12, 13, 14, 15]
    loop = FakeLoop(greedy, prompt_len=3, ring=True)
    loop.kv.seq_len = 3
    out = measure_verify(loop, greedy, k=2)
    assert out["greedy_match"] is True, out
    assert out["n_blocks"] == 3, out["n_blocks"]
    assert len(out["walls"]) == 3
    assert all(w >= 0.0 for w in out["walls"])
    assert out["h2d_bytes"] == [200, 200, 100], out["h2d_bytes"]
    assert out["h2d_copies"] == [1, 1, 1]
    assert all(flag for _, _, flag in loop.forwards)


def test_measure_verify_no_ring() -> None:
    loop = FakeLoop([1, 2, 3], prompt_len=1, ring=False)
    loop.kv.seq_len = 1
    out = measure_verify(loop, [1, 2, 3], k=3)
    assert out["h2d_bytes"] is None
    assert out["h2d_copies"] is None
    assert out["greedy_match"] is True


# --------------------------------------------------------------------------- #
# generate()
# --------------------------------------------------------------------------- #


def test_generate_defaults() -> None:
    sig = inspect.signature(TokenLoop.generate)
    assert sig.parameters["speculate"].default == 1
    assert sig.parameters["draft"].default == "none"


def test_generate_none_is_step_loop() -> None:
    orig = _silence_cuda_sync()
    try:
        greedy = [21, 22, 23, 24]
        loop = FakeLoop(greedy, prompt_len=3)
        prompt = torch.tensor([1, 2, 3], dtype=torch.long)
        out = loop.generate(prompt, max_new_tokens=4)
        assert out.tokens == greedy, out.tokens
        assert not any(flag for _, _, flag in loop.forwards), loop.forwards
        assert loop.kv.seq_len == 3 + 3, loop.kv.seq_len  # last token not stepped
    finally:
        torch.cuda.synchronize = orig


def test_generate_lookup_matches_greedy() -> None:
    orig = _silence_cuda_sync()
    try:
        greedy = [21, 22, 23, 24, 25, 26]
        prompt = torch.tensor([1, 2, 3, 1, 2], dtype=torch.long)  # trigram fuel
        loop = FakeLoop(greedy, prompt_len=5)
        out = loop.generate(prompt, max_new_tokens=6, speculate=4, draft="lookup")
        assert out.tokens == greedy, out.tokens
        # Prompt n-gram proposes 3; leftover is 21. Skip verify, same tokens as step().
        assert not any(flag for _, _, flag in loop.forwards), loop.forwards
        assert loop.kv.seq_len == 5 + 5, loop.kv.seq_len
    finally:
        torch.cuda.synchronize = orig


def test_generate_lookup_rewinds_rejected_tail() -> None:
    orig = _silence_cuda_sync()
    try:
        greedy = [3, 9, 9, 9]
        # Prompt has no n-gram for (last two) that continues as greedy, so draft
        # repeats last=1; leftover argmax is 3 → reject first, emit 3, step.
        loop = FakeLoop(greedy, prompt_len=3)
        prompt = torch.tensor([8, 8, 1], dtype=torch.long)
        out = loop.generate(prompt, max_new_tokens=4, speculate=3, draft="lookup")
        assert out.tokens == greedy, out.tokens
        hist = loop.kv.hist
        # First leftover (3) != pad-last draft (1): step from seq 3, no k=3 tape.
        assert hist[:3] == [0, 3, 4], hist
    finally:
        torch.cuda.synchronize = orig


def test_generate_lookup_verify_then_rewind() -> None:
    orig = _silence_cuda_sync()
    try:
        greedy = [3, 9, 9, 9]
        # 2-gram: 7 → 3 matches leftover; draft continues 7,7 and mismatches.
        loop = FakeLoop(greedy, prompt_len=3)
        prompt = torch.tensor([7, 3, 7], dtype=torch.long)
        out = loop.generate(prompt, max_new_tokens=4, speculate=3, draft="lookup")
        assert out.tokens == greedy, out.tokens
        assert any(flag for _, _, flag in loop.forwards), loop.forwards
        hist = loop.kv.hist
        assert any(
            hist[i] == 6 and hist[i + 1] == 4 for i in range(len(hist) - 1)
        ), hist
    finally:
        torch.cuda.synchronize = orig


def test_generate_unknown_draft_raises() -> None:
    orig = _silence_cuda_sync()
    try:
        loop = FakeLoop([1], prompt_len=1)
        try:
            loop.generate(torch.tensor([0]), max_new_tokens=1, draft="hf")
        except ValueError as exc:
            assert "lookup" in str(exc)
        else:
            raise AssertionError("unknown draft must raise")
    finally:
        torch.cuda.synchronize = orig


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

TESTS = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("test_") and callable(value)
]


def main_runner() -> int:
    print(f"gpu/loop speculate, {len(TESTS)} tests, CPU\n")
    for fn in TESTS:
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                fn()
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

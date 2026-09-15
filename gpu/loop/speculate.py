"""Greedy speculative verify for TokenLoop. No second model, no sampling.

Lab path (``verify-k`` / ``spec-lookup`` in ``docs/plan-h2-accel.md``):

- :func:`verify_block` is one ``forward(..., all_positions=True)`` with
  ``k`` in ``1..LIVE_MAX_N``. ``k>32`` is two tapes; this file refuses to hide
  that.
- :func:`lookup_draft` is an n-gram (length 2 or 3) copy from the known
  prompt+prefix. A miss still returns ``k`` ids so verify can reject them.
- :func:`accept_greedy` is Leviathan greedy: leftover checks ``draft[0]``,
  ``block_logits[i]`` checks ``draft[i+1]``. KV rewind is the caller's job.

``TokenLoop.generate(..., speculate=1, draft="none")`` stays the ``step()``
loop. ``draft="lookup"`` with ``speculate>=2`` calls these helpers.
"""

from __future__ import annotations

import time
from typing import Sequence

import torch

from gpu.nf4.plan import LIVE_MAX_N

__all__ = [
    "verify_block",
    "measure_verify",
    "lookup_draft",
    "accept_greedy",
    "spec_lookup_generate",
]


def _as_long_1d(ids) -> torch.Tensor:
    if isinstance(ids, torch.Tensor):
        return ids.reshape(-1).to(dtype=torch.long)
    return torch.tensor(list(ids), dtype=torch.long)


def _id_list(ids) -> list[int]:
    if isinstance(ids, torch.Tensor):
        return [int(x) for x in ids.reshape(-1).tolist()]
    return [int(x) for x in ids]


def _argmax_id(logits: torch.Tensor) -> int:
    return int(logits.reshape(-1).argmax())


def _sync(loop) -> None:
    """Synchronize the loop's CUDA device when there is one. CPU is a no-op."""
    if not torch.cuda.is_available():
        return
    device = getattr(loop, "device", None)
    if device is None:
        return
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def verify_block(loop, ids, start_pos: int) -> torch.Tensor:
    """Teacher-forced logits ``[k, vocab]`` for ``ids`` at ``start_pos``.

    One :meth:`TokenLoop.forward` with ``all_positions=True``. ``k`` must be
    in ``1..LIVE_MAX_N`` (32). A larger block is two forwards, hence two HOST
    tapes, and this function raises rather than splitting.
    """
    ids = _as_long_1d(ids)
    k = int(ids.numel())
    if k > LIVE_MAX_N:
        raise ValueError(
            f"verify_block k={k} > {LIVE_MAX_N}: that needs two forwards "
            f"(two tapes); split the block instead of hiding the second tape"
        )
    if k < 1:
        raise ValueError(f"verify_block k={k}; expected 1..{LIVE_MAX_N}")
    device = getattr(loop, "device", ids.device)
    ids = ids.to(device=device, dtype=torch.long)
    y = loop.forward(ids, int(start_pos), all_positions=True)
    if y is None:
        raise RuntimeError("verify_block: forward returned None")
    if y.dim() == 1:
        y = y.view(1, -1)
    if int(y.shape[0]) != k:
        raise RuntimeError(
            f"verify_block expected logits [{k}, vocab], got {tuple(y.shape)}"
        )
    return y


def measure_verify(loop, greedy_token_ids, k: int) -> dict:
    """Replay already-prefilled greedy tokens in blocks of ``k``.

    Caller has already run the same prefill as the baseline. This does **not**
    ``reset``. It feeds ``greedy_token_ids`` through :func:`verify_block` and
    records wall time per block (``time.perf_counter``, CUDA sync when the
    loop is on GPU).

    ``greedy_match`` is True iff every next-token argmax from a block agrees
    with the remainder of ``greedy_token_ids``: ``block_logits[i]`` must match
    ``ids[lo + i + 1]`` when that token exists. The first greedy token is not
    checked here (that leftover lives on the caller after prefill).

    Returns ``walls`` (seconds per block), ``h2d_bytes`` / ``h2d_copies``
    deltas when ``loop._ring`` exists else ``None``, and ``greedy_match``.
    """
    ids = _as_long_1d(greedy_token_ids)
    k = int(k)
    n = int(ids.numel())
    walls: list[float] = []
    h2d_bytes: list[int] | None = None
    h2d_copies: list[int] | None = None
    ring = getattr(loop, "_ring", None)
    if ring is not None:
        h2d_bytes, h2d_copies = [], []
    match = True
    pos = int(loop.kv.seq_len)
    device = getattr(loop, "device", ids.device)
    ids = ids.to(device=device, dtype=torch.long)

    for lo in range(0, n, k):
        hi = min(lo + k, n)
        block = ids[lo:hi]
        b0 = 0 if ring is None else int(ring.total_bytes)
        c0 = 0 if ring is None else int(ring.total_copies)
        _sync(loop)
        t0 = time.perf_counter()
        logits = verify_block(loop, block, pos)
        _sync(loop)
        walls.append(time.perf_counter() - t0)
        if ring is not None:
            h2d_bytes.append(int(ring.total_bytes) - b0)
            h2d_copies.append(int(ring.total_copies) - c0)
        pred = logits.reshape(int(block.numel()), -1).argmax(dim=-1)
        kb = int(block.numel())
        for i in range(kb):
            nxt = lo + i + 1
            if nxt >= n:
                break
            if int(pred[i]) != int(ids[nxt]):
                match = False
        pos += kb

    return {
        "k": k,
        "n_tokens": n,
        "n_blocks": len(walls),
        "walls": walls,
        "h2d_bytes": h2d_bytes,
        "h2d_copies": h2d_copies,
        "greedy_match": match,
    }


def _pad_k(toks: list[int], k: int) -> list[int]:
    if len(toks) >= k:
        return toks[:k]
    if not toks:
        return [0] * k
    return toks + [toks[-1]] * (k - len(toks))


def _ngram_follow(seq: list[int], ctx: Sequence[int], k: int) -> list[int] | None:
    """Rightmost occurrence of ``ctx`` that is not the suffix, then copy."""
    n = len(seq)
    c = len(ctx)
    if c < 1 or n < c + 1:
        return None
    ctx_t = tuple(int(x) for x in ctx)
    for i in range(n - c - 1, -1, -1):
        if tuple(seq[i : i + c]) != ctx_t:
            continue
        followed = seq[i + c :]
        if not followed:
            continue
        return _pad_k(followed, k)
    return None


def lookup_draft(ids, k: int) -> torch.Tensor:
    """Propose ``k`` token ids from a length-2 or length-3 n-gram match.

    The haystack is ``ids`` (prompt + already emitted prefix). Deterministic:

    1. Prefer a **3-gram** continuation: last two tokens as context. Search
       **rightmost** occurrence that starts at ``i <= n-3`` (so it is not the
       suffix itself and at least one token follows). Copy what follows.
    2. Else a **2-gram**: last token as context, rightmost ``i <= n-2``.
    3. If the copied span is shorter than ``k``, **repeat the last copied
       token** to length ``k``.
    4. If nothing matches, return **k copies of the last token**, or **k
       zeros** if ``ids`` is empty.

    A miss still returns ``k`` ids so :func:`verify_block` can run; greedy
    verify will reject a bad draft. Never loads a second model.
    """
    k = int(k)
    if k < 1:
        raise ValueError(f"lookup_draft k={k}; expected k>=1")
    seq = _id_list(ids)
    n = len(seq)
    drafted: list[int] | None = None
    if n >= 2:
        drafted = _ngram_follow(seq, seq[-2:], k)
    if drafted is None and n >= 1:
        drafted = _ngram_follow(seq, seq[-1:], k)
    if drafted is None:
        drafted = _pad_k([seq[-1]] if seq else [], k)
    return torch.tensor(drafted, dtype=torch.long)


def accept_greedy(
    leftover_logits: torch.Tensor,
    draft_ids,
    block_logits: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Leviathan greedy verify. KV rewind is the caller's job (``seq_len``).

    ``leftover_logits`` is ``[vocab]`` (or ``[1, vocab]``) from the previous
    position and checks ``draft_ids[0]``. ``block_logits`` is ``[k, vocab]``
    from :func:`verify_block`; ``block_logits[i]`` checks ``draft_ids[i+1]``.

    On the first mismatch, emit the target argmax at that position and stop.
    ``n_accepted`` is the number of **draft** tokens accepted (not the bonus).
    ``next_leftover_logits`` is ``[vocab]``: ``block_logits[-1]`` if every
    draft token was accepted, otherwise the logits that produced the bonus.
    ``emitted_ids`` is 1-D long: accepted drafts, plus the bonus on mismatch.

    Does not sample. Does not rewind KV.
    """
    draft = _as_long_1d(draft_ids)
    k = int(draft.numel())
    if k < 1:
        raise ValueError("accept_greedy: empty draft")
    block = block_logits if block_logits.dim() == 2 else block_logits.view(1, -1)
    if int(block.shape[0]) != k:
        raise ValueError(
            f"accept_greedy: block_logits has {int(block.shape[0])} rows, draft k={k}"
        )
    device = draft.device if draft.device.type != "cpu" else leftover_logits.device

    target = _argmax_id(leftover_logits)
    if target != int(draft[0]):
        leftover = leftover_logits.reshape(-1)
        emitted = torch.tensor([target], dtype=torch.long, device=device)
        return 0, leftover, emitted

    accepted = [int(draft[0])]
    for i in range(k - 1):
        target = _argmax_id(block[i])
        if target != int(draft[i + 1]):
            accepted.append(target)
            leftover = block[i].reshape(-1)
            emitted = torch.tensor(accepted, dtype=torch.long, device=device)
            return i + 1, leftover, emitted
        accepted.append(int(draft[i + 1]))

    leftover = block[-1].reshape(-1)
    emitted = torch.tensor(accepted, dtype=torch.long, device=device)
    return k, leftover, emitted


def spec_lookup_generate(
    loop,
    out,
    prompt_ids: torch.Tensor,
    leftover: torch.Tensor,
    max_new_tokens: int,
    stop_set: frozenset,
    on_token,
    speculate: int,
    should_stop=None,
) -> None:
    """Fill ``out`` with n-gram draft + greedy verify. Mutates ``loop.kv.seq_len``.

    Used by :meth:`TokenLoop.generate` when ``draft=="lookup"`` and
    ``speculate>=2``. Sets ``kv.seq_len`` to the accepted draft length (bonus
    token is consumed with :meth:`TokenLoop.step` only if generation continues).
    """

    def emit(token: int) -> bool:
        if should_stop is not None and should_stop():
            out.interrupted = True
            return False
        token = int(token)
        out.tokens.append(token)
        if len(out.tokens) > 1:
            out.decode_steps += 1
        if on_token is not None:
            on_token(token)
        if token in stop_set:
            out.stop_token = token
            return False
        return len(out.tokens) < int(max_new_tokens)

    known = [int(x) for x in prompt_ids.reshape(-1).tolist()]
    spec_k = max(2, int(speculate))

    while len(out.tokens) < int(max_new_tokens):
        remaining = int(max_new_tokens) - len(out.tokens)
        room = int(loop.max_seq) - int(loop.kv.seq_len)
        if remaining <= 0:
            break
        if room <= 0:
            emit(_argmax_id(leftover))
            break

        k = min(spec_k, remaining, room, int(LIVE_MAX_N))
        if k < 2:
            token = _argmax_id(leftover)
            if not emit(token):
                break
            if int(loop.kv.seq_len) >= int(loop.max_seq):
                break
            leftover = loop.step(token)
            continue

        draft_ids = lookup_draft(known + out.tokens, k)
        # Leftover already rejects draft[0]: do not pay a full overflow tape.
        if _argmax_id(leftover) != int(draft_ids.reshape(-1)[0]):
            token = _argmax_id(leftover)
            if not emit(token):
                break
            if int(loop.kv.seq_len) >= int(loop.max_seq):
                break
            leftover = loop.step(token)
            continue

        start = int(loop.kv.seq_len)
        block_logits = verify_block(loop, draft_ids, start)
        n_acc, leftover, emitted = accept_greedy(leftover, draft_ids, block_logits)

        stopped = False
        n_emitted = 0
        for tok in emitted.reshape(-1).tolist():
            n_emitted += 1
            if not emit(int(tok)):
                stopped = True
                break

        n_drafts = min(int(n_acc), n_emitted)
        loop.kv.seq_len = start + n_drafts
        if stopped:
            break
        if n_emitted > int(n_acc):
            if int(loop.kv.seq_len) >= int(loop.max_seq):
                break
            leftover = loop.step(int(out.tokens[-1]))

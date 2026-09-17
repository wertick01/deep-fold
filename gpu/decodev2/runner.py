"""Greedy runner: prompt is chunked MMA prefill; decode tokens stay on GPU."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .linear import DeviceWeights
from .prefill import forward_prefill, prefill_chunk_width
from .state import DecodeState
from .step import greedy_decode

__all__ = ["consume_prompt", "generate"]


def consume_prompt(
    state: DecodeState,
    weights: DeviceWeights,
    token_ids: Sequence[int],
    *,
    chunk: int | None = None,
    start: int = 0,
) -> None:
    """Fill KV for ``token_ids`` in MMA chunks, starting at slot ``start``.

    ``start == 0`` is a cold prompt (call after ``reset``). Chat session KV
    passes ``start=kv.seq_len`` for the new suffix only.
    """
    if not token_ids:
        raise ValueError("prompt must be non-empty")
    ids = [int(x) for x in token_ids]
    n = len(ids)
    pos = int(start)
    if pos < 0:
        raise ValueError(f"start {pos} is negative")
    if pos + n > state.spec.max_seq:
        raise ValueError(
            f"prompt {pos}+{n} exceeds max_seq={state.spec.max_seq}"
        )
    width = prefill_chunk_width(state, chunk)
    buf = torch.tensor(ids, dtype=torch.long, device=state.device)
    for lo in range(0, n, width):
        hi = min(lo + width, n)
        forward_prefill(
            state,
            weights,
            buf[lo:hi],
            start=pos + lo,
            logits=(hi == n),
        )


def generate(
    state: DecodeState,
    weights: DeviceWeights,
    prompt_ids: Sequence[int],
    n_new: int,
    *,
    step=None,
    ignore_eos: bool = False,
    chunk: int | None = None,
) -> list[int]:
    """Host list of new ids. The next forward does not wait on this list."""
    consume_prompt(state, weights, prompt_ids, chunk=chunk)
    decode = greedy_decode if step is None else step
    out: list[int] = []
    first = int(state.next_token.item())
    out.append(first)
    if not ignore_eos:
        hit = (state.next_token == state.eos_id).to(dtype=state.finished.dtype)
        state.finished.bitwise_or_(hit)
        if int(state.finished.item()) == 1:
            return out
    state.token.copy_(state.next_token)
    for _ in range(max(0, int(n_new) - 1)):
        decode(state, weights)
        out.append(int(state.token.item()))
        if not ignore_eos and int(state.finished.item()) == 1:
            break
    return out

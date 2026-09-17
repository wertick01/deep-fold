"""Greedy runner: prompt is teacher-forced; decode tokens stay on GPU."""

from __future__ import annotations

from collections.abc import Sequence

from .linear import DeviceWeights
from .state import DecodeState
from .step import greedy_decode, teacher_force_token

__all__ = ["consume_prompt", "generate"]


def consume_prompt(state: DecodeState, weights: DeviceWeights, token_ids: Sequence[int]) -> None:
    if not token_ids:
        raise ValueError("prompt must be non-empty")
    for tok in token_ids:
        teacher_force_token(state, weights, int(tok))


def generate(
    state: DecodeState,
    weights: DeviceWeights,
    prompt_ids: Sequence[int],
    n_new: int,
    *,
    step=None,
    ignore_eos: bool = False,
) -> list[int]:
    """Host list of new ids. The next forward does not wait on this list."""
    consume_prompt(state, weights, prompt_ids)
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

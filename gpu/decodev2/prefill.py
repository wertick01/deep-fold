"""Chunked prompt prefill. MMA GEMM N>1; not captured into the decode CUDA graph.

Decode stays N=1 (GEMV graph). Prefill walks ``LIVE_MAX_N`` columns, the same
ceiling as TokenLoop. CopyRing / HOST overflow is out of scope.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gpu.nf4.plan import LIVE_MAX_N

from .linear import DeviceWeights, nf4_linear
from .ops import rms, split_qkv, swiglu
from .plan import ArchSpec
from .rope import apply_rope_torch
from .state import DecodeState

__all__ = ["forward_prefill", "prefill_chunk_width"]

_GQA: bool | None = None


def prefill_chunk_width(state: DecodeState, chunk: int | None = None) -> int:
    """Columns per MMA launch. CPU numpy can take the full live cap."""
    cap = min(int(LIVE_MAX_N), int(state.spec.max_seq))
    if chunk is not None:
        return max(1, min(int(chunk), cap))
    if state.device.type != "cuda":
        return cap
    from gpu.loop.graph import nf4_max_n

    return max(1, min(int(nf4_max_n(LIVE_MAX_N)), cap))


def _sdpa_gqa() -> bool:
    global _GQA
    if _GQA is None:
        q = torch.zeros((1, 2, 1, 4))
        k = torch.zeros((1, 1, 2, 4))
        try:
            F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
            _GQA = True
        except TypeError:
            _GQA = False
    return _GQA


def _causal_mask(start: int, n: int, seq: int, device: torch.device) -> torch.Tensor:
    """``[1, 1, n, seq]`` bool, True = attend. Same as TokenLoop._causal_mask."""
    q_pos = torch.arange(start, start + n, device=device).unsqueeze(1)
    k_pos = torch.arange(seq, device=device).unsqueeze(0)
    return (k_pos <= q_pos)[None, None]


def _attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spec: ArchSpec,
    mask: torch.Tensor,
) -> torch.Tensor:
    """``q`` is ``[n, n_q, hd]``; ``k``/``v`` are ``[1, n_kv, seq, hd]``."""
    n = int(q.shape[0])
    q_sdpa = q.permute(1, 0, 2).unsqueeze(0)
    scale = spec.attn_scale
    if spec.n_q != spec.n_kv and not _sdpa_gqa():
        k = k.repeat_interleave(spec.n_rep, dim=1)
        v = v.repeat_interleave(spec.n_rep, dim=1)
        gqa = False
    else:
        gqa = spec.n_q != spec.n_kv
    try:
        if gqa:
            a = F.scaled_dot_product_attention(
                q_sdpa, k, v, attn_mask=mask, scale=scale, enable_gqa=True
            )
        else:
            a = F.scaled_dot_product_attention(
                q_sdpa, k, v, attn_mask=mask, scale=scale
            )
    except (RuntimeError, TypeError):
        if spec.n_q != spec.n_kv and k.shape[1] != spec.n_q:
            k = k.repeat_interleave(spec.n_rep, dim=1)
            v = v.repeat_interleave(spec.n_rep, dim=1)
        scores = torch.matmul(q_sdpa.float(), k.float().transpose(-2, -1)) * scale
        scores = scores.masked_fill(~mask, -1.0e9)
        w = torch.softmax(scores, dim=-1)
        a = torch.matmul(w, v.float()).to(dtype=q.dtype)
    return a.squeeze(0).permute(1, 0, 2).reshape(n, spec.q_dim)


def forward_prefill(
    state: DecodeState,
    weights: DeviceWeights,
    token_ids: torch.Tensor,
    start: int,
    *,
    logits: bool = True,
) -> None:
    """Teacher-force ``token_ids`` into KV slots ``start..start+n-1``. Eager only.

    Linears go through ``nf4_linear`` so CUDA N>1 is MMA ``chr_nf4_gemm``.
    ``lm_head`` runs only when ``logits`` (last chunk): TTFT does not score
    intermediate prompt rows.
    """
    ids = token_ids.reshape(-1).to(device=state.device, dtype=torch.long)
    n = int(ids.numel())
    if n < 1:
        raise ValueError("prefill chunk must be non-empty")
    start = int(start)
    seq = start + n
    if seq > state.spec.max_seq:
        raise ValueError(f"prefill {start}..{seq} exceeds max_seq={state.spec.max_seq}")
    spec = state.spec
    x = weights.embed.index_select(0, ids)
    cos = state.cos[start:seq]
    sin = state.sin[start:seq]
    mask = _causal_mask(start, n, seq, state.device)
    for li, layer in enumerate(weights.layers):
        h = rms(x, layer.norm1, spec.rms_eps)
        q, k, v = split_qkv(layer, h, spec)
        q = apply_rope_torch(q, cos, sin)
        k = apply_rope_torch(k, cos, sin)
        state.kv.write_range(li, start, k, v)
        k_all, v_all = state.kv.view(li, seq)
        attn = _attend(q, k_all, v_all, spec, mask)
        x = x + nf4_linear(layer.o, attn)
        h = rms(x, layer.norm2, spec.rms_eps)
        down_in = swiglu(nf4_linear(layer.gate, h), nf4_linear(layer.up, h))
        x = x + nf4_linear(layer.down, down_in)
    state.position.fill_(seq)
    state.valid_len.fill_(seq)
    if not logits:
        return
    hidden = rms(x[-1:], weights.final_norm, spec.rms_eps)
    logits_t = nf4_linear(weights.lm_head, hidden)
    state.arena.logits[0].copy_(logits_t.reshape(-1))
    state.next_token.copy_(logits_t.argmax(dim=-1).reshape(()).to(dtype=torch.int64))

"""Graph-friendly decode ops. Attention is matmul+mask+softmax (synthetic max_seq)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gpu.loop.generate import split_internlm_wqkv

from .linear import DeviceLayer, nf4_linear
from .plan import ArchSpec
from .rope import apply_rope_torch
from .state import DecodeState

__all__ = ["attend", "rms", "split_qkv", "swiglu"]


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    if x.device.type == "cuda" and hasattr(F, "rms_norm"):
        return F.rms_norm(x, w.shape, w, eps)
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (w.float() * v).to(dtype=x.dtype)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def split_qkv(
    layer: DeviceLayer, h: torch.Tensor, spec: ArchSpec
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(h.shape[0])
    if spec.family == "internlm_gqa":
        if layer.wqkv is None:
            raise RuntimeError("internlm_gqa layer missing wqkv")
        y = nf4_linear(layer.wqkv, h)
        return split_internlm_wqkv(y, spec.n_q, spec.n_kv, spec.head_dim)
    if layer.q is None or layer.k is None or layer.v is None:
        raise RuntimeError("llama_swiglu layer missing q/k/v")
    q = nf4_linear(layer.q, h).reshape(n, spec.n_q, spec.head_dim)
    k = nf4_linear(layer.k, h).reshape(n, spec.n_kv, spec.head_dim)
    v = nf4_linear(layer.v, h).reshape(n, spec.n_kv, spec.head_dim)
    return q, k, v


def attend(state: DecodeState, layer: int, q: torch.Tensor) -> torch.Tensor:
    """GQA without repeating KV. Mask from GPU ``valid_len``, full ``max_seq`` axis."""
    spec = state.spec
    q4 = q.reshape(1, spec.n_kv, spec.n_rep, spec.head_dim)
    k, v = state.kv.attn(layer)
    mask = state.kv.additive_mask(state.arena.attn_mask)
    if q.device.type == "cuda":
        a = F.scaled_dot_product_attention(
            q4, k, v, attn_mask=mask.to(dtype=q4.dtype), scale=spec.attn_scale
        )
        return a.to(dtype=q.dtype).reshape(1, spec.q_dim)
    scores = torch.matmul(q4.float(), k.float().transpose(-2, -1))
    scores = scores * spec.attn_scale + mask
    w = torch.softmax(scores, dim=-1)
    a = torch.matmul(w, v.float())
    return a.to(dtype=q.dtype).reshape(1, spec.q_dim)


def rope_qkv(state: DecodeState, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = state.rope_at_position()
    return apply_rope_torch(q, cos, sin), apply_rope_torch(k, cos, sin)

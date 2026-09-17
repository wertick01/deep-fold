"""Eager resident decode step. No Python seq_len, no .item() on the hot path."""

from __future__ import annotations

import torch

from .linear import DeviceLayer, DeviceLinear, DeviceWeights, linear_backend, nf4_linear
from .ops import attend, rms, rope_qkv, split_qkv, swiglu
from .state import DecodeState

__all__ = ["forward_decode", "greedy_decode", "teacher_force_token"]


def _into(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    dst.copy_(src.reshape(dst.shape))
    return dst


def _rms_into(x: torch.Tensor, w: torch.Tensor, eps: float, out: torch.Tensor) -> torch.Tensor:
    if x.device.type == "cuda":
        from gpu.nf4 import nf4_rms

        nf4_rms(x.reshape(-1), w.reshape(-1), out.reshape(-1), eps)
        return out
    return _into(out, rms(x, w, eps))


def _attend_into(state: DecodeState, layer_i: int) -> None:
    ar = state.arena
    if ar.q.device.type == "cuda":
        from gpu.nf4 import ATTN_SPLIT, nf4_attn

        nf4_attn(
            ar.q,
            state.kv.k[layer_i],
            state.kv.v[layer_i],
            ar.attn,
            state.valid_len,
            state.spec.attn_scale,
            ar.attn_ws,
            ATTN_SPLIT,
            ar.k,
            ar.v,
            state.cos,
            state.sin,
            state.position,
        )
        return
    _into(ar.attn, attend(state, layer_i, ar.q))


def _cuda_gemv() -> bool:
    return linear_backend() == "gemv"


def _qkv_into_arena(state: DecodeState, layer: DeviceLayer, h: torch.Tensor) -> None:
    spec = state.spec
    ar = state.arena
    if (
        h.device.type == "cuda"
        and _cuda_gemv()
        and layer.q is not None
        and layer.k is not None
        and layer.v is not None
        and int(h.shape[0]) == 1
    ):
        from gpu.nf4 import nf4_qkv

        nf4_qkv(
            layer.q.packed,
            layer.q.scale,
            layer.k.packed,
            layer.k.scale,
            layer.v.packed,
            layer.v.scale,
            h,
            layer.q.M,
            layer.k.M,
            layer.v.M,
            layer.q.K,
            layer.q.K_pad,
            ar.q.view(-1),
            ar.k.view(-1),
            ar.v.view(-1),
            layer.q.bias,
            layer.k.bias,
            layer.v.bias,
        )
        return
    q, k, v = split_qkv(layer, h, spec)
    _into(ar.q, q)
    _into(ar.k, k)
    _into(ar.v, v)


def _rope_and_cache(state: DecodeState, layer_i: int) -> None:
    ar = state.arena
    if ar.q.device.type == "cuda":
        from gpu.nf4 import nf4_rope_kv

        nf4_rope_kv(
            ar.q,
            ar.k,
            ar.v,
            state.kv.k[layer_i],
            state.kv.v[layer_i],
            state.cos,
            state.sin,
            state.position,
        )
        return
    q, k = rope_qkv(state, ar.q, ar.k)
    _into(ar.q, q)
    _into(ar.k, k)
    state.kv.write(layer_i, state.position, ar.k, ar.v)


def _gemv_accum(mat: DeviceLinear, src: torch.Tensor, dest: torch.Tensor) -> None:
    """``dest += W @ src`` (residual). CUDA GEMV writes in-place, one launch."""
    if dest.device.type == "cuda" and _cuda_gemv():
        from gpu.nf4 import nf4_gemv

        nf4_gemv(
            mat.packed,
            mat.scale,
            src.reshape(mat.K),
            mat.M,
            mat.K,
            mat.K_pad,
            dest.view(-1),
            dest.view(-1),
        )
        if mat.bias is not None:
            dest.view(-1).add_(mat.bias.to(dtype=dest.dtype))
        return
    dest.add_(nf4_linear(mat, src).reshape(dest.shape))


def _mlp_down_in(state: DecodeState, layer: DeviceLayer, h: torch.Tensor) -> None:
    ar = state.arena
    if h.device.type == "cuda" and _cuda_gemv() and int(h.shape[0]) == 1:
        from gpu.nf4 import nf4_swiglu

        nf4_swiglu(
            layer.gate.packed,
            layer.gate.scale,
            layer.up.packed,
            layer.up.scale,
            h,
            layer.gate.M,
            layer.gate.K,
            layer.gate.K_pad,
            ar.down_in.view(-1),
        )
        return
    gate = _into(ar.gate, nf4_linear(layer.gate, h))
    up = _into(ar.up, nf4_linear(layer.up, h))
    _into(ar.down_in, swiglu(gate, up))


def forward_decode(state: DecodeState, weights: DeviceWeights) -> None:
    spec = state.spec
    ar = state.arena
    x = _into(ar.x, weights.embed.index_select(0, state.token.reshape(1)))
    for li, layer in enumerate(weights.layers):
        h = _rms_into(x, layer.norm1, spec.rms_eps, ar.h)
        _qkv_into_arena(state, layer, h)
        if ar.q.device.type != "cuda":
            _rope_and_cache(state, li)
        if li == 0:
            state.kv.mark_written(state.position)
        _attend_into(state, li)
        _gemv_accum(layer.o, ar.attn, ar.x)
        h = _rms_into(ar.x, layer.norm2, spec.rms_eps, ar.h)
        _mlp_down_in(state, layer, h)
        _gemv_accum(layer.down, ar.down_in, ar.x)
        x = ar.x
    hidden = _rms_into(x, weights.final_norm, spec.rms_eps, ar.h)
    logits = nf4_linear(weights.lm_head, hidden)
    _into(ar.logits, logits)
    state.next_token.copy_(ar.logits.argmax(dim=-1).reshape(()).to(dtype=torch.int64))


def greedy_decode(state: DecodeState, weights: DeviceWeights) -> None:
    forward_decode(state, weights)
    state.commit_step()


def teacher_force_token(state: DecodeState, weights: DeviceWeights, token_id: int) -> None:
    state.load_token(token_id)
    forward_decode(state, weights)
    state.position.add_(1)

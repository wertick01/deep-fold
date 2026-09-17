"""CPU float32 teacher-forced oracle for synthetic NF4 models."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gpu.tests.nf4_oracle import decode_nf4, matmul_f32

from .plan import ArchSpec
from .rope import apply_rope_numpy, rope_tables_numpy
from .synth import Nf4Matrix, SynthModel

__all__ = [
    "OracleCache",
    "greedy_ids",
    "nf4_linear",
    "rms",
    "silu",
    "teacher_logits",
]

_NEG = np.float32(-1.0e9)


def rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    v = x.astype(np.float32, copy=False)
    mean = (v * v).mean(axis=-1, keepdims=True)
    return (v * np.reciprocal(np.sqrt(mean + np.float32(eps))) * w.astype(np.float32)).astype(
        np.float32
    )


def silu(x: np.ndarray) -> np.ndarray:
    v = x.astype(np.float32, copy=False)
    return (v * (1.0 / (1.0 + np.exp(-v)))).astype(np.float32)


def nf4_linear(mat: Nf4Matrix, x: np.ndarray) -> np.ndarray:
    """``x`` is ``[N, K]`` float32 → ``[N, M]`` float32."""
    w = decode_nf4(mat.packed, mat.scale, mat.M, mat.K)
    xn = x.astype(np.float32, copy=False)
    y = matmul_f32(w, xn.T.copy())
    out = y.T
    if mat.bias is not None:
        out = out + mat.bias.astype(np.float32)
    return out.astype(np.float32)


def split_internlm(y: np.ndarray, spec: ArchSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(y.shape[0])
    packed = y.reshape(n, spec.n_kv, 2 + spec.n_rep, spec.head_dim)
    q = packed[:, :, : spec.n_rep, :].reshape(n, spec.n_q, spec.head_dim)
    k = packed[:, :, -2, :]
    v = packed[:, :, -1, :]
    return q, k, v


def attend(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    spec: ArchSpec,
    valid_len: int,
) -> np.ndarray:
    """GQA without repeating KV. ``q`` ``[N, n_q, hd]``; ``k``/``v`` ``[max_seq, n_kv, hd]``."""
    n = int(q.shape[0])
    q4 = q.reshape(n, spec.n_kv, spec.n_rep, spec.head_dim)
    k4 = np.transpose(k, (1, 0, 2))[None, ...]  # [1, n_kv, max_seq, hd]
    v4 = np.transpose(v, (1, 0, 2))[None, ...]
    if n != 1:
        k4 = np.broadcast_to(k4, (n,) + k4.shape[1:])
        v4 = np.broadcast_to(v4, (n,) + v4.shape[1:])
    scores = np.einsum("bhrd,bhtd->bhrt", q4, k4).astype(np.float32)
    scores *= np.float32(spec.attn_scale)
    scores[..., int(valid_len) :] = _NEG
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=-1, keepdims=True)
    a = np.einsum("bhrt,bhtd->bhrd", w, v4).astype(np.float32)
    return a.reshape(n, spec.q_dim)


class OracleCache:
    """Numpy KV used by tests that poison the tail."""

    def __init__(self, spec: ArchSpec) -> None:
        self.spec = spec
        shape = (spec.n_layers, spec.max_seq, spec.n_kv, spec.head_dim)
        self.k = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.pos = 0
        self.cos, self.sin = rope_tables_numpy(spec.max_seq, spec.head_dim, spec.rope_theta)

    def reset(self) -> None:
        self.pos = 0


def _qkv(layer, h: np.ndarray, spec: ArchSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if spec.family == "internlm_gqa":
        if layer.wqkv is None:
            raise RuntimeError("internlm_gqa layer missing wqkv")
        return split_internlm(nf4_linear(layer.wqkv, h), spec)
    if layer.q is None or layer.k is None or layer.v is None:
        raise RuntimeError("llama_swiglu layer missing q/k/v")
    q = nf4_linear(layer.q, h).reshape(h.shape[0], spec.n_q, spec.head_dim)
    k = nf4_linear(layer.k, h).reshape(h.shape[0], spec.n_kv, spec.head_dim)
    v = nf4_linear(layer.v, h).reshape(h.shape[0], spec.n_kv, spec.head_dim)
    return q, k, v


def step_logits(model: SynthModel, cache: OracleCache, token_id: int) -> np.ndarray:
    spec = model.spec
    if cache.pos >= spec.max_seq:
        raise ValueError(f"position {cache.pos} past max_seq={spec.max_seq}")
    x = model.embed[int(token_id) : int(token_id) + 1]
    pos = cache.pos
    cos = cache.cos[pos : pos + 1]
    sin = cache.sin[pos : pos + 1]
    for li, layer in enumerate(model.layers):
        h = rms(x, layer.norm1, spec.rms_eps)
        q, k, v = _qkv(layer, h, spec)
        q = apply_rope_numpy(q, cos, sin)
        k = apply_rope_numpy(k, cos, sin)
        cache.k[li, pos] = k[0]
        cache.v[li, pos] = v[0]
        valid = pos + 1
        a = attend(q, cache.k[li], cache.v[li], spec, valid)
        x = x + nf4_linear(layer.o, a)
        h = rms(x, layer.norm2, spec.rms_eps)
        gate = nf4_linear(layer.gate, h)
        up = nf4_linear(layer.up, h)
        x = x + nf4_linear(layer.down, silu(gate) * up)
    hidden = rms(x, model.final_norm, spec.rms_eps)
    logits = nf4_linear(model.lm_head, hidden)[0]
    cache.pos = pos + 1
    return logits.astype(np.float32)


def teacher_logits(model: SynthModel, token_ids: Sequence[int]) -> np.ndarray:
    cache = OracleCache(model.spec)
    rows = [step_logits(model, cache, int(t)) for t in token_ids]
    if not rows:
        return np.zeros((0, model.spec.vocab), dtype=np.float32)
    return np.stack(rows, axis=0)


def greedy_ids(model: SynthModel, prompt_ids: Sequence[int], n_new: int) -> list[int]:
    if not prompt_ids:
        raise ValueError("prompt_ids must be non-empty")
    cache = OracleCache(model.spec)
    last = None
    for t in prompt_ids:
        last = step_logits(model, cache, int(t))
    assert last is not None
    out: list[int] = []
    tok = int(last.argmax())
    eos = int(model.spec.eos_id)
    for _ in range(int(n_new)):
        out.append(tok)
        if tok == eos:
            break
        last = step_logits(model, cache, tok)
        tok = int(last.argmax())
    return out

"""Seeded tiny NF4 transformers. No checkpoints, no CHR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gpu.tests.nf4_oracle import encode_nf4, k_pad

from .plan import ArchSpec

__all__ = ["Nf4Matrix", "SynthLayer", "SynthModel", "build"]


@dataclass
class Nf4Matrix:
    packed: np.ndarray
    scale: np.ndarray
    M: int
    K: int
    K_pad: int
    bias: np.ndarray | None = None


@dataclass
class SynthLayer:
    norm1: np.ndarray
    norm2: np.ndarray
    q: Nf4Matrix | None
    k: Nf4Matrix | None
    v: Nf4Matrix | None
    wqkv: Nf4Matrix | None
    o: Nf4Matrix
    gate: Nf4Matrix
    up: Nf4Matrix
    down: Nf4Matrix


@dataclass
class SynthModel:
    spec: ArchSpec
    seed: int
    embed: np.ndarray
    final_norm: np.ndarray
    lm_head: Nf4Matrix
    layers: list[SynthLayer]


def _matrix(rng: np.random.Generator, m: int, k: int, scale: float) -> Nf4Matrix:
    w = rng.standard_normal((m, k), dtype=np.float32) * np.float32(scale)
    packed, sc = encode_nf4(w)
    return Nf4Matrix(packed=packed, scale=sc, M=m, K=k, K_pad=k_pad(k), bias=None)


def _vec(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    return rng.standard_normal(n, dtype=np.float32) * np.float32(scale)


def build(spec: ArchSpec, seed: int) -> SynthModel:
    """Finite NF4 toy model. ``llama_swiglu`` split QKV or ``internlm_gqa`` fused wqkv."""
    if spec.family not in {"llama_swiglu", "internlm_gqa"}:
        raise ValueError(f"unsupported family {spec.family!r}")
    rng = np.random.default_rng(int(seed))
    h, kv, inter = spec.hidden, spec.kv_dim, spec.intermediate
    layers: list[SynthLayer] = []
    for _ in range(spec.n_layers):
        if spec.family == "llama_swiglu":
            q, k, v, wqkv = (
                _matrix(rng, h, h, 0.05),
                _matrix(rng, kv, h, 0.05),
                _matrix(rng, kv, h, 0.05),
                None,
            )
        else:
            q = k = v = None
            wqkv = _matrix(rng, spec.wqkv_out, h, 0.05)
        layers.append(
            SynthLayer(
                norm1=_vec(rng, h, 0.5),
                norm2=_vec(rng, h, 0.5),
                q=q,
                k=k,
                v=v,
                wqkv=wqkv,
                o=_matrix(rng, h, h, 0.05),
                gate=_matrix(rng, inter, h, 0.05),
                up=_matrix(rng, inter, h, 0.05),
                down=_matrix(rng, h, inter, 0.05),
            )
        )
    return SynthModel(
        spec=spec,
        seed=int(seed),
        embed=_vec(rng, spec.vocab * h, 0.5).reshape(spec.vocab, h),
        final_norm=_vec(rng, h, 0.5),
        lm_head=_matrix(rng, spec.vocab, h, 0.05),
        layers=layers,
    )

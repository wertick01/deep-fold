"""Rotate-half RoPE tables shared by the CPU oracle and the GPU step.

Matches ``gpu.loop.generate._rope``: ``rot = cat(-t[..., d:], t[..., :d])``
then ``t * cos + rot * sin``. Tables are ``[max_seq, head_dim]`` with the
HuggingFace ``cat(freqs, freqs)`` layout (not interleaved pairs).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "apply_rope_numpy",
    "apply_rope_torch",
    "rope_tables_numpy",
    "rope_tables_torch",
]


def _inv_freq(head_dim: int, theta: float) -> np.ndarray:
    i = np.arange(0, head_dim, 2, dtype=np.float32)
    return (1.0 / (np.float32(theta) ** (i / np.float32(head_dim)))).astype(np.float32)


def rope_tables_numpy(max_seq: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """``cos``, ``sin`` as float32 ``[max_seq, head_dim]``."""
    if head_dim % 2 != 0:
        raise ValueError("even head_dim required")
    inv = _inv_freq(head_dim, theta)
    pos = np.arange(max_seq, dtype=np.float32)
    freqs = np.outer(pos, inv).astype(np.float32)  # [S, hd/2]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope_numpy(t: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """``t`` is ``[N, heads, hd]``; ``cos``/``sin`` ``[N, hd]`` or ``[N, 1, hd]``."""
    if cos.ndim == 2:
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    d = t.shape[-1] // 2
    rot = np.concatenate((-t[..., d:], t[..., :d]), axis=-1)
    return (t * cos + rot * sin).astype(np.float32)


def rope_tables_torch(max_seq: int, head_dim: int, theta: float, *, device, dtype):
    import torch

    cos, sin = rope_tables_numpy(max_seq, head_dim, theta)
    return (
        torch.from_numpy(cos).to(device=device, dtype=dtype),
        torch.from_numpy(sin).to(device=device, dtype=dtype),
    )


def apply_rope_torch(t, cos, sin):
    """Same rotate-half as ``gpu.loop.generate._rope``."""
    import torch

    if cos.ndim == 2:
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    d = t.shape[-1] // 2
    rot = torch.cat((-t[..., d:], t[..., :d]), dim=-1)
    return t * cos + rot * sin

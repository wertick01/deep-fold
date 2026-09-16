"""i4c pack/decode oracle. No CUDA, no GGUF."""

from __future__ import annotations

import numpy as np

GROUP = 64


def k_pad_i4c(k: int) -> int:
    return GROUP * ((int(k) + GROUP - 1) // GROUP)


def pack_i4c(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``w`` float32 ``[M, K]`` → packed uint8 ``[M, K_pad/2]``, fp16 scale ``[M, G]``."""
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"w must be [M,K], got {w.shape}")
    if not np.isfinite(w).all():
        raise ValueError("i4c: non-finite weight")
    m, k = int(w.shape[0]), int(w.shape[1])
    kp = k_pad_i4c(k)
    if kp != k:
        w = np.pad(w, ((0, 0), (0, kp - k)))
    ng = kp // GROUP
    blocks = w.reshape(m, ng, GROUP)
    amax = np.max(np.abs(blocks), axis=-1)
    amax = np.where(amax == 0.0, 1.0, amax)
    s16 = (amax / 8.0).astype(np.float16)
    s = s16.astype(np.float32)
    q = np.rint(np.clip(blocks / s[..., None], -8.0, 7.0)).astype(np.int32) + 8
    q = np.clip(q, 0, 15).astype(np.uint8)
    packed = (q[..., 0::2] | (q[..., 1::2].astype(np.uint16) << 4)).astype(np.uint8)
    packed = packed.reshape(m, kp // 2)
    return packed, s16


def decode_i4c(packed: np.ndarray, scale: np.ndarray, m: int, k: int) -> np.ndarray:
    """Float32 ``W_hat[M, K]`` (tests only; the kernel must not write this)."""
    packed = np.asarray(packed, dtype=np.uint8)
    scale = np.asarray(scale, dtype=np.float16).astype(np.float32)
    kp = k_pad_i4c(k)
    ng = kp // GROUP
    q = np.empty((m, ng, GROUP), dtype=np.int32)
    raw = packed.reshape(m, ng, GROUP // 2)
    q[..., 0::2] = raw & 0x0F
    q[..., 1::2] = (raw >> 4) & 0x0F
    w = (q - 8).astype(np.float32) * scale.reshape(m, ng, 1)
    return w.reshape(m, kp)[:, :k]


def toy_i4c(m: int, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((m, k), dtype=np.float32).astype(np.float32)
    return pack_i4c(w)

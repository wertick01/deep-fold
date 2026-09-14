"""Host overflow layout: packed‖scale in one 1d uint8 arena (one future H2D).

Matches ``gpu.chr0.loader`` views: packed is a prefix of the arena, scale is
the suffix as float16. No mmap, no ``cudaHostRegister``. Pin is optional and
skipped when CUDA is missing so CPU tests do not need a device.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "GROUP_SIZE",
    "HostImage",
    "cpu_is_pinned",
    "maybe_pin",
    "pack_arena",
    "unpack_arena",
]

GROUP_SIZE = 64


def k_pad(k: int) -> int:
    """``64 * ceil(K / 64)`` — same as ChrMatrix / ``gpu.host.linear.k_pad``."""
    return GROUP_SIZE * ((int(k) + GROUP_SIZE - 1) // GROUP_SIZE)


def pack_arena(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Concatenate C-contiguous packed bytes and ``scale.view(uint8)`` into one 1d tensor."""
    if packed.dtype is not torch.uint8:
        raise TypeError(f"packed must be uint8, got {packed.dtype}")
    if scale.dtype is not torch.float16:
        raise TypeError(f"scale must be float16, got {scale.dtype}")
    if packed.dim() != 2 or scale.dim() != 2:
        raise ValueError(
            f"packed {tuple(packed.shape)} / scale {tuple(scale.shape)}: expected rank 2"
        )
    m, packed_w = int(packed.shape[0]), int(packed.shape[1])
    if int(scale.shape[0]) != m:
        raise ValueError(f"M mismatch: packed {m} vs scale {int(scale.shape[0])}")
    k_pad_ = packed_w * 2
    n_groups = k_pad_ // GROUP_SIZE
    if int(scale.shape[1]) != n_groups:
        raise ValueError(
            f"scale n_groups {int(scale.shape[1])} != K_pad/64 ({n_groups})"
        )

    packed_flat = packed.contiguous().reshape(-1)
    scale_u8 = scale.contiguous().view(torch.uint8).reshape(-1)
    arena = torch.empty(packed_flat.numel() + scale_u8.numel(), dtype=torch.uint8)
    arena[: packed_flat.numel()].copy_(packed_flat)
    arena[packed_flat.numel() :].copy_(scale_u8)
    return arena


def unpack_arena(
    arena: torch.Tensor,
    m: int,
    k: int,
    *,
    k_pad_: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """packed and scale views into ``arena``. Same storage, offset; no copy."""
    if arena.dtype is not torch.uint8 or arena.dim() != 1:
        raise TypeError(f"arena must be uint8 1d, got {arena.dtype} {tuple(arena.shape)}")
    m = int(m)
    pad = int(k_pad_) if k_pad_ is not None else k_pad(k)
    packed_n = m * (pad // 2)
    n_groups = pad // GROUP_SIZE
    scale_n = m * n_groups * 2
    if int(arena.numel()) != packed_n + scale_n:
        raise ValueError(
            f"arena {int(arena.numel())} bytes != packed {packed_n} + scale {scale_n}"
        )
    packed = arena[:packed_n].view(m, pad // 2)
    scale = arena[packed_n:].view(torch.float16).view(m, n_groups)
    return packed, scale


def cpu_is_pinned(t: torch.Tensor) -> bool:
    """True when ``t`` is a pinned CPU tensor.

    ``Tensor.is_pinned`` is a method. ``bool(t.is_pinned)`` is always True
    (the bound method is truthy), which is how attach_host skipped pin and
    ``host_pin_snapshot`` reported ``all_pinned=True`` on pageable arenas.
    """
    if not t.is_cpu:
        return False
    flag = t.is_pinned
    return bool(flag() if callable(flag) else flag)


def maybe_pin(t: torch.Tensor) -> torch.Tensor:
    """``pin_memory()`` when CUDA is present; otherwise the same CPU tensor."""
    if not t.is_cpu:
        return t
    if not torch.cuda.is_available():
        return t
    if cpu_is_pinned(t):
        return t
    return t.pin_memory()


@dataclass(frozen=True)
class HostImage:
    """One NF4 matrix on the host as packed‖scale. One future H2D of ``arena``."""

    arena: torch.Tensor
    M: int
    K: int
    K_pad: int

    @property
    def packed(self) -> torch.Tensor:
        n = self.M * (self.K_pad // 2)
        return self.arena[:n].view(self.M, self.K_pad // 2)

    @property
    def scale(self) -> torch.Tensor:
        n = self.M * (self.K_pad // 2)
        n_groups = self.K_pad // GROUP_SIZE
        return self.arena[n:].view(torch.float16).view(self.M, n_groups)

    @property
    def nbytes(self) -> int:
        return int(self.arena.numel())

    @classmethod
    def from_blobs(cls, packed: torch.Tensor, scale: torch.Tensor, k: int) -> HostImage:
        pad = int(packed.shape[1]) * 2
        if pad != k_pad(k):
            raise ValueError(f"packed width {packed.shape[1]} != K_pad/2 ({k_pad(k) // 2})")
        arena = pack_arena(packed, scale)
        return cls(arena, int(packed.shape[0]), int(k), pad)

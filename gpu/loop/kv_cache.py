"""Preallocated KV. The cache is written in place and never grown.

``docs/token-loop.md`` §1.2 is blunt about why this file exists instead of
``transformers.DynamicCache``: a ``cat`` along the length axis on every decode
step reallocates and copies the whole cache, and on 12 GB that dies a few
hundred tokens in even when the arithmetic said it fit. So the shape is fixed at
session start:

    k, v: [n_layers, max_seq, n_kv_heads, head_dim]  bf16

and a step does ``k[layer, t] = k_new`` -- one copy into an existing slot.
There is no ``torch.cat`` in this module, and :meth:`KVCache.view` hands out
strided views, not clones.

The layout is the one the token loop wants to *write*; SDPA wants
``[batch, heads, seq, head_dim]``, which is a ``permute`` away and costs
nothing (``head_dim`` stays the fastest-moving axis, so the flash/mem-efficient
backends still take it).
"""

from __future__ import annotations

import torch

__all__ = ["KVCache"]

MIB = 1024 * 1024


class KVCache:
    """Two fixed ``[n_layers, max_seq, n_kv_heads, head_dim]`` tensors.

    ``seq_len`` is the number of *valid* slots, owned by the caller (the token
    loop advances it after a write). Slots past it are zeros that nothing reads.
    """

    __slots__ = (
        "k",
        "v",
        "n_layers",
        "max_seq",
        "n_kv_heads",
        "head_dim",
        "seq_len",
        "_k",
        "_v",
    )

    def __init__(
        self,
        n_layers: int,
        max_seq: int,
        n_kv_heads: int,
        head_dim: int,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if min(n_layers, max_seq, n_kv_heads, head_dim) < 1:
            raise ValueError(
                f"bad shape: layers={n_layers} max_seq={max_seq} "
                f"kv_heads={n_kv_heads} head_dim={head_dim}"
            )
        self.n_layers = int(n_layers)
        self.max_seq = int(max_seq)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)

        shape = (self.n_layers, self.max_seq, self.n_kv_heads, self.head_dim)
        # zeros, not empty: the unwritten tail is never read, but paying for the
        # pages up front is what makes `nvidia-smi` flat during generation.
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.seq_len = 0

        # Per-layer views, built once. `self.k[layer]` would be an advanced
        # index on every layer of every token; these are the same 36 views.
        self._k = [self.k[i] for i in range(self.n_layers)]
        self._v = [self.v[i] for i in range(self.n_layers)]

    # --- accounting --------------------------------------------------------
    @property
    def dtype(self) -> torch.dtype:
        return self.k.dtype

    @property
    def device(self) -> torch.device:
        return self.k.device

    @property
    def nbytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()

    @property
    def mib(self) -> float:
        return self.nbytes / MIB

    @property
    def bytes_per_token(self) -> int:
        """KV cost of one more token, all layers (token-loop.md §1.2 table)."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.k.element_size()

    def reset(self) -> None:
        """New request, same memory. Nothing is freed and nothing is zeroed."""
        self.seq_len = 0

    # --- the two operations a step needs -----------------------------------
    def write(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """``k``/``v`` are ``[n, n_kv_heads, head_dim]`` for slots ``start..start+n``.

        Assignment into a slice of an existing tensor: one copy kernel each, no
        allocation, no ``cat``.
        """
        n = k.shape[0]
        end = start + n
        if end > self.max_seq:
            raise ValueError(
                f"KV overflow: writing slots {start}..{end} into max_seq={self.max_seq}"
            )
        self._k[layer][start:end] = k
        self._v[layer][start:end] = v

    def view(self, layer: int, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``[1, n_kv_heads, seq, head_dim]`` views for SDPA. No copy."""
        k = self._k[layer][:seq].permute(1, 0, 2).unsqueeze(0)
        v = self._v[layer][:seq].permute(1, 0, 2).unsqueeze(0)
        return k, v

    def __repr__(self) -> str:
        return (
            f"KVCache(layers={self.n_layers}, max_seq={self.max_seq}, "
            f"kv_heads={self.n_kv_heads}, head_dim={self.head_dim}, "
            f"dtype={self.dtype}, {self.mib:.1f} MiB, "
            f"{self.bytes_per_token / 1024:.0f} KiB/token, seq_len={self.seq_len})"
        )

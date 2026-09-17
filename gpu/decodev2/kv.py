"""KV with static capacity. Attention never takes a Python ``:seq`` slice."""

from __future__ import annotations

import torch

from .plan import ArchSpec

__all__ = ["GraphSafeKV"]

_NEG = -1.0e9


class GraphSafeKV:
    """``[layers, max_seq, n_kv, head_dim]`` plus a device ``valid_len``.

    Attention reads the full ``max_seq`` axis. The tail is hidden by an
    additive mask derived from ``valid_len``, not by zeros in storage.
    """

    __slots__ = (
        "k",
        "v",
        "valid_len",
        "n_layers",
        "max_seq",
        "n_kv",
        "head_dim",
        "_k",
        "_v",
        "_k_attn",
        "_v_attn",
        "_pos",
        "_zero",
        "_neg",
    )

    def __init__(
        self,
        spec: ArchSpec,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        valid_len: torch.Tensor | None = None,
    ) -> None:
        self.n_layers = spec.n_layers
        self.max_seq = spec.max_seq
        self.n_kv = spec.n_kv
        self.head_dim = spec.head_dim
        shape = (spec.n_layers, spec.max_seq, spec.n_kv, spec.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        if valid_len is None:
            self.valid_len = torch.zeros((), dtype=torch.int32, device=device)
        else:
            if valid_len.device != self.k.device:
                raise ValueError("valid_len must live on the KV device")
            self.valid_len = valid_len.view(())
        self._k = [self.k[i] for i in range(self.n_layers)]
        self._v = [self.v[i] for i in range(self.n_layers)]
        # SDPA layout [1, n_kv, max_seq, hd] — full axis, never sliced.
        self._k_attn = [t.permute(1, 0, 2).unsqueeze(0) for t in self._k]
        self._v_attn = [t.permute(1, 0, 2).unsqueeze(0) for t in self._v]
        self._pos = torch.arange(self.max_seq, device=self.k.device, dtype=torch.int32)
        self._zero = torch.zeros((), dtype=torch.float32, device=self.k.device)
        self._neg = torch.tensor(_NEG, dtype=torch.float32, device=self.k.device)

    @property
    def device(self) -> torch.device:
        return self.k.device

    @property
    def dtype(self) -> torch.dtype:
        return self.k.dtype

    def attn(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Full-capacity ``[1, n_kv, max_seq, hd]`` views."""
        return self._k_attn[layer], self._v_attn[layer]

    def additive_mask(self, out: torch.Tensor | None = None) -> torch.Tensor:
        """``[1, 1, 1, max_seq]`` float mask: 0 on live slots, large negative on the tail.

        When ``out`` is set, fill it in place (arena buffer, no step-path alloc).
        """
        live = self._pos < self.valid_len
        if out is None:
            return torch.where(live, self._zero, self._neg).view(1, 1, 1, self.max_seq)
        out.fill_(_NEG)
        out.view(-1).masked_fill_(live, 0.0)
        return out

    def write(self, layer: int, position: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write decode ``N=1`` rows. ``position`` is a GPU integer tensor."""
        idx = position.reshape(1).to(dtype=torch.long)
        k_row = k.reshape(1, self.n_kv, self.head_dim)
        v_row = v.reshape(1, self.n_kv, self.head_dim)
        self._k[layer].index_copy_(0, idx, k_row)
        self._v[layer].index_copy_(0, idx, v_row)

    def mark_written(self, position: torch.Tensor) -> None:
        """``valid_len = position + 1``. Tensor math, safe to capture."""
        self.valid_len.copy_((position.reshape(()).to(dtype=torch.int32) + 1).to(torch.int32))

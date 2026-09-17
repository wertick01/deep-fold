"""Preallocated activations for one decode token. Nothing on the step path allocates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .plan import ArchSpec

__all__ = ["Arena"]

_ATTN_SPLIT = 32


@dataclass
class Arena:
    spec: ArchSpec
    x: torch.Tensor
    h: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    attn: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    down_in: torch.Tensor
    logits: torch.Tensor
    attn_mask: torch.Tensor
    attn_ws: torch.Tensor

    @classmethod
    def allocate(
        cls,
        spec: ArchSpec,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "Arena":
        z = lambda *shape: torch.zeros(shape, dtype=dtype, device=device)
        return cls(
            spec=spec,
            x=z(1, spec.hidden),
            h=z(1, spec.hidden),
            q=z(1, spec.n_q, spec.head_dim),
            k=z(1, spec.n_kv, spec.head_dim),
            v=z(1, spec.n_kv, spec.head_dim),
            attn=z(1, spec.q_dim),
            gate=z(1, spec.intermediate),
            up=z(1, spec.intermediate),
            down_in=z(1, spec.intermediate),
            logits=z(1, spec.vocab),
            attn_mask=torch.zeros((1, 1, 1, spec.max_seq), dtype=torch.float32, device=device),
            attn_ws=torch.zeros(
                spec.n_q * _ATTN_SPLIT * (spec.head_dim + 2),
                dtype=torch.float32,
                device=device,
            ),
        )

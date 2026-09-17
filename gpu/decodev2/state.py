"""GPU-resident decode registers: token, position, valid_len, next id."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .arena import Arena
from .kv import GraphSafeKV
from .plan import ArchSpec
from .rope import rope_tables_torch

__all__ = ["DecodeState"]


@dataclass
class DecodeState:
    spec: ArchSpec
    token: torch.Tensor
    position: torch.Tensor
    valid_len: torch.Tensor
    next_token: torch.Tensor
    finished: torch.Tensor
    kv: GraphSafeKV
    arena: Arena
    cos: torch.Tensor
    sin: torch.Tensor
    embed: torch.Tensor
    eos_id: torch.Tensor

    @classmethod
    def allocate(
        cls,
        spec: ArchSpec,
        embed: torch.Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> "DecodeState":
        if embed.ndim != 2 or embed.shape != (spec.vocab, spec.hidden):
            raise ValueError(f"embed shape {tuple(embed.shape)} != {(spec.vocab, spec.hidden)}")
        if device is None:
            device = embed.device
        device = torch.device(device)
        embed = embed.to(device=device, dtype=dtype)
        valid_len = torch.zeros((), dtype=torch.int32, device=device)
        kv = GraphSafeKV(spec, device=device, dtype=dtype, valid_len=valid_len)
        arena = Arena.allocate(spec, device=device, dtype=dtype)
        if cos is None or sin is None:
            cos, sin = rope_tables_torch(
                spec.max_seq, spec.head_dim, spec.rope_theta, device=device, dtype=dtype
            )
        else:
            cos = cos.to(device=device, dtype=dtype)
            sin = sin.to(device=device, dtype=dtype)
        return cls(
            spec=spec,
            token=torch.zeros((), dtype=torch.int64, device=device),
            position=torch.zeros((), dtype=torch.int64, device=device),
            valid_len=valid_len,
            next_token=torch.zeros((), dtype=torch.int64, device=device),
            finished=torch.zeros((), dtype=torch.uint8, device=device),
            kv=kv,
            arena=arena,
            cos=cos,
            sin=sin,
            embed=embed,
            eos_id=torch.tensor(spec.eos_id, dtype=torch.int64, device=device),
        )

    @property
    def device(self) -> torch.device:
        return self.token.device

    def reset(self) -> None:
        """New request. Dirty KV is legal; ``valid_len`` hides it."""
        self.token.zero_()
        self.position.zero_()
        self.valid_len.zero_()
        self.next_token.zero_()
        self.finished.zero_()

    def load_token(self, token_id: int) -> None:
        self.token.fill_(int(token_id))

    def commit_step(self) -> None:
        """After greedy: feed next id back without a CPU round-trip."""
        self.token.copy_(self.next_token)
        self.position.add_(1)
        hit = (self.next_token == self.eos_id).to(dtype=self.finished.dtype)
        self.finished.bitwise_or_(hit)

    def rope_at_position(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.position.reshape(1)
        return self.cos.index_select(0, idx), self.sin.index_select(0, idx)

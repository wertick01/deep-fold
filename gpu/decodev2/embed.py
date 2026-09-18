"""NF4 embed rows for Decode V2. No dense ``[vocab, hidden]`` table."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .plan import ArchSpec

__all__ = ["PackedEmbed", "bind_embed", "gather_embed"]


@dataclass
class PackedEmbed:
    """Packed NF4 rows + a persistent LUT. Gather never densifies the table."""

    packed: torch.Tensor
    scale: torch.Tensor
    vocab: int
    hidden: int
    lut: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.packed.device

    @property
    def shape(self) -> tuple[int, int]:
        return (self.vocab, self.hidden)

    @property
    def nbytes(self) -> int:
        return (
            int(self.packed.numel()) * int(self.packed.element_size())
            + int(self.scale.numel()) * int(self.scale.element_size())
            + int(self.lut.numel()) * int(self.lut.element_size())
        )

    def gather(self, ids: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        from gpu.host.embedding import dequant_nf4_rows

        return dequant_nf4_rows(
            self.packed,
            self.scale,
            ids,
            self.hidden,
            lut=self.lut,
            dtype=dtype,
        )


def bind_embed(mod, spec: ArchSpec, dtype: torch.dtype) -> torch.Tensor | PackedEmbed:
    """Alias packed embed rows, or a dense weight if the module is already dense.

    Does not call ``dequant_table``. 3B packed embed is ~156 MiB; the dense
    BF16 table was ~594 MiB.
    """
    packed = getattr(mod, "packed", None)
    scale = getattr(mod, "scale", None)
    if packed is not None and int(packed.numel()) > 0 and scale is not None:
        from gpu.host.embedding import NF4_LEVELS

        m = int(packed.shape[0])
        k = int(getattr(mod, "embedding_dim", spec.hidden))
        if (m, k) != (spec.vocab, spec.hidden):
            raise ValueError(f"embed [{m},{k}] != spec [{spec.vocab},{spec.hidden}]")
        return PackedEmbed(
            packed=packed,
            scale=scale,
            vocab=spec.vocab,
            hidden=spec.hidden,
            lut=torch.tensor(NF4_LEVELS, dtype=torch.float32, device=packed.device),
        )
    weight = getattr(mod, "weight", None)
    if weight is None or int(weight.numel()) == 0:
        raise RuntimeError("embed has neither NF4 packed rows nor a dense weight")
    table = weight.to(dtype=dtype)
    if tuple(table.shape) != (spec.vocab, spec.hidden):
        raise ValueError(f"embed weight {tuple(table.shape)} != {(spec.vocab, spec.hidden)}")
    return table


def gather_embed(
    embed: torch.Tensor | PackedEmbed,
    ids: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``[n, hidden]`` rows. Packed path decodes only those rows."""
    if isinstance(embed, PackedEmbed):
        return embed.gather(ids, dtype=dtype)
    flat = ids.reshape(-1).to(device=embed.device, dtype=torch.long)
    return embed.index_select(0, flat).to(dtype=dtype)

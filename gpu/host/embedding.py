"""``embed_tokens`` when the file says ``codec=nf4``.

``qwen25-3b.nf4.chr`` has one codec per file (chr0.md §2.3), so the embedding
table is NF4 too. An embedding is a lookup, not a GEMM, and wave 2 explicitly
has no kernel for it (wave2-gpu.md, non-goals), so the rows are decoded in
plain PyTorch here.

Two modes, both without a BF16 ``[vocab, hidden]`` weight inside a Linear:

* :class:`Nf4Embedding` -- decode only the gathered rows, per step. A token
  costs one ``[1, 1024]`` byte row; VRAM stays at the size of the packed table
  (155.6 MiB for 3B), which is what keeps a full generate inside the ``.chr``
  budget.
* :func:`dequant_table` -- one BF16 ``[vocab, hidden]`` table (594 MiB for 3B)
  for a stock ``nn.Embedding``. Allowed for the embedding only; a Linear must
  never be dequantized this way.

The LUT is the 16 literals of docs/spec/nf4.md §1 and the nibble order is
§5.1 (low nibble = even column) -- the same bits the CPU oracle and the kernel
use, asserted at import.
"""

from __future__ import annotations

import struct

import torch
import torch.nn as nn

from ._deps import ChrMatrix

__all__ = ["NF4_LEVELS", "Nf4Embedding", "dequant_nf4_rows", "dequant_table"]

GROUP_SIZE = 64

# docs/spec/nf4.md §1, literal for literal.
NF4_LEVELS: tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

# The binary32 bit patterns of the same table. A typo in the literals above
# would shift a whole level and go unnoticed in an eyeball diff.
NF4_BITS: tuple[int, ...] = (
    0xBF800000, 0xBF3239B1, 0xBF066B30, 0xBECA32A0,
    0xBE91A24D, 0xBE3D353F, 0xBDBA7871, 0x00000000,
    0x3DA2FAFF, 0x3E24CAE3, 0x3E7C04DD, 0x3EAD033A,
    0x3EE1A4B8, 0x3F1007AB, 0x3F3913B3, 0x3F800000,
)
for _i, (_v, _b) in enumerate(zip(NF4_LEVELS, NF4_BITS)):
    if struct.unpack("<I", struct.pack("<f", _v))[0] != _b:
        raise AssertionError(f"NF4 LUT level {_i} is not 0x{_b:08X}")
del _i, _v, _b


def _lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(NF4_LEVELS, dtype=torch.float32, device=device)


def dequant_nf4_rows(
    packed: torch.Tensor,
    scale: torch.Tensor,
    ids: torch.Tensor,
    K: int,  # noqa: N803
    *,
    lut: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode rows ``ids`` of an NF4 matrix to ``[len(ids), K]``.

    ``decode = float32(LUT[nib]) * float32(scale)`` (nf4.md §3), then one cast
    to ``dtype``. Only the requested rows are touched, so this never
    materializes the table.
    """
    if packed.dim() != 2 or packed.dtype is not torch.uint8:
        raise TypeError(f"packed must be uint8 [M, K_pad/2], got {packed.dtype} {tuple(packed.shape)}")
    if scale.dtype is not torch.float16:
        raise TypeError(f"scale must be float16, got {scale.dtype}")
    k_padded = packed.shape[1] * 2
    n_groups = k_padded // GROUP_SIZE
    if scale.shape[1] != n_groups:
        raise ValueError(f"scale has {scale.shape[1]} groups, packed implies {n_groups}")

    flat = ids.reshape(-1).to(packed.device, torch.long)
    rows = packed.index_select(0, flat)                          # [n, K_pad/2] uint8
    nib = torch.stack((rows & 0x0F, rows >> 4), dim=-1)          # low nibble = even column
    nib = nib.reshape(flat.numel(), k_padded).long()

    table = _lut(packed.device) if lut is None else lut
    w = table[nib]                                               # float32 [n, K_pad]
    s = scale.index_select(0, flat).to(torch.float32)             # exact fp16 -> f32
    w = w.view(flat.numel(), n_groups, GROUP_SIZE) * s[:, :, None]
    return w.view(flat.numel(), k_padded)[:, :K].to(dtype)


def dequant_table(
    matrix: ChrMatrix,
    *,
    dtype: torch.dtype = torch.bfloat16,
    chunk: int = 8192,
) -> torch.Tensor:
    """The whole matrix as one ``[M, K]`` tensor of ``dtype``.

    Only ever called for ``embed_tokens`` (594 MiB on 3B). Decoding happens in
    ``chunk``-row slices so the float32 scratch stays a few MiB instead of
    2.4 GiB.
    """
    out = torch.empty((matrix.M, matrix.K), dtype=dtype, device=matrix.packed.device)
    lut = _lut(matrix.packed.device)
    for lo in range(0, matrix.M, chunk):
        hi = min(lo + chunk, matrix.M)
        ids = torch.arange(lo, hi, device=matrix.packed.device)
        out[lo:hi] = dequant_nf4_rows(
            matrix.packed, matrix.scale, ids, matrix.K, lut=lut, dtype=dtype
        )
    return out


class Nf4Embedding(nn.Module):
    """``nn.Embedding`` that decodes NF4 rows on the fly.

    Holds the packed table (shared with a tied ``lm_head`` when there is one)
    and no BF16 copy of it.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        padding_idx: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.padding_idx = padding_idx
        self.compute_dtype = dtype
        dev = torch.device(device) if device is not None else torch.device("cpu")
        self.register_buffer("packed", torch.empty(0, dtype=torch.uint8, device=dev), persistent=False)
        self.register_buffer("scale", torch.empty(0, dtype=torch.float16, device=dev), persistent=False)
        self.chr_name: str | None = None

    @property
    def weight(self) -> torch.Tensor:
        """0-numel stand-in. The table is never materialized (see module doc)."""
        return torch.empty(0, dtype=self.compute_dtype, device=self.packed.device)

    @property
    def is_loaded(self) -> bool:
        return self.packed.numel() > 0

    @property
    def nbytes(self) -> int:
        return self.packed.numel() + self.scale.numel() * 2

    def attach(self, matrix: ChrMatrix) -> None:
        if (int(matrix.M), int(matrix.K)) != (self.num_embeddings, self.embedding_dim):
            raise ValueError(
                f"{matrix.name}: matrix is [{matrix.M},{matrix.K}], embedding is "
                f"[{self.num_embeddings},{self.embedding_dim}]"
            )
        self.packed = matrix.packed
        self.scale = matrix.scale
        self.chr_name = matrix.name

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.is_loaded:
            raise RuntimeError("Nf4Embedding has no weights: call load_chr_nf4 first")
        rows = dequant_nf4_rows(
            self.packed, self.scale, input_ids, self.embedding_dim, dtype=self.compute_dtype
        )
        return rows.view(*input_ids.shape, self.embedding_dim)

    def extra_repr(self) -> str:
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, codec=nf4-g{GROUP_SIZE}, "
            f"loaded={self.is_loaded}"
        )

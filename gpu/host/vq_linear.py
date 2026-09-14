"""``CompressedVqLinear``: an ``nn.Linear`` seat for the VQ 2x8 kernel.

The NF4 twin of this module is :mod:`gpu.host.linear`, and the contract is the
same one stitch-gpu.md writes down: the module holds the file's bytes
(``index``, ``book``) and nothing else. No ``[out, in]`` parameter is ever
registered, so ``model.to(...)`` cannot resurrect a BF16 weight and nothing
dequantizes ``W`` into HBM -- 2 bits per weight plus an 8 KiB book is all that
is resident.

Transposition lives here, as gpu-abi.md §6 requires: HuggingFace hands us
``[..., K]``, the kernel wants ``[K, N]`` and answers ``[M, N]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ._deps import Header  # importing it also puts the repo root on sys.path
from .vq_blobs import (
    CODEBOOK_SIZE,
    N_CODEBOOKS,
    VQ_GROUP_SIZE,
    VqMatrix,
    dequant_vq_rows,
    k_pad_vq,
    materialize_vq,
)

__all__ = ["CompressedVqLinear", "VqEmbedding", "vq_linear", "load_chr_vq", "k_pad_vq"]

_gemm = None


def _vq_gemm(index, book, x, M, K, K_pad):  # noqa: N803
    """Lazy handle on the VQ extension: importing it may trigger a JIT build.

    Resolved once and cached -- this is on the token path, and re-entering the
    import machinery per matrix per token buys nothing.
    """
    global _gemm
    if _gemm is None:
        from gpu.vq import vq_gemm as fn

        _gemm = fn
    return _gemm(index, book, x, M, K, K_pad)


def vq_linear(
    x: torch.Tensor,
    index: torch.Tensor,
    book: torch.Tensor,
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    K_pad: int,  # noqa: N803
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``y = (C1[i1] + C2[i2]) @ x`` for activations shaped ``[..., K]``.

    Thin wrapper over ``gpu/vq``: it only permutes and adds the bias. ``N``
    (the kernel's sequence length) must be 1 -- the wave-3 VQ kernel is
    decode-only, and pretending otherwise would silently compute one column.
    """
    if x.shape[-1] != K:
        raise ValueError(f"x has {x.shape[-1]} features, this linear takes K={K}")
    lead = tuple(x.shape[:-1])
    n = 1
    for d in lead:
        n *= int(d)
    if n != 1:
        raise NotImplementedError(
            f"CompressedVqLinear: N={n} (x{tuple(x.shape)}). The wave-3 VQ kernel is "
            "decode-only (N==1, docs/spec/stitch-gpu.md); there is no prefill launch. "
            "Feed one token at a time."
        )
    if x.dtype is not torch.bfloat16:
        x = x.to(torch.bfloat16)

    # [..., K] with prod(lead) == 1 -> [K, 1]: same bytes, the kernel's layout.
    xk = x.reshape(K, 1)
    y = _vq_gemm(index, book, xk, M, K, K_pad)  # bf16 [M, 1]
    out = y.view(M)
    if bias is not None:
        out = out + bias
    return out.view(*lead, M)


class CompressedVqLinear(nn.Module):
    """Drop-in for ``nn.Linear`` whose weight lives as CHR0 VQ 2x8 bytes.

    Buffers are 0-numel until :meth:`attach` gets a :class:`VqMatrix`, so the
    replacement pass costs nothing and a skeleton can be built on ``meta``
    first. ``weight`` is a property returning an empty tensor: utilities that
    poke ``module.weight.dtype`` keep working without ``M*K`` ever existing.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        if self.in_features < 1 or self.out_features < 1:
            raise ValueError(f"bad shape: in={in_features} out={out_features}")
        self.K = self.in_features
        self.M = self.out_features
        self.K_pad = k_pad_vq(self.K)
        self.G = self.K_pad // VQ_GROUP_SIZE
        self.compute_dtype = dtype

        dev = torch.device(device) if device is not None else torch.device("cpu")
        # persistent=False: these are file bytes, not a checkpoint of ours.
        self.register_buffer("index", torch.empty(0, dtype=torch.uint8, device=dev), persistent=False)
        self.register_buffer("book", torch.empty(0, dtype=torch.float16, device=dev), persistent=False)
        if bias:
            self.register_buffer("bias", torch.empty(0, dtype=dtype, device=dev), persistent=False)
        else:
            self.register_buffer("bias", None)
        self.chr_name: str | None = None

    # --- the weight that is not there ------------------------------------
    @property
    def weight(self) -> torch.Tensor:
        """A 0-numel stand-in with the right dtype/device. Never ``[M, K]``.

        Assigning to it raises (``property`` has no setter) -- that is the
        point: a stray ``module.weight = ...`` from a generic utility must fail
        loudly instead of allocating ``out x in`` BF16.
        """
        return torch.empty(0, dtype=self.compute_dtype, device=self.index.device)

    @property
    def is_loaded(self) -> bool:
        return self.index.numel() > 0

    @property
    def nbytes(self) -> int:
        """Device bytes held by this layer's weight."""
        return self.index.numel() + self.book.numel() * 2

    # --- loading ----------------------------------------------------------
    def attach(self, matrix: VqMatrix) -> None:
        """Point this layer at an already materialized CHR0 VQ matrix.

        The tensors are shared, not copied: two tied layers can hold the same
        :class:`VqMatrix` without a second copy in VRAM. The book is per matrix
        (vq.md: a book per matrix, not per layer), so it travels with the index
        and is never pooled across layers.
        """
        if (int(matrix.M), int(matrix.K)) != (self.M, self.K):
            raise ValueError(
                f"{matrix.name}: matrix is [{matrix.M},{matrix.K}], "
                f"this linear is [{self.M},{self.K}]"
            )
        if int(matrix.K_pad) != self.K_pad:
            raise ValueError(f"{matrix.name}: K_pad {matrix.K_pad} != {self.K_pad}")
        if matrix.index.dtype is not torch.uint8 or matrix.book.dtype is not torch.float16:
            raise TypeError(
                f"{matrix.name}: expected uint8 index / float16 book, "
                f"got {matrix.index.dtype} / {matrix.book.dtype}"
            )
        want = (N_CODEBOOKS, CODEBOOK_SIZE, VQ_GROUP_SIZE)
        if tuple(matrix.book.shape) != want:
            raise ValueError(f"{matrix.name}: book {tuple(matrix.book.shape)} != {want}")
        if matrix.index.numel() != self.M * self.G * N_CODEBOOKS:
            raise ValueError(
                f"{matrix.name}: index has {matrix.index.numel()} bytes, "
                f"M*G*2 = {self.M * self.G * N_CODEBOOKS}"
            )
        self.index = matrix.index
        self.book = matrix.book
        self.chr_name = matrix.name

    def set_bias(self, bias: torch.Tensor) -> None:
        if self.bias is None:
            raise ValueError(f"{self.chr_name or self}: built with bias=False")
        if bias.shape != (self.M,):
            raise ValueError(f"bias {tuple(bias.shape)} != {(self.M,)}")
        self.bias = bias.to(self.compute_dtype)

    # --- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_loaded:
            raise RuntimeError(
                f"CompressedVqLinear[{self.M},{self.K}] "
                f"({self.chr_name or 'unnamed'}) has no weights: call load_chr_vq first"
            )
        return vq_linear(x, self.index, self.book, self.M, self.K, self.K_pad, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, codec=vq2x8-g{VQ_GROUP_SIZE}, "
            f"K_pad={self.K_pad}, loaded={self.is_loaded}"
        )


class VqEmbedding(nn.Module):
    """``nn.Embedding`` that reconstructs VQ rows on the fly. No dense table."""

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
        self.K_pad = k_pad_vq(self.embedding_dim)
        dev = torch.device(device) if device is not None else torch.device("cpu")
        self.register_buffer("index", torch.empty(0, dtype=torch.uint8, device=dev), persistent=False)
        self.register_buffer("book", torch.empty(0, dtype=torch.float16, device=dev), persistent=False)
        self.chr_name: str | None = None

    @property
    def weight(self) -> torch.Tensor:
        return torch.empty(0, dtype=self.compute_dtype, device=self.index.device)

    @property
    def is_loaded(self) -> bool:
        return self.index.numel() > 0

    @property
    def nbytes(self) -> int:
        return self.index.numel() + self.book.numel() * 2

    def attach(self, matrix: VqMatrix) -> None:
        if (int(matrix.M), int(matrix.K)) != (self.num_embeddings, self.embedding_dim):
            raise ValueError(
                f"{matrix.name}: matrix is [{matrix.M},{matrix.K}], embedding is "
                f"[{self.num_embeddings},{self.embedding_dim}]"
            )
        self.index = matrix.index
        self.book = matrix.book
        self.chr_name = matrix.name

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.is_loaded:
            raise RuntimeError("VqEmbedding has no weights: call load_chr_vq first")
        rows = dequant_vq_rows(
            self.index, self.book, input_ids, self.embedding_dim, dtype=self.compute_dtype
        )
        return rows.view(*input_ids.shape, self.embedding_dim)

    def extra_repr(self) -> str:
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, codec=vq2x8-g{VQ_GROUP_SIZE}, "
            f"loaded={self.is_loaded}"
        )


def load_chr_vq(
    path: str,
    name: str,
    device: str | torch.device = "cuda",
    *,
    header: Header | None = None,
    bias: torch.Tensor | None = None,
) -> CompressedVqLinear:
    """One CHR0 ``codec=vq`` matrix -> a ready :class:`CompressedVqLinear`.

    The single-matrix entry point wave3-vq.md asks for: shapes come from the
    header, so the caller does not have to know them, and the module is
    attached to the bytes that were just copied to ``device``.
    """
    matrix = materialize_vq(path, name, device, header=header)
    layer = CompressedVqLinear(
        in_features=matrix.K,
        out_features=matrix.M,
        bias=bias is not None,
        device=matrix.device,
    )
    layer.attach(matrix)
    if bias is not None:
        layer.set_bias(bias.to(matrix.device))
    return layer

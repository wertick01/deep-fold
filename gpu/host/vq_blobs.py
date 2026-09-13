"""VQ 2x8 blobs of a CHR0 file -> device tensors.

``gpu/chr0.materialize_nf4`` refuses ``codec=vq`` (gpu-abi.md §8) and is not
ours to extend, so the VQ half of the loader lives here, next to
:mod:`gpu.host.blobs`. Same discipline as the NF4 loader: the header says where
the bytes are, one positioned read per matrix, one host buffer that dies right
after the host-to-device copy, and views into a single device allocation.

Nothing is decoded on the way in. The reconstruct ``g = C1[i1] + C2[i2]``
(vq.md §7.2) happens in the kernel; the only CPU implementation here is
:func:`reconstruct_vq`, which exists to be the oracle the kernel is checked
against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

from ._deps import Chr0Error, Header, TensorInfo, TruncatedError, load_header

__all__ = [
    "VQ_GROUP_SIZE",
    "N_CODEBOOKS",
    "CODEBOOK_SIZE",
    "CODEBOOK_NBYTES",
    "VqCodecError",
    "VqMatrix",
    "k_pad_vq",
    "materialize_vq",
    "iter_vq",
    "reconstruct_vq",
]

VQ_GROUP_SIZE = 8
N_CODEBOOKS = 2
CODEBOOK_SIZE = 256
# [2, 256, 8] FP16 -- 8192 bytes whatever the matrix is (vq.md §7.1).
CODEBOOK_NBYTES = N_CODEBOOKS * CODEBOOK_SIZE * VQ_GROUP_SIZE * 2


class VqCodecError(Chr0Error):
    """Asked for the VQ blobs of a tensor whose codec is not ``vq``."""


def k_pad_vq(k: int) -> int:
    """``8 * ceil(K / 8)`` -- vq.md §1.1."""
    return VQ_GROUP_SIZE * ((int(k) + VQ_GROUP_SIZE - 1) // VQ_GROUP_SIZE)


@dataclass(frozen=True)
class VqMatrix:
    """One additive-VQ matrix resident on ``index.device``.

    The mirror of :class:`gpu.chr0.ChrMatrix` for ``codec=vq``, laid out like
    ``chr_vq_dev_t`` in ``gpu/include/chr_gpu.h``: the kernel takes exactly
    ``index``, ``book``, ``M``, ``K``, ``K_pad``.

    ``index`` and ``book`` are views into one allocation, so their
    ``data_ptr()`` stay valid for as long as this object is alive.
    """

    name: str
    M: int
    K: int
    K_pad: int
    index: torch.Tensor  # uint8,   [M, K_pad // 8, 2]
    book: torch.Tensor  # float16, [2, 256, 8]

    @property
    def G(self) -> int:
        """Groups of 8 along ``n_in`` -- ``K_pad // 8``."""
        return self.K_pad // VQ_GROUP_SIZE

    @property
    def device(self) -> torch.device:
        return self.index.device

    @property
    def nbytes(self) -> int:
        """Device bytes owned by this matrix (2 bits/weight plus 8 KiB)."""
        return self.index.numel() + self.book.numel() * 2


def _info(header: Header, name: str) -> TensorInfo:
    info = header.tensor(name)
    if info.codec != "vq":
        raise VqCodecError(f"{name}: codec {info.codec!r}, materialize_vq handles vq only")
    # load_header already enforces these three; saying so at the codec boundary
    # too keeps a hand-written header from reaching the kernel through this path.
    if info.group_size != VQ_GROUP_SIZE:
        raise VqCodecError(f"{name}: group_size {info.group_size}, must be {VQ_GROUP_SIZE}")
    if info.n_codebooks != N_CODEBOOKS or info.codebook_bits != 8:
        raise VqCodecError(
            f"{name}: n_codebooks={info.n_codebooks} codebook_bits={info.codebook_bits}; "
            "this host reads 2x256x8 books only"
        )
    return info


def _read_range(path: str, start: int, nbytes: int, what: str) -> bytearray:
    """One positioned read into a fresh host buffer. The file is opened read-only."""
    buf = bytearray(nbytes)
    with open(path, "rb") as f:
        f.seek(start)
        got = f.readinto(memoryview(buf))
    if got != nbytes:
        raise TruncatedError(f"truncated: {what} wanted {nbytes} bytes at {start}, got {got}")
    return buf


def _fill(arena: torch.Tensor, offset: int, buf) -> None:
    host = torch.frombuffer(buf, dtype=torch.uint8)
    arena[offset : offset + host.numel()].copy_(host)


def materialize_vq(
    path: str,
    name: str,
    device: str | torch.device = "cuda",
    *,
    header: Header | None = None,
) -> VqMatrix:
    """Load the ``codebook`` and ``index`` blobs of ``name`` onto ``device``.

    ``header=`` reuses an already parsed header, exactly like
    ``materialize_nf4``: 300+ matrices must not reparse the JSON each.

    No decoding: ``index`` is the file's byte range verbatim and ``book`` is its
    FP16 bytes reinterpreted, little-endian on both sides (gpu-abi.md §3).
    """
    path = str(path)
    hdr = header if header is not None else load_header(path)
    info = _info(hdr, name)

    M, K, K_pad = info.M, info.K, info.K_pad
    book_blob = info.blobs["codebook"]
    index_blob = info.blobs["index"]

    # Read host-side first: nothing touches the device until the bytes are in
    # hand, so a short file fails without having allocated VRAM. The writer puts
    # the two blobs back to back (either order), so the common case is one pread.
    lo = min(book_blob.start, index_blob.start)
    hi = max(book_blob.end, index_blob.end)
    if hi - lo == book_blob.nbytes + index_blob.nbytes:
        whole = memoryview(_read_range(path, lo, hi - lo, name))
        book_buf = whole[book_blob.start - lo : book_blob.end - lo]
        index_buf = whole[index_blob.start - lo : index_blob.end - lo]
    else:
        book_buf = _read_range(path, book_blob.start, book_blob.nbytes, f"{name}.codebook")
        index_buf = _read_range(path, index_blob.start, index_blob.nbytes, f"{name}.index")

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")

    # Book first: its 8192 bytes keep the uint8 -> float16 reinterpret aligned
    # and leave the index tail 64-byte aligned for the kernel's vector loads.
    arena = torch.empty(CODEBOOK_NBYTES + index_blob.nbytes, dtype=torch.uint8, device=dev)
    _fill(arena, 0, book_buf)
    _fill(arena, CODEBOOK_NBYTES, index_buf)
    del book_buf, index_buf  # the bytes live in VRAM now; drop the host copy

    book = arena[:CODEBOOK_NBYTES].view(torch.float16)
    book = book.view(N_CODEBOOKS, CODEBOOK_SIZE, VQ_GROUP_SIZE)
    index = arena[CODEBOOK_NBYTES:].view(M, K_pad // VQ_GROUP_SIZE, N_CODEBOOKS)

    return VqMatrix(name=name, M=M, K=K, K_pad=K_pad, index=index, book=book)


def iter_vq(header: Header) -> Iterator[str]:
    """CHR0 names whose ``codec`` is ``vq``, in header key order.

    The VQ mirror of ``chr0.iter_linears``. ``embed_tokens`` and ``lm_head`` are
    in here too when the file quantized them; which of those actually go through
    the GEMM is the host's business, not the file's.
    """
    for name, info in header.tensors.items():
        if info.codec == "vq":
            yield name


def reconstruct_vq(
    index: torch.Tensor,
    book: torch.Tensor,
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    K_pad: int,  # noqa: N803
) -> torch.Tensor:
    """``W_hat[r, 8j+d] = f32(book[0, i1, d]) + f32(book[1, i2, d])`` -> ``[M, K]``.

    vq.md §9.2 verbatim: FP16 widened exactly, the add in float32, the
    ``K_pad - K`` padding columns dropped. This is the reference the kernel is
    checked against, so it materializes ``M x K`` float32 -- never call it on
    the token path.
    """
    G = K_pad // VQ_GROUP_SIZE
    idx = index.reshape(M, G, N_CODEBOOKS).long()
    b = book.reshape(N_CODEBOOKS, CODEBOOK_SIZE, VQ_GROUP_SIZE).to(torch.float32)
    g = b[0].index_select(0, idx[:, :, 0].reshape(-1))
    g = g + b[1].index_select(0, idx[:, :, 1].reshape(-1))
    return g.reshape(M, K_pad)[:, :K]

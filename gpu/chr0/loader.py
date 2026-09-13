"""Map an NF4 tensor of a CHR0 file onto device memory.

The bytes on disk are the bytes in VRAM: this module slices, it never
decodes. Nibble order, the NF4 LUT and dequantisation belong to the kernel
(``docs/spec/nf4.md`` §1, §3).

Copy strategy is fixed by ``docs/spec/gpu-abi.md`` §4: one ``pread`` of the
matrix's blobs into a host ``bytearray``, one host-to-device copy into one
device allocation, then ``packed`` / ``scale`` as views into it. Peak host RAM
is one matrix, so loading a whole model never holds a host copy and a device
copy of everything at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .errors import CodecError, GroupSizeError, TruncatedError
from .header import NF4_GROUP_SIZE, Blob, Header, load_header

__all__ = ["ChrMatrix", "materialize_nf4"]


@dataclass(frozen=True)
class ChrMatrix:
    """One NF4 matrix resident on ``packed.device``.

    Field names and order are frozen: they are the Python mirror of
    ``chr_nf4_dev_t`` in ``gpu/include/chr_gpu.h``.

    ``packed`` and ``scale`` are views into a single allocation, so their
    ``data_ptr()`` stay valid for as long as this object is alive.
    """

    name: str
    M: int
    K: int
    K_pad: int
    packed: torch.Tensor  # uint8,   [M, K_pad // 2]
    scale: torch.Tensor  # float16, [M, K_pad // 64]

    @property
    def n_groups(self) -> int:
        return self.K_pad // NF4_GROUP_SIZE

    @property
    def device(self) -> torch.device:
        return self.packed.device

    @property
    def nbytes(self) -> int:
        """Device bytes owned by this matrix."""
        return self.packed.numel() + self.scale.numel() * 2


def _read_range(path: str, start: int, nbytes: int, what: str) -> bytearray:
    """One positioned read into a fresh host buffer. The file is opened read-only."""
    buf = bytearray(nbytes)
    with open(path, "rb") as f:
        f.seek(start)
        got = f.readinto(memoryview(buf))
    if got != nbytes:
        raise TruncatedError(f"truncated: {what} wanted {nbytes} bytes at {start}, got {got}")
    return buf


def _fill(arena: torch.Tensor, offset: int, buf: bytearray) -> None:
    host = torch.frombuffer(buf, dtype=torch.uint8)
    arena[offset : offset + host.numel()].copy_(host)


def materialize_nf4(
    path: str,
    name: str,
    device: str | torch.device = "cuda",
    *,
    header: Header | None = None,
) -> ChrMatrix:
    """Load the NF4 blobs of ``name`` from ``path`` onto ``device``.

    ``header`` lets a caller that loads many matrices reuse one parsed header
    instead of re-reading the JSON per matrix; when omitted the header is read
    and fully validated first, so every malformed range is rejected before any
    device allocation happens.

    No dequantisation: ``packed`` is the file's byte range verbatim.
    """
    path = str(path)
    hdr = header if header is not None else load_header(path)
    info = hdr.tensor(name)

    if info.codec != "nf4":
        raise CodecError(
            f"{name}: codec {info.codec!r}; wave 2.0 materializes nf4 only"
        )
    if info.group_size != NF4_GROUP_SIZE:
        raise GroupSizeError(f"{name}: group_size {info.group_size}, must be {NF4_GROUP_SIZE}")

    M, K, K_pad = info.M, info.K, info.K_pad
    n_groups = info.n_groups
    data: Blob = info.blobs["data"]
    scale: Blob = info.blobs["scale"]

    # Read host-side first: nothing touches the device until the bytes are in
    # hand, so a short file fails without having allocated VRAM.
    # Fast path: the CHR0 writer emits `data` then `scale` back to back, so the
    # whole matrix is one pread. Only a pad gap between them costs a second.
    if scale.start == data.end:
        chunks = [(0, _read_range(path, data.start, data.nbytes + scale.nbytes, name))]
    else:
        chunks = [
            (0, _read_range(path, data.start, data.nbytes, f"{name}.data")),
            (data.nbytes, _read_range(path, scale.start, scale.nbytes, f"{name}.scale")),
        ]

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")

    arena = torch.empty(data.nbytes + scale.nbytes, dtype=torch.uint8, device=dev)
    for offset, buf in chunks:
        _fill(arena, offset, buf)
    del chunks, buf  # the bytes live in VRAM now; drop the host copy

    packed_t = arena[: data.nbytes].view(M, K_pad // 2)
    scale_t = arena[data.nbytes :].view(torch.float16).view(M, n_groups)

    return ChrMatrix(name=name, M=M, K=K, K_pad=K_pad, packed=packed_t, scale=scale_t)

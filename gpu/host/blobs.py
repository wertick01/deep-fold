"""BF16 blobs of a CHR0 file -> device tensors.

``gpu/chr0`` materializes ``nf4`` only (gpu-abi.md §8), and it is not ours to
extend. Norms, biases and any ``codec=bf16`` payload therefore get read here,
with the same discipline the NF4 loader uses: the header says where the bytes
are, the file is opened ``"rb"``, one positioned read per tensor, one host
buffer that dies right after the host-to-device copy.

No decoding happens here either -- BF16 on disk is BF16 in VRAM, little-endian
both sides (gpu-abi.md §3).
"""

from __future__ import annotations

from typing import Iterator

import torch

from ._deps import Chr0Error, Header, TensorInfo, TruncatedError, load_header

__all__ = ["load_bf16", "iter_bf16", "read_bf16_host"]


class Bf16CodecError(Chr0Error):
    """Asked for a BF16 blob of a tensor whose codec is not ``bf16``."""


def _info(header: Header, name: str) -> TensorInfo:
    info = header.tensor(name)
    if info.codec != "bf16":
        raise Bf16CodecError(f"{name}: codec {info.codec!r}, load_bf16 handles bf16 only")
    return info


def read_bf16_host(path: str, name: str, *, header: Header | None = None) -> torch.Tensor:
    """The tensor's bytes as a host BF16 tensor of its declared shape.

    One ``pread`` into one buffer. The returned tensor aliases that buffer and
    keeps it alive; callers that want it on a device should use
    :func:`load_bf16` instead, which drops the host copy immediately.
    """
    path = str(path)
    hdr = header if header is not None else load_header(path)
    info = _info(hdr, name)
    blob = info.blobs["data"]

    buf = bytearray(blob.nbytes)
    with open(path, "rb") as f:
        f.seek(blob.start)
        got = f.readinto(memoryview(buf))
    if got != blob.nbytes:
        raise TruncatedError(
            f"truncated: {name}.data wanted {blob.nbytes} bytes at {blob.start}, got {got}"
        )

    host = torch.frombuffer(buf, dtype=torch.uint8).view(torch.bfloat16)
    return host.view(info.shape)


def load_bf16(
    path: str,
    name: str,
    device: str | torch.device = "cuda",
    *,
    header: Header | None = None,
) -> torch.Tensor:
    """Load one ``codec=bf16`` tensor of ``path`` onto ``device``.

    ``header=`` reuses an already parsed header, exactly like
    ``materialize_nf4``: parsing 65 KiB of JSON once per norm is the only thing
    this parameter is about.
    """
    host = read_bf16_host(path, name, header=header)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
    out = torch.empty(host.shape, dtype=torch.bfloat16, device=dev)
    out.copy_(host)
    del host  # the bytes live on the device now
    return out


def iter_bf16(header: Header) -> Iterator[str]:
    """CHR0 names whose ``codec`` is ``bf16``, in header key order.

    The mirror of ``chr0.iter_linears``: norms, biases and anything else the
    compressor left uncompressed. Which module each name belongs to is the
    host's business, not the file's.
    """
    for name, info in header.tensors.items():
        if info.codec == "bf16":
            yield name

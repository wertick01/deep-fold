"""Test fixtures: write a tiny safetensors, and craft malformed CHR0 files.

Only used by ``test_chr0.py``. Nothing here is part of the loader ABI.
"""

from __future__ import annotations

import json
import struct
from typing import Any, Mapping

import numpy as np

from .header import align64

__all__ = ["write_safetensors", "build_chr", "nf4_blob_sizes"]


def write_safetensors(path, tensors: Mapping[str, np.ndarray]) -> None:
    """Write ``tensors`` (float32 arrays) as an F32 safetensors file."""
    header: dict[str, Any] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        blobs.append(raw)
    js = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(js)))
        f.write(js)
        for raw in blobs:
            f.write(raw)


def nf4_blob_sizes(m: int, k: int) -> tuple[int, int]:
    """``(data_nbytes, scale_nbytes)`` for a logical ``[m, k]`` NF4 matrix."""
    k_pad = 64 * ((k + 63) // 64)
    return m * k_pad // 2, m * (k_pad // 64) * 2


def build_chr(path, tensors: Mapping[str, Any], /, *, file_size: int | None = None, **root_overrides) -> int:
    """Write a CHR0 file with hand-written offsets and zeroed payload.

    Used to craft the malformed cases that the Go writer cannot produce.
    Returns the number of bytes written. ``file_size`` truncates or extends the
    payload region, which is how the "``end`` past EOF" case is built.
    """
    root: dict[str, Any] = {
        "magic": "CHR0",
        "version": 1,
        "arch": "toy",
        "hidden_size": 128,
        "intermediate_size": 128,
        "num_layers": 1,
        "vocab_size": 0,
        "tile": {"row": 64, "col_group": 8},
        "tensors": dict(tensors),
    }
    root.update(root_overrides)

    js = json.dumps(root, separators=(",", ":")).encode("utf-8")
    body = bytearray(struct.pack("<Q", len(js)) + js)
    body += b"\x00" * (align64(len(body)) - len(body))

    max_end = 0
    for entry in root["tensors"].values():
        for key in ("data", "scale", "zero", "codebook", "index"):
            if key in entry:
                max_end = max(max_end, int(entry[key][1]))
    size = max(align64(max_end), len(body)) if file_size is None else file_size
    if size > len(body):
        body += b"\x00" * (size - len(body))
    else:
        del body[size:]

    with open(path, "wb") as f:
        f.write(body)
    return len(body)

"""Read-only CHR0 loader: ``.chr`` on disk -> NF4 tensors on a CUDA device.

Wave 2, agent 1. Owns the file format side of the GPU path and nothing else:
no kernels, no dequantisation, no HuggingFace modules. The contract is
``docs/spec/gpu-abi.md``; the C mirror is ``gpu/include/chr_gpu.h``.

    from chr0 import load_header, iter_linears, materialize_nf4

    hdr = load_header(r"C:\\dev\\models\\qwen25-3b.nf4.chr")
    w = materialize_nf4(hdr.path, "model.layers.0.mlp.gate_proj", header=hdr)
    w.packed.shape, w.scale.shape   # (11008, 1024), (11008, 32)

``gpu`` has no ``__init__.py`` on purpose -- each agent owns a subdirectory and
nobody owns the parent. Import either as a namespace package (``import
gpu.chr0`` with the repo root on ``sys.path``) or by putting ``gpu/`` itself on
``sys.path`` and doing ``import chr0``.
"""

from __future__ import annotations

from .errors import (
    AlignmentError,
    Chr0Error,
    CodecError,
    GroupSizeError,
    HeaderError,
    OverlapError,
    SizeMismatchError,
    TensorNotFoundError,
    TruncatedError,
)
from .header import (
    NF4_GROUP_SIZE,
    Blob,
    Header,
    TensorInfo,
    align64,
    iter_linears,
    load_header,
    quantized_codec,
)

__all__ = [
    "load_header",
    "iter_linears",
    "quantized_codec",
    "materialize_nf4",
    "ChrMatrix",
    "Header",
    "TensorInfo",
    "Blob",
    "align64",
    "NF4_GROUP_SIZE",
    "Chr0Error",
    "TruncatedError",
    "HeaderError",
    "AlignmentError",
    "SizeMismatchError",
    "OverlapError",
    "CodecError",
    "GroupSizeError",
    "TensorNotFoundError",
]


def __getattr__(name: str):
    # torch is only needed to materialize; parsing a header must not pay for it.
    if name in ("materialize_nf4", "ChrMatrix"):
        from . import loader

        return getattr(loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

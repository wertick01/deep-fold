"""PyTorch host for CHR0 NF4 weights: our algorithm drives ``nn.Linear``.

Wave 2, agent 3. Owns ``gpu/host/`` and nothing else: the file format is
``gpu/chr0`` (agent 1), the GEMM is ``gpu/nf4`` (agent 2). This package is the
seat they plug into -- Python owns every long-lived tensor, the extension only
computes ``y = dequant_nf4(packed, scale) @ x`` for ``N == 1``.

    import sys; sys.path.insert(0, r"C:\\dev\\deep-fold")
    from gpu.host import build_skeleton, replace_linears, load_chr_nf4

    model = build_skeleton(r"C:\\dev\\models\\Qwen2.5-3B-Instruct")  # meta, config.json only
    replace_linears(model)                                          # no [out, in] weights left
    report = load_chr_nf4(model, r"C:\\dev\\models\\qwen25-3b.nf4.chr")

Or in one call: ``model, report = load_model(model_id, chr_path)``.

Wave 3 adds the ``codec=vq`` seat next to it -- :class:`CompressedVqLinear`,
``materialize_vq`` and ``load_chr_vq`` (``vq_linear.py`` / ``vq_blobs.py``),
driving ``gpu/vq`` the same way. Same rule there: no ``[out, in]`` weight, ever.

    from gpu.host import load_chr_vq
    layer = load_chr_vq(r"C:\\dev\\models\\qwen25-3b.vq2.chr", "model.layers.0.mlp.gate_proj")

Contracts: ``docs/spec/stitch-gpu.md`` (wins ties), ``docs/spec/gpu-abi.md``,
``docs/spec/vq.md``, ``docs/token-loop.md`` §1 and §3. Acceptance:
``python gpu/host/verify.py`` (NF4) and ``python gpu/host/verify_vq.py`` (VQ).
"""

from __future__ import annotations

from .attach import (
    AttachError,
    DriverPlan,
    LayerPlan,
    attach,
    attach_dir,
    attach_module,
    plan_violations,
)
from .blobs import iter_bf16, load_bf16, read_bf16_host
from .embedding import NF4_LEVELS, Nf4Embedding, dequant_nf4_rows, dequant_table
from .linear import CompressedLinear, k_pad, nf4_linear
from .model import (
    LINEAR_KINDS,
    LoadReport,
    build_skeleton,
    linear_modules,
    load_chr_nf4,
    load_model,
    replace_linears,
)
from .vq_blobs import VqMatrix, iter_vq, k_pad_vq, materialize_vq, reconstruct_vq
from .vq_linear import CompressedVqLinear, load_chr_vq, vq_linear

__all__ = [
    # wave10 P1: one attach path (docs/tz/wave8-arch.md §2)
    "attach",
    "attach_dir",
    "attach_module",
    "plan_violations",
    "AttachError",
    "DriverPlan",
    "LayerPlan",
    "CompressedLinear",
    "Nf4Embedding",
    "nf4_linear",
    "k_pad",
    "build_skeleton",
    "replace_linears",
    "load_chr_nf4",
    "load_model",
    "linear_modules",
    "LoadReport",
    "LINEAR_KINDS",
    "load_bf16",
    "read_bf16_host",
    "iter_bf16",
    "dequant_nf4_rows",
    "dequant_table",
    "NF4_LEVELS",
    # wave 3, codec=vq (docs/spec/vq.md)
    "CompressedVqLinear",
    "vq_linear",
    "load_chr_vq",
    "materialize_vq",
    "reconstruct_vq",
    "iter_vq",
    "VqMatrix",
    "k_pad_vq",
]

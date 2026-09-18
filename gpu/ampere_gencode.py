"""NVCC fatbinary for the NF4/VQ GEMM. No torch.

Measured generate numbers remain RTX 3080 (sm_86). Ampere-family cards
(sm_80 / sm_87 / sm_89) load the same .cu via SASS or PTX. SM120 (GeForce
RTX 50, including the remote RTX 5070 Ti) is the same MMA model: native
``sm_120`` cubin when nvcc is 12.8+, otherwise PTX ``compute_80`` JIT.
Hopper and SM100 (datacenter Blackwell) are not in this list.
"""

from __future__ import annotations

from gpu.arch_family import (  # noqa: F401 - re-export for existing imports
    AMPERE_CAPABILITIES,
    GENERATE_CAPABILITIES,
    MEASURED_CAPABILITY,
    SM120_CAPABILITIES,
)
from gpu.cuda_env import nvcc_supports_sm120

__all__ = [
    "AMPERE_CAPABILITIES",
    "FAMILY_CAPABILITIES",
    "GENERATE_CAPABILITIES",
    "KERNEL_GENCODE",
    "MEASURED_CAPABILITY",
    "NVCC_GENCODE_FLAGS",
    "SM120_CAPABILITIES",
    "kernel_gencode",
    "nvcc_cflags",
    "nvcc_gencode_flags",
]

FAMILY_CAPABILITIES = AMPERE_CAPABILITIES
NVCC_GENCODE_FLAGS = (
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_80,code=compute_80",
)
_SM120_GENCODE = ("-gencode=arch=compute_120,code=sm_120",)
KERNEL_GENCODE = "sm_80/sm_86/sm_89 + PTX compute_80"


def nvcc_gencode_flags() -> tuple[str, ...]:
    """Ampere cubins + PTX, plus native SM120 when the toolkit can emit it."""
    if nvcc_supports_sm120():
        return NVCC_GENCODE_FLAGS + _SM120_GENCODE
    return NVCC_GENCODE_FLAGS


def kernel_gencode() -> str:
    if nvcc_supports_sm120():
        return "sm_80/sm_86/sm_89/sm_120 + PTX compute_80"
    return KERNEL_GENCODE


def nvcc_cflags() -> list[str]:
    """Flags for setup.py and torch JIT ``extra_cuda_cflags``."""
    return ["-O3", *nvcc_gencode_flags(), "--expt-relaxed-constexpr", "-lineinfo"]

"""NVCC fatbinary for the Ampere-family NF4/VQ GEMM. No torch.

Measured generate numbers remain RTX 3080 (sm_86). sm_80 / sm_87 / sm_89
load the same .cu via SASS or PTX. Hopper and Blackwell are not in this list.
"""

from __future__ import annotations

__all__ = [
    "FAMILY_CAPABILITIES",
    "KERNEL_GENCODE",
    "MEASURED_CAPABILITY",
    "NVCC_GENCODE_FLAGS",
    "nvcc_cflags",
]

MEASURED_CAPABILITY = (8, 6)
FAMILY_CAPABILITIES = frozenset({(8, 0), (8, 6), (8, 7), (8, 9)})
NVCC_GENCODE_FLAGS = (
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_80,code=compute_80",
)
KERNEL_GENCODE = "sm_80/sm_86/sm_89 + PTX compute_80"


def nvcc_cflags() -> list[str]:
    """Flags for setup.py and torch JIT ``extra_cuda_cflags``."""
    return ["-O3", *NVCC_GENCODE_FLAGS, "--expt-relaxed-constexpr", "-lineinfo"]

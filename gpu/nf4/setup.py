"""Build the NF4 GEMM CUDA extension.

    C:\\Users\\Professional\\anaconda3\\envs\\torch-gpu\\python.exe setup.py build_ext --inplace
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
INCLUDE = REPO / "gpu" / "include"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from gpu.ampere_gencode import nvcc_cflags  # noqa: E402

cxx_flags = ["/O2"] if os.name == "nt" else ["-O3"]
nvcc_flags = nvcc_cflags()

setup(
    name="chr_nf4",
    ext_modules=[
        CUDAExtension(
            name="chr_nf4_ext",
            sources=[
                str(ROOT / "bindings.cpp"),
                str(ROOT / "nf4_gemm.cu"),
                str(ROOT / "nf4_gemv.cu"),
            ],
            include_dirs=[str(INCLUDE)],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

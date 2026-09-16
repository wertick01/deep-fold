"""Build the CPU fused NF4 GEMV (AVX2). No CUDA.

    C:\\Users\\Professional\\anaconda3\\envs\\torch-gpu\\python.exe gpu/cpu/setup.py build_ext --inplace
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpu.win_toolchain import inject_msvc_env  # noqa: E402

inject_msvc_env()
os.chdir(ROOT)

if os.name == "nt":
    cxx_flags = ["/O2", "/arch:AVX2", "/std:c++17", "/DCHR_NF4_FORCE_AVX2"]
else:
    cxx_flags = ["-O3", "-mavx2", "-mfma", "-mf16c", "-std=c++17", "-DCHR_NF4_FORCE_AVX2"]

setup(
    name="chr_nf4_cpu",
    ext_modules=[
        CppExtension(
            name="chr_nf4_cpu_ext",
            sources=[
                str(ROOT / "bindings.cpp"),
                str(ROOT / "nf4_gemv.cpp"),
            ],
            extra_compile_args=cxx_flags,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)

"""``deepfold setup``: catch up torch CUDA + chr in *this* interpreter.

Does not create a venv (that is ``scripts/setup.ps1`` / ``setup.sh``) and
never pip-installs into conda env ``torch-gpu``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from argparse import Namespace

from gpu.cuda_env import PREFER_SM120_NVCC_ENV

from . import messages
from .go_toolchain import GoToolchainError, ensure_chr
from .kernel_build import KernelBuildError, ensure_kernel
from .paths import REPO, find_chr_bin

__all__ = [
    "TORCH_INDEX",
    "TORCH_INDEX_CU128",
    "plan_lines",
    "prefix_is_protected",
    "setup",
    "torch_wheel_index",
    "wants_sm120_stack",
]

TORCH_INDEX = messages.TORCH_INDEX_CU124
TORCH_INDEX_CU128 = messages.TORCH_INDEX_CU128


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def prefix_is_protected(prefix: str | None = None) -> bool:
    """Author lab interpreter. A broken wheel there takes ``chr_nf4_ext`` with it."""
    raw = (prefix or sys.prefix).replace("\\", "/").rstrip("/")
    return raw.lower().endswith("/envs/torch-gpu") or raw.lower().endswith("/torch-gpu")


def wants_sm120_stack() -> bool:
    """RTX 50 / 5070 Ti: cu128 + nvcc 12.8. Spy seam for tests."""
    raw = os.environ.get("DEEPFOLD_TORCH_INDEX", "").strip().lower()
    if raw in {"cu128", "cu129", "sm120"}:
        return True
    if raw in {"cu124", "ampere"}:
        return False
    if os.environ.get(PREFER_SM120_NVCC_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "sm120",
    }:
        return True
    from .smi import visible_gpu_looks_sm120

    return visible_gpu_looks_sm120()


def torch_wheel_index() -> str:
    """cu124 on the 3080 lab; cu128 when nvidia-smi says RTX 50."""
    raw = os.environ.get("DEEPFOLD_TORCH_INDEX", "").strip()
    if raw.startswith("http"):
        return raw
    return TORCH_INDEX_CU128 if wants_sm120_stack() else TORCH_INDEX


def plan_lines(*, python: str | None = None, torch_index: str | None = None) -> list[str]:
    py = python or sys.executable
    index = torch_index or torch_wheel_index()
    return [
        f"{py} -m pip install torch --index-url {index}",
        f'{py} -m pip install -e ".[hub,chat]"',
        f"{py} -m pip install ninja",
        f"{py} -m gpu.cli setup --chr-only",
        f"{py} -m gpu.cli setup --kernel-only",
        f"{py} -m gpu.cli doctor",
    ]


def _run(cmd: list[str], *, cwd: str | None = None) -> int:
    """Spy seam. Tests replace this; live setup is the only caller."""
    return int(subprocess.call(cmd, cwd=cwd))


def setup(args: Namespace) -> int:
    dry = bool(getattr(args, "dry_run", False))
    if bool(getattr(args, "chr_only", False)):
        if dry:
            print(f"{sys.executable} -m gpu.cli setup --chr-only")
            return 0
        try:
            print(ensure_chr())
            return 0
        except GoToolchainError as exc:
            _err(f"setup: {exc}")
            return 1

    if bool(getattr(args, "kernel_only", False)):
        if dry:
            print(f"{sys.executable} -m gpu.cli setup --kernel-only")
            return 0
        try:
            print(
                ensure_kernel(
                    install=not bool(getattr(args, "no_install_tools", False))
                )
            )
            return 0
        except KernelBuildError as exc:
            _err(f"setup: {exc}")
            return 1

    lines = plan_lines()
    if wants_sm120_stack():
        os.environ.setdefault(PREFER_SM120_NVCC_ENV, "1")
    if prefix_is_protected():
        if dry:
            _err("dry-run: would refuse to pip-install into torch-gpu")
            for line in lines:
                print(line)
            print("powershell -File scripts/setup.ps1")
            print("bash scripts/setup.sh")
            return 0
        _err(messages.SETUP_REFUSE_TORCH_GPU)
        return 1

    if dry:
        for line in lines:
            print(line)
        return 0

    py = sys.executable
    index = torch_wheel_index()
    code = _run([py, "-m", "pip", "install", "torch", "--index-url", index])
    if code != 0:
        _err("setup: pip install torch failed")
        return code

    code = _run([py, "-m", "pip", "install", "-e", ".[hub,chat]"], cwd=str(REPO))
    if code != 0:
        _err("setup: pip install -e \".[hub,chat]\" failed")
        return code

    code = _run([py, "-m", "pip", "install", "ninja"])
    if code != 0:
        _err("setup: pip install ninja failed")
        return code

    if find_chr_bin(getattr(args, "chr_bin", None)) is None:
        try:
            ensure_chr()
        except GoToolchainError as exc:
            _err(f"setup: {exc}")
            return 1

    try:
        ensure_kernel(install=not bool(getattr(args, "no_install_tools", False)))
    except KernelBuildError as exc:
        _err(f"setup: {exc}")
        return 1

    from . import doctor as doctor_mod

    return int(doctor_mod.doctor(args))

"""``deepfold setup``: catch up torch CUDA + chr in *this* interpreter.

Does not create a venv (that is ``scripts/setup.ps1`` / ``setup.sh``) and
never pip-installs into conda env ``torch-gpu``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from argparse import Namespace

from . import messages
from .paths import REPO, find_chr_bin

__all__ = ["TORCH_INDEX", "plan_lines", "prefix_is_protected", "setup"]

TORCH_INDEX = "https://download.pytorch.org/whl/cu124"


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def prefix_is_protected(prefix: str | None = None) -> bool:
    """Author lab interpreter. A broken wheel there takes ``chr_nf4_ext`` with it."""
    raw = (prefix or sys.prefix).replace("\\", "/").rstrip("/")
    return raw.lower().endswith("/envs/torch-gpu") or raw.lower().endswith("/torch-gpu")


def plan_lines(*, python: str | None = None) -> list[str]:
    py = python or sys.executable
    chr_name = "chr.exe" if os.name == "nt" else "chr"
    return [
        f"{py} -m pip install torch --index-url {TORCH_INDEX}",
        f"go build -o {chr_name} ./cmd/chr",
        f"{py} -m gpu.cli doctor",
    ]


def _run(cmd: list[str], *, cwd: str | None = None) -> int:
    """Spy seam. Tests replace this; live setup is the only caller."""
    return int(subprocess.call(cmd, cwd=cwd))


def setup(args: Namespace) -> int:
    dry = bool(getattr(args, "dry_run", False))
    lines = plan_lines()
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
    code = _run([py, "-m", "pip", "install", "torch", "--index-url", TORCH_INDEX])
    if code != 0:
        _err("setup: pip install torch failed")
        return code

    chr_name = "chr.exe" if os.name == "nt" else "chr"
    if find_chr_bin() is None:
        go = shutil.which("go")
        if go is None:
            _err(
                "setup: Go is not on PATH and chr was not found. "
                "Install Go 1.22+ or set DEEPFOLD_CHR_BIN."
            )
            return 1
        code = _run([go, "build", "-o", chr_name, "./cmd/chr"], cwd=str(REPO))
        if code != 0:
            _err("setup: go build chr failed")
            return code

    from . import doctor as doctor_mod

    return int(doctor_mod.doctor(args))

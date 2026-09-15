"""``deepfold test``: CPU CLI acceptance. ``--live`` is a skip unless 3B is on disk."""

from __future__ import annotations

import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from .paths import REPO, models_root

__all__ = ["selftest"]


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def _live(args: Namespace) -> int:
    """No Hub, no generate. Skip is not a silent pass (D12)."""
    from .doctor import probe, verdict
    from .hub import source_complete

    m = probe(chr_bin=getattr(args, "chr_bin", None))
    v = verdict(m)
    if not v.allowed:
        _err(f"SKIP: generate not allowed on this machine ({v.line})")
        return 0
    path = models_root() / "Qwen2.5-3B-Instruct"
    if not source_complete(path):
        _err(
            f"SKIP: no 3B tree at {path}. "
            "deepfold pull Qwen/Qwen2.5-3B-Instruct --yes"
        )
        return 0
    _err(f"LIVE: {path} present; doctor arch={v.arch} (no generate in --live)")
    return 0


def selftest(args: Namespace) -> int:
    if getattr(args, "live", False):
        return _live(args)
    scripts = (
        REPO / "gpu" / "cli" / "test_cli.py",
        REPO / "gpu" / "cli" / "test_plate.py",
    )
    code = 0
    for script in scripts:
        code |= int(subprocess.call([sys.executable, str(script)]))
    return int(code)

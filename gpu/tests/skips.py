"""Absent hardware is a skip, never a silent pass (wave8-runtime D12).

Several suites in this tree run on a laptop with no NVIDIA device. The rule is
that a check which *could not be made* must say so: under pytest it becomes a
``skip``, and under the standalone ``python -m gpu.<pkg>.test_<x>`` runners it
prints ``SKIP``. What it must never do is ``return`` quietly, because pytest
records a function that returns as **passed** -- a green tick for a check that
never ran is worse than a red one.

The kernel is Ampere ``sm_86`` SASS only (``-gencode=arch=compute_86,code=sm_86``,
no PTX), so "CUDA is available" and "this card can launch the shipped kernel"
are two different questions and get two different helpers.

Import is deliberately cheap: ``torch`` is only imported once something
actually asks about the device.
"""

from __future__ import annotations

import importlib.util
from typing import NoReturn

__all__ = [
    "Skip",
    "skip",
    "cuda_reason",
    "sm86_reason",
    "requires_cuda",
    "requires_sm86",
    "requires_module",
]

try:  # let pytest report these as skips rather than as passes or errors
    import pytest

    Skip: type[BaseException] = pytest.skip.Exception
except ImportError:

    class Skip(Exception):  # type: ignore[no-redef]
        """Raised instead of a test that needs absent hardware, deps or files."""


#: The one capability the shipped kernel image was built for.
SHIP_CAPABILITY = (8, 6)


def skip(reason: str) -> NoReturn:
    """Abandon this check, loudly."""
    raise Skip(reason)


def cuda_reason() -> str | None:
    """Why CUDA is unusable here, or ``None`` when a device answered."""
    if importlib.util.find_spec("torch") is None:
        return "torch is not importable"
    import torch

    try:
        if not torch.cuda.is_available():
            return "no CUDA device (torch.cuda.is_available() is False)"
    except Exception as exc:  # noqa: BLE001 - a broken driver is still "no device"
        return f"torch.cuda.is_available() raised {type(exc).__name__}: {exc}"
    return None


def sm86_reason() -> str | None:
    """Why the shipped kernel cannot launch here, or ``None`` on an sm_86 card."""
    reason = cuda_reason()
    if reason is not None:
        return reason
    import torch

    try:
        cap = tuple(torch.cuda.get_device_capability(0))
    except Exception as exc:  # noqa: BLE001
        return f"device capability unreadable: {type(exc).__name__}: {exc}"
    if cap != SHIP_CAPABILITY:
        return (
            f"this GPU is sm_{cap[0]}{cap[1]}; the shipped kernel is sm_86 SASS "
            "only (-gencode=arch=compute_86,code=sm_86; no PTX)"
        )
    return None


def requires_cuda() -> None:
    reason = cuda_reason()
    if reason is not None:
        skip(reason)


def requires_sm86() -> None:
    reason = sm86_reason()
    if reason is not None:
        skip(reason)


def requires_module(name: str, *, extra: str = "") -> None:
    """Skip when an optional dependency is absent, naming how to install it."""
    if importlib.util.find_spec(name) is not None:
        return
    hint = f"; pip install deepfold[{extra}]" if extra else ""
    skip(f"{name} is not installed{hint}")

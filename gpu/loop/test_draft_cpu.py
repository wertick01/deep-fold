"""CPU: CpuHfDraft helpers without loading a 3B checkpoint.

    python gpu/loop/test_draft_cpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.loop.draft_cpu import _clone_past  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def test_clone_past_tuple() -> None:
    past = ((torch.arange(4).view(1, 1, 2, 2), torch.ones(1, 1, 2, 2)),)
    cloned = _clone_past(past)
    check("clone is not same object", cloned is not past, "")
    check("values match", torch.equal(cloned[0][0], past[0][0]), "")
    cloned[0][0].fill_(0)
    check("clone is a copy", int(past[0][0].sum()) == 6, str(int(past[0][0].sum())))


def test_clone_none() -> None:
    check("None stays None", _clone_past(None) is None, "")


TESTS = [test_clone_past_tuple, test_clone_none]


def main() -> int:
    print("gpu/loop draft_cpu CPU\n")
    for fn in TESTS:
        fn()
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

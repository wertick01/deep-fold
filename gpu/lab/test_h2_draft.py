"""CPU: h2_draft parser and oracle_draft wiring. No GPU.

    python gpu/lab/test_h2_draft.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab.h2_draft import _parser, main
from gpu.loop.speculate import oracle_draft

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def test_parser() -> None:
    p = _parser()
    args = p.parse_args(["--plan-only", "--speculate", "8", "--max-seq", "512"])
    check("plan_only", args.plan_only is True, "")
    check("speculate 8", args.speculate == 8, str(args.speculate))
    check("max_seq", args.max_seq == 512, str(args.max_seq))


def test_plan_only_writes_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "out"
        rc = main(["--plan-only", "--out", str(dest)])
        check("plan-only exit 0", rc == 0, str(rc))
        payload = json.loads((dest / "draft.json").read_text(encoding="utf-8"))
        check("schema", payload.get("schema") == "deepfold.h2_draft.v1", str(payload))
        check("plan_only flag", payload.get("plan_only") is True, str(payload))


def test_oracle_draft_import() -> None:
    d = oracle_draft([1, 2], 3, [1, 2, 9, 8, 7])
    check("teacher tail", d.tolist() == [9, 8, 7], str(d.tolist()))


TESTS = [test_parser, test_plan_only_writes_schema, test_oracle_draft_import]


def main_runner() -> int:
    print("gpu/lab h2_draft CPU\n")
    for fn in TESTS:
        fn()
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main_runner())

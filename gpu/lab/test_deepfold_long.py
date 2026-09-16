"""CPU: deepfold_long parser and plan-only. No GPU.

    python gpu/lab/test_deepfold_long.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab.deepfold_long import PRESETS, _parser, main
from gpu.lab.llamacpp_h2 import LONG_PROMPT as LLAMA_LONG
from gpu.lab.script import LONG_PROMPT

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def test_parser() -> None:
    args = _parser().parse_args(["--size", "3B", "--plan-only", "--long-n", "64"])
    check("size 3B", args.size == "3B", "")
    check("plan_only", args.plan_only is True, "")
    check("long_n 64", args.long_n == 64, str(args.long_n))
    check("3B compare id", PRESETS["3B"]["compare_id"] == "deepfold-nf4-3B-resident", "")
    check("32B compare id", PRESETS["32B"]["compare_id"] == "deepfold-nf4-32B-overflow", "")
    check("prompt matches llama.cpp", LONG_PROMPT == LLAMA_LONG, "")


def test_plan_only_writes_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "out"
        rc = main(["--plan-only", "--size", "32B", "--out", str(dest)])
        check("plan-only exit 0", rc == 0, str(rc))
        payload = json.loads((dest / "plate.json").read_text(encoding="utf-8"))
        check("schema", payload.get("schema") == "deepfold.long.v1", str(payload.get("schema")))
        check("plan_only flag", payload.get("plan_only") is True, "")
        check("compare_id", payload.get("compare_id") == "deepfold-nf4-32B-overflow", "")
        check("long_n", payload.get("long_n") == 64, str(payload.get("long_n")))


def main_test() -> int:
    print("gpu/lab deepfold_long, CPU only\n")
    test_parser()
    test_plan_only_writes_schema()
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main_test())

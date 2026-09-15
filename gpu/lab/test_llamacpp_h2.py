"""CPU: llamacpp_h2 parser and plan-only. No GPU, no GGUF required.

    python gpu/lab/test_llamacpp_h2.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab.llamacpp_h2 import EXPECTED_GGUF_BYTES, _parser, main

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def test_parser() -> None:
    p = _parser()
    args = p.parse_args(["--plan-only", "--ctx", "2048", "--ngl", "99"])
    check("plan_only", args.plan_only is True, "")
    check("ctx 2048", args.ctx == 2048, str(args.ctx))
    check("ngl 99", args.ngl == 99, str(args.ngl))
    check("parallel 1", args.parallel == 1, str(args.parallel))
    check("long_n 64", args.long_n == 64, str(args.long_n))
    check("bench off", args.bench is False, str(args.bench))


def test_plan_only_writes_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "out"
        rc = main(["--plan-only", "--out", str(dest)])
        check("plan-only exit 0", rc == 0, str(rc))
        payload = json.loads((dest / "plate.json").read_text(encoding="utf-8"))
        check("schema", payload.get("schema") == "deepfold.llamacpp_h2.v1", str(payload.get("schema")))
        check("plan_only flag", payload.get("plan_only") is True, "")
        check("quant", payload.get("quant") == "Q4_K_M", str(payload.get("quant")))
        check("expected GGUF size", EXPECTED_GGUF_BYTES == 19_851_336_576, str(EXPECTED_GGUF_BYTES))


TESTS = [test_parser, test_plan_only_writes_schema]


def main_runner() -> int:
    print("gpu/lab llamacpp_h2 CPU\n")
    for fn in TESTS:
        fn()
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main_runner())

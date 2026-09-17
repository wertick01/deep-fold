"""CPU: ollama_hard plan-only. No Ollama server required.

    python gpu/lab/test_ollama_hard.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab.hard import load_fixture, score_item
from gpu.lab.ollama_hard import main

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def test_plan_only() -> None:
    items = load_fixture()["independent"]
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "out"
        rc = main(["--plan-only", "--out", str(dest)])
        check("plan-only exit 0", rc == 0, str(rc))
        payload = json.loads((dest / "plate.json").read_text(encoding="utf-8"))
        check("schema", payload.get("schema") == "deepfold.ollama_hard.v1", str(payload.get("schema")))
        check("twelve items", payload.get("n_items") == 12, str(payload.get("n_items")))
        check("ids match fixture", payload.get("item_ids") == [item.id for item in items], "")


def test_score_item_smoke_path() -> None:
    items = {item.id: item for item in load_fixture()["independent"]}
    lamps = items["gsm8k-lamps"]
    ok = score_item(lamps, "Monday 35, Tuesday 47, Wednesday 82, total 164\n#### 164")
    check("lamps pass", ok.correct is True, ok.extracted)
    miss = score_item(lamps, "I think 100\n#### 100")
    check("lamps miss", miss.correct is False, miss.extracted)


TESTS = [test_plan_only, test_score_item_smoke_path]


def main_runner() -> int:
    print("gpu/lab ollama_hard CPU\n")
    for fn in TESTS:
        fn()
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main_runner())

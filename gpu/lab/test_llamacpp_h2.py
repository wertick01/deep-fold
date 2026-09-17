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

from gpu.lab.llamacpp_h2 import EXPECTED_GGUF_BYTES, _compare_id, _parser, _server_cmd, _size_and_model, main

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
    auto = p.parse_args(["--plan-only"])
    check("ngl default auto", auto.ngl == -1, str(auto.ngl))


def test_server_cmd_omits_ngl_when_negative() -> None:
    auto = _server_cmd(
        "llama-server.exe",
        "m.gguf",
        ctx=2048,
        ngl=-1,
        host="127.0.0.1",
        port=8765,
        parallel=1,
    )
    fill = _server_cmd(
        "llama-server.exe",
        "m.gguf",
        ctx=2048,
        ngl=99,
        host="127.0.0.1",
        port=8765,
        parallel=1,
    )
    check("auto omits flag", "--n-gpu-layers" not in auto, " ".join(auto))
    check("99 keeps flag", fill[-2:] == ["--n-gpu-layers", "99"], " ".join(fill[-4:]))


def test_size_and_model_from_filename() -> None:
    size, model = _size_and_model("Qwen2.5-32B-Instruct-Q4_K_M.gguf")
    check("32B before 3B substring", size == "32B" and "32B" in model, f"{size} {model}")
    size, model = _size_and_model("internlm2_5-20b-chat-Q4_K_M.gguf")
    check("internlm 20b lowercase", size == "20B" and "internlm" in model, f"{size} {model}")
    size, model = _size_and_model("Qwen2.5-14B-Instruct-Q4_K_M.gguf")
    check("14B before 3B substring", size == "14B" and "14B" in model, f"{size} {model}")
    size, model = _size_and_model("Qwen2.5-3B-Instruct-Q4_K_M.gguf")
    check("3B qwen", size == "3B", f"{size} {model}")


def test_compare_id_keeps_ngl99_row() -> None:
    check("ngl 99 id", _compare_id("32B", 99) == "llamacpp-q4-32B-Q4_K_M", "")
    check("auto-fit id", _compare_id("32B", -1) == "llamacpp-q4-32B-Q4_K_M-autofit", "")
    check("other ngl", _compare_id("32B", 33) == "llamacpp-q4-32B-Q4_K_M-ngl33", "")


def test_plan_only_writes_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "out"
        rc = main(["--plan-only", "--out", str(dest)])
        check("plan-only exit 0", rc == 0, str(rc))
        payload = json.loads((dest / "plate.json").read_text(encoding="utf-8"))
        check("schema", payload.get("schema") == "deepfold.llamacpp_h2.v1", str(payload.get("schema")))
        check("plan_only flag", payload.get("plan_only") is True, "")
        check("quant", payload.get("quant") == "Q4_K_M", str(payload.get("quant")))
        check("ngl omitted by default", payload.get("ngl_omitted") is True, str(payload.get("ngl")))
        check("expected GGUF size", EXPECTED_GGUF_BYTES == 19_851_336_576, str(EXPECTED_GGUF_BYTES))


TESTS = [
    test_parser,
    test_server_cmd_omits_ngl_when_negative,
    test_compare_id_keeps_ngl99_row,
    test_size_and_model_from_filename,
    test_plan_only_writes_schema,
]


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

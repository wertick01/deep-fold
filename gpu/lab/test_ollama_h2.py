"""CPU: Ollama timing labels. No server required.

    python gpu/lab/test_ollama_h2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab.ollama_h2 import _fold_ndjson, _size_label, _timings

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def test_cached_prompt_is_not_ttft() -> None:
    times = _timings(
        {
            "prompt_eval_count": 42,
            "prompt_eval_cached_count": 24,
            "prompt_eval_duration": 986_005_000,
            "eval_count": 8,
            "eval_duration": 2_824_142_000,
            "_wall_ms": 3838.7,
        }
    )
    check("records cache", times["prompt_cached_n"] == 24, str(times["prompt_cached_n"]))
    check("not comparable TTFT", times["ttft_comparable"] is False, "")
    check("client ttft unset", times["client_ttft_ms"] is None, str(times["client_ttft_ms"]))
    check("still has decode", abs(times["decode_tok_s"] - 8 / 2.824142) < 1e-6, str(times["decode_tok_s"]))


def test_uncached_prompt_is_comparable() -> None:
    times = _timings(
        {
            "prompt_eval_count": 40,
            "prompt_eval_cached_count": 0,
            "prompt_eval_duration": 1_000_000_000,
            "eval_count": 64,
            "eval_duration": 25_000_000_000,
            "_client_ttft_ms": 912.5,
        }
    )
    check("zero cache comparable", times["ttft_comparable"] is True, "")
    check("server prompt 1000 ms", abs(times["prompt_ms"] - 1000.0) < 1e-6, str(times["prompt_ms"]))
    check("client ttft kept", abs(times["client_ttft_ms"] - 912.5) < 1e-9, str(times["client_ttft_ms"]))


def test_fold_ndjson_joins_stream_chunks() -> None:
    folded = _fold_ndjson(
        [
            {"message": {"role": "assistant", "content": "The "}, "done": False},
            {"message": {"role": "assistant", "content": "capital"}, "done": False},
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "eval_count": 8,
                "eval_duration": 1_000_000_000,
                "prompt_eval_cached_count": 0,
            },
        ]
    )
    check("joined text", folded["message"]["content"] == "The capital", folded["message"]["content"])
    check("keeps timers", folded.get("eval_count") == 8, str(folded.get("eval_count")))


def test_size_label_14b_is_not_3b() -> None:
    from argparse import Namespace

    check("explicit 14B", _size_label(Namespace(size="14B", model="qwen2.5:14b")) == "14B", "")
    check("14b tag not 3B", _size_label(Namespace(size="", model="qwen2.5:14b")) == "14B", "")
    check("3b tag", _size_label(Namespace(size="", model="qwen2.5:3b")) == "3B", "")
    check("20b internlm", _size_label(Namespace(size="", model="internlm2.5:20b-q4km")) == "20B", "")
    check("32b tag", _size_label(Namespace(size="", model="qwen2.5:32b")) == "32B", "")


TESTS = [
    test_cached_prompt_is_not_ttft,
    test_uncached_prompt_is_comparable,
    test_fold_ndjson_joins_stream_chunks,
    test_size_label_14b_is_not_3b,
]


def main() -> int:
    print("gpu/lab ollama_h2 CPU\n")
    for fn in TESTS:
        fn()
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

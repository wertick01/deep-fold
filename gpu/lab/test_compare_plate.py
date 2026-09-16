"""CPU tests for the 3080 compare plate. No model, no CUDA.

    python -m gpu.lab.test_compare_plate
    python gpu/lab/test_compare_plate.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def _is_text(artist: Any) -> bool:
    from matplotlib.text import Text

    return isinstance(artist, Text)


def _plate_text(figure: Any) -> str:
    return "\n".join(artist.get_text() for artist in figure.findobj(match=_is_text))


def gate_no_torch() -> None:
    before = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
    from gpu.lab import compare_plate as plate

    after = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
    leaked = after - before
    check("importing compare_plate does not import torch", not leaked, str(sorted(leaked)))
    source = Path(plate.__file__).read_text(encoding="utf-8")
    check(
        "compare_plate does not import gpu.nf4 or torch",
        "import torch" not in source
        and "import gpu.nf4" not in source
        and "from gpu.nf4" not in source,
        plate.__file__,
    )


def gate_tokens() -> None:
    from gpu.lab.compare_plate import (
        H2_32B_LONG,
        H2_3B_LONG,
        H2_STREAM_MATRICES,
        H2_STREAM_MIB,
        LLAMA_32B_LONG,
        OLLAMA_32B_LONG,
        OLLAMA_3B_LONG,
        OLLAMA_GPU_LAYERS,
        OLLAMA_TOTAL_LAYERS,
    )
    from gpu.lab.h2_plate import DECODE_TOK_S, STREAMED_MIB

    check("Ollama 32B long is 2.54", OLLAMA_32B_LONG == 2.54, str(OLLAMA_32B_LONG))
    check("H2 32B long is 2.49", H2_32B_LONG == 2.49, str(H2_32B_LONG))
    check("llama.cpp 32B long is 1.52", LLAMA_32B_LONG == 1.52, str(LLAMA_32B_LONG))
    check("Ollama 3B long is 187.3", OLLAMA_3B_LONG == 187.3, str(OLLAMA_3B_LONG))
    check("H2 3B long is 35.2", H2_3B_LONG == 35.2, str(H2_3B_LONG))
    check("33/65 layers", (OLLAMA_GPU_LAYERS, OLLAMA_TOTAL_LAYERS) == (33, 65), "")
    check("stream 96 / 6885 matches H2 plate", H2_STREAM_MATRICES == 96 and H2_STREAM_MIB == 6885, "")
    check("H2 product smoke stays 2.31", DECODE_TOK_S == 2.31, str(DECODE_TOK_S))
    check("H2 streamed MiB matches", STREAMED_MIB == 6885.0, str(STREAMED_MIB))
    check("does not claim H2 faster than Ollama", H2_32B_LONG < OLLAMA_32B_LONG, "")


def gate_honesty() -> None:
    from gpu.lab.compare_plate import compare_plate

    figure = compare_plate()
    text = _plate_text(figure)
    import matplotlib.pyplot as plt

    plt.close(figure)
    check("mentions 64-token plateau", "64-token" in text and "plateau" in text.lower(), "")
    check("quotes 2.54 and 2.49", "2.54" in text and "2.49" in text, "")
    check("shows 33/65", "33/65" in text, "")
    check("names CopyRing", "CopyRing" in text, "")
    check("names SKIP", "SKIP" in text, "")
    check("warns against smoke 3.18", "3.18" in text, "")
    check("mentions ngl 99 / fit abort", "ngl 99" in text and "fit abort" in text, "")
    check("does not say faster than Ollama", "faster than Ollama" not in text.lower(), "")
    check("does not quote 0.8 as design", "0.8" not in text, "")


def main() -> int:
    print("gpu/lab compare_plate, CPU only\n")
    gate_no_torch()
    gate_tokens()
    gate_honesty()
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

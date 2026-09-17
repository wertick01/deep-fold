"""CPU tests for the Decode V2 vs Ollama hard-12 table. No model, no CUDA.

    python -m gpu.lab.test_hard_v2_plate
    python gpu/lab/test_hard_v2_plate.py
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
    from gpu.lab import hard_v2_plate as plate

    after = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
    leaked = after - before
    check("importing hard_v2_plate does not import torch", not leaked, str(sorted(leaked)))
    source = Path(plate.__file__).read_text(encoding="utf-8")
    check(
        "hard_v2_plate does not import gpu.nf4 or torch",
        "import torch" not in source
        and "import gpu.nf4" not in source
        and "from gpu.nf4" not in source,
        plate.__file__,
    )


def gate_tokens() -> None:
    from gpu.lab.decodev2_plate import (
        COMING_SOON,
        OLLAMA_20B_HARD,
        OLLAMA_20B_HARD_TOK,
        OLLAMA_3B_HARD,
        OLLAMA_3B_HARD_TOK,
        V2_14B_HARD,
        V2_20B_HARD,
        V2_20B_HARD_TOK,
        V2_3B_HARD,
        V2_3B_HARD_TOK,
    )

    check("V2 3B hard is 8/12 at 190.9", V2_3B_HARD == "8/12" and V2_3B_HARD_TOK == 190.9, "")
    check("V2 14B hard is 11/12", V2_14B_HARD == "11/12", V2_14B_HARD)
    check("V2 20B hard is 9/12 at 35.2", V2_20B_HARD == "9/12" and V2_20B_HARD_TOK == 35.2, "")
    check("Ollama 3B hard is 7/12 at 197.2", OLLAMA_3B_HARD == "7/12" and OLLAMA_3B_HARD_TOK == 197.2, "")
    check("Ollama 20B hard is 9/12 at 13.2", OLLAMA_20B_HARD == "9/12" and OLLAMA_20B_HARD_TOK == 13.2, "")
    check("3B V2 hard is not faster than Ollama", V2_3B_HARD_TOK < OLLAMA_3B_HARD_TOK, "")
    check("Coming soon label is frozen", COMING_SOON == "Coming soon", COMING_SOON)


def gate_honesty() -> None:
    from gpu.lab.hard_v2_plate import hard_v2_plate

    figure = hard_v2_plate()
    text = _plate_text(figure)
    import matplotlib.pyplot as plt

    plt.close(figure)
    check("names Decode V2 and Ollama", "Decode V2" in text and "Ollama" in text, "")
    check("quotes 8/12 and 7/12", "8/12" in text and "7/12" in text, "")
    check("quotes 11/12 and both 9/12", "11/12" in text and "9/12" in text, "")
    check("quotes V2 190.9 and Ollama 197.2", "190.9" in text and "197.2" in text, "")
    check("quotes V2 54.8", "54.8" in text, "")
    check("quotes V2 35.2 and Ollama 13.2", "35.2" in text and "13.2" in text, "")
    check("14B Ollama is Coming soon", "Coming soon" in text, "")
    check("mentions median 184.5", "184.5" in text, "")
    check("does not say faster than Ollama", "faster than Ollama" not in text.lower(), "")
    check("20B V2 is not ignore-EOS 40", "not ignore-EOS 40" in text, "")
    check("llama.cpp hard-12 is Coming soon", "llama.cpp hard-12" in text and "Coming soon" in text, "")
    check("keeps TokenLoop MMA as old path", "TokenLoop MMA" in text and "16.9" in text, "")


def main() -> int:
    print("gpu/lab hard_v2_plate, CPU only\n")
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

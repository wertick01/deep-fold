"""CPU tests for the H2 32B overflow plate. No model, no CUDA.

    python -m gpu.lab.test_h2_plate
    python gpu/lab/test_h2_plate.py
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
    from gpu.lab import h2_plate as plate

    after = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
    leaked = after - before
    check("importing h2_plate does not import torch", not leaked, str(sorted(leaked)))
    source = Path(plate.__file__).read_text(encoding="utf-8")
    check(
        "h2_plate does not import gpu.nf4 or torch",
        "import torch" not in source
        and "import gpu.nf4" not in source
        and "from gpu.nf4" not in source,
        plate.__file__,
    )


def gate_tokens() -> None:
    from gpu.lab.hard_plate import CODEC_STYLE as HARD_STYLE
    from gpu.lab.hard_plate import INK, INK_SOFT, LIMIT
    from gpu.lab.h2_plate import CODEC_STYLE, CARD_MIB, DECODE_TOK_S, PACKED_MIB
    from gpu.lab.h2_plate import INK as H_INK
    from gpu.lab.h2_plate import INK_SOFT as H_SOFT
    from gpu.lab.h2_plate import LIMIT as H_LIMIT
    from gpu.lab.h2_plate import RESIDENT_MIB, STREAMED_MIB

    check("nf4 colour matches hard_plate", CODEC_STYLE["nf4"].color == HARD_STYLE["nf4"].color)
    check("INK matches", H_INK == INK, H_INK)
    check("INK_SOFT matches", H_SOFT == INK_SOFT, H_SOFT)
    check("LIMIT matches", H_LIMIT == LIMIT, H_LIMIT)
    check("card is 12288", CARD_MIB == 12288.0, str(CARD_MIB))
    check("packed is 16599", PACKED_MIB == 16599.0, str(PACKED_MIB))
    check("resident is 9716", RESIDENT_MIB == 9716.0, str(RESIDENT_MIB))
    check("streamed is 6885", STREAMED_MIB == 6885.0, str(STREAMED_MIB))
    check("product decode is 2.31", DECODE_TOK_S == 2.31, str(DECODE_TOK_S))


def gate_honesty() -> None:
    from gpu.lab.h2_plate import h2_plate

    figure = h2_plate()
    text = _plate_text(figure)
    import matplotlib.pyplot as plt

    plt.close(figure)

    must = [
        "Qwen2.5-32B-Instruct",
        "NF4 overflow",
        "2.31 tok/s",
        "9,716",
        "6,885",
        "16,599",
        "12,288",
        "policy D",
        "CopyRing",
        "chr_nf4_gemm",
        "pinned",
        "Not VQ",
        "No dense [M,K]",
        "hard-12 not started",
        "smoke 3/3",
        "WDDM",
        "Tensor.is_pinned",
        "not a kernel ranking",
        "10 tok/s is not a claim",
        "No BF16 32B",
        "python -m gpu.lab.h2_plate --redraw",
    ]
    for needle in must:
        check(f"plate names {needle!r}", needle in text, needle if needle not in text else "")

    check('does not claim all weights in HBM', 'not "all weights in HBM"' in text, "HBM claim")
    # The plate uses curly quotes in the honesty footer.
    check(
        "fits via overflow",
        "fits via overflow" in text,
        "overflow",
    )

    forbidden = [
        "faster than Marlin",
        "faster than llama.cpp",
        "faster than AWQ",
        "WikiText",
        "GSM8K",
        "PPL",
        "codec vq",
        "all-resident 32B",
        "10 tok/s target",
        "BF16 32B tok/s",
    ]
    for needle in forbidden:
        check(f"plate does not say {needle!r}", needle not in text, needle)

    check("0.8 is labeled a bug / diagnostic", "pin bug" in text and "not the product" in text, "0.8")
    check("277 ms is not measured wall", "277 ms ≠ measured wall" in text or "Serial floor 277" in text, "floor")


def main() -> int:
    print("gpu.lab.test_h2_plate")
    gate_no_torch()
    gate_tokens()
    gate_honesty()
    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

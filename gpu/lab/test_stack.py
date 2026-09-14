"""CPU tests for the stack architecture plate. No model, no CUDA.

    python -m gpu.lab.test_stack
    python gpu/lab/test_stack.py
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
    from gpu.lab import stack_plate as plate

    after = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
    leaked = after - before
    check("importing stack_plate does not import torch", not leaked, str(sorted(leaked)))
    source = Path(plate.__file__).read_text(encoding="utf-8")
    check(
        "stack_plate does not import gpu.nf4 or torch",
        "import torch" not in source
        and "import gpu.nf4" not in source
        and "from gpu.nf4" not in source,
        plate.__file__,
    )


def gate_tokens() -> None:
    from gpu.lab.hard_plate import CODEC_STYLE as HARD_STYLE
    from gpu.lab.hard_plate import INK, INK_SOFT, LIMIT
    from gpu.lab.stack_plate import CODEC_STYLE, INK as S_INK
    from gpu.lab.stack_plate import INK_SOFT as S_SOFT
    from gpu.lab.stack_plate import LIMIT as S_LIMIT
    from gpu.lab.stack_plate import LIVE_MAX_N

    for codec in ("bf16", "nf4"):
        check(
            f"{codec} colour matches hard_plate",
            CODEC_STYLE[codec].color == HARD_STYLE[codec].color,
            CODEC_STYLE[codec].color,
        )
    check("INK matches", S_INK == INK, S_INK)
    check("INK_SOFT matches", S_SOFT == INK_SOFT, S_SOFT)
    check("LIMIT matches", S_LIMIT == LIMIT, S_LIMIT)
    check("LIVE_MAX_N is 16", LIVE_MAX_N == 16, str(LIVE_MAX_N))


def gate_honesty() -> None:
    from gpu.lab.stack_plate import stack_plate

    figure = stack_plate()
    text = _plate_text(figure)
    import matplotlib.pyplot as plt

    plt.close(figure)

    must = [
        "CHR0",
        "NF4",
        "4.25",
        "CompressedLinear",
        "packed",
        "scale",
        "registers",
        "mma.sync.aligned.m16n8k16",
        "LIVE_MAX_N",
        "TokenLoop",
        "transformers.generate",
        "llama_swiglu",
        "internlm_gqa",
        "Gemma",
        "Phi-3",
        "MoE",
        "sm_86",
        "prior art",
        "cmd/chr",
        "gpu/host/model.py",
        "gpu/nf4/nf4_gemm.cu",
        "Nf4Embedding",
        "[out, in]",
        "from_pretrained",
        "HBM",
        "GGUF",
        "Ada",
        "Hopper",
        "Blackwell",
        "69",
        "113",
        "1.63",
        "python -m gpu.cli doctor",
        "python -m gpu.cli run",
        "chr compress",
    ]
    for needle in must:
        check(f"plate names {needle!r}", needle in text, needle if needle not in text else "")

    check("LIVE_MAX_N = 16 is on the plate", "LIVE_MAX_N = 16" in text, "LIVE_MAX_N")
    check("n32 is labeled not live", "Not the live TokenLoop path" in text or "not the live TokenLoop" in text, "n32")
    check("LIVE_MAX_N was not raised", "LIVE_MAX_N was not raised" in text, "raised")
    check(
        "does not call transformers.generate",
        "Does not call transformers.generate" in text,
        "generate",
    )
    check("reconstruct in registers, discard", "discarded" in text or "dropped" in text, "discard")
    check("packed table is read-only", "read-only" in text, "RO")
    check("prior art names Linear4bit", "Linear4bit" in text, "Linear4bit")

    forbidden = [
        "17.0",
        "31.6",
        "28.4",
        "WikiText",
        "GSM8K",
        "PPL",
        "faster than Marlin",
        "faster than AWQ",
        "faster than bitsandbytes",
        "faster than llama.cpp",
        "unpack a layer into video memory",
        "unpack layer into VRAM, multiply, pack",
    ]
    for needle in forbidden:
        check(f"plate does not say {needle!r}", needle not in text, needle)

    check(
        "footer refuses the unpack-layer cartoon",
        "not unpack-layer-into-VRAM" in text,
        "cartoon",
    )
    check("nvidia-smi is not subtracted", "not subtracted" in text, "smi")
    check("not a tok/s leaderboard", "not a tok/s leaderboard" in text, "leaderboard")


def gate_flow() -> None:
    from gpu.lab.stack_flow import LIVE_MAX_N, stack_flow
    from gpu.lab.stack_plate import LIVE_MAX_N as POSTER_N

    check("flow LIVE_MAX_N matches the poster", LIVE_MAX_N == POSTER_N == 16, str(LIVE_MAX_N))
    source = (_REPO / "gpu" / "lab" / "stack_flow.py").read_text(encoding="utf-8")
    check(
        "stack_flow does not import gpu.nf4 or torch",
        "import torch" not in source
        and "import gpu.nf4" not in source
        and "from gpu.nf4" not in source,
        "imports",
    )
    figure = stack_flow()
    text = _plate_text(figure)
    import matplotlib.pyplot as plt

    plt.close(figure)
    for needle in (
        "chr compress",
        "CHR0",
        "CompressedLinear",
        "TokenLoop",
        "LIVE_MAX_N = 16",
        "registers",
        "transformers.generate",
        "tile-local",
        "llama_swiglu",
        "internlm_gqa",
    ):
        check(f"flow names {needle!r}", needle in text, needle if needle not in text else "")
    check("flow does not raise LIVE_MAX_N", "TokenLoop does not launch it" in text, "n32 live")
    for needle in (
        "17.0",
        "31.6",
        "28.4",
        "faster than Marlin",
        "unpack a layer into VRAM, multiply, pack",
    ):
        check(f"flow does not say {needle!r}", needle not in text, needle)


def main() -> int:
    print("gpu.lab.test_stack")
    gate_no_torch()
    gate_tokens()
    gate_honesty()
    gate_flow()
    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

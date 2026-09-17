"""CPU tests for the Decode V2 plate. No model, no CUDA.

    python -m gpu.lab.test_decodev2_plate
    python gpu/lab/test_decodev2_plate.py
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
    from gpu.lab import decodev2_plate as plate

    after = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
    leaked = after - before
    check("importing decodev2_plate does not import torch", not leaked, str(sorted(leaked)))
    source = Path(plate.__file__).read_text(encoding="utf-8")
    check(
        "decodev2_plate does not import gpu.nf4 or torch",
        "import torch" not in source
        and "import gpu.nf4" not in source
        and "from gpu.nf4" not in source,
        plate.__file__,
    )


def gate_tokens() -> None:
    from gpu.lab.decodev2_plate import (
        COMING_SOON,
        LLAMA_14B_LONG,
        LLAMA_20B_LONG,
        LLAMA_3B,
        OLLAMA_14B_LONG,
        OLLAMA_20B_LONG,
        OLLAMA_3B,
        TL_14B,
        TL_20B,
        TL_20B_LONG,
        TL_3B_LONG,
        V2_14B_HOST,
        V2_14B_HOST_512,
        V2_20B_DEVICE_SECOND,
        V2_20B_HOST,
        V2_20B_HOST_512,
        V2_3B_HOST,
        V2_3B_HOST_512,
    )

    check("V2 3B host is 197 at max_seq=2048", V2_3B_HOST == 197.0, str(V2_3B_HOST))
    check("V2 14B host is 57.5 at max_seq=2048", V2_14B_HOST == 57.5, str(V2_14B_HOST))
    check("V2 20B host is 40.0 at max_seq=2048", V2_20B_HOST == 40.0, str(V2_20B_HOST))
    check("512-era 3B stays 196 in the footnote", V2_3B_HOST_512 == 196.0, str(V2_3B_HOST_512))
    check("512-era 14B stays 55.6 in the footnote", V2_14B_HOST_512 == 55.6, str(V2_14B_HOST_512))
    check("512-era 20B stays 41.0 in the footnote", V2_20B_HOST_512 == 41.0, str(V2_20B_HOST_512))
    check("TokenLoop 3B stays 35.2", TL_3B_LONG == 35.2, str(TL_3B_LONG))
    check("TokenLoop 14B stays 6.56", TL_14B == 6.56, str(TL_14B))
    check("TokenLoop 20B smoke stays 5.01", TL_20B == 5.01, str(TL_20B))
    check("TokenLoop 20B long is 4.59", TL_20B_LONG == 4.59, str(TL_20B_LONG))
    check("Ollama 3B long stays 187.3", OLLAMA_3B == 187.3, str(OLLAMA_3B))
    check("llama.cpp 3B long stays 187.0", LLAMA_3B == 187.0, str(LLAMA_3B))
    check("Ollama 14B exclusive long is 58.9", OLLAMA_14B_LONG == 58.9, str(OLLAMA_14B_LONG))
    check("llama.cpp 14B exclusive long is 69.9", LLAMA_14B_LONG == 69.9, str(LLAMA_14B_LONG))
    check("Ollama 20B long is 11.53", OLLAMA_20B_LONG == 11.53, str(OLLAMA_20B_LONG))
    check("llama.cpp 20B long is 11.87", LLAMA_20B_LONG == 11.87, str(LLAMA_20B_LONG))
    check("14B V2 is not faster than exclusive Ollama", V2_14B_HOST < OLLAMA_14B_LONG, "")
    check("14B V2 is not faster than exclusive llama.cpp", V2_14B_HOST < LLAMA_14B_LONG, "")
    check("20B second-pass 25.6 is labeled, not the bar", V2_20B_DEVICE_SECOND == 25.6, "")
    check("does not treat 25.6 as faster than 40", V2_20B_DEVICE_SECOND < V2_20B_HOST, "")
    check("Coming soon label is frozen", COMING_SOON == "Coming soon", COMING_SOON)


def gate_honesty() -> None:
    from gpu.lab.decodev2_plate import decodev2_plate

    figure = decodev2_plate()
    text = _plate_text(figure)
    import matplotlib.pyplot as plt

    plt.close(figure)
    check("names Decode V2", "Decode V2" in text, "")
    check("quotes 197 and 57.5 and 40", "197" in text and "57.5" in text and "40" in text, "")
    check("quotes 14B Q4_K 58.9 and 69.9", "58.9" in text and "69.9" in text, "")
    check("quotes 20B Q4_K 11.53 and 11.87", "11.53" in text and "11.87" in text, "")
    check("keeps TokenLoop 35.2", "35.2" in text, "")
    check("headline max_seq is 2048", "max_seq 2048" in text or "max_seq=2048" in text, "")
    check("mentions ctx 2048 for Q4_K", "2048" in text, "")
    check("keeps 512-era footnote", "512" in text and "55.6" in text and "41.0" in text, "")
    check("warns against 25.6", "25.6" in text, "")
    check("warns against 20.7", "20.7" in text, "")
    check("discards overlapping 14B 5.95", "5.95" in text, "")
    check("mentions overlapping / paging", "paging" in text.lower() or "Overlapping" in text, "")
    check("32B stays CopyRing", "CopyRing" in text and "32B" in text, "")
    check("names --executor auto", "--executor auto" in text, "")
    check("does not say faster than Ollama", "faster than Ollama" not in text.lower(), "")
    check("does not quote 7–10 as a plate", "7–10 tok/s" in text or "7-10" in text, "")
    check("hard-12 3B is 8/12", "8/12" in text, "")
    check("hard-12 14B is 11/12", "11/12" in text, "")
    check("hard-12 20B is 9/12", "9/12" in text, "")
    check("20B hard mean is 35.2", "35.2" in text, "")
    check("Ollama 14B hard is Coming soon", "Coming soon" in text and "Ollama 14B hard-12" in text, "")
    check("llama.cpp hard-12 is Coming soon", "llama.cpp hard-12" in text and "Coming soon" in text, "")
    check("Nsight 70–85 is Coming soon", "Nsight" in text and "70–85" in text, "")
    check("CLI chrome is in progress", "in progress" in text.lower(), "")
    check("V2 still attends the full buffer", "full buffer" in text or "full axis" in text, "")


def main() -> int:
    print("gpu/lab decodev2_plate, CPU only\n")
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

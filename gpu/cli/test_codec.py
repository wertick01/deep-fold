"""CPU: NF4 vs VQ auto-pick from config.json shapes. No GPU.

    python gpu/cli/test_codec.py
    python -m pytest gpu/cli/test_codec.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.cli.codec import (  # noqa: E402
    DEFAULT_VRAM_MIB,
    RUNTIME_OVERHEAD_MIB,
    CodecFitError,
    budget_for,
    decide,
    params_from_config,
)

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


# docs/vram-3080.md Qwen2.5-14B / 32B shapes (not tied).
QWEN_14B = dict(
    hidden_size=5120,
    intermediate_size=13824,
    num_hidden_layers=48,
    num_attention_heads=40,
    num_key_value_heads=8,
    vocab_size=152064,
    tie_word_embeddings=False,
    model_type="qwen2",
)
QWEN_32B = dict(
    hidden_size=5120,
    intermediate_size=27648,
    num_hidden_layers=64,
    num_attention_heads=40,
    num_key_value_heads=8,
    vocab_size=152064,
    tie_word_embeddings=False,
    model_type="qwen2",
)
# Tiny 3B-class: must still pick NF4 on a 12 GB card.
QWEN_3B = dict(
    hidden_size=2048,
    intermediate_size=11008,
    num_hidden_layers=36,
    num_attention_heads=16,
    num_key_value_heads=2,
    vocab_size=151936,
    tie_word_embeddings=True,
    model_type="qwen2",
)


def test_14b_param_count_matches_vram_doc() -> None:
    n = params_from_config(QWEN_14B)
    # 14.770B on the card; allow 2M slack for rounding.
    check("14B params ~14.770e9", abs(n - 14_769_192_960) < 2_000_000, f"{n}")


def test_14b_nf4_fits_32b_does_not() -> None:
    b14 = budget_for(QWEN_14B, DEFAULT_VRAM_MIB)
    b32 = budget_for(QWEN_32B, DEFAULT_VRAM_MIB)
    check("14B NF4 leftover > 0", b14.leftover_nf4 > 0, f"{b14.leftover_nf4:.0f}")
    check("32B NF4 leftover < 0", b32.leftover_nf4 < 0, f"{b32.leftover_nf4:.0f}")
    check("32B VQ leftover > 0", b32.leftover_vq > 0, f"{b32.leftover_vq:.0f}")


def test_auto_picks_nf4_when_it_fits() -> None:
    d = decide(QWEN_3B, DEFAULT_VRAM_MIB, requested="auto")
    check("3B auto is nf4", d.codec == "nf4" and not d.overflow, d.reason)
    d14 = decide(QWEN_14B, DEFAULT_VRAM_MIB, requested="auto")
    check("14B auto is nf4", d14.codec == "nf4" and not d14.overflow, d14.reason)


def test_auto_refuses_vq_after_3b_canary() -> None:
    """H2-1: 32B auto on 12 GB is NF4 overflow, never VQ."""
    d = decide(QWEN_32B, DEFAULT_VRAM_MIB, requested="auto")
    check("32B auto is nf4", d.codec == "nf4" and d.codec != "vq", d.codec)
    check("32B auto overflow True", d.overflow, d.reason)
    check(
        "32B auto reason overflow H2",
        "overflow" in d.reason and "H2" in d.reason,
        d.reason,
    )


def test_force_nf4_on_32b_refuses() -> None:
    d = decide(QWEN_32B, DEFAULT_VRAM_MIB, requested="nf4")
    check("32B --codec nf4 is nf4", d.codec == "nf4", d.codec)
    check("32B --codec nf4 overflow True", d.overflow, d.reason)


def test_3b_auto_overflow_on_tiny_vram() -> None:
    """Canary without a 32B file: 3B on a card that holds VQ but not NF4."""
    full = budget_for(QWEN_3B, DEFAULT_VRAM_MIB)
    tiny = int(full.nf4_mib + RUNTIME_OVERHEAD_MIB) - 1
    b = budget_for(QWEN_3B, tiny)
    check("3B tiny leftover_nf4 < 0", b.leftover_nf4 < 0, f"{b.leftover_nf4:.1f}")
    check("3B tiny leftover_vq >= 0", b.leftover_vq >= 0, f"{b.leftover_vq:.1f}")
    d = decide(QWEN_3B, tiny, requested="auto")
    check("3B auto overflow True", d.codec == "nf4" and d.overflow, d.reason)
    check(
        "3B overflow reason H2",
        "overflow" in d.reason and "H2" in d.reason,
        d.reason,
    )


def test_force_vq_on_3b_allowed() -> None:
    d = decide(QWEN_3B, DEFAULT_VRAM_MIB, requested="vq")
    check("3B --codec vq allowed", d.codec == "vq" and not d.overflow, d.reason)


def test_32b_on_24gb_stays_nf4() -> None:
    d = decide(QWEN_32B, 24576, requested="auto")
    check("32B on 24 GB is nf4", d.codec == "nf4" and not d.overflow, d.reason)


def test_detect_vram_reads_nvidia_smi() -> None:
    from gpu.cli import codec as codec_mod

    original_smi = codec_mod._smi_total_mib
    original_torch = codec_mod._torch_vram_mib
    try:
        codec_mod._smi_total_mib = lambda: 24576  # type: ignore[assignment]
        codec_mod._torch_vram_mib = lambda: 8192  # type: ignore[assignment]
        check("smi 24 GB wins over torch", codec_mod.detect_vram_mib() == 24576, "")

        codec_mod._smi_total_mib = lambda: None  # type: ignore[assignment]
        check("torch fallback 8 GB", codec_mod.detect_vram_mib() == 8192, "")

        codec_mod._torch_vram_mib = lambda: None  # type: ignore[assignment]
        check("no probe falls back", codec_mod.detect_vram_mib() == DEFAULT_VRAM_MIB, "")
    finally:
        codec_mod._smi_total_mib = original_smi  # type: ignore[assignment]
        codec_mod._torch_vram_mib = original_torch  # type: ignore[assignment]


def test_70b_refuses_even_vq() -> None:
    cfg = dict(QWEN_32B)
    cfg["num_hidden_layers"] = 80
    cfg["hidden_size"] = 8192
    cfg["intermediate_size"] = 28672
    cfg["num_attention_heads"] = 64
    cfg["vocab_size"] = 128256
    try:
        decide(cfg, DEFAULT_VRAM_MIB, requested="auto")
    except CodecFitError as exc:
        check("70B-class auto raises", "2-bit" in str(exc) or "VQ" in str(exc), str(exc)[:80])
        return
    check("70B-class auto raises", False, "no error")


TESTS = [
    test_14b_param_count_matches_vram_doc,
    test_14b_nf4_fits_32b_does_not,
    test_auto_picks_nf4_when_it_fits,
    test_auto_refuses_vq_after_3b_canary,
    test_force_nf4_on_32b_refuses,
    test_3b_auto_overflow_on_tiny_vram,
    test_force_vq_on_3b_allowed,
    test_32b_on_24gb_stays_nf4,
    test_detect_vram_reads_nvidia_smi,
    test_70b_refuses_even_vq,
]


def main() -> int:
    print("gpu/cli/codec acceptance, no GPU\n")
    for fn in TESTS:
        fn()
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Decode V2 VRAM estimate before ``--executor auto``. CPU only.

    python gpu/decodev2/test_budget.py
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.decodev2.budget import (  # noqa: E402
    GRAPH_MIB,
    WDDM_RESERVE_MIB,
    estimate_v2_from_config,
    estimate_v2_mib,
)
from gpu.decodev2.session import decodev2_refused, pick_executor  # noqa: E402
from gpu.tests.skips import Skip  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []
SKIPPED: list[tuple[str, str]] = []

# InternLM2.5-20B shapes; packed MiB is the codec-fit number on the 3080 plate.
CFG_20B = {
    "hidden_size": 6144,
    "intermediate_size": 16384,
    "num_hidden_layers": 48,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "vocab_size": 92544,
    "head_dim": 128,
}
PACKED_20B_MIB = 10062.0
CARD_MIB = 12288
PLAN_20B = SimpleNamespace(family="internlm_gqa")
REPORT_20B = SimpleNamespace(overflow=False, codec="nf4", device_mib=PACKED_20B_MIB)


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def test_20b_packed_saves_about_a_gib() -> None:
    packed = estimate_v2_from_config(
        PACKED_20B_MIB, CFG_20B, max_seq=2048, packed_embed=True
    )
    dense = estimate_v2_from_config(
        PACKED_20B_MIB, CFG_20B, max_seq=2048, packed_embed=False
    )
    assert dense.dense_embed_extra_mib > 1000
    assert dense.total_mib - packed.total_mib == dense.dense_embed_extra_mib
    assert packed.graph_mib == GRAPH_MIB
    assert packed.wddm_reserve_mib == WDDM_RESERVE_MIB
    assert packed.kv_mib > 300
    assert packed.total_mib < CARD_MIB


def test_pick_executor_uses_estimate() -> None:
    packed = estimate_v2_from_config(
        PACKED_20B_MIB, CFG_20B, max_seq=2048, packed_embed=True
    )
    tight = int(packed.total_mib) - 1
    chosen, why = pick_executor(
        "auto",
        REPORT_20B,
        PLAN_20B,
        max_seq=2048,
        vram_mib=tight,
        config=CFG_20B,
    )
    assert chosen == "tokenloop" and why is not None and "estimated" in why
    chosen, why = pick_executor(
        "auto",
        REPORT_20B,
        PLAN_20B,
        max_seq=2048,
        vram_mib=CARD_MIB,
        config=CFG_20B,
    )
    assert chosen == "decodev2" and why is None
    chosen, why = pick_executor(
        "auto",
        REPORT_20B,
        PLAN_20B,
        max_seq=2048,
        vram_mib=CARD_MIB,
        config=CFG_20B,
        packed_embed=False,
    )
    assert chosen == "tokenloop" and why is not None and "dense embed" in why


def test_forced_decodev2_keeps_reason() -> None:
    packed = estimate_v2_mib(
        packed_mib=PACKED_20B_MIB,
        n_layers=48,
        n_q=48,
        n_kv=8,
        head_dim=128,
        hidden=6144,
        intermediate=16384,
        vocab=92544,
        max_seq=2048,
        packed_embed=True,
    )
    chosen, why = pick_executor(
        "decodev2",
        REPORT_20B,
        PLAN_20B,
        max_seq=2048,
        vram_mib=int(packed.total_mib) - 1,
        config=CFG_20B,
    )
    assert chosen == "decodev2" and why is not None


def test_budget_skipped_without_vram() -> None:
    assert decodev2_refused(REPORT_20B, PLAN_20B) is None
    assert pick_executor("auto", REPORT_20B, PLAN_20B) == ("decodev2", None)


TESTS = [
    value
    for name, value in sorted(globals().items())
    if name.startswith("test_") and callable(value)
]


def main_runner() -> int:
    print(f"gpu/decodev2 budget, {len(TESTS)} tests\n")
    for fn in TESTS:
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                fn()
        except Skip as exc:
            SKIPPED.append((fn.__name__, str(exc)))
            print(f"  SKIP  {fn.__name__}  -- {exc}")
        except AssertionError as exc:
            check(fn.__name__, False, str(exc).splitlines()[0] if str(exc) else "assert")
        except Exception as exc:  # pragma: no cover
            check(fn.__name__, False, f"{type(exc).__name__}: {exc}")
        else:
            check(fn.__name__, True)

    failed = [name for name, ok, _ in CHECKS if not ok]
    tail = f", {len(SKIPPED)} skipped" if SKIPPED else ""
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed{tail}")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main_runner())

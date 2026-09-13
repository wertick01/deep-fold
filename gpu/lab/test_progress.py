"""CPU tests for the progress plate. Never loads a model, never touches CUDA.

    python -m gpu.lab.test_progress
    python gpu/lab/test_progress.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def _gate(name: str, body: Callable[[], tuple[bool, str]]) -> None:
    try:
        ok, detail = body()
    except Exception as error:  # noqa: BLE001
        check(name, False, f"{type(error).__name__}: {error}")
        return
    check(name, ok, detail)


def _is_text(artist: Any) -> bool:
    from matplotlib.text import Text

    return isinstance(artist, Text)


def _plate_text(figure: Any) -> str:
    return "\n".join(artist.get_text() for artist in figure.findobj(match=_is_text))


def gate_no_torch() -> None:
    """gpu.nf4.__init__ imports torch. This plate must not."""
    before = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
    from gpu.lab import progress_plate as plate

    after = {name for name in sys.modules if name == "torch" or name.startswith("torch.")}
    leaked = after - before
    check("importing progress_plate does not import torch", not leaked, str(sorted(leaked)))
    source = Path(plate.__file__).read_text(encoding="utf-8")
    check(
        "progress_plate does not import gpu.nf4 or torch",
        "import torch" not in source
        and "import gpu.nf4" not in source
        and "from gpu.nf4" not in source,
        plate.__file__,
    )


def gate_tokens() -> None:
    from gpu.lab.hard_plate import CODEC_STYLE as HARD_STYLE
    from gpu.lab.hard_plate import INK, INK_SOFT, LIMIT
    from gpu.lab.progress_plate import CODEC_STYLE, INK as P_INK
    from gpu.lab.progress_plate import INK_SOFT as P_SOFT
    from gpu.lab.progress_plate import LIMIT as P_LIMIT
    from gpu.lab.progress_plate import (
        FIX_KV_CTAS,
        FIX_QO_CTAS,
        LIVE_MAX_N,
        SMS_3080,
        STARVED_KV_CTAS,
        STARVED_QO_CTAS,
    )

    for codec in ("bf16", "nf4"):
        check(
            f"{codec} colour matches hard_plate",
            CODEC_STYLE[codec].color == HARD_STYLE[codec].color,
            CODEC_STYLE[codec].color,
        )
    check("nf4 hatch is the second encoding", CODEC_STYLE["nf4"].hatch == "///", CODEC_STYLE["nf4"].hatch)
    check(
        "ink tokens match hard_plate",
        (P_INK, P_SOFT, P_LIMIT) == (INK, INK_SOFT, LIMIT),
        f"{P_INK}/{P_SOFT}/{P_LIMIT}",
    )
    check(
        "starved grid is 16 and 2 against 70 SMs",
        (STARVED_QO_CTAS, STARVED_KV_CTAS, SMS_3080) == (16, 2, 70),
        f"{STARVED_QO_CTAS}/{STARVED_KV_CTAS}/{SMS_3080}",
    )
    check(
        "occupancy-fix grid is 128 and 64, LIVE_MAX_N=16",
        (FIX_QO_CTAS, FIX_KV_CTAS, LIVE_MAX_N) == (128, 64, 16),
        f"{FIX_QO_CTAS}/{FIX_KV_CTAS}/{LIVE_MAX_N}",
    )


def gate_load_committed() -> None:
    from gpu.lab.progress_plate import load_progress, default_paths

    story = load_progress(default_paths(_REPO))
    bf16 = story.committed_3b.codecs["bf16"]
    nf4 = story.committed_3b.codecs["nf4"]
    check(
        "committed 3B BF16 is 23.1 tok/s / 52 ms",
        abs(float(bf16.tok_s) - 23.1482) < 1e-3 and abs(float(bf16.ttft_ms) - 52.0049) < 1e-2,
        f"{bf16.tok_s} / {bf16.ttft_ms}",
    )
    check(
        "committed 3B NF4 is 16.9953 tok/s / 212 ms (starved, not WAVE 2)",
        abs(float(nf4.tok_s) - 16.9953) < 1e-3 and abs(float(nf4.ttft_ms) - 212.422) < 1e-2,
        f"{nf4.tok_s} / {nf4.ttft_ms}",
    )
    check(
        "committed 3B smi after load is 7477 / 3142",
        abs(float(bf16.smi_after_mib) - 7477) < 1 and abs(float(nf4.smi_after_mib) - 3142) < 1,
        f"{bf16.smi_after_mib} / {nf4.smi_after_mib}",
    )
    ws_bf16 = story.fit_14b.codecs["bf16"].working_set_mib
    ws_nf4 = story.fit_14b.codecs["nf4"].working_set_mib
    check(
        "14B working set is ~28270 vs ~7539, not the smi cap",
        ws_bf16 is not None
        and ws_nf4 is not None
        and abs(ws_bf16 - 28269.6) < 1
        and abs(ws_nf4 - 7539.01) < 1,
        f"{ws_bf16} / {ws_nf4}",
    )
    tok_14 = (
        story.fit_14b.codecs["bf16"].tok_s,
        story.fit_14b.codecs["nf4"].tok_s,
    )
    check(
        "14B decode is 0.92 vs 6.56",
        tok_14[0] is not None
        and tok_14[1] is not None
        and abs(tok_14[0] - 0.922588) < 1e-3
        and abs(tok_14[1] - 6.56418) < 1e-3,
        str(tok_14),
    )
    twenty_bf16 = story.internlm_20b.codecs["bf16"]
    twenty_nf4 = story.internlm_20b.codecs["nf4"]
    check(
        "20B BF16 generate is empty (recorded miss)",
        twenty_bf16.tok_s is None and twenty_bf16.ttft_ms is None,
        str(twenty_bf16.tok_s),
    )
    check(
        "20B NF4 is 5.01 tok/s, working set 10273 vs 37882",
        twenty_nf4.tok_s is not None
        and abs(twenty_nf4.tok_s - 5.01495) < 1e-3
        and abs(float(twenty_nf4.working_set_mib) - 10273.3) < 1
        and abs(float(twenty_bf16.working_set_mib) - 37882.2) < 1,
        f"{twenty_nf4.tok_s} / {twenty_nf4.working_set_mib} / {twenty_bf16.working_set_mib}",
    )
    labels = {(run.size, run.codec, run.n_ok, run.n_items) for run in story.hard}
    check(
        "hard eval is 7/12, 9/12, 8/12, 10/12",
        labels
        == {
            ("3B", "bf16", 7, 12),
            ("14B", "bf16", 9, 12),
            ("3B", "nf4", 8, 12),
            ("14B", "nf4", 10, 12),
        },
        str(labels),
    )
    qn1 = next(case for case in story.ncu if case.case.endswith("q_proj-n1"))
    qn16 = next(case for case in story.ncu if case.case.endswith("q_proj-n16"))
    check(
        "ncu q_proj decode DRAM ~5%, tensor ~1.4%, occupancy ~16%",
        abs(float(qn1.dram_pct) - 5.12) < 0.05
        and abs(float(qn1.tensor_pct) - 1.38) < 0.05
        and abs(float(qn1.occupancy_pct) - 15.73) < 0.05,
        f"{qn1.dram_pct}/{qn1.tensor_pct}/{qn1.occupancy_pct}",
    )
    check(
        "ncu q_proj N=16 occupancy ~29%",
        abs(float(qn16.occupancy_pct) - 28.8) < 0.05,
        str(qn16.occupancy_pct),
    )
    check(
        "committed competitors have no invented tok/s",
        story.competitor.n_measured == 0 and story.competitor.n_skip == story.competitor.n_stacks,
        f"{story.competitor.n_skip}/{story.competitor.n_stacks}",
    )
    check(
        "live bnb e2e is quoted from the CSV or left as SKIP — never invented",
        "invented" not in story.competitor.bnb_note.lower()
        and (
            "not on disk" in story.competitor.bnb_note.lower()
            or "still skip" in story.competitor.bnb_note.lower()
            or "live measured" in story.competitor.bnb_note.lower()
        ),
        story.competitor.bnb_note[:160],
    )
    if story.competitor.live_tok_s is not None:
        check(
            "live bitsandbytes-nf4 is 22.8 tok/s / 57 ms, smi 4442, from the e2e CSV",
            abs(story.competitor.live_tok_s - 22.7678) < 1e-3
            and story.competitor.live_ttft_ms is not None
            and abs(story.competitor.live_ttft_ms - 56.9505) < 0.05
            and story.competitor.live_smi is not None
            and abs(story.competitor.live_smi - 4442) < 1
            and story.competitor.live_smoke is True,
            f"{story.competitor.live_tok_s}/{story.competitor.live_ttft_ms}/{story.competitor.live_smi}",
        )
        check(
            "live bitsandbytes kernel microbench is 0/63 SKIP",
            story.competitor.micro_n == 63 and story.competitor.micro_skip == 63,
            f"{story.competitor.micro_skip}/{story.competitor.micro_n}",
        )


def gate_optional_live() -> None:
    from gpu.lab.progress_plate import load_progress, default_paths

    story = load_progress(default_paths(_REPO))
    if story.occupancy_nf4.present:
        nf4 = story.occupancy_nf4.codecs.get("nf4")
        check(
            "WAVE 2 occupancy NF4 is unpaired 31.6 tok/s / 167 ms",
            nf4 is not None
            and nf4.tok_s is not None
            and abs(nf4.tok_s - 31.5665) < 1e-3
            and abs(float(nf4.ttft_ms) - 167.264) < 1e-2,
            f"{None if nf4 is None else nf4.tok_s}",
        )
        check(
            "WAVE 2 occupancy folder has no BF16 row",
            "bf16" not in story.occupancy_nf4.codecs,
            str(list(story.occupancy_nf4.codecs)),
        )
    else:
        check(
            "WAVE 2 occupancy dir absent (gap, not a remembered 31.6)",
            True,
            story.occupancy_nf4.note[:80],
        )
    if story.paired_3b.present:
        bf16 = story.paired_3b.codecs["bf16"]
        nf4 = story.paired_3b.codecs["nf4"]
        check(
            "honest pair is 24.3 vs 28.4 tok/s, 45 vs 139 ms",
            abs(float(bf16.tok_s) - 24.3366) < 1e-3
            and abs(float(nf4.tok_s) - 28.3752) < 1e-3
            and abs(float(bf16.ttft_ms) - 44.73) < 0.05
            and abs(float(nf4.ttft_ms) - 139.017) < 0.05,
            f"{bf16.tok_s}/{nf4.tok_s} {bf16.ttft_ms}/{nf4.ttft_ms}",
        )
        check(
            "honest pair peaks are 8910 / 4036, weights 5886 / 1563",
            abs(float(bf16.smi_peak_mib) - 8910) < 1
            and abs(float(nf4.smi_peak_mib) - 4036) < 1
            and abs(float(bf16.weight_mib) - 5885.96) < 0.1
            and abs(float(nf4.weight_mib) - 1563.34) < 0.1,
            f"{bf16.smi_peak_mib}/{nf4.smi_peak_mib}",
        )
    else:
        check("honest pair dir absent (gap)", True, story.paired_3b.note[:80])
    if story.gsm8k.present:
        check(
            "GSM8K slice is 84/200 vs 97/200",
            story.gsm8k.n_ok.get("bf16") == 84
            and story.gsm8k.n_ok.get("nf4") == 97
            and story.gsm8k.n_items == 200
            and story.gsm8k.corpus_n == 1319,
            f"{story.gsm8k.n_ok} n={story.gsm8k.n_items}",
        )
        check(
            "GSM8K disagreements are 45",
            story.gsm8k.disagreements == 45,
            str(story.gsm8k.disagreements),
        )
        bf16 = story.gsm8k.codecs.get("bf16")
        nf4 = story.gsm8k.codecs.get("nf4")
        check(
            "GSM8K speed is ~25.46 vs ~32.15 tok/s, TTFT ~45.4 vs ~311, peaks 8795 / 4616",
            bf16 is not None
            and nf4 is not None
            and abs(float(bf16.tok_s) - 25.4594) < 1e-3
            and abs(float(nf4.tok_s) - 32.1525) < 1e-3
            and abs(float(bf16.ttft_ms) - 45.3623) < 0.05
            and abs(float(nf4.ttft_ms) - 310.627) < 0.05
            and abs(float(bf16.smi_peak_mib) - 8795) < 1
            and abs(float(nf4.smi_peak_mib) - 4616) < 1,
            "speeds",
        )
        check(
            "GSM8K quality_all_ok is false on both codecs",
            bf16.quality_all_ok is False and nf4.quality_all_ok is False,
            f"{bf16.quality_all_ok}/{nf4.quality_all_ok}",
        )
    else:
        check("GSM8K dir absent (gap, not a headline)", True, story.gsm8k.note[:80])
    if story.internlm_hard is not None:
        check(
            "20B hard is live NF4 8/12, no BF16 pair",
            story.internlm_hard.codec == "nf4"
            and story.internlm_hard.n_ok == 8
            and story.internlm_hard.n_items == 12
            and story.internlm_hard.size == "20B",
            f"{story.internlm_hard.n_ok}/{story.internlm_hard.n_items}",
        )
    else:
        check("20B hard dir absent (gap, not a remembered 8/12)", True, "no internlm hard CSV")
    if story.n32.present:
        check(
            "true n32 is ~1.63× faster than two n16 on q_proj",
            story.n32.true_n32_us is not None
            and story.n32.two_n16_us is not None
            and story.n32.ratio is not None
            and abs(story.n32.true_n32_us - 69.02) < 0.05
            and abs(story.n32.two_n16_us - 112.66) < 0.05
            and abs(story.n32.ratio - 1.632) < 0.01
            and "prefill_n32" in story.n32.n32_kernel
            and "prefill_n16" in story.n32.two_kernel,
            f"{story.n32.true_n32_us}/{story.n32.two_n16_us}/{story.n32.ratio}",
        )
    else:
        check("n32 comparison dir absent (gap)", True, "no comparison.csv")


def gate_render() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    from gpu.lab.progress_plate import load_progress, progress_plate, write_plate, default_paths

    story = load_progress(default_paths(_REPO))
    figure = progress_plate(story)
    with tempfile.TemporaryDirectory(prefix="progress-plate-") as tmp:
        path = write_plate(figure, Path(tmp) / "progress.png")[0]
        check(
            "plate renders a PNG without kaleido",
            path.is_file() and path.stat().st_size > 40_000,
            f"{path.stat().st_size / 1024:.0f} KiB",
        )
    text = _plate_text(figure)
    check("committed 3B plate is named", "docs/runs/qwen25-3b" in text or "lab-qwen25-3b" in text, "3B plate")
    check("17.0 and 23.1 both appear as labeled committed numbers", "23.1" in text and "17.0" in text, "17/23.1")
    if story.occupancy_nf4.present:
        check(
            "31.6 is labeled unpaired / occupancy, not mixed into the pair",
            "31.6" in text
            and (
                "unpaired" in text.lower()
                or "not a same-session" in text.lower()
                or "Occupancy fix" in text
            ),
            "31.6",
        )
    if story.paired_3b.present:
        check("28.4 and 24.3 appear on the honest pair", "28.4" in text and "24.3" in text, "pair")
        check("pair is not claimed vs Marlin/AWQ/bnb", "Marlin" in text or "not vs Marlin" in text, "disclaimer")
    check("14B working-set caveat is on the plate", "11,955" in text or "11,955 vs 8,913" in text, "smi caveat")
    check("card limit 12288 is drawn as text", "12,288" in text, "12288")
    check("hard 7/12 is on the plate", "7/12" in text, "7/12")
    check("hard plate is named", "hard-eval-qwen25" in text, "hard plate name")
    if story.internlm_hard is not None:
        check(
            "20B NF4 8/12 is labeled as live, not a BF16 pair",
            "20B NF4 8/12" in text and "no BF16" in text,
            "20B hard",
        )
    if story.n32.present:
        check(
            "n32 vs two n16 is on footer F and TokenLoop stays 16",
            "True n32" in text and "1.63" in text and "chunks at 16" in text,
            "n32 footer",
        )
    check(
        "occupancy labels sit outside the bars",
        "q/o starved" in text and "q/o after fix" in text and "k/v after fix" in text,
        "cta ticks",
    )
    check("committed competitor matrix stays SKIP", "SKIP" in text, "SKIP")
    if story.competitor.live_tok_s is not None:
        check(
            "live bitsandbytes 22.8 is on the plate and not ranked as a kernel win",
            "22.8" in text and "not a kernel ranking" in text.lower(),
            "22.8",
        )
    check("stack claim is not a 4-bit engine win", "prior art" in text.lower() or "The claim is the stack" in text, "stack")
    if story.gsm8k.present:
        check("GSM8K 84/200 and 97/200 are on the plate", "84/200" in text and "97/200" in text, "gsm8k frac")
        check("GSM8K 1319 disclaimer is on the plate", "1,319" in text or "1319" in text, "1319")
        check("GSM8K is not headlined as quality", "not a GSM8K quality" in text or "not a published" in text.lower() or "quality headline" in text.lower(), "headline")
    import matplotlib.pyplot as plt

    plt.close(figure)


def gate_cli() -> None:
    from gpu.lab.progress_plate import _parser

    args = _parser().parse_args(["--redraw", "--out", "extra.png"])
    check("--redraw is a flag", args.redraw is True, str(args.redraw))
    check("--out is repeatable", args.out == ["extra.png"], str(args.out))


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print("gpu/lab progress plate (CSV only, no GPU)\n")
    gate_no_torch()
    gate_tokens()
    gate_load_committed()
    gate_optional_live()
    gate_render()
    gate_cli()
    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

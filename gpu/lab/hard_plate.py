"""The hard-eval summary plate: four runs of bars, plus the fixture's answer table.

``gpu.lab.plot`` draws the smoke plate (Plotly, one model, two codecs). This is
the *other* plate: one matplotlib sheet for the whole 3B/14B x BF16/NF4 matrix,
written at the bottom of notebook 06. Same visual language as the smoke plate --
one typeface, the same two Okabe-Ito codec colours, a hatch as the second
encoding, and a card-limit rule in red -- but matplotlib, so a PNG falls out
without kaleido or a headless Chrome.

The sheet reads as two blocks:

    A  CUDA working set ..... what torch actually holds. 14B BF16 is ~28 GiB on
                             a 12 GiB card; the part above the rule is Windows
                             shared GPU memory (system RAM).
    B  nvidia-smi peak ...... dedicated VRAM only, and it stops at 12288 MiB.
                             Capped, so it is not the compression win.
    C  mean TTFT ............ prefill per item.
    D  decode tok/s ......... HF dense GEMM against the fused NF4 loop.
    E  hard accuracy ........ auto-scored items of the hard set, per run.

    the answer table ....... every fixture item: the question, the gold, and
                             what each run actually answered.

The table is the part that has to work before the GPU does: it renders from
``gpu/lab/data/hard_items.json`` alone, and the run columns fill in as the
per-run ``hard_scores.csv`` files appear.

    from gpu.lab.hard import load_fixture, runs_from_matrix
    from gpu.lab.hard_plate import hard_plate, write_plate

    items = load_fixture()["independent"]
    fig = hard_plate(items, runs_from_matrix(run_root))
    write_plate(fig, r"docs\\img\\hard-eval-qwen25.png")
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .hard import RUN_ORDER, HardItem, HardRun, answer_rows, pad_runs

__all__ = [
    "CARD_MIB",
    "CODEC_STYLE",
    "PANELS",
    "hard_plate",
    "write_plate",
]

CARD_MIB = 12288.0

# --------------------------------------------------------------------------- #
# design tokens -- mirrored from gpu.lab.plot so the two plates match.
# gpu.lab.test_hard asserts the colours are still identical, so this copy
# cannot drift; importing plot here would drag Plotly into a matplotlib sheet.
# --------------------------------------------------------------------------- #

FONT = ["Arial", "Helvetica", "DejaVu Sans"]

INK = "#151515"
INK_SOFT = "#5B5B5B"
RULE = "#9A9A9A"
GRID = "#E9E9E9"
LIMIT = "#8C1D18"  # the only red: the card limit and bad news


@dataclass(frozen=True)
class _Codec:
    label: str
    color: str
    hatch: str


CODEC_STYLE: dict[str, _Codec] = {
    "bf16": _Codec("BF16 dense", "#E69F00", ""),
    "nf4": _Codec("NF4 driver", "#0072B2", "///"),
}

SIZE_TITLE = 15.0
SIZE_SUB = 10.5
SIZE_CLAIM = 11.0
SIZE_NOTE = 8.6
SIZE_BODY = 9.6
SIZE_SMALL = 8.4

# Panel letter, HardRun attribute, heading, unit note, value format, card rule.
PANELS: tuple[tuple[str, str, str, str, str, bool], ...] = (
    (
        "A",
        "working_set_mib",
        "CUDA working set",
        "torch, MiB — above the rule is shared GPU memory",
        "{:,.0f}",
        True,
    ),
    (
        "B",
        "smi_peak_mib",
        "nvidia-smi peak",
        "dedicated VRAM, MiB — this meter stops at the card",
        "{:,.0f}",
        True,
    ),
    (
        "C",
        "mean_ttft_ms",
        "mean TTFT",
        "prefill per item, ms — lower is better",
        "{:,.0f}",
        False,
    ),
    (
        "D",
        "mean_tok_s",
        "decode throughput",
        "tokens/s — dense GEMM vs fused NF4 loop",
        "{:,.1f}",
        False,
    ),
    (
        "E",
        "accuracy_pct",
        "hard accuracy",
        "% of auto-scored hard items — not Paris/Berlin/323",
        "{:.0f}%",
        False,
    ),
)

# Answer-table verdict -> (colour, weight). A miss is the plate's red; nothing
# is encoded by colour alone, the cell also spells out "ok" / "miss".
_STATE_INK: dict[str, tuple[str, str]] = {
    "ok": (INK, "normal"),
    "miss": (LIMIT, "bold"),
    "human": (INK_SOFT, "normal"),
    "none": (RULE, "normal"),
}

# --------------------------------------------------------------------------- #
# geometry, in inches
# --------------------------------------------------------------------------- #

FIG_W = 16.0
_M_LEFT, _M_RIGHT = 0.62, 0.34
_HEADER_H = 1.36  # title, subtitle, provenance band
_PANEL_H = 2.55
_PANEL_GAP = 0.62
_CLAIM_H = 0.60  # room above the panels for their letter, heading and unit note
_TABLE_HEAD = 0.78  # answer-table claim line plus its column headers
_ROW_H = 0.235
_TABLE_PAD = 0.30
_CAPTION_H = 1.55  # three wrapped paragraphs of provenance

# Answer table column starts, in axes fraction. `question` is the wide one.
_COL_N = 0.004
_COL_ITEM = 0.032
_COL_QUESTION = 0.152
_COL_GOLD = 0.556
_COL_RUNS = 0.636


def _fmt(value: float | None, pattern: str) -> str:
    return "n/a" if value is None else pattern.format(float(value))


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def _title_for(runs: Sequence[HardRun]) -> str:
    titles: list[str] = []
    for run in runs:
        if run.title not in titles:
            titles.append(run.title)
    models = " and ".join(titles) if titles else "Qwen2.5-3B-Instruct and Qwen2.5-14B-Instruct"
    return f"Hard eval on one RTX 3080 12 GB — {models}, dense BF16 against the NF4 driver"


def _subtitle_for(runs: Sequence[HardRun], n_items: int) -> str:
    if not runs:
        return (
            f"{n_items} fixture items with their gold answers. No live session in this "
            "plate yet — run the four cells above and redraw."
        )
    order = " → ".join(run.label for run in runs)
    ran = sum(1 for run in runs if run.ran)
    pending = sum(1 for run in runs if run.pending)
    tail = (
        f", {pending} still queued behind the card"
        if pending
        else ", greedy, one isolated child process per run so the card never holds two models"
    )
    return (
        f"{order}. {ran} of {len(runs)} runs produced replies over {n_items} hard items{tail}."
    )


def _band_for(runs: Sequence[HardRun]) -> tuple[str, str, str]:
    """Provenance band: measured, in progress, partial, or fixture-only."""
    ran = [run for run in runs if run.ran]
    if not runs or not ran:
        return (
            "FIXTURE QUESTIONS ONLY",
            "the questions and golds below are the committed fixture; no run has "
            "reported a reply yet, so every live cell is a dash, not a zero",
            "#FBECEA",
        )
    pending = [run for run in runs if run.pending]
    if pending:
        queued = ", ".join(run.label for run in pending)
        return (
            "RUN IN PROGRESS",
            f"{queued} has not reported yet — those bars say \"not run yet\" and those "
            "answer cells are dashes; redraw from hard_matrix.csv when the queue finishes",
            "#FDF3E3",
        )
    if len(ran) < len(runs):
        missing = ", ".join(run.label for run in runs if not run.ran)
        return (
            "PARTIAL RUN",
            f"{missing} recorded no replies — see notes in hard_matrix.csv; those bars "
            "and cells are blank, not zero",
            "#FDF3E3",
        )
    return (
        "MEASURED RUN",
        "nvidia-smi and torch.cuda sampled throughout every session; numbers are as "
        "recorded, with nothing corrected or subtracted",
        "#F2F4F5",
    )


CAPTION_PARAGRAPHS = (
    "Sources: hard_matrix.csv (one row per run), plus hard_scores.csv and summary.csv in each "
    "<lab>-<codec>/ directory; questions and golds from gpu/lab/data/hard_items.json — the item id carries "
    "the kind (gsm8k / trap / yesno / needle / open).",
    "Answer cells hold this run's extracted answer against the gold: \"ok\" matched, \"miss\" did not or "
    "there was no reply, \"human\" is an open item that two people have to rate and is excluded from panel E, "
    "and \"-\" means no session at all — a skip, an OOM, or not run yet. Never a zero.",
    "Panel A is torch.cuda; panel B is nvidia-smi, which reports dedicated VRAM only and stops at 12,288 MiB. "
    "14B BF16 does not fit: the remainder is Windows shared GPU memory (system RAM), so the two nvidia-smi "
    "peaks are not the compression win. Panels C and D are different stacks — HuggingFace generate against "
    "CompressedLinear + TokenLoop — and on 3B, where both fit, dense BF16 decodes faster.",
)

_CAPTION_WRAP = 214


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #


def _letter(ax: Any, letter: str, heading: str, *, y: float = 1.175) -> None:
    """``A   CUDA working set`` -- bold letter, then the claim, in axes fraction."""
    ax.text(
        0.0,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=SIZE_CLAIM,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.078,
        y,
        heading,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=SIZE_CLAIM,
        color=INK,
    )


def _style_panel(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=3, color=RULE, labelsize=SIZE_SMALL, labelcolor=INK_SOFT)
    ax.tick_params(axis="x", length=0, labelsize=SIZE_SMALL, labelcolor=INK)


def _bar_panel(
    ax: Any,
    runs: Sequence[HardRun],
    *,
    letter: str,
    key: str,
    heading: str,
    note: str,
    pattern: str,
    card: bool,
    total_mib: float,
) -> None:
    _style_panel(ax)
    # The claim band above the panel: bold letter, heading, then the unit note
    # in a softer, smaller face. Drawn by hand so the letters of all five
    # panels stay on one line however the notes wrap.
    _letter(ax, letter, heading)
    ax.text(
        0.0,
        1.055,
        textwrap.fill(note, width=42),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=SIZE_NOTE,
        color=INK_SOFT,
        linespacing=1.35,
    )

    if not runs:
        ax.text(
            0.5,
            0.5,
            "no live session yet",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=SIZE_BODY,
            color=INK_SOFT,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    values = [getattr(run, key) for run in runs]
    present = [float(value) for value in values if value is not None]
    if key == "accuracy_pct":
        ceiling = 100.0
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 25, 50, 75, 100])
    else:
        ceiling = max(present + ([total_mib] if card else []) + [1.0])
        ax.set_ylim(0, ceiling * 1.30)

    for index, (run, value) in enumerate(zip(runs, values)):
        style = CODEC_STYLE.get(run.codec, CODEC_STYLE["bf16"])
        if value is None:
            # A pending slot is the queue, not a failure: grey and quiet. A run
            # that reported nothing is the plate's red. Neither is ever a zero.
            ax.text(
                index,
                ceiling * 0.04,
                "not run\nyet" if run.pending else "no\ndata",
                ha="center",
                va="bottom",
                fontsize=SIZE_SMALL,
                color=RULE if run.pending else (LIMIT if run.notes else INK_SOFT),
                linespacing=1.2,
            )
            continue
        ax.bar(
            index,
            float(value),
            width=0.62,
            color=style.color,
            edgecolor=style.color,
            linewidth=0.8,
            hatch=style.hatch,
        )
        ax.annotate(
            _fmt(value, pattern),
            (index, float(value)),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=SIZE_SMALL,
            color=INK,
        )

    if card:
        from matplotlib.ticker import StrMethodFormatter

        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.axhline(total_mib, color=LIMIT, linewidth=1.1, linestyle=(0, (2, 2)))
        ax.annotate(
            f"card {total_mib:,.0f} MiB",
            (len(runs) - 0.45, total_mib),
            textcoords="offset points",
            xytext=(0, 3),
            ha="right",
            fontsize=SIZE_SMALL,
            color=LIMIT,
        )

    ax.set_xlim(-0.62, len(runs) - 0.38)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"{run.size}\n{run.codec.upper()}" for run in runs])
    for tick, run in zip(ax.get_xticklabels(), runs):
        tick.set_color(
            RULE if run.pending else CODEC_STYLE.get(run.codec, CODEC_STYLE["bf16"]).color
        )


# --------------------------------------------------------------------------- #
# the answer table
# --------------------------------------------------------------------------- #


def _answer_table(
    ax: Any,
    items: Sequence[HardItem],
    runs: Sequence[HardRun],
    *,
    height_in: float,
    letter: str = "F",
) -> None:
    """Every fixture item: question, gold, and what each run answered.

    Laid out in inches (converted to axes fraction) so the row pitch is the
    same whether the script is the 12 independent items or the 4 history turns.
    """
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    rows = answer_rows(items, runs)
    labels = [run.label for run in runs]
    n_runs = max(1, len(labels))
    step = (1.0 - _COL_RUNS) / n_runs
    run_x = [_COL_RUNS + index * step for index in range(n_runs)]

    verdicts = [cell for row in rows for cell in row["cells"].values()]
    correct = sum(1 for cell in verdicts if cell.state == "ok")
    graded = sum(1 for cell in verdicts if cell.state in ("ok", "miss"))
    if graded:
        claim = (
            f"The questions and the answers: {correct} of {graded} graded answers across "
            f"{len(runs)} run(s) matched the gold."
        )
    else:
        claim = (
            "The questions and the answers: these are the committed fixture golds. "
            "No run has reported an answer yet."
        )

    def frac(inches: float) -> float:
        return inches / height_in

    ax.text(
        _COL_N,
        1.0,
        letter,
        ha="left",
        va="top",
        fontsize=SIZE_CLAIM,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        _COL_N + 0.016,
        1.0,
        claim,
        ha="left",
        va="top",
        fontsize=SIZE_CLAIM,
        color=INK if graded else INK_SOFT,
    )

    n_rows = len(rows)
    rule_top = 1.0 - frac(0.34)
    header_y = rule_top - frac(0.155)
    rule_head = rule_top - frac(0.30)
    row_step = frac(_ROW_H)

    def rule(y: float, width: float, color: str) -> None:
        ax.plot([0, 1], [y, y], color=color, linewidth=width, clip_on=False)

    rule(rule_top, 1.1, INK)
    rule(rule_head, 0.8, RULE)
    rule(rule_head - row_step * n_rows, 1.1, INK)

    def cell(
        x: float,
        y: float,
        text: str,
        *,
        color: str = INK,
        size: float = SIZE_BODY,
        weight: str = "normal",
    ) -> None:
        ax.text(x, y, text, ha="left", va="center", fontsize=size, color=color, fontweight=weight)

    head = header_y
    cell(_COL_N, head, "#", color=INK_SOFT, size=SIZE_SMALL)
    cell(_COL_ITEM, head, "item", color=INK_SOFT, size=SIZE_SMALL)
    cell(
        _COL_QUESTION,
        head,
        "question (identical for every run; full text in hard_script.json)",
        color=INK_SOFT,
        size=SIZE_SMALL,
    )
    cell(_COL_GOLD, head, "gold", color=INK_SOFT, size=SIZE_SMALL)
    if labels:
        for x, run in zip(run_x, runs):
            style = CODEC_STYLE.get(run.codec, CODEC_STYLE["bf16"])
            cell(
                x,
                head,
                run.label if not run.pending else f"{run.label} (queued)",
                color=RULE if run.pending else style.color,
                size=SIZE_SMALL,
                weight="normal" if run.pending else "bold",
            )
    else:
        cell(run_x[0], head, "live answer", color=RULE, size=SIZE_SMALL)

    for index, row in enumerate(rows):
        y = rule_head - row_step * (index + 0.5)
        cell(_COL_N, y, str(row["n"]), color=INK_SOFT, size=SIZE_SMALL)
        cell(_COL_ITEM, y, _clip(row["item"], 20), color=INK_SOFT, size=SIZE_SMALL)
        cell(_COL_QUESTION, y, _clip(row["question"], 74))
        cell(_COL_GOLD, y, _clip(row["gold"], 12), weight="bold")
        if not labels:
            cell(run_x[0], y, "-", color=RULE)
            continue
        for x, label in zip(run_x, labels):
            verdict = row["cells"][label]
            color, weight = _STATE_INK.get(verdict.state, (INK, "normal"))
            cell(x, y, verdict.text, color=color, size=SIZE_SMALL, weight=weight)


# --------------------------------------------------------------------------- #
# the sheet
# --------------------------------------------------------------------------- #


def hard_plate(
    items: Sequence[HardItem],
    runs: Sequence[HardRun] = (),
    *,
    total_mib: float = CARD_MIB,
    title: str | None = None,
    subtitle: str | None = None,
    fig_width: float = FIG_W,
    order: Sequence[tuple[str, str]] | None = RUN_ORDER,
) -> Any:
    """The summary sheet for notebook 06. Draws with or without live runs.

    ``items`` is the fixture script that was (or will be) asked -- the answer
    table is built from it, so the sheet is meaningful before the first worker
    has started. ``runs`` are :class:`gpu.lab.hard.HardRun` rows, in run order.

    As soon as there is one live run, the panels are padded out to ``order``
    (:data:`gpu.lab.hard.RUN_ORDER`) so all four ticks are present and the
    pairs still queued read *not run yet* instead of quietly vanishing --
    otherwise a plate drawn mid-matrix looks like a finished two-run
    comparison. Pass ``order=None`` to draw exactly the runs given.
    """
    import matplotlib.pyplot as plt

    items = list(items)
    runs = list(runs)
    if runs and order:
        runs = pad_runs(runs, order=order)
    n_rows = max(len(items), 1)

    table_h = _TABLE_HEAD + _ROW_H * n_rows + _TABLE_PAD
    fig_h = (
        _HEADER_H + _PANEL_H + _PANEL_GAP + _CLAIM_H + table_h + _CAPTION_H
    )

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": RULE,
            "text.color": INK,
            "axes.labelcolor": INK_SOFT,
            "savefig.facecolor": "#FFFFFF",
        }
    ):
        fig = plt.figure(figsize=(fig_width, fig_h), dpi=110)

        plot_w = fig_width - _M_LEFT - _M_RIGHT
        n_panels = len(PANELS)
        gap = 0.46
        panel_w = (plot_w - gap * (n_panels - 1)) / n_panels
        panel_bottom = fig_h - _HEADER_H - _PANEL_H - _CLAIM_H

        for index, (letter, key, heading, note, pattern, card) in enumerate(PANELS):
            left = _M_LEFT + index * (panel_w + gap)
            ax = fig.add_axes(
                [
                    left / fig_width,
                    panel_bottom / fig_h,
                    panel_w / fig_width,
                    _PANEL_H / fig_h,
                ]
            )
            _bar_panel(
                ax,
                runs,
                letter=letter,
                key=key,
                heading=heading,
                note=note,
                pattern=pattern,
                card=card,
                total_mib=total_mib,
            )

        ax_table = fig.add_axes(
            [
                _M_LEFT / fig_width,
                _CAPTION_H / fig_h,
                plot_w / fig_width,
                table_h / fig_h,
            ]
        )
        _answer_table(ax_table, items, runs, height_in=table_h)

        # --- header -------------------------------------------------------
        x0 = _M_LEFT / fig_width
        fig.text(
            x0,
            1.0 - 0.34 / fig_h,
            title or _title_for(runs),
            ha="left",
            va="top",
            fontsize=SIZE_TITLE,
            color=INK,
        )
        fig.text(
            x0,
            1.0 - 0.62 / fig_h,
            subtitle or _subtitle_for(runs, len(items)),
            ha="left",
            va="top",
            fontsize=SIZE_SUB,
            color=INK_SOFT,
        )
        band_label, band_text, band_fill = _band_for(runs)
        band_ink = LIMIT if band_fill == "#FBECEA" else INK_SOFT
        band_h = 0.30
        band_bottom = fig_h - _HEADER_H + 0.16
        fig.add_artist(
            plt.Rectangle(
                (x0, band_bottom / fig_h),
                plot_w / fig_width,
                band_h / fig_h,
                transform=fig.transFigure,
                facecolor=band_fill,
                edgecolor=band_ink,
                linewidth=0.7,
                zorder=0,
            )
        )
        fig.text(
            x0 + 0.006,
            (band_bottom + band_h / 2) / fig_h,
            f"{band_label} — {band_text}",
            ha="left",
            va="center",
            fontsize=SIZE_NOTE,
            color=band_ink,
        )

        caption = "\n".join(
            textwrap.fill(paragraph, width=_CAPTION_WRAP)
            for paragraph in CAPTION_PARAGRAPHS
        )
        fig.text(
            x0,
            (_CAPTION_H - 0.20) / fig_h,
            caption,
            ha="left",
            va="top",
            fontsize=SIZE_NOTE,
            color=INK_SOFT,
            linespacing=1.5,
        )
    return fig


def write_plate(figure: Any, *paths: str | Path, dpi: float = 140.0) -> list[Path]:
    """Save the sheet to every path given. PNG needs no kaleido and no browser."""
    written: list[Path] = []
    for path in paths:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(dest, dpi=dpi, facecolor="#FFFFFF")
        written.append(dest)
    return written

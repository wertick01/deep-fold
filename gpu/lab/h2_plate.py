"""H2 overflow plate: Qwen2.5-32B NF4 on a 12 GB card.

Same visual language as :mod:`gpu.lab.hard_plate` / :mod:`gpu.lab.stack_plate`
(Okabe–Ito NF4, ink, the card-limit red). Matplotlib Agg. No torch, no GPU.

    python -m gpu.lab.h2_plate --redraw

Writes ``docs/img/h2-qwen25-32b.png``. Numbers are a WAVE freeze of the product
smoke ``C:\\dev\\models\\runs\\h2-qwen25-32b-20260914-234048`` (outside git).
This sheet is the overflow path, not a tok/s leaderboard and not a 14B/20B pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from .hard_plate import (
    CODEC_STYLE,
    FONT,
    GRID,
    INK,
    INK_SOFT,
    LIMIT,
    RULE,
    write_plate,
)

__all__ = [
    "CARD_MIB",
    "DECODE_TOK_S",
    "PACKED_MIB",
    "RESIDENT_MIB",
    "STREAMED_MIB",
    "default_png",
    "h2_plate",
    "redraw",
]

_REPO = Path(__file__).resolve().parents[2]

# WAVE freeze — product CLI ``python -m gpu.lab.h2_trace --no-timing``.
# Packed from decide() (16599), not the 4.5-bit paper figure 17577.
PACKED_MIB = 16599.0
RESIDENT_MIB = 9716.0
STREAMED_MIB = 6885.0
SLOT_MIB = 71.72
CARD_MIB = 12288.0
SMI_AFTER_LOAD = 11268.0
SMI_PEAK = 11933.0
TORCH_ALLOC_LOAD = 9933.0
COPY_FLOOR_MS = 277.3
DECODE_TOK_S = 2.31
TTFT_MS = 1006.0
MS_PER_TOK = 432.4
COPY_FLOOR_TOK_S = 3.6
PAGEABLE_BUG_TOK_S = 0.8
STREAMED_MATRICES = 96
KV_MIB = 512.0
MAX_SEQ = 2048
PINNED_GB_S = 24.3
LIVE_MAX_N = 32

BF16 = CODEC_STYLE["bf16"].color
NF4 = CODEC_STYLE["nf4"].color

PAPER = "#FFFFFF"
BAND = "#F2F4F5"
CHIP = "#F4F5F6"
BF16_FILL = "#FDF3E3"
NF4_FILL = "#E7F1F8"
LIMIT_FILL = "#FBECEA"
HOST_FILL = "#F4F5F6"

MONO = ["Consolas", "DejaVu Sans Mono", "Courier New"]

FIG_W = 13.40
_M_LEFT = 0.28
_M_RIGHT = 0.22
_M_TOP = 0.16
_M_BOTTOM = 0.16

_T_TITLE = 13.4
_T_SUB = 8.6
_T_HEAD = 9.4
_T_CLAIM = 8.5
_T_BODY = 7.6
_T_SMALL = 7.15
_T_TINY = 7.05
_T_CHIP = 7.2


def default_png(repo: str | Path | None = None) -> Path:
    root = Path(repo) if repo else _REPO
    return root / "docs" / "img" / "h2-qwen25-32b.png"


class _Sheet:
    def __init__(self, fig: Any, fig_h: float) -> None:
        self.fig = fig
        self.h = fig_h

    def x(self, xin: float) -> float:
        return xin / FIG_W

    def y(self, yin_from_top: float) -> float:
        return 1.0 - yin_from_top / self.h

    def w(self, win: float) -> float:
        return win / FIG_W

    def hh(self, hin: float) -> float:
        return hin / self.h

    def rect(
        self,
        xin: float,
        yin: float,
        win: float,
        hin: float,
        *,
        fc: str = PAPER,
        ec: str = RULE,
        lw: float = 0.7,
        z: int = 1,
        hatch: str = "",
        alpha: float = 1.0,
    ) -> None:
        self.fig.add_artist(
            Rectangle(
                (self.x(xin), self.y(yin + hin)),
                self.w(win),
                self.hh(hin),
                transform=self.fig.transFigure,
                facecolor=fc,
                edgecolor=ec,
                linewidth=lw,
                hatch=hatch,
                alpha=alpha,
                zorder=z,
                clip_on=False,
            )
        )

    def text(
        self,
        xin: float,
        yin: float,
        text: str,
        *,
        fs: float,
        color: str = INK,
        ha: str = "left",
        va: str = "top",
        weight: str = "normal",
        family: Sequence[str] | None = None,
        z: int = 4,
        spacing: float = 1.22,
    ) -> None:
        self.fig.text(
            self.x(xin),
            self.y(yin),
            text,
            ha=ha,
            va=va,
            fontsize=fs,
            color=color,
            fontweight=weight,
            fontfamily=list(family) if family is not None else FONT,
            transform=self.fig.transFigure,
            zorder=z,
            linespacing=spacing,
        )

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = INK,
        lw: float = 0.95,
    ) -> None:
        self.fig.add_artist(
            FancyArrowPatch(
                (self.x(x1), self.y(y1)),
                (self.x(x2), self.y(y2)),
                transform=self.fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=9.5,
                linewidth=lw,
                color=color,
                zorder=5,
                shrinkA=0.5,
                shrinkB=0.5,
            )
        )

    def chip(
        self,
        xin: float,
        yin: float,
        win: float,
        hin: float,
        label: str,
        *,
        fc: str = CHIP,
        ec: str = RULE,
        tc: str = INK,
        fs: float | None = None,
        weight: str = "normal",
        hatch: str = "",
        family: Sequence[str] | None = None,
    ) -> None:
        self.rect(xin, yin, win, hin, fc=fc, ec=ec, lw=0.65, hatch=hatch, z=2)
        self.text(
            xin + win / 2,
            yin + hin / 2,
            label,
            fs=fs if fs is not None else _T_CHIP,
            color=tc,
            ha="center",
            va="center",
            weight=weight,
            family=family,
        )


def _fig_h() -> float:
    return (
        _M_TOP
        + 1.12
        + 0.08
        + 3.22
        + 0.10
        + 2.62
        + 0.10
        + 3.55
        + 0.10
        + 1.32
        + _M_BOTTOM
    )


def h2_plate() -> Any:
    fig_h = _fig_h()
    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor=PAPER, dpi=105.0)
    s = _Sheet(fig, fig_h)
    y = _M_TOP
    y = _header(s, y)
    y += 0.08
    y = _memory(s, y)
    y += 0.10
    y = _speed(s, y)
    y += 0.10
    y = _path(s, y)
    y += 0.10
    _footer(s, y)
    return fig


def _header(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    s.rect(x0, y, 0.10, 1.12, fc=NF4, ec=NF4, lw=0.0, z=2)
    s.text(x0 + 0.22, y + 0.08, "Qwen2.5-32B-Instruct  ·  NF4 overflow (H2)", fs=_T_TITLE, weight="bold")
    s.text(
        x0 + 0.22,
        y + 0.38,
        "RTX 3080 12 GB  ·  Ampere sm_86  ·  WDDM (display occupies VRAM)  ·  greedy smoke Paris / Berlin / 323",
        fs=_T_SUB,
        color=INK_SOFT,
    )
    s.text(
        x0 + 0.22,
        y + 0.62,
        "Packed NF4 is 16,599 MiB and does not fit. Resident packed weights 9,716 MiB; a pinned host tail of "
        "6,885 MiB (96 matrices) is copied one matrix at a time into two static slots.\n"
        "Dequant only in registers of chr_nf4_gemm. No dense [M,K] BF16 in HBM. Not VQ. Not a 14B/20B pair.",
        fs=_T_BODY,
        color=INK,
        spacing=1.28,
    )
    s.rect(x0 + w - 2.55, y + 0.12, 2.42, 0.88, fc=NF4_FILL, ec=NF4, lw=0.8)
    s.text(x0 + w - 1.34, y + 0.22, "product decode", fs=_T_TINY, color=INK_SOFT, ha="center")
    s.text(
        x0 + w - 1.34,
        y + 0.42,
        f"{DECODE_TOK_S:.2f} tok/s",
        fs=12.4,
        color=NF4,
        ha="center",
        weight="bold",
        family=MONO,
    )
    s.text(x0 + w - 1.34, y + 0.72, f"TTFT {TTFT_MS:.0f} ms  ·  smoke 3/3", fs=_T_TINY, color=INK, ha="center")
    return y + 1.12


def _panel_frame(s: _Sheet, y: float, h: float, letter: str, title: str) -> tuple[float, float, float]:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    s.rect(x0, y, w, h, fc=PAPER, ec=RULE, lw=0.7)
    s.rect(x0, y, 0.36, h, fc=BAND, ec=RULE, lw=0.55)
    s.text(x0 + 0.18, y + 0.18, letter, fs=11.0, weight="bold", ha="center", color=INK)
    s.text(x0 + 0.50, y + 0.14, title, fs=_T_HEAD, weight="bold")
    return x0, w, h


def _memory(s: _Sheet, y: float) -> float:
    h = 3.22
    x0, w, _ = _panel_frame(s, y, h, "A", "Where 16.6 GiB of packed NF4 go  ·  MiB")
    inner_x = x0 + 0.50
    label_w = 2.55
    bar_x = inner_x + label_w
    bar_w = w - label_w - 2.05
    xmax = 18000.0
    card_frac = CARD_MIB / xmax
    rule_x = bar_x + bar_w * card_frac

    s.text(
        inner_x,
        y + 0.42,
        "The card line is 12,288 MiB. Packed NF4 crosses it. Resident HBM weights do not.\n"
        "nvidia-smi includes the Windows desktop; leftover is not an empty card.",
        fs=_T_SMALL,
        color=INK_SOFT,
        spacing=1.25,
    )

    rows = [
        ("Packed NF4 (would not fit)", PACKED_MIB, NF4, NF4_FILL, "///", "4.25 bit  ·  decide()"),
        ("Resident HBM weights", RESIDENT_MIB, NF4, NF4_FILL, "", "report.device_mib  ·  not 16,601"),
        ("Streamed host tail", STREAMED_MIB, INK, HOST_FILL, "", f"{STREAMED_MATRICES} matrices  ·  pinned"),
        ("nvidia-smi peak (decode)", SMI_PEAK, LIMIT, LIMIT_FILL, "", "dedicated VRAM  ·  desktop inside"),
    ]
    row_y = y + 0.92
    row_h = 0.42
    gap = 0.10
    for label, value, ec, fc, hatch, note in rows:
        s.text(inner_x, row_y + 0.04, label, fs=_T_SMALL, weight="bold")
        s.text(inner_x, row_y + 0.22, note, fs=_T_TINY, color=INK_SOFT)
        bw = bar_w * min(value / xmax, 1.0)
        s.rect(bar_x, row_y, bar_w, row_h, fc="#F7F7F7", ec=GRID, lw=0.4)
        s.rect(bar_x, row_y, bw, row_h, fc=fc, ec=ec, lw=0.7, hatch=hatch)
        s.text(
            bar_x + bar_w + 0.12,
            row_y + row_h / 2,
            f"{value:,.0f}",
            fs=_T_CLAIM,
            weight="bold",
            va="center",
            family=MONO,
            color=ec if ec == LIMIT else INK,
        )
        row_y += row_h + gap

    s.rect(rule_x - 0.015, y + 0.86, 0.03, 2.08, fc=LIMIT, ec=LIMIT, lw=0.0, z=6)
    s.text(rule_x, y + 0.78, "card 12,288", fs=_T_TINY, color=LIMIT, ha="center", weight="bold")

    s.text(
        inner_x,
        y + h - 0.28,
        f"After load: nvidia-smi {SMI_AFTER_LOAD:,.0f}  ·  torch allocated {TORCH_ALLOC_LOAD:,.0f}  ·  "
        f"KV {KV_MIB:.0f} MiB at max_seq={MAX_SEQ}  ·  slot {SLOT_MIB:.2f} MiB × 2",
        fs=_T_TINY,
        color=INK,
    )
    return y + h


def _speed(s: _Sheet, y: float) -> float:
    h = 2.62
    x0, w, _ = _panel_frame(s, y, h, "B", "Decode tok/s  ·  not a kernel ranking")
    inner_x = x0 + 0.50
    label_w = 3.35
    bar_x = inner_x + label_w
    bar_w = w - label_w - 1.85
    xmax = 4.5

    s.text(
        inner_x,
        y + 0.42,
        "Floor to beat: 1–2 tok/s (“ship the whole packed tail over PCIe every token”). "
        "Ceiling ≈ 3 if those weights were already in HBM.\n"
        "10 tok/s is not a claim. No BF16 32B generate (does not fit). Hard-12 not run.",
        fs=_T_SMALL,
        color=INK_SOFT,
        spacing=1.25,
    )

    rows = [
        (
            "Pageable H2D (pin bug)",
            PAGEABLE_BUG_TOK_S,
            INK_SOFT,
            HOST_FILL,
            "",
            "diagnostic  ·  not the product",
        ),
        (
            "Product chat  (--no-timing)",
            DECODE_TOK_S,
            NF4,
            NF4_FILL,
            "///",
            f"{MS_PER_TOK:.0f} ms/tok  ·  CopyRing + TokenLoop",
        ),
        (
            "Serial copy floor",
            COPY_FLOOR_TOK_S,
            LIMIT,
            LIMIT_FILL,
            "",
            f"{COPY_FLOOR_MS:.1f} ms if wall = copy  ·  not measured wall",
        ),
    ]
    row_y = y + 0.92
    row_h = 0.40
    gap = 0.12
    for label, value, ec, fc, hatch, note in rows:
        s.text(inner_x, row_y + 0.02, label, fs=_T_SMALL, weight="bold")
        s.text(inner_x, row_y + 0.20, note, fs=_T_TINY, color=INK_SOFT)
        bw = bar_w * min(value / xmax, 1.0)
        s.rect(bar_x, row_y, bar_w, row_h, fc="#F7F7F7", ec=GRID, lw=0.4)
        s.rect(bar_x, row_y, max(bw, 0.04), row_h, fc=fc, ec=ec, lw=0.7, hatch=hatch)
        shown = f"{value:.1f}" if value == COPY_FLOOR_TOK_S else f"{value:.2f}"
        s.text(
            bar_x + bar_w + 0.12,
            row_y + row_h / 2,
            shown,
            fs=_T_CLAIM,
            weight="bold",
            va="center",
            family=MONO,
            color=ec,
        )
        row_y += row_h + gap

    floor_x = bar_x + bar_w * (1.5 / xmax)
    s.rect(floor_x - 0.012, y + 0.88, 0.024, 1.48, fc=RULE, ec=RULE, lw=0.0, z=5)
    s.text(floor_x, y + h - 0.22, "1–2 floor", fs=_T_TINY, color=INK_SOFT, ha="center")
    return y + h


def _path(s: _Sheet, y: float) -> float:
    h = 3.55
    x0, w, _ = _panel_frame(s, y, h, "C", "Data path  ·  policy D  ·  CopyRing")
    inner_x = x0 + 0.50
    s.text(
        inner_x,
        y + 0.42,
        "Pinned host image of overflow NF4 (packed + scale, one arena per matrix). "
        "One copy_stream. Prefetch depth 1. WDDM: timing copy events + CPU join.",
        fs=_T_SMALL,
        color=INK_SOFT,
    )

    chip_y = y + 0.72
    chip_h = 0.48
    chips = [
        (2.15, "pinned HostImage\n6,885 MiB", HOST_FILL, INK),
        (2.00, "copy_stream\n24.3 GB/s pin", NF4_FILL, NF4),
        (2.15, "two static slots\n71.72 MiB each", NF4_FILL, NF4),
        (2.55, "chr_nf4_gemm\nregisters only", NF4_FILL, NF4),
    ]
    cx = inner_x
    last = len(chips) - 1
    for i, (width, label, fc, tc) in enumerate(chips):
        s.chip(cx, chip_y, width, chip_h, label, fc=fc, tc=tc, fs=_T_TINY, weight="bold")
        if i != last:
            nx = cx + width + 0.28
            s.arrow(
                cx + width + 0.04,
                chip_y + chip_h / 2,
                nx - 0.04,
                chip_y + chip_h / 2,
                color=INK,
                lw=0.9,
            )
            cx = nx
        else:
            cx = cx + width

    map_y = y + 1.38
    s.text(inner_x, map_y, "Policy D residency (64 layers)", fs=_T_CLAIM, weight="bold")

    def _row(ry: float, name: str, device: str, host: str, note: str, host_hot: bool) -> None:
        s.text(inner_x, ry + 0.08, name, fs=_T_SMALL, weight="bold", family=MONO)
        s.chip(inner_x + 1.55, ry, 3.55, 0.32, device, fc=NF4_FILL, ec=NF4, tc=NF4, fs=_T_TINY, weight="bold")
        host_ec = INK if host_hot else RULE
        s.chip(
            inner_x + 5.22,
            ry,
            3.55,
            0.32,
            host,
            fc=HOST_FILL,
            ec=host_ec,
            tc=INK,
            fs=_T_TINY,
            weight="bold",
        )
        s.text(inner_x + 8.90, ry + 0.08, note, fs=_T_TINY, color=INK_SOFT)

    _row(map_y + 0.28, "q k v o", "DEVICE  ·  all 64", "HOST  ·  none", "CUDA graphs when captured", False)
    _row(map_y + 0.68, "gate / up", "DEVICE  ·  L0–47", "HOST  ·  L48–63", "tail FFN pairs streamed", True)
    _row(map_y + 1.08, "down", "DEVICE  ·  none", "HOST  ·  all 64", "every layer, every token", True)
    _row(map_y + 1.48, "embed / head", "DEVICE  ·  both", "HOST  ·  none", "lm_head prefetch on N=1", False)

    s.text(
        inner_x,
        y + h - 0.20,
        f"LIVE_MAX_N = {LIVE_MAX_N}  ·  prefill: the overflow tape once per chunk, not per column  ·  "
        "177/257 groups graphed (DEVICE); HOST eager  ·  auto never picks VQ",
        fs=_T_TINY,
        color=INK,
    )
    return y + h


def _footer(s: _Sheet, y: float) -> None:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    h = 1.32
    s.rect(x0, y, w, h, fc=BAND, ec=RULE, lw=0.6)
    s.text(x0 + 0.12, y + 0.10, "Honesty  ·  this sheet", fs=_T_CLAIM, weight="bold")
    s.text(
        x0 + 0.12,
        y + 0.32,
        '32B fits via overflow, not "all weights in HBM". Reconstruct is still tile-local; a dense layer is still never resident.\n'
        "Decode tok/s is CopyRing + TokenLoop + chr_nf4_gemm, not vs Marlin / llama.cpp / BF16 32B. Smoke 3/3 is not quality; hard-12 not started.\n"
        "Pageable ~0.8 tok/s was Tensor.is_pinned used as a bool (always true). Do not quote 0.8 as the H2 design.\n"
        "Serial floor 277 ms ≠ measured wall ~432 ms (WDDM join + incomplete overlap). nvidia-smi is not subtracted. Generate: Ampere sm_86 only.\n"
        "Redraw: python -m gpu.lab.h2_plate --redraw   ·   live dump: C:\\dev\\models\\runs\\h2-qwen25-32b-20260914-234048",
        fs=_T_TINY,
        color=INK,
        spacing=1.32,
    )


def redraw(*extra: str | Path, repo: str | Path | None = None) -> list[Path]:
    """Write the PNG. No GPU, no torch."""
    figure = h2_plate()
    targets: list[Path] = [default_png(repo)]
    for path in extra:
        if path:
            dest = Path(path)
            if dest not in targets:
                targets.append(dest)
    written = write_plate(figure, *targets, dpi=105.0)
    plt.close(figure)
    return written


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gpu.lab.h2_plate",
        description="Redraw docs/img/h2-qwen25-32b.png. Matplotlib only, no GPU.",
    )
    parser.add_argument("--redraw", action="store_true", help="write docs/img/h2-qwen25-32b.png")
    parser.add_argument("--out", action="append", default=[], help="extra PNG path (repeatable)")
    parser.add_argument("--repo", default="", help="repository root (default: this tree)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.redraw and not args.out:
        parser.error("pass --redraw and/or --out PATH")
    extra = [Path(item) for item in args.out]
    written = redraw(*extra, repo=args.repo or None)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

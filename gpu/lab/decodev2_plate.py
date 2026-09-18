"""Decode V2 resident plate: 3B / 14B / 20B on one RTX 3080.

Same visual language as :mod:`gpu.lab.compare_plate`. Matplotlib Agg. No torch.

    python -m gpu.lab.decodev2_plate --redraw

Writes ``docs/img/decodev2-3080.png``. WAVE freeze of
``docs/decode-v2-lab.md`` (2026-09-18, max_seq=2048). 32B is still CopyRing —
not this sheet.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .compare_plate import _hbar
from .h2_plate import (
    BAND,
    FIG_W,
    MONO,
    NF4,
    NF4_FILL,
    PAPER,
    _Sheet,
)
from .h2_plate import _T_CLAIM as T_CLAIM
from .h2_plate import _T_HEAD as T_HEAD
from .h2_plate import _T_SMALL as T_SMALL
from .h2_plate import _T_SUB as T_SUB
from .h2_plate import _T_TINY as T_TINY
from .h2_plate import _T_TITLE as T_TITLE
from .hard_plate import FONT, GRID, INK, INK_SOFT, RULE, write_plate

__all__ = [
    "V2_3B_HOST",
    "V2_14B_HOST",
    "V2_20B_HOST",
    "OLLAMA_14B_LONG",
    "LLAMA_14B_LONG",
    "OLLAMA_20B_LONG",
    "LLAMA_20B_LONG",
    "OLLAMA_3B_HARD_TOK",
    "OLLAMA_14B_HARD",
    "OLLAMA_14B_HARD_TOK",
    "OLLAMA_20B_HARD_TOK",
    "COMING_SOON",
    "decodev2_plate",
    "default_png",
    "redraw",
]

_REPO = Path(__file__).resolve().parents[2]

# WAVE freeze — exclusive ignore-EOS 64, max_seq=2048, match Q4_K ctx 2048.
# Host 196.956 / 57.495 / 40.025. V2 still attends the full buffer.
V2_3B_HOST = 197.0
V2_3B_DEVICE = 198.9
V2_3B_PREFILL_MS = 86.0
V2_14B_HOST = 57.5
V2_14B_DEVICE = 58.1
V2_14B_PREFILL_MS = 322.0
V2_20B_HOST = 40.0
V2_20B_PREFILL_MS = 220.0
# Second 64-token pass at VRAM cap. Not the decode number (512-era 25.6; 2048-era 20.7).
V2_20B_DEVICE_SECOND = 25.6
V2_20B_DEVICE_SECOND_2048 = 20.7

# 2026-09-17 max_seq=512 ignore-EOS (kept in the footer, not the bars).
V2_3B_HOST_512 = 196.0
V2_14B_HOST_512 = 55.6
V2_20B_HOST_512 = 41.0

TL_3B_LONG = 35.2
TL_3B_PREFILL_SAME_MS = 86.0
TL_14B = 6.56
TL_14B_PREFILL_MS = 759.0
TL_20B = 5.01
TL_20B_LONG = 4.59
TL_20B_PREFILL_MS = 605.0

OLLAMA_3B = 187.3
LLAMA_3B = 187.0
# Exclusive 2026-09-18. ctx 2048 travelogue. Overlapping Ollama 14B 5.95 is not a number.
OLLAMA_14B_LONG = 58.9
LLAMA_14B_LONG = 69.9
OLLAMA_20B_LONG = 11.53
LLAMA_20B_LONG = 11.87

# Independent hard-12, max_new=256, max_seq=2048. Not ignore-EOS 64.
# 20B packed NF4 rows (2026-09-18 reheat): mean 38.96 → 39.0; item-1 434 ms.
V2_3B_HARD = "8/12"
V2_3B_HARD_TOK = 190.9
V2_14B_HARD = "11/12"
V2_14B_HARD_TOK = 54.8
V2_20B_HARD = "9/12"
V2_20B_HARD_TOK = 39.0
# Ollama hard-12 mean decode_tok_s over the same 12 items (plate.json).
# 3B mean 197.2 vs median 184.5 (logic-yesno is 2 tokens at 330).
OLLAMA_3B_HARD = "7/12"
OLLAMA_3B_HARD_TOK = 197.2
OLLAMA_3B_HARD_MEDIAN = 184.5
# Exclusive 2026-09-18 qwen2.5:14b. Mean 66.23 → 66.2; median 62.0 (yesno 2 tok).
OLLAMA_14B_HARD = "10/12"
OLLAMA_14B_HARD_TOK = 66.2
OLLAMA_14B_HARD_MEDIAN = 62.0
OLLAMA_20B_HARD = "9/12"
OLLAMA_20B_HARD_TOK = 13.2
OLLAMA_20B_HARD_MEDIAN = 11.6
OLLAMA_32B_HARD = "11/12"
OLLAMA_32B_HARD_TOK = 3.1
# TokenLoop MMA hard-12 (old resident path / 32B CopyRing). Not Decode V2.
TL_3B_HARD = "8/12"
TL_3B_HARD_TOK = 16.9
TL_14B_HARD = "10/12"
TL_14B_HARD_TOK = 6.59
TL_20B_HARD = "8/12"
TL_20B_HARD_TOK = 4.4
TL_32B_HARD = "12/12"
TL_32B_HARD_TOK = 2.12

COMING_SOON = "Coming soon"

OLLAMA = "#009E73"
LLAMA = "#E69F00"
LOOP = "#5B5B5B"

_M_LEFT = 0.28
_M_RIGHT = 0.22
_M_TOP = 0.16
_M_BOTTOM = 0.16


def default_png(repo: str | Path | None = None) -> Path:
    root = Path(repo) if repo else _REPO
    return root / "docs" / "img" / "decodev2-3080.png"


def _fig_h() -> float:
    return (
        _M_TOP
        + 0.92
        + 0.08
        + 0.78
        + 0.10
        + 2.55
        + 0.10
        + 2.90
        + 0.10
        + 1.72
        + 0.10
        + 2.22
        + _M_BOTTOM
    )


def _header(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    h = 0.92
    s.rect(x0, y, w, h, fc=PAPER, ec=RULE, lw=0.7)
    s.text(x0 + 0.14, y + 0.10, "Decode V2 on one RTX 3080 12 GB", fs=T_TITLE, weight="bold")
    s.text(
        x0 + 0.14,
        y + 0.42,
        "Greedy  ·  V2 max_seq 2048  ·  ignore-EOS 64  ·  CHR0 NF4 GEMV graph + MMA prefill chunks of 32\n"
        "CLI --executor auto on resident 3B/14B/20B. Same buffer size as Q4_K ctx 2048; V2 still attends the full axis.\n"
        "Hard-12 tok/s vs size is on this sheet (256 new). 3B Q4_K ~187 lives on compare-3080.png.",
        fs=T_SUB,
        color=INK_SOFT,
        spacing=1.28,
    )
    return y + h


def _stats(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    gap = 0.10
    w = (FIG_W - _M_LEFT - _M_RIGHT - 2 * gap) / 3.0
    h = 0.78
    cells = (
        (f"{V2_3B_HOST:.0f}", "3B host tok/s", NF4, f"device-window {V2_3B_DEVICE:.0f}  ·  prefill {V2_3B_PREFILL_MS:.0f} ms"),
        (f"{V2_14B_HOST:.1f}", "14B host tok/s", NF4, f"device-window {V2_14B_DEVICE:.1f}  ·  prefill {V2_14B_PREFILL_MS:.0f} ms"),
        (f"{V2_20B_HOST:.0f}", "20B host tok/s", NF4, f"prefill {V2_20B_PREFILL_MS:.0f} ms  ·  exclusive card"),
    )
    for i, (value, label, color, note) in enumerate(cells):
        x = x0 + i * (w + gap)
        s.rect(x, y, w, h, fc=BAND, ec=RULE, lw=0.65)
        s.rect(x, y, 0.08, h, fc=color, ec=color, lw=0.0)
        s.text(x + 0.20, y + 0.08, value, fs=13.0, weight="bold", family=MONO, color=color)
        s.text(x + 0.20, y + 0.38, label, fs=T_SMALL, weight="bold")
        s.text(x + 0.20, y + 0.54, note, fs=T_TINY, color=INK_SOFT)
    return y + h


def _bars(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    gap = 0.12
    col_w = (FIG_W - _M_LEFT - _M_RIGHT - gap) / 2.0
    h = 2.55
    s.rect(x0, y, col_w, h, fc=PAPER, ec=RULE, lw=0.7)
    s.text(x0 + 0.12, y + 0.08, "14B decode, tok/s", fs=T_HEAD, weight="bold")
    s.text(
        x0 + 0.12,
        y + 0.30,
        "Exclusive. V2 max_seq=2048. Q4_K long is ctx 2048.",
        fs=T_TINY,
        color=INK_SOFT,
    )
    rows_14 = (
        (OLLAMA, "Ollama", "Q4_K_M  ·  library tag  ·  ctx 2048", OLLAMA_14B_LONG),
        (LLAMA, "llama.cpp", "Q4_K_M  ·  auto-fit  ·  ctx 2048", LLAMA_14B_LONG),
        (NF4, "Decode V2", "NF4 GEMV  ·  host 57.5", V2_14B_HOST),
        (LOOP, "TokenLoop", "NF4 MMA  ·  product CSV 6.56", TL_14B),
    )
    for i, (color, name, cap, val) in enumerate(rows_14):
        _hbar(s, x0 + 0.12, y + 0.52 + i * 0.48, col_w - 0.20, 0.42, val, 80.0, color, name, cap, label_w=2.42)

    x1 = x0 + col_w + gap
    s.rect(x1, y, col_w, h, fc=PAPER, ec=RULE, lw=0.7)
    s.text(x1 + 0.12, y + 0.08, "20B decode, tok/s", fs=T_HEAD, weight="bold")
    s.text(
        x1 + 0.12,
        y + 0.30,
        "InternLM Q4_K vs NF4. Do not quote 20B 25.6 or 20.7.",
        fs=T_TINY,
        color=INK_SOFT,
    )
    rows_20 = (
        (OLLAMA, "Ollama", "Q4_K_M  ·  ctx 2048", OLLAMA_20B_LONG),
        (LLAMA, "llama.cpp", "Q4_K_M  ·  auto-fit  ·  ctx 2048", LLAMA_20B_LONG),
        (NF4, "Decode V2", "exclusive host 40.0", V2_20B_HOST),
        (LOOP, "TokenLoop", "NF4 MMA  ·  long max_seq 1024", TL_20B_LONG),
    )
    for i, (color, name, cap, val) in enumerate(rows_20):
        _hbar(s, x1 + 0.12, y + 0.52 + i * 0.48, col_w - 0.20, 0.42, val, 50.0, color, name, cap, label_w=2.42)
    return y + h


def _hard(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    h = 2.90
    s.rect(x0, y, w, h, fc=PAPER, ec=RULE, lw=0.7)
    s.text(x0 + 0.12, y + 0.08, "Hard-12  ·  decode tok/s vs model size", fs=T_HEAD, weight="bold")
    s.text(
        x0 + 0.12,
        y + 0.30,
        f"Independent turns  ·  256 new  ·  ctx / max_seq 2048  ·  mean of 12 decode_tok_s. "
        f"32B is CopyRing overflow, not Decode V2. llama.cpp hard-12 is Coming soon.",
        fs=T_TINY,
        color=INK_SOFT,
    )

    resident = (3.0, 14.0, 20.0)
    overflow = (3.0, 14.0, 20.0, 32.0)
    v2 = (V2_3B_HARD_TOK, V2_14B_HARD_TOK, V2_20B_HARD_TOK)
    oll = (OLLAMA_3B_HARD_TOK, OLLAMA_14B_HARD_TOK, OLLAMA_20B_HARD_TOK, OLLAMA_32B_HARD_TOK)
    loop = (TL_3B_HARD_TOK, TL_14B_HARD_TOK, TL_20B_HARD_TOK, TL_32B_HARD_TOK)

    pad_l, pad_r, pad_t, pad_b = 0.82, 0.32, 0.54, 0.50
    ax_w = w - pad_l - pad_r
    ax_h = h - pad_t - pad_b
    ax = s.fig.add_axes(
        [
            (x0 + pad_l) / FIG_W,
            1.0 - (y + pad_t + ax_h) / s.h,
            ax_w / FIG_W,
            ax_h / s.h,
        ]
    )
    ax.set_facecolor(PAPER)
    ax.set_zorder(3)
    ax.plot(overflow, loop, color=LOOP, linewidth=1.05, linestyle=(0, (3.0, 2.4)), marker="^", markersize=5.0, zorder=3, label="TokenLoop MMA")
    ax.plot(overflow, oll, color=OLLAMA, linewidth=1.65, marker="s", markersize=5.5, zorder=4, label="Ollama Q4_K")
    ax.plot(resident, v2, color=NF4, linewidth=1.85, marker="o", markersize=6.0, zorder=5, label="Decode V2")
    ax.set_xlim(1.4, 35.5)
    ax.set_ylim(0.0, 255.0)
    ax.set_xticks(list(overflow))
    ax.set_xticklabels(("3B", "14B", "20B", "32B"))
    ax.set_yticks((0.0, 50.0, 100.0, 150.0, 200.0))
    ax.set_xlabel("model size (parameters)", fontsize=T_TINY, color=INK_SOFT, fontfamily=FONT, labelpad=3.0)
    ax.set_ylabel("hard-12 decode tok/s", fontsize=T_TINY, color=INK_SOFT, fontfamily=FONT, labelpad=4.0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(RULE)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=INK_SOFT, labelsize=T_TINY, length=2.6, width=0.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT)
        label.set_color(INK_SOFT)

    points = (
        (3.0, V2_3B_HARD_TOK, f"{V2_3B_HARD}  ·  {V2_3B_HARD_TOK:.1f}", NF4, 8, -11, "left"),
        (3.0, OLLAMA_3B_HARD_TOK, f"{OLLAMA_3B_HARD}  ·  {OLLAMA_3B_HARD_TOK:.1f}", OLLAMA, 8, 10, "left"),
        (14.0, V2_14B_HARD_TOK, f"{V2_14B_HARD}  ·  {V2_14B_HARD_TOK:.1f}", NF4, 7, -12, "left"),
        (14.0, OLLAMA_14B_HARD_TOK, f"{OLLAMA_14B_HARD}  ·  {OLLAMA_14B_HARD_TOK:.1f}", OLLAMA, 7, 10, "left"),
        (20.0, V2_20B_HARD_TOK, f"{V2_20B_HARD}  ·  {V2_20B_HARD_TOK:.1f}", NF4, 7, 11, "left"),
        (20.0, OLLAMA_20B_HARD_TOK, f"{OLLAMA_20B_HARD}  ·  {OLLAMA_20B_HARD_TOK:.1f}", OLLAMA, 7, 10, "left"),
        (32.0, TL_32B_HARD_TOK, f"{TL_32B_HARD}  ·  {TL_32B_HARD_TOK:.2f}", LOOP, -8, 28, "right"),
        (32.0, OLLAMA_32B_HARD_TOK, f"{OLLAMA_32B_HARD}  ·  {OLLAMA_32B_HARD_TOK:.1f}", OLLAMA, -8, 12, "right"),
    )
    for x, yy, text, color, dx, dy, ha in points:
        ax.annotate(
            text,
            xy=(x, yy),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=T_TINY,
            color=color,
            ha=ha,
            va="center",
            fontfamily=MONO,
            zorder=6,
            annotation_clip=False,
        )

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor=RULE,
        facecolor=PAPER,
        fontsize=T_TINY,
        borderpad=0.45,
        handlelength=2.2,
        labelcolor=INK,
    )
    legend.get_frame().set_linewidth(0.55)
    for text in legend.get_texts():
        text.set_fontfamily(FONT)
    return y + h


def _prefill(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    h = 1.72
    s.rect(x0, y, w, h, fc=NF4_FILL, ec=RULE, lw=0.7)
    s.text(x0 + 0.12, y + 0.08, "Prefill  ·  MMA chunks of 32, not the N=1 GEMV graph", fs=T_HEAD, weight="bold")
    s.text(
        x0 + 0.12,
        y + 0.32,
        "Token-at-a-time V2 prefill is retired (3B was 918 ms). CopyRing / 32B is out of scope.",
        fs=T_TINY,
        color=INK_SOFT,
    )
    cells = (
        ("3B", f"{V2_3B_PREFILL_MS:.0f} ms", f"same-process TokenLoop {TL_3B_PREFILL_SAME_MS:.0f} ms"),
        ("14B", f"{V2_14B_PREFILL_MS:.0f} ms", f"TokenLoop product {TL_14B_PREFILL_MS:.0f} ms"),
        ("20B", f"{V2_20B_PREFILL_MS:.0f} ms", f"TokenLoop product {TL_20B_PREFILL_MS:.0f} ms"),
    )
    gap = 0.10
    cw = (w - 0.24 - 2 * gap) / 3.0
    for i, (name, value, note) in enumerate(cells):
        x = x0 + 0.12 + i * (cw + gap)
        s.rect(x, y + 0.58, cw, 0.92, fc=PAPER, ec=RULE, lw=0.55)
        s.text(x + 0.10, y + 0.66, name, fs=T_SMALL, weight="bold")
        s.text(x + 0.10, y + 0.88, value, fs=12.0, weight="bold", family=MONO, color=NF4)
        s.text(x + 0.10, y + 1.20, note, fs=T_TINY, color=INK_SOFT)
    return y + h


def _footer(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    h = 2.22
    s.rect(x0, y, w, h, fc=BAND, ec=RULE, lw=0.6)
    s.text(x0 + 0.12, y + 0.08, "Honesty  ·  this sheet", fs=T_CLAIM, weight="bold")
    s.text(
        x0 + 0.12,
        y + 0.30,
        "Ignore-EOS 64, V2 max_seq=2048 vs Q4_K ctx 2048. V2 still attends the full buffer; Q4_K attends live length.\n"
        f"3B {V2_3B_HOST:.0f} vs Q4_K ~187 is the same capacity class. TokenLoop MMA 3B product {TL_3B_LONG:.1f} stays the old resident path.\n"
        f"14B: llama.cpp 69.9 / Ollama 58.9 / V2 57.5. Overlapping 5.95 is not a number.\n"
        f"20B Q4_K: Ollama 11.53 / llama.cpp 11.87. Ignore-EOS host {V2_20B_HOST:.1f} (not 25.6 / 20.7). Hard-12 20B mean {V2_20B_HARD_TOK:.1f} is packed NF4 rows, not the 41/40 plateau.\n"
        f"Hard-12 Ollama tok/s is the same 12-item mean: 3B {OLLAMA_3B_HARD_TOK:.1f} (median {OLLAMA_3B_HARD_MEDIAN:.1f}) / 14B {OLLAMA_14B_HARD_TOK:.1f} (median {OLLAMA_14B_HARD_MEDIAN:.1f}) / 20B {OLLAMA_20B_HARD_TOK:.1f} / 32B {OLLAMA_32B_HARD_TOK:.1f}. Table: docs/img/hard-v2-ollama.png.\n"
        f"14B V2 {V2_14B_HARD_TOK:.1f} sits under Ollama {OLLAMA_14B_HARD_TOK:.1f}. 32B hard is CopyRing {TL_32B_HARD} · {TL_32B_HARD_TOK:.2f} vs Ollama {OLLAMA_32B_HARD} · {OLLAMA_32B_HARD_TOK:.1f}, not Decode V2.\n"
        f"2026-09-17 max_seq=512 was {V2_3B_HOST_512:.0f} / {V2_14B_HOST_512:.1f} / {V2_20B_HOST_512:.1f}. Long 32B stays CopyRing 2.49 / 2.53 vs Ollama 2.54.\n"
        f"{COMING_SOON}: llama.cpp hard-12, 3B Nsight 70–85%. CLI TTY / agent chrome: in progress.\n"
        "Overlapping 20B jobs (7–10 tok/s) are WDDM paging. CLI: --executor auto. Redraw: python -m gpu.lab.decodev2_plate --redraw",
        fs=T_TINY,
        color=INK,
        spacing=1.28,
    )
    return y + h


def decodev2_plate() -> Any:
    fig_h = _fig_h()
    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor=PAPER, dpi=105.0)
    s = _Sheet(fig, fig_h)
    y = _M_TOP
    y = _header(s, y)
    y += 0.08
    y = _stats(s, y)
    y += 0.10
    y = _bars(s, y)
    y += 0.10
    y = _hard(s, y)
    y += 0.10
    y = _prefill(s, y)
    y += 0.10
    _footer(s, y)
    return fig


def redraw(*extra: str | Path, repo: str | Path | None = None) -> list[Path]:
    figure = decodev2_plate()
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
        prog="python -m gpu.lab.decodev2_plate",
        description="Redraw docs/img/decodev2-3080.png. Matplotlib only, no GPU.",
    )
    parser.add_argument("--redraw", action="store_true", help="write docs/img/decodev2-3080.png")
    parser.add_argument("--out", action="append", default=[], help="extra PNG path (repeatable)")
    parser.add_argument("--repo", default="", help="repository root (default: this tree)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.redraw and not args.out:
        parser.error("pass --redraw and/or --out PATH")
    extra = [Path(p) for p in args.out]
    written = redraw(*extra, repo=args.repo or None)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

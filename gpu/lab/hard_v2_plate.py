"""Hard-12 table: Decode V2 vs Ollama on one RTX 3080.

Same visual language as :mod:`gpu.lab.decodev2_plate`. Matplotlib Agg. No torch.

    python -m gpu.lab.hard_v2_plate --redraw

Writes ``docs/img/hard-v2-ollama.png``. Numbers are WAVE-frozen in
:mod:`gpu.lab.decodev2_plate`. llama.cpp hard-12 and Ollama 14B stay Coming soon.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .decodev2_plate import (
    COMING_SOON,
    OLLAMA,
    OLLAMA_20B_HARD,
    OLLAMA_20B_HARD_MEDIAN,
    OLLAMA_20B_HARD_TOK,
    OLLAMA_32B_HARD,
    OLLAMA_32B_HARD_TOK,
    OLLAMA_3B_HARD,
    OLLAMA_3B_HARD_MEDIAN,
    OLLAMA_3B_HARD_TOK,
    TL_14B_HARD,
    TL_14B_HARD_TOK,
    TL_20B_HARD,
    TL_20B_HARD_TOK,
    TL_32B_HARD,
    TL_32B_HARD_TOK,
    TL_3B_HARD,
    TL_3B_HARD_TOK,
    V2_14B_HARD,
    V2_14B_HARD_TOK,
    V2_20B_HARD,
    V2_20B_HARD_TOK,
    V2_3B_HARD,
    V2_3B_HARD_TOK,
)
from .h2_plate import (
    BAND,
    FIG_W,
    MONO,
    NF4,
    PAPER,
    _Sheet,
)
from .h2_plate import _T_CLAIM as T_CLAIM
from .h2_plate import _T_HEAD as T_HEAD
from .h2_plate import _T_SMALL as T_SMALL
from .h2_plate import _T_SUB as T_SUB
from .h2_plate import _T_TINY as T_TINY
from .h2_plate import _T_TITLE as T_TITLE
from .hard_plate import INK, INK_SOFT, RULE, write_plate

__all__ = ["hard_v2_plate", "default_png", "redraw"]

_REPO = Path(__file__).resolve().parents[2]

_M_LEFT = 0.28
_M_RIGHT = 0.22
_M_TOP = 0.16
_M_BOTTOM = 0.16


def default_png(repo: str | Path | None = None) -> Path:
    root = Path(repo) if repo else _REPO
    return root / "docs" / "img" / "hard-v2-ollama.png"


def _fig_h() -> float:
    return _M_TOP + 0.92 + 0.10 + 2.95 + 0.10 + 1.72 + _M_BOTTOM


def _header(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    h = 0.92
    s.rect(x0, y, w, h, fc=PAPER, ec=RULE, lw=0.7)
    s.text(x0 + 0.14, y + 0.10, "Hard-12  ·  Decode V2 vs Ollama on one RTX 3080 12 GB", fs=T_TITLE, weight="bold")
    s.text(
        x0 + 0.14,
        y + 0.42,
        "Independent turns  ·  256 new tokens  ·  ctx / max_seq 2048  ·  greedy  ·  mean tok/s over 12 items\n"
        "Decode V2 is CHR0 NF4 GEMV. Ollama 0.34.0 is Q4_K_M. llama.cpp hard-12 is Coming soon.\n"
        "Not the ignore-EOS 64 plateau. 20B V2 first-item TTFT is WDDM, not 40 tok/s.",
        fs=T_SUB,
        color=INK_SOFT,
        spacing=1.28,
    )
    return y + h


def _table(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    h = 2.95
    s.rect(x0, y, w, h, fc=PAPER, ec=RULE, lw=0.7)
    s.text(x0 + 0.12, y + 0.08, "Correct answers and decode tok/s", fs=T_HEAD, weight="bold")
    s.text(
        x0 + 0.12,
        y + 0.28,
        "Same 12-item fixture. tok/s is the mean of per-item decode_tok_s. 14B Ollama is Coming soon.",
        fs=T_TINY,
        color=INK_SOFT,
    )

    pad = 0.12
    gap = 0.04
    model_w = 2.70
    rest = w - 2 * pad - model_w - 4 * gap
    col_w = rest / 4.0
    xs = [
        x0 + pad,
        x0 + pad + model_w + gap,
        x0 + pad + model_w + gap + col_w + gap,
        x0 + pad + model_w + gap + 2 * (col_w + gap),
        x0 + pad + model_w + gap + 3 * (col_w + gap),
    ]
    widths = [model_w, col_w, col_w, col_w, col_w]
    row_h = 0.42
    head_y = y + 0.50
    group_h = 0.32

    s.rect(xs[1], head_y, widths[1] + gap + widths[2], group_h, fc=NF4, ec=NF4, lw=0.0)
    s.rect(xs[3], head_y, widths[3] + gap + widths[4], group_h, fc=OLLAMA, ec=OLLAMA, lw=0.0)
    s.text(xs[1] + (widths[1] + gap + widths[2]) / 2, head_y + 0.07, "Decode V2", fs=T_SMALL, weight="bold", ha="center", color=PAPER)
    s.text(xs[3] + (widths[3] + gap + widths[4]) / 2, head_y + 0.07, "Ollama Q4_K", fs=T_SMALL, weight="bold", ha="center", color=PAPER)

    labels_y = head_y + group_h + 0.04
    headers = ("Model", "correct", "tok/s", "correct", "tok/s")
    for x, width, label in zip(xs, widths, headers, strict=True):
        s.text(x + width / 2, labels_y, label, fs=T_TINY, ha="center", color=INK_SOFT, weight="bold")

    rows = (
        (
            "Qwen2.5-3B",
            V2_3B_HARD,
            f"{V2_3B_HARD_TOK:.1f}",
            OLLAMA_3B_HARD,
            f"{OLLAMA_3B_HARD_TOK:.1f}",
        ),
        (
            "Qwen2.5-14B",
            V2_14B_HARD,
            f"{V2_14B_HARD_TOK:.1f}",
            COMING_SOON,
            COMING_SOON,
        ),
        (
            "InternLM2.5-20B",
            V2_20B_HARD,
            f"{V2_20B_HARD_TOK:.1f}",
            OLLAMA_20B_HARD,
            f"{OLLAMA_20B_HARD_TOK:.1f}",
        ),
    )
    body_y = labels_y + 0.26
    for i, row in enumerate(rows):
        ry = body_y + i * row_h
        bg = BAND if i % 2 == 0 else PAPER
        s.rect(x0 + pad, ry, w - 2 * pad, row_h - 0.04, fc=bg, ec=RULE, lw=0.35)
        colors = (INK, NF4, NF4, OLLAMA if row[3] != COMING_SOON else INK_SOFT, OLLAMA if row[4] != COMING_SOON else INK_SOFT)
        for x, width, cell, color in zip(xs, widths, row, colors, strict=True):
            s.text(
                x + (0.10 if x == xs[0] else width / 2),
                ry + 0.10,
                cell,
                fs=T_SMALL,
                ha="left" if x == xs[0] else "center",
                family=MONO if x != xs[0] else None,
                color=color,
                weight="bold" if x == xs[0] else "normal",
            )

    note_y = body_y + 3 * row_h + 0.08
    s.text(
        x0 + 0.12,
        note_y,
        f"TokenLoop MMA (old resident path, not V2): 3B {TL_3B_HARD} · {TL_3B_HARD_TOK:.1f}   "
        f"14B {TL_14B_HARD} · {TL_14B_HARD_TOK:.2f}   "
        f"20B {TL_20B_HARD} · {TL_20B_HARD_TOK:.1f}\n"
        f"32B overflow is CopyRing, not Decode V2: TokenLoop {TL_32B_HARD} · {TL_32B_HARD_TOK:.2f} vs "
        f"Ollama {OLLAMA_32B_HARD} · {OLLAMA_32B_HARD_TOK:.1f}.",
        fs=T_TINY,
        color=INK_SOFT,
        spacing=1.32,
    )
    return y + h


def _footer(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    h = 1.72
    s.rect(x0, y, w, h, fc=BAND, ec=RULE, lw=0.6)
    s.text(x0 + 0.12, y + 0.08, "Honesty  ·  this sheet", fs=T_CLAIM, weight="bold")
    s.text(
        x0 + 0.12,
        y + 0.30,
        f"Ollama 3B mean {OLLAMA_3B_HARD_TOK:.1f} is pulled up by logic-yesno (2 tokens, 330 tok/s); median {OLLAMA_3B_HARD_MEDIAN:.1f}. "
        f"V2 3B {V2_3B_HARD_TOK:.1f} is the same 12-item mean.\n"
        f"3B V2 {V2_3B_HARD} vs Ollama {OLLAMA_3B_HARD}. 20B quality is tied {V2_20B_HARD}; V2 {V2_20B_HARD_TOK:.1f} vs Ollama {OLLAMA_20B_HARD_TOK:.1f} (median {OLLAMA_20B_HARD_MEDIAN:.1f}).\n"
        f"20B V2 {V2_20B_HARD_TOK:.1f} is not ignore-EOS 40. Do not say faster than Ollama.\n"
        f"{COMING_SOON}: Ollama 14B hard-12, llama.cpp hard-12 (3B / 14B / 20B), 3B Nsight 70–85%.\n"
        "Evidence: docs/runs/hard-decodev2-3b|14b|20b/ and docs/runs/ollama-hard-3b|20b/. "
        "Redraw: python -m gpu.lab.hard_v2_plate --redraw",
        fs=T_TINY,
        color=INK,
        spacing=1.28,
    )
    return y + h


def hard_v2_plate() -> Any:
    fig_h = _fig_h()
    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor=PAPER, dpi=105.0)
    s = _Sheet(fig, fig_h)
    y = _M_TOP
    y = _header(s, y)
    y += 0.10
    y = _table(s, y)
    y += 0.10
    _footer(s, y)
    return fig


def redraw(*extra: str | Path, repo: str | Path | None = None) -> list[Path]:
    figure = hard_v2_plate()
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
        prog="python -m gpu.lab.hard_v2_plate",
        description="Redraw docs/img/hard-v2-ollama.png. Matplotlib only, no GPU.",
    )
    parser.add_argument("--redraw", action="store_true", help="write docs/img/hard-v2-ollama.png")
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

"""Matched 3080 compare plate: Ollama / llama.cpp / Decode V2 / TokenLoop.

Same visual language as :mod:`gpu.lab.h2_plate` (Okabe–Ito, ink, card-limit
red). Matplotlib Agg. No torch, no GPU.

    python -m gpu.lab.compare_plate --redraw

Writes ``docs/img/compare-3080.png``. WAVE freeze of
``docs/runs/compare-3080/compare.json`` plus Decode V2 3B from
``docs/decode-v2-lab.md``. Quote 32B **long** 64-token plateaus,
not Ollama's short-EOS smoke mean.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .h2_plate import (
    BAND,
    CHIP,
    FIG_W,
    HOST_FILL,
    LIMIT_FILL,
    MONO,
    NF4,
    NF4_FILL,
    PAPER,
    _Sheet,
)
from .h2_plate import _T_BODY as T_BODY
from .h2_plate import _T_CLAIM as T_CLAIM
from .h2_plate import _T_HEAD as T_HEAD
from .h2_plate import _T_SMALL as T_SMALL
from .h2_plate import _T_SUB as T_SUB
from .h2_plate import _T_TINY as T_TINY
from .h2_plate import _T_TITLE as T_TITLE
from .hard_plate import INK, INK_SOFT, LIMIT, RULE, write_plate

__all__ = [
    "OLLAMA_32B_LONG",
    "H2_32B_LONG",
    "H2_32B_EVAL",
    "LLAMA_32B_LONG",
    "LLAMA_32B_NGL99",
    "V2_3B_HOST",
    "compare_plate",
    "default_png",
    "redraw",
]

_REPO = Path(__file__).resolve().parents[2]

# WAVE freeze — docs/runs/compare-3080/compare.json. Long = ignore-EOS 64.
OLLAMA_32B_LONG = 2.54
H2_32B_LONG = 2.49
H2_32B_EVAL = 2.53
LLAMA_32B_LONG = 2.54
LLAMA_32B_NGL99 = 1.52
OLLAMA_32B_SMOKE = 3.18
H2_32B_SMOKE = 2.31
OLLAMA_3B_LONG = 187.3
LLAMA_3B_LONG = 187.0
H2_3B_LONG = 35.2
V2_3B_HOST = 197.0
BNB_3B_SMOKE = 22.2
H2_COPY_FLOOR_MS = 277.3
OLLAMA_GPU_LAYERS = 33
OLLAMA_TOTAL_LAYERS = 65
OLLAMA_CUDA_MIB = 9559
OLLAMA_HOST_MIB = 9367
H2_STREAM_MIB = 6885
H2_STREAM_MATRICES = 96
CARD_MIB = 12288

OLLAMA = "#009E73"
LLAMA = "#E69F00"
BNB = "#5B5B5B"

_M_LEFT = 0.28
_M_RIGHT = 0.22
_M_TOP = 0.16
_M_BOTTOM = 0.16


def default_png(repo: str | Path | None = None) -> Path:
    root = Path(repo) if repo else _REPO
    return root / "docs" / "img" / "compare-3080.png"


def _fig_h() -> float:
    return (
        _M_TOP
        + 0.92
        + 0.08
        + 0.78
        + 0.10
        + 2.55
        + 0.10
        + 2.72
        + 0.10
        + 1.48
        + _M_BOTTOM
    )


def _header(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    h = 0.92
    s.rect(x0, y, w, h, fc=PAPER, ec=RULE, lw=0.7)
    s.text(x0 + 0.14, y + 0.10, "Matched decode on one RTX 3080 12 GB", fs=T_TITLE, weight="bold")
    s.text(
        x0 + 0.14,
        y + 0.42,
        "Greedy  ·  ctx 2048  ·  Qwen2.5 Instruct  ·  quote the 64-token ignore-EOS plateau on 32B\n"
        "Ollama 0.34.0 is llama-server (Q4_K_M). 32B H2 is NF4 CopyRing. "
        "3B Decode V2 is max_seq=2048; Q4_K long is ctx 2048. V2 still attends the full buffer. "
        "-ngl 99 was 1.52.",
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
        (f"{OLLAMA_32B_LONG:.2f}", "Ollama 32B long, tok/s", OLLAMA, "33/65 layers on GPU; CPU suffix"),
        (f"{H2_32B_LONG:.2f}", "H2 NF4 32B long, tok/s", NF4, f"{H2_STREAM_MATRICES} matrices / {H2_STREAM_MIB} MiB H2D"),
        (f"{LLAMA_32B_LONG:.2f}", "llama.cpp 32B long, tok/s", LLAMA, "ngl omitted; auto-fit (ngl 99 was 1.52)"),
    )
    for i, (value, label, color, note) in enumerate(cells):
        x = x0 + i * (w + gap)
        s.rect(x, y, w, h, fc=BAND, ec=RULE, lw=0.65)
        s.rect(x, y, 0.08, h, fc=color, ec=color, lw=0.0)
        s.text(x + 0.20, y + 0.08, value, fs=13.0, weight="bold", family=MONO, color=color)
        s.text(x + 0.20, y + 0.38, label, fs=T_SMALL, weight="bold")
        s.text(x + 0.20, y + 0.54, note, fs=T_TINY, color=INK_SOFT)
    return y + h


def _hbar(
    s: _Sheet,
    x: float,
    y: float,
    w: float,
    h: float,
    value: float,
    vmax: float,
    color: str,
    name: str,
    caption: str,
    label_w: float = 1.72,
) -> None:
    s.text(x, y + 0.02, name, fs=T_SMALL, weight="bold")
    s.text(x, y + 0.20, caption, fs=T_TINY, color=INK_SOFT)
    bar_x = x + label_w
    bar_w = w - label_w - 0.83
    s.rect(bar_x, y + 0.06, bar_w, 0.22, fc=CHIP, ec=RULE, lw=0.45)
    fill = max(0.04, bar_w * (value / vmax))
    s.rect(bar_x, y + 0.06, fill, 0.22, fc=color, ec=color, lw=0.0)
    s.text(
        bar_x + bar_w + 0.10,
        y + 0.08,
        f"{value:.2f}" if value < 10 else f"{value:.1f}",
        fs=T_SMALL,
        family=MONO,
        weight="bold",
        color=INK,
    )


def _bars(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    gap = 0.12
    col_w = (FIG_W - _M_LEFT - _M_RIGHT - gap) / 2.0
    h = 2.55
    s.rect(x0, y, col_w, h, fc=PAPER, ec=RULE, lw=0.7)
    s.text(x0 + 0.12, y + 0.08, "3B long decode, tok/s", fs=T_HEAD, weight="bold")
    s.text(
        x0 + 0.12,
        y + 0.30,
        "Resident. V2 GEMV vs Q4_K. TokenLoop MMA is the old 3B plate.",
        fs=T_TINY,
        color=INK_SOFT,
    )
    rows_3b = (
        (OLLAMA, "Ollama", "Q4_K_M  ·  37/37 GPU  ·  ctx 2048", OLLAMA_3B_LONG),
        (LLAMA, "llama.cpp", "Q4_K_M  ·  ctx 2048 long", LLAMA_3B_LONG),
        (NF4, "Decode V2", "NF4 GEMV  ·  host 197  ·  max_seq 2048", V2_3B_HOST),
        (BNB, "TokenLoop", "NF4 MMA  ·  product plate 35.2", H2_3B_LONG),
    )
    for i, (color, name, cap, val) in enumerate(rows_3b):
        _hbar(s, x0 + 0.12, y + 0.52 + i * 0.48, col_w - 0.20, 0.42, val, 220.0, color, name, cap)

    x1 = x0 + col_w + gap
    s.rect(x1, y, col_w, h, fc=PAPER, ec=RULE, lw=0.7)
    s.text(x1 + 0.12, y + 0.08, "32B long decode, tok/s", fs=T_HEAD, weight="bold")
    s.text(
        x1 + 0.12,
        y + 0.30,
        "Do not quote Ollama smoke 3.18 (short EOS). ngl 99 was 1.52 (fit abort).",
        fs=T_TINY,
        color=INK_SOFT,
    )
    rows_32 = (
        (OLLAMA, "Ollama", "Q4_K_M  ·  33/65 GPU + CPU suffix", OLLAMA_32B_LONG),
        (LLAMA, "llama.cpp", "Q4_K_M  ·  ngl omitted auto-fit", LLAMA_32B_LONG),
        (NF4, "H2 NF4", "CopyRing  ·  96 HOST matrices", H2_32B_LONG),
    )
    for i, (color, name, cap, val) in enumerate(rows_32):
        _hbar(s, x1 + 0.12, y + 0.52 + i * 0.48, col_w - 0.20, 0.42, val, 3.0, color, name, cap)
    s.text(
        x1 + 0.12,
        y + h - 0.28,
        "Tie is two bottlenecks: 5950X Q4_K on 32 CPU layers vs 6885 MiB H2D/tok.",
        fs=T_TINY,
        color=INK,
    )
    return y + h


def _split(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    gap = 0.12
    col_w = (FIG_W - _M_LEFT - _M_RIGHT - gap) / 2.0
    h = 2.72
    s.rect(x0, y, col_w, h, fc=NF4_FILL, ec=RULE, lw=0.7)
    s.text(x0 + 0.12, y + 0.08, "Ollama overflow  ·  layers stay put", fs=T_HEAD, weight="bold")
    s.text(
        x0 + 0.12,
        y + 0.32,
        "llama-server auto-fit. Weights do not move each token.\n"
        "Hidden state (~10 KiB) crosses GPU/CPU once at the split.\n"
        "Decode graph splits = 2 at batch 1. CUDA graphs reused.",
        fs=T_BODY,
        color=INK,
        spacing=1.32,
    )
    gpu_h = 0.72
    cpu_h = 0.72
    s.rect(x0 + 0.12, y + 1.18, col_w - 0.24, gpu_h, fc=OLLAMA, ec=OLLAMA, lw=0.0, alpha=0.22)
    s.rect(x0 + 0.12, y + 1.18, 0.08, gpu_h, fc=OLLAMA, ec=OLLAMA, lw=0.0)
    s.text(
        x0 + 0.28,
        y + 1.24,
        f"GPU  {OLLAMA_GPU_LAYERS}/{OLLAMA_TOTAL_LAYERS} layers   ·   CUDA0 {OLLAMA_CUDA_MIB} MiB",
        fs=T_SMALL,
        weight="bold",
    )
    s.text(
        x0 + 0.28,
        y + 1.50,
        "32 repeating + lm_head  ·  Flash Attention auto  ·  KV 256 MiB",
        fs=T_TINY,
        color=INK_SOFT,
    )
    s.rect(x0 + 0.12, y + 1.98, col_w - 0.24, cpu_h, fc=HOST_FILL, ec=RULE, lw=0.65)
    s.text(
        x0 + 0.28,
        y + 2.04,
        f"CPU  32 layers   ·   CUDA_Host {OLLAMA_HOST_MIB} MiB  ·  AVX2 x16",
        fs=T_SMALL,
        weight="bold",
    )
    s.text(
        x0 + 0.28,
        y + 2.30,
        "Q4_K SIMD  ·  --load-mode none (Windows CUDA, no mmap faults)",
        fs=T_TINY,
        color=INK_SOFT,
    )

    x1 = x0 + col_w + gap
    s.rect(x1, y, col_w, h, fc=LIMIT_FILL, ec=RULE, lw=0.7)
    s.text(x1 + 0.12, y + 0.08, "H2 overflow  ·  matrices move", fs=T_HEAD, weight="bold")
    s.text(
        x1 + 0.12,
        y + 0.32,
        "Every layer still runs on the GPU. Policy D streams packed NF4\n"
        "into two slots. WDDM CPU-joins e_copy before the next prefetch.\n"
        "Serial copy floor 277 ms; measured long wall ~402 ms/tok.",
        fs=T_BODY,
        color=INK,
        spacing=1.32,
    )
    s.rect(x1 + 0.12, y + 1.18, col_w - 0.24, 0.72, fc=NF4, ec=NF4, lw=0.0, alpha=0.18)
    s.rect(x1 + 0.12, y + 1.18, 0.08, 0.72, fc=NF4, ec=NF4, lw=0.0)
    s.text(
        x1 + 0.28,
        y + 1.24,
        "DEVICE  q/k/v/o all 64  ·  gate/up L0–47  ·  embed + lm_head",
        fs=T_SMALL,
        weight="bold",
    )
    s.text(
        x1 + 0.28,
        y + 1.50,
        "Resident packed ~9716 MiB  ·  KV 512 MiB  ·  chr_nf4_gemm",
        fs=T_TINY,
        color=INK_SOFT,
    )
    s.rect(x1 + 0.12, y + 1.98, col_w - 0.24, 0.72, fc=HOST_FILL, ec=RULE, lw=0.65)
    s.text(
        x1 + 0.28,
        y + 2.04,
        f"HOST tape  {H2_STREAM_MATRICES} matrices  ·  {H2_STREAM_MIB} MiB / token",
        fs=T_SMALL,
        weight="bold",
    )
    s.text(
        x1 + 0.28,
        y + 2.30,
        "all down_proj + gate/up L48–63  ·  2 slots  ·  depth 1",
        fs=T_TINY,
        color=INK_SOFT,
    )
    return y + h


def _footer(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    h = 1.48
    s.rect(x0, y, w, h, fc=BAND, ec=RULE, lw=0.6)
    s.text(x0 + 0.12, y + 0.08, "Honesty  ·  this sheet", fs=T_CLAIM, weight="bold")
    s.text(
        x0 + 0.12,
        y + 0.30,
        "Not a ranking of Marlin / AWQ / ExLlamaV2 / vLLM: those rows are SKIP, not borrowed tok/s.\n"
        "32B Ollama smoke 3.18 tok/s is 8+8+4 tokens; quote long 2.54 vs H2 2.49 steps / 2.53 eval vs llama.cpp auto-fit 2.54 (ngl 99 was 1.52).\n"
        "The 32B tie is not a better GPU kernel. Ollama leaves 32 layers on the 5950X; we copy 6885 MiB over PCIe every token.\n"
        "3B Decode V2 197 (max_seq=2048) sits next to Q4_K ~187 (ctx 2048). V2 still attends the full buffer — not a kernel ranking. TokenLoop MMA 35.2 stays the old plate.\n"
        "Redraw: python -m gpu.lab.compare_plate --redraw   ·   JSON: docs/runs/compare-3080/   ·   V2 sheet: docs/img/decodev2-3080.png",
        fs=T_TINY,
        color=INK,
        spacing=1.32,
    )
    return y + h


def compare_plate() -> Any:
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
    y = _split(s, y)
    y += 0.10
    _footer(s, y)
    return fig


def redraw(*extra: str | Path, repo: str | Path | None = None) -> list[Path]:
    figure = compare_plate()
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
        prog="python -m gpu.lab.compare_plate",
        description="Redraw docs/img/compare-3080.png. Matplotlib only, no GPU.",
    )
    parser.add_argument("--redraw", action="store_true", help="write docs/img/compare-3080.png")
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

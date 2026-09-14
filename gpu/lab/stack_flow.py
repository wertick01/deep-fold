"""Arrow flowchart of the stack. Does not replace ``docs/img/stack.png``.

The poster in :mod:`gpu.lab.stack_plate` stays. This sheet is the same path
drawn as boxes and arrows so a reader can follow one run.

    python -m gpu.lab.stack_flow --redraw

Writes ``docs/img/stack-flow.png``. Matplotlib Agg. No torch, no GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from .hard_plate import CODEC_STYLE, FONT, INK, INK_SOFT, RULE, write_plate

__all__ = ["LIVE_MAX_N", "default_png", "redraw", "stack_flow"]

_REPO = Path(__file__).resolve().parents[2]

LIVE_MAX_N = 16

BF16 = CODEC_STYLE["bf16"].color
NF4 = CODEC_STYLE["nf4"].color
PAPER = "#FFFFFF"
BAND = "#F2F4F5"
BF16_FILL = "#FDF3E3"
NF4_FILL = "#E7F1F8"
CPU_FILL = "#F4F5F6"
GPU_FILL = "#F4F7FA"
REG_FILL = "#D9E8F4"

FIG_W = 13.40
FIG_H = 9.15
XMAX = 134.0
YMAX = 91.5


@dataclass(frozen=True)
class _Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def left(self) -> tuple[float, float]:
        return self.x, self.cy

    @property
    def right(self) -> tuple[float, float]:
        return self.x + self.w, self.cy

    @property
    def top(self) -> tuple[float, float]:
        return self.cx, self.y

    @property
    def bottom(self) -> tuple[float, float]:
        return self.cx, self.y + self.h


def default_png(repo: str | Path | None = None) -> Path:
    root = Path(repo) if repo else _REPO
    return root / "docs" / "img" / "stack-flow.png"


def _rounded(
    ax: Any,
    box: _Box,
    *,
    fc: str,
    ec: str,
    lw: float = 0.9,
    z: int = 3,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (box.x, box.y),
            box.w,
            box.h,
            boxstyle="round,pad=0.18,rounding_size=0.55",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            mutation_aspect=1.0,
            zorder=z,
            clip_on=False,
        )
    )


def _arrow(
    ax: Any,
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    label: str = "",
    color: str = INK,
    lw: float = 1.25,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color,
            shrinkA=1.2,
            shrinkB=1.2,
            zorder=5,
            clip_on=False,
        )
    )
    if not label:
        return
    mx = (a[0] + b[0]) / 2
    my = (a[1] + b[1]) / 2
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) >= abs(dy):
        ax.text(
            mx,
            my - 1.15,
            label,
            ha="center",
            va="top",
            fontsize=6.6,
            color=INK_SOFT,
            fontfamily=FONT,
            zorder=6,
        )
    else:
        ax.text(
            mx + 0.55,
            my,
            label,
            ha="left",
            va="center",
            fontsize=6.6,
            color=INK_SOFT,
            fontfamily=FONT,
            zorder=6,
        )


def _title(ax: Any, box: _Box, title: str, body: str, *, tc: str = INK) -> None:
    ax.text(
        box.x + 0.7,
        box.y + 0.85,
        title,
        ha="left",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        color=tc,
        fontfamily=FONT,
        zorder=4,
    )
    if body:
        ax.text(
            box.x + 0.7,
            box.y + 2.55,
            body,
            ha="left",
            va="top",
            fontsize=6.55,
            color=INK,
            fontfamily=FONT,
            linespacing=1.28,
            zorder=4,
        )


def _band(ax: Any, x: float, y: float, w: float, h: float, fc: str) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=fc,
            edgecolor=RULE,
            linewidth=0.65,
            zorder=0,
            clip_on=False,
        )
    )


def stack_flow() -> Any:
    """Boxes and arrows: HF tree → .chr → CompressedLinear → TokenLoop → GEMM tile."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=110)
    fig.patch.set_facecolor(PAPER)
    ax.set_xlim(0, XMAX)
    ax.set_ylim(YMAX, 0)
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(
        2.2,
        1.7,
        "deep-fold flow  —  one run, Ampere sm_86",
        fontsize=13.0,
        fontweight="bold",
        color=INK,
        fontfamily=FONT,
        va="top",
    )
    ax.text(
        2.2,
        4.15,
        "No full-size copy of any linear layer is ever created on the GPU. "
        "Packed residency is prior art (Linear4bit, Marlin, AWQ W4A16); the claim is this path.",
        fontsize=7.5,
        color=INK_SOFT,
        fontfamily=FONT,
        va="top",
    )

    # --- CPU ---------------------------------------------------------------
    _band(ax, 1.6, 6.0, 130.8, 16.8, CPU_FILL)
    ax.text(2.4, 7.05, "CPU", fontsize=8.2, fontweight="bold", color=INK, fontfamily=FONT)

    hf = _Box(3.2, 8.5, 32.0, 12.8)
    compress = _Box(47.0, 8.5, 32.0, 12.8)
    blob = _Box(90.8, 8.5, 39.4, 12.8)
    _rounded(ax, hf, fc=BF16_FILL, ec=BF16)
    _rounded(ax, compress, fc=PAPER, ec=RULE)
    _rounded(ax, blob, fc=NF4_FILL, ec=NF4)
    _title(
        ax,
        hf,
        "HuggingFace tree",
        "config.json  ·  tokenizer\nBF16 / FP16 safetensors\nGGUF / ~/.ollama  —  refused",
        tc=BF16,
    )
    _title(
        ax,
        compress,
        "chr compress  (Go)",
        "cmd/chr  ·  --codec nf4\nNF4: 16 levels, G=64\none FP16 scale / group\n4.25 bit / weight",
    )
    _title(
        ax,
        blob,
        ".chr  ·  CHR0",
        "packed uint8  +  scale FP16\nGPU reads; it does not recode\n4.25 = 4 + 16/64",
        tc=NF4,
    )
    _arrow(ax, hf.right, compress.left, label="once, on the CPU")
    _arrow(ax, compress.right, blob.left, label="writes one file")

    # --- GPU ---------------------------------------------------------------
    _band(ax, 1.6, 24.2, 130.8, 61.0, GPU_FILL)
    ax.text(
        2.4,
        25.25,
        "GPU  ·  Ampere sm_86  ·  Ada / Hopper / Blackwell refused",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
        fontfamily=FONT,
    )

    cfg = _Box(3.2, 27.4, 28.0, 9.6)
    load = _Box(43.8, 27.4, 48.4, 9.6)
    lin = _Box(103.4, 27.4, 26.8, 9.6)
    _rounded(ax, cfg, fc=BF16_FILL, ec=BF16)
    _rounded(ax, load, fc=PAPER, ec=RULE)
    _rounded(ax, lin, fc=NF4_FILL, ec=NF4)
    _title(
        ax,
        cfg,
        "config.json + tokenizer",
        "meta skeleton only\nshards never opened here",
        tc=BF16,
    )
    _title(
        ax,
        load,
        "gpu.host.load_model",
        "build_skeleton  →  replace_linears\nthen bind CHR0 bytes\nnever from_pretrained on W",
    )
    _title(
        ax,
        lin,
        "CompressedLinear",
        "packed + scale only\nno [out, in] weight",
        tc=NF4,
    )
    _arrow(ax, blob.bottom, lin.top, label=".chr bytes")
    _arrow(ax, cfg.right, load.left)
    _arrow(ax, load.right, lin.left, label="bind bytes")

    loop = _Box(3.2, 40.6, 127.0, 10.6)
    _rounded(ax, loop, fc=PAPER, ec=INK, lw=1.15)
    _title(
        ax,
        loop,
        "gpu.loop.TokenLoop",
        "Drives the transformer graph. Does not call transformers.generate.  "
        "prefill chunks of LIVE_MAX_N = 16  ·  decode N = 1  ·  host chunks N > 16.",
    )
    _arrow(ax, lin.bottom, (lin.cx, loop.y))

    glue = _Box(3.2, 55.4, 38.0, 13.6)
    _rounded(ax, glue, fc=BF16_FILL, ec=BF16)
    _title(
        ax,
        glue,
        "Stays BF16  ·  PyTorch",
        "rms_norm  ·  RoPE  ·  SDPA\nKV cache (slot write, no cat)\nactivations  ·  embed lookup",
        tc=BF16,
    )
    _arrow(ax, (loop.x + 19.0, loop.y + loop.h), glue.top, label="attention / norms")

    kx, ky, kh = 44.6, 55.4, 13.6
    gap = 3.4
    widths = (18.4, 19.2, 18.0, 20.4)
    titles = (
        "HBM tile",
        "Registers",
        "MMA",
        "Discard W",
    )
    bodies = (
        "packed + scale\nread, not rewritten",
        "LUT × FP16 scale\n→ BF16 A-fragment",
        "m16n8k16\n× BF16 x [K,N]",
        "reconstructed W gone\npacked stays in HBM",
    )
    kboxes: list[_Box] = []
    x = kx
    for w in widths:
        kboxes.append(_Box(x, ky, w, kh))
        x += w + gap
    ax.text(
        kx,
        69.35,
        "chr_nf4_gemm  ·  one tile, not a layer  ·  gpu/nf4/nf4_gemm.cu",
        fontsize=7.2,
        fontweight="bold",
        color=NF4,
        fontfamily=FONT,
        va="top",
    )
    for box, title, body in zip(kboxes, titles, bodies):
        _rounded(ax, box, fc=REG_FILL if title != "HBM tile" else NF4_FILL, ec=NF4)
        _title(ax, box, title, body, tc=NF4)
    for left, right in zip(kboxes, kboxes[1:]):
        _arrow(ax, left.right, right.left)
    _arrow(
        ax,
        (kboxes[0].cx, loop.y + loop.h),
        kboxes[0].top,
        label="q / k / v / o / gate / up / down",
    )

    note = _Box(44.6, 71.4, 85.6, 12.2)
    _rounded(ax, note, fc=PAPER, ec=RULE, lw=0.7)
    _title(
        ax,
        note,
        "What the GEMM arrow is not",
        "Not unpack-layer-into-VRAM, multiply, pack again. Reconstruction is tile-local "
        "in registers.\n"
        "The packed table is read-only for the whole run. "
        f"Live launch N ∈ [1, {LIVE_MAX_N}]. Planned n32 exists; TokenLoop does not launch it.\n"
        "Generate: Ampere sm_86 only. Glue: llama_swiglu, internlm_gqa. "
        "Refusals: Gemma, Phi-3, MoE, vision, GGUF.",
    )

    ax.text(
        2.2,
        86.4,
        "Always two inputs after compress: HuggingFace directory (config + tokenizer) and one .chr. "
        "Safetensor shards are not opened at generate time. Display VRAM is inside nvidia-smi (not subtracted). "
        "Not a tok/s leaderboard.",
        fontsize=6.5,
        color=INK_SOFT,
        fontfamily=FONT,
        va="top",
    )
    return fig


def redraw(*extra: str | Path, repo: str | Path | None = None) -> list[Path]:
    figure = stack_flow()
    targets: list[Path] = [default_png(repo)]
    for path in extra:
        if path:
            dest = Path(path)
            if dest not in targets:
                targets.append(dest)
    written = write_plate(figure, *targets, dpi=110.0)
    plt.close(figure)
    return written


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gpu.lab.stack_flow",
        description="Redraw docs/img/stack-flow.png. Does not overwrite stack.png.",
    )
    parser.add_argument("--redraw", action="store_true", help="write docs/img/stack-flow.png")
    parser.add_argument("--out", action="append", default=[], help="extra PNG path")
    parser.add_argument("--repo", default="", help="repository root")
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

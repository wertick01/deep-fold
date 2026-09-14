"""Architecture plate: the four-part stack, no dense W on the GPU.

Same visual language as :mod:`gpu.lab.hard_plate` / :mod:`gpu.lab.progress_plate`
(Okabe–Ito BF16/NF4, ink, the card-limit red). Matplotlib Agg. No torch, no GPU.

    python -m gpu.lab.stack_plate --redraw

Writes ``docs/img/stack.png``. The README caption is the place for a paragraph;
this sheet is the path, not a tok/s table.
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
    INK,
    INK_SOFT,
    LIMIT,
    RULE,
    write_plate,
)

__all__ = [
    "LIVE_MAX_N",
    "default_png",
    "redraw",
    "stack_plate",
]

_REPO = Path(__file__).resolve().parents[2]

# WAVE freeze — same constant as gpu.nf4.plan, duplicated so this module never
# imports the CUDA package (and never pulls torch).
LIVE_MAX_N = 16

BF16 = CODEC_STYLE["bf16"].color
NF4 = CODEC_STYLE["nf4"].color

PAPER = "#FFFFFF"
BAND = "#F2F4F5"
CHIP = "#F4F5F6"
BF16_FILL = "#FDF3E3"
NF4_FILL = "#E7F1F8"
LIMIT_FILL = "#FBECEA"
GPU_FILL = "#F4F7FA"
CPU_FILL = "#F4F5F6"

MONO = ["Consolas", "DejaVu Sans Mono", "Courier New"]

FIG_W = 13.40
_M_LEFT = 0.28
_M_RIGHT = 0.22
_M_TOP = 0.14
_M_BOTTOM = 0.16

# Point sizes chosen for a ~1400 px-wide PNG (dpi 105).
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
    return root / "docs" / "img" / "stack.png"


# --------------------------------------------------------------------------- #
# figure helpers (inches from the top-left of the sheet)
# --------------------------------------------------------------------------- #


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

    def stripe(
        self,
        xin: float,
        yin: float,
        hin: float,
        color: str,
        *,
        win: float = 0.07,
    ) -> None:
        self.rect(xin, yin, win, hin, fc=color, ec=color, lw=0.0, z=2)

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

    def badge(self, xin: float, yin: float, letter: str, *, fc: str = NF4) -> None:
        s = 0.28
        self.rect(xin, yin, s, s, fc=fc, ec=fc, lw=0.0, z=3)
        self.text(
            xin + s / 2,
            yin + s / 2,
            letter,
            fs=9.0,
            color=PAPER,
            ha="center",
            va="center",
            weight="bold",
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
        family: Sequence[str] | None = None,
        weight: str = "normal",
        hatch: str = "",
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
        + 1.18  # header
        + 0.08
        + 0.22  # CPU rail
        + 2.78  # part 1
        + 0.10
        + 0.22  # GPU rail
        + 5.58  # parts 2+3
        + 0.10
        + 5.18  # part 4
        + 0.10
        + 0.92  # footer
        + _M_BOTTOM
    )


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #


def stack_plate() -> Any:
    """Draw the stack. Numbers on this sheet are structure, not tok/s."""
    fig_h = _fig_h()
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "text.color": INK,
        }
    ):
        fig = plt.figure(figsize=(FIG_W, fig_h), dpi=105)
        s = _Sheet(fig, fig_h)
        y = _M_TOP
        y = _header(s, y)
        y = _cpu_rail(s, y)
        y = _part1(s, y)
        y = _gpu_rail(s, y)
        y = _parts23(s, y)
        y = _part4(s, y)
        _footer(s, y)
        return fig


def _header(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    s.text(
        x0,
        y,
        "deep-fold stack — 16-bit HuggingFace tree to tokens, Ampere sm_86",
        fs=_T_TITLE,
        weight="bold",
    )
    s.text(
        x0,
        y + 0.28,
        "No full-size copy of any linear layer is ever created on the GPU. Packed residency is prior art "
        "(Linear4bit, Marlin, AWQ W4A16); the claim is this path.",
        fs=_T_SUB,
        color=INK_SOFT,
    )
    # CLI chips
    cy = y + 0.52
    ch = 0.30
    gap = 0.10
    cmds = [
        ("python -m gpu.cli doctor", NF4_FILL, NF4),
        ("python -m gpu.cli run --model <HF dir>", NF4_FILL, NF4),
        ("chr compress --in DIR --out FILE --codec nf4", BF16_FILL, BF16),
    ]
    cx = x0
    widths = (3.55, 4.55, 4.55)
    for (label, fc, ec), w in zip(cmds, widths):
        s.chip(
            cx,
            cy,
            w,
            ch,
            label,
            fc=fc,
            ec=ec,
            fs=6.7,
            family=MONO,
        )
        cx += w + gap

    # legend
    ly = y + 0.90
    s.text(x0, ly + 0.07, "On this sheet", fs=_T_TINY, color=INK_SOFT, va="center")
    lx = x0 + 1.15
    s.chip(lx, ly, 1.55, 0.22, "BF16 (stays)", fc=BF16_FILL, ec=BF16, tc=INK, fs=_T_TINY)
    s.chip(lx + 1.65, ly, 1.85, 0.22, "NF4 packed (resident)", fc=NF4_FILL, ec=NF4, tc=INK, fs=_T_TINY)
    s.chip(lx + 3.60, ly, 1.55, 0.22, "named refusal", fc=LIMIT_FILL, ec=LIMIT, tc=LIMIT, fs=_T_TINY)
    s.chip(
        lx + 5.25,
        ly,
        2.55,
        0.22,
        "registers: reconstruct + discard",
        fc=PAPER,
        ec=INK,
        tc=INK,
        fs=_T_TINY,
    )
    return y + 1.18


def _cpu_rail(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    s.rect(x0, y, w, 0.22, fc=CPU_FILL, ec=RULE, lw=0.6)
    s.stripe(x0, y, 0.22, INK, win=0.08)
    s.text(
        x0 + 0.18,
        y + 0.11,
        "CPU  —  compressor only. No CUDA. GGUF is refused; the path is never opened.",
        fs=_T_SMALL,
        va="center",
        weight="bold",
    )
    return y + 0.22 + 0.08


def _gpu_rail(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    s.rect(x0, y, w, 0.22, fc=GPU_FILL, ec=NF4, lw=0.7)
    s.stripe(x0, y, 0.22, NF4, win=0.08)
    s.text(
        x0 + 0.18,
        y + 0.11,
        "GPU  —  Ampere sm_86 (RTX 3080 class). Generate is this arch only; Ada / Hopper / Blackwell are refused.",
        fs=_T_SMALL,
        va="center",
        weight="bold",
    )
    return y + 0.22 + 0.10


def _part1(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    h = 2.78
    s.rect(x0, y, w, h, fc=CPU_FILL, ec=RULE, lw=0.8)
    s.badge(x0 + 0.10, y + 0.10, "1")
    s.text(
        x0 + 0.46,
        y + 0.12,
        "CPU compressor  chr  (Go)   ·  cmd/chr",
        fs=_T_HEAD,
        weight="bold",
    )
    s.text(
        x0 + 0.46,
        y + 0.36,
        "Reads a HuggingFace directory of BF16/FP16 safetensors. Writes one .chr. Container CHR0. Codec NF4.",
        fs=_T_BODY,
        color=INK_SOFT,
    )

    box_y = y + 0.58
    box_h = 2.06
    gap = 0.12
    bw = (w - 0.24 - 2 * gap) / 3
    bx = x0 + 0.12

    # input
    s.rect(bx, box_y, bw, box_h, fc=PAPER, ec=BF16, lw=1.0)
    s.rect(bx, box_y, bw, 0.28, fc=BF16_FILL, ec=BF16, lw=0.8)
    s.text(bx + bw / 2, box_y + 0.14, "HuggingFace tree  (disk)", fs=_T_CLAIM, ha="center", va="center", weight="bold")
    s.text(
        bx + 0.10,
        box_y + 0.40,
        "config.json\n"
        "tokenizer\n"
        "*.safetensors   BF16 / FP16\n"
        "\n"
        "GGUF / ~/.ollama  —  refused\n"
        "(CLI never opens the path)\n"
        "\n"
        "Not a new numerical code.\n"
        "NF4 is the QLoRA / bitsandbytes\n"
        "family: 16 reconstruction levels.",
        fs=_T_SMALL,
        color=INK,
    )

    s.arrow(bx + bw + 0.02, box_y + box_h / 2, bx + bw + gap - 0.02, box_y + box_h / 2, color=INK)

    # compress
    bx2 = bx + bw + gap
    s.rect(bx2, box_y, bw, box_h, fc=PAPER, ec=INK, lw=1.0)
    s.rect(bx2, box_y, bw, 0.28, fc=CHIP, ec=RULE, lw=0.8)
    s.text(bx2 + bw / 2, box_y + 0.14, "chr compress  --codec nf4", fs=_T_CLAIM, ha="center", va="center", weight="bold")
    s.text(
        bx2 + 0.10,
        box_y + 0.40,
        "Go, CPU, no CGO. Minutes on 14B/20B.\n"
        "\n"
        "Each weight → one of 16 NF4 levels.\n"
        "Groups of 64.\n"
        "One FP16 scale per group.\n"
        "\n"
        "4-bit codes + 16/64 scale bits\n"
        "→  4.25 bit / weight  (not 16).\n"
        "\n"
        "python -m gpu.cli compress wraps this;\n"
        "the writer is still the Go binary.",
        fs=_T_SMALL,
    )

    s.arrow(bx2 + bw + 0.02, box_y + box_h / 2, bx2 + bw + gap - 0.02, box_y + box_h / 2, color=NF4)

    # CHR0
    bx3 = bx2 + bw + gap
    s.rect(bx3, box_y, bw, box_h, fc=PAPER, ec=NF4, lw=1.1)
    s.rect(bx3, box_y, bw, 0.28, fc=NF4_FILL, ec=NF4, lw=0.8)
    s.text(bx3 + bw / 2, box_y + 0.14, "one .chr   magic CHR0", fs=_T_CLAIM, ha="center", va="center", weight="bold")
    s.text(
        bx3 + 0.10,
        box_y + 0.40,
        "JSON header + blobs. GPU reads;\n"
        "it does not recode.\n"
        "\n"
        "NF4 linear / embed:\n"
        "  packed  uint8  [M, K_pad/2]\n"
        "  scale   fp16   [M, n_groups]\n"
        "  K_pad = 64·ceil(K/64)\n"
        "\n"
        "Same file also carries BF16 blobs\n"
        "for RMSNorm weights and biases.\n"
        "group_size = 64  (not 32, not 128).",
        fs=_T_SMALL,
    )
    return y + h + 0.10


def _parts23(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    h = 5.58
    gap = 0.12
    left_w = (w - gap) * 0.48
    right_w = w - gap - left_w
    _part2(s, x0, y, left_w, h)
    _part3(s, x0 + left_w + gap, y, right_w, h)
    return y + h + 0.10


def _part2(s: _Sheet, x: float, y: float, w: float, h: float) -> None:
    s.rect(x, y, w, h, fc=GPU_FILL, ec=NF4, lw=0.8)
    s.badge(x + 0.10, y + 0.10, "2")
    s.text(
        x + 0.46,
        y + 0.12,
        "Loader / CompressedLinear",
        fs=_T_HEAD,
        weight="bold",
    )
    s.text(
        x + 0.46,
        y + 0.34,
        "gpu/host/model.py   ·  gpu/host/linear.py",
        fs=_T_SMALL,
        color=INK_SOFT,
    )

    # order
    inner_x = x + 0.12
    inner_w = w - 0.24
    oy = y + 0.54
    steps = [
        (
            "A",
            "build_skeleton",
            "from_config on torch.device(\"meta\"). config.json only.\n"
            "Shards never opened. Meta tensors: a shape, no storage.",
        ),
        (
            "B",
            "attach_module",
            "Walk the tree → DriverPlan (family, GEMM slots).\n"
            "Fail closed. No device is touched.",
        ),
        (
            "C",
            "replace_linears",
            "Every planned nn.Linear → CompressedLinear\n"
            "*before* any device is touched.",
        ),
        (
            "D",
            "load_chr_nf4",
            "Bind CHR0 packed bytes. The .chr is the only\n"
            "weight file read. Bytes land on cuda here.",
        ),
    ]
    step_h = 0.72
    for i, (letter, title, body) in enumerate(steps):
        sy = oy + i * (step_h + 0.06)
        s.rect(inner_x, sy, inner_w, step_h, fc=PAPER, ec=RULE, lw=0.65)
        s.chip(inner_x + 0.06, sy + 0.18, 0.28, 0.28, letter, fc=NF4, ec=NF4, tc=PAPER, fs=8.0, weight="bold")
        s.text(inner_x + 0.42, sy + 0.08, title, fs=_T_CLAIM, weight="bold")
        s.text(inner_x + 0.42, sy + 0.28, body, fs=_T_TINY, color=INK)

    # never + module
    ny = oy + 4 * (step_h + 0.06) + 0.04
    never_h = y + h - ny - 0.12
    col = (inner_w - 0.08) / 2
    s.rect(inner_x, ny, col, never_h, fc=LIMIT_FILL, ec=LIMIT, lw=0.8)
    s.text(inner_x + 0.08, ny + 0.08, "Never (structural)", fs=_T_SMALL, weight="bold", color=LIMIT)
    s.text(
        inner_x + 0.08,
        ny + 0.28,
        "from_pretrained on original shards\n"
        "model.to(dtype) of dense W\n"
        "load_state_dict of BF16 weights\n"
        "\n"
        "After compress, generate does not\n"
        "open safetensor shards.\n"
        "Runtime needs: HF dir (config +\n"
        "tokenizer) + one .chr.",
        fs=_T_TINY,
        color=INK,
    )
    mx = inner_x + col + 0.08
    s.rect(mx, ny, col, never_h, fc=PAPER, ec=NF4, lw=0.9)
    s.text(mx + 0.08, ny + 0.08, "CompressedLinear", fs=_T_SMALL, weight="bold")
    s.text(
        mx + 0.08,
        ny + 0.28,
        "No [out, in] weight tensor — not a\n"
        "parameter, not a buffer.\n"
        "Resident: packed + scale only.\n"
        "weight is a 0-numel stand-in.\n"
        "model.to(\"cuda\") has nothing dense\n"
        "to move, so cannot resurrect W.\n"
        "\n"
        "embed: Nf4Embedding row lookup\n"
        "lm_head: this GEMM, or tied packed",
        fs=_T_TINY,
    )


def _part3(s: _Sheet, x: float, y: float, w: float, h: float) -> None:
    s.rect(x, y, w, h, fc=GPU_FILL, ec=NF4, lw=0.8)
    s.badge(x + 0.10, y + 0.10, "3")
    s.text(x + 0.46, y + 0.12, "Ampere kernel  chr_nf4_gemm", fs=_T_HEAD, weight="bold")
    s.text(
        x + 0.46,
        y + 0.34,
        "gpu/nf4/nf4_gemm.cu   ·  sm_86   ·  no BF16 W in HBM",
        fs=_T_SMALL,
        color=INK_SOFT,
    )

    inner_x = x + 0.12
    inner_w = w - 0.24

    # memory hierarchy
    hy = y + 0.54
    hh = 2.22
    s.rect(inner_x, hy, inner_w, hh, fc=PAPER, ec=INK, lw=0.85)
    s.text(inner_x + 0.10, hy + 0.08, "One tile — not a layer", fs=_T_CLAIM, weight="bold")

    # registers (top of chip)
    ry = hy + 0.32
    rh = 0.92
    s.rect(inner_x + 0.08, ry, inner_w - 0.16, rh, fc=CHIP, ec=INK, lw=0.8)
    s.text(inner_x + 0.16, ry + 0.08, "Registers  (on-chip, discarded after the MMA)", fs=_T_SMALL, weight="bold")
    s.text(
        inner_x + 0.16,
        ry + 0.32,
        "nibble → NF4 LUT × FP16 group scale  →  BF16 A-fragment\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
        "against BF16 activations x  [K, N]   →  y  BF16  [M, N]\n"
        "reconstructed weights are dropped. Never written back to HBM.",
        fs=_T_TINY,
    )

    s.text(
        inner_x + inner_w / 2,
        ry + rh + 0.10,
        "↑ tile-local reconstruct     ↓ packed table is read-only for the whole run",
        fs=_T_TINY,
        color=INK_SOFT,
        ha="center",
        va="center",
    )

    # HBM
    hby = ry + rh + 0.20
    hbh = hy + hh - ry - rh - 0.28
    s.rect(inner_x + 0.08, hby, inner_w - 0.16, hbh, fc=NF4_FILL, ec=NF4, lw=0.9)
    s.text(inner_x + 0.16, hby + 0.06, "HBM / VRAM", fs=_T_SMALL, weight="bold")
    s.text(
        inner_x + 0.16,
        hby + 0.26,
        "packed uint8 + scale fp16   —  the only stored form of a linear\n"
        "activations, KV cache, RMSNorm, RoPE, SDPA  stay BF16\n"
        "split-K workspace = FP32 partials, not a dense W",
        fs=_T_TINY,
    )

    # ABI + live N
    ay = hy + hh + 0.10
    ah = 1.18
    s.rect(inner_x, hy + hh + 0.10, inner_w, ah, fc=PAPER, ec=RULE, lw=0.65)
    s.text(inner_x + 0.10, ay + 0.08, "Host ABI  ·  live launch", fs=_T_CLAIM, weight="bold")
    s.text(
        inner_x + 0.10,
        ay + 0.28,
        "HuggingFace hands [..., K]; the host transposes to [K, N].\n"
        "Embed is a lookup (Nf4Embedding rows), not this GEMM.\n"
        "lm_head is this GEMM, or tied (shares packed with embed).\n"
        "\n"
        f"Live N ∈ [1, {LIVE_MAX_N}]. Decode N=1. Prefill N=2..8 BN=8; N=9..16 BN=16.\n"
        f"N>{LIVE_MAX_N}: chr_nf4_gemm_ws returns -2; the host chunks.\n"
        "n32/n64 tiles are planned (gpu.nf4.plan). Not launched here.",
        fs=_T_TINY,
    )

    # stays BF16 chips
    by = ay + ah + 0.10
    s.text(inner_x, by, "Stays BF16 on the card", fs=_T_SMALL, weight="bold", color=BF16)
    chips = ["activations", "KV cache", "RMSNorm", "RoPE", "SDPA"]
    cw = (inner_w - 0.16) / 5
    for i, name in enumerate(chips):
        s.chip(
            inner_x + i * (cw + 0.04),
            by + 0.22,
            cw - 0.04,
            0.28,
            name,
            fc=BF16_FILL,
            ec=BF16,
            fs=_T_TINY,
        )


def _part4(s: _Sheet, y: float) -> float:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    h = 5.18
    s.rect(x0, y, w, h, fc=GPU_FILL, ec=NF4, lw=0.8)
    s.badge(x0 + 0.10, y + 0.10, "4")
    s.text(x0 + 0.46, y + 0.12, "gpu.loop.TokenLoop  —  drives the transformer graph itself", fs=_T_HEAD, weight="bold")
    s.text(
        x0 + 0.46,
        y + 0.34,
        "Does not call transformers.generate. Owns KV (slot write, no torch.cat), the RoPE table, RMS, SDPA.",
        fs=_T_BODY,
        color=INK_SOFT,
    )

    gap = 0.12
    inner_x = x0 + 0.12
    inner_w = w - 0.24
    left_w = inner_w * 0.46
    right_w = inner_w - gap - left_w
    gy = y + 0.54
    gh = 2.55

    # layer body
    s.rect(inner_x, gy, left_w, gh, fc=PAPER, ec=RULE, lw=0.7)
    s.text(inner_x + 0.10, gy + 0.08, "One layer  (TokenLoop.forward)", fs=_T_CLAIM, weight="bold")
    body = [
        ("BF16", "embed(ids)          lookup, not GEMM"),
        ("BF16", "rms(x)              PyTorch"),
        ("NF4", "qkv / wqkv          chr_nf4_gemm"),
        ("BF16", "rope(q, k, pos)     table"),
        ("BF16", "kv[layer, pos]=k,v  slot, no cat"),
        ("BF16", "sdpa(q, kv[:seq])   PyTorch"),
        ("NF4", "o_proj              chr_nf4_gemm"),
        ("NF4", "SwiGLU gate/up/down chr_nf4_gemm"),
        ("NF4", "lm_head(rms(x))     chr_nf4_gemm"),
    ]
    for i, (kind, line) in enumerate(body):
        ly = gy + 0.32 + i * 0.23
        fc, ec = (NF4_FILL, NF4) if kind == "NF4" else (BF16_FILL, BF16)
        s.chip(inner_x + 0.08, ly, 0.48, 0.20, kind, fc=fc, ec=ec, fs=6.2, weight="bold")
        s.text(inner_x + 0.64, ly + 0.10, line, fs=_T_TINY, va="center", family=MONO)

    # N policy
    rx = inner_x + left_w + gap
    s.rect(rx, gy, right_w, gh, fc=PAPER, ec=NF4, lw=0.9)
    s.text(rx + 0.10, gy + 0.08, f"N  ·  LIVE_MAX_N = {LIVE_MAX_N}", fs=_T_CLAIM, weight="bold")
    s.chip(rx + 0.10, gy + 0.36, right_w * 0.44, 0.46, "Prefill\nchunks of 16", fc=NF4_FILL, ec=NF4, fs=_T_SMALL, weight="bold")
    s.chip(
        rx + 0.10 + right_w * 0.46,
        gy + 0.36,
        right_w * 0.44,
        0.46,
        "Decode\nN = 1",
        fc=PAPER,
        ec=INK,
        fs=_T_SMALL,
        weight="bold",
    )
    s.text(
        rx + 0.10,
        gy + 0.92,
        "TokenLoop.prefill_chunk = 16. It does not launch n32/n64.\n"
        "If a leading dim is N>16, CompressedLinear.nf4_linear\n"
        "chunks it into n16 launches (kernel returns -2 above 16).\n"
        "The host does not pad tails to 16.\n"
        "\n"
        "Planned n32 tile (not the live TokenLoop path):\n"
        "one q_proj GEMM, 69 µs vs two n16 113 µs (1.63×).\n"
        "Measured. Not wired. LIVE_MAX_N was not raised.",
        fs=_T_TINY,
    )

    # glue + refusals
    fy = gy + gh + 0.10
    fh = y + h - gy - gh - 0.22
    glue_w = inner_w * 0.56
    s.rect(inner_x, fy, glue_w, fh, fc=PAPER, ec=RULE, lw=0.65)
    s.text(inner_x + 0.10, fy + 0.08, "Glue families that exist", fs=_T_CLAIM, weight="bold")
    s.chip(
        inner_x + 0.10,
        fy + 0.34,
        glue_w * 0.46,
        0.28,
        "llama_swiglu",
        fc=NF4_FILL,
        ec=NF4,
        fs=_T_SMALL,
        weight="bold",
    )
    s.chip(
        inner_x + 0.10 + glue_w * 0.48,
        fy + 0.34,
        glue_w * 0.46,
        0.28,
        "internlm_gqa",
        fc=NF4_FILL,
        ec=NF4,
        fs=_T_SMALL,
        weight="bold",
    )
    s.text(
        inner_x + 0.10,
        fy + 0.72,
        "llama_swiglu  —  split q/k/v/o + SwiGLU + Llama RMS + full causal.\n"
        "Measured: Qwen2.5 3B/14B. Also Llama 3.x without qk-norm;\n"
        "Mistral only when the sliding window is inert.\n"
        "\n"
        "internlm_gqa  —  same body, fused wqkv + InternLM GQA packer.\n"
        "Measured: internlm2.5-20B. Packer is the matrix name (wqkv),\n"
        "never “fused, so InternLM”.",
        fs=_T_TINY,
    )

    ref_x = inner_x + glue_w + gap
    ref_w = inner_w - glue_w - gap
    s.rect(ref_x, fy, ref_w, fh, fc=LIMIT_FILL, ec=LIMIT, lw=0.8)
    s.text(ref_x + 0.10, fy + 0.08, "Named refusals", fs=_T_CLAIM, weight="bold", color=LIMIT)
    refuses = [
        "Gemma  (gemma_gelu)",
        "Phi-3  (phi3_concat)",
        "MoE experts",
        "vision / multimodal",
        "Ada / Hopper / Blackwell",
        "GGUF  (never opened)",
        "live Mistral SWA, qk-norm",
    ]
    for i, name in enumerate(refuses):
        s.text(ref_x + 0.12, fy + 0.34 + i * 0.18, "·  " + name, fs=_T_TINY, color=INK)
    return y + h + 0.10


def _footer(s: _Sheet, y: float) -> None:
    x0 = _M_LEFT
    w = FIG_W - _M_LEFT - _M_RIGHT
    h = 0.92
    s.rect(x0, y, w, h, fc=BAND, ec=RULE, lw=0.6)
    s.text(x0 + 0.12, y + 0.10, "Runtime inputs  ·  after compress", fs=_T_CLAIM, weight="bold")
    s.text(
        x0 + 0.12,
        y + 0.32,
        "Always two things: a HuggingFace directory (config.json + tokenizer; InternLM2 also remote-code Python) "
        "and one .chr of packed NF4 weights.\n"
        "Safetensor shards are not opened at generate time. Reconstruction is tile-local in registers — "
        "not unpack-layer-into-VRAM, multiply, pack again.\n"
        "Packed residency is prior art. Display VRAM is inside nvidia-smi (not subtracted). "
        "This sheet is the path, not a tok/s leaderboard. Generate: Ampere sm_86 only.",
        fs=_T_TINY,
        color=INK,
    )


def redraw(*extra: str | Path, repo: str | Path | None = None) -> list[Path]:
    """Write the PNG. No GPU, no torch."""
    figure = stack_plate()
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
        prog="python -m gpu.lab.stack_plate",
        description="Redraw docs/img/stack.png. Matplotlib only, no GPU.",
    )
    parser.add_argument("--redraw", action="store_true", help="write docs/img/stack.png")
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

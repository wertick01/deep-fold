"""The lab plate: one printable figure that argues the NF4 driver, panel by panel.

This is a paper figure, not a dashboard. Geometry is fixed in pixels and the
axis domains are computed by hand, so the plate looks the same in ``lab.html``,
in ``lab.png`` and on a poster board. One typeface, two codec colours that
survive grayscale, a line style per codec so colour is never the only encoding.

The plate reads top to bottom as an argument, one claim per panel:

    A  VRAM is the product ......... two graphs side by side (uncompressed |
                                     compressed), one 0..12288 MiB scale.
    F  GPU memory past VRAM ........ shown only when CUDA holds more than the
                                     card: working set and shared GPU memory
                                     (system RAM), same Y on both columns.
    B  Weights stay packed ......... where the VRAM after load actually goes:
                                     packed weights, the rest of torch, what sits
                                     outside torch, and the free headroom.
    C  The model still talks ....... the three replies and their needles.
    D  TTFT is prefill ............. first token per turn.
    E  Decode is a different stack .  tok/s per turn, with the caveat in the panel.

Panel titles are written from the data, so the plate cannot claim a win the
numbers do not support: if NF4 climbs into the BF16 trace, panel A says a layer
was materialised; if BF16 decodes faster, panel E says so.

``comparison_figure(bundle)`` returns that plate. ``telemetry_figure(bundle)``
is the optional second sheet (allocator, utilisation, power) for readers who
want the raw instrument traces; ``write_artifacts`` puts either next to the CSVs.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .bundle import CODECS, LabBundle
from .script import NEEDLES

__all__ = [
    "CODEC_STYLE",
    "comparison_figure",
    "telemetry_figure",
    "write_artifacts",
    "write_png",
]

# --------------------------------------------------------------------------- #
# design tokens -- one typeface, one ink, one accent per codec
# --------------------------------------------------------------------------- #

FONT = "Arial, Helvetica, sans-serif"

INK = "#151515"
INK_SOFT = "#5B5B5B"
RULE = "#9A9A9A"
GRID = "#E9E9E9"
LIMIT = "#8C1D18"  # the only red on the plate: the card limit and bad news

SIZE_TITLE = 18.0
SIZE_SUB = 12.0
SIZE_CLAIM = 13.0
SIZE_NOTE = 10.5
SIZE_BODY = 11.0
SIZE_TICK = 10.5
SIZE_SMALL = 9.5
SIZE_CAPTION = 10.0

# Okabe-Ito orange and blue: colourblind-safe, and far enough apart in
# luminance to stay two different grays on a black-and-white printer. The dash
# is the second encoding, so neither colour nor style carries the codec alone.
CODEC_STYLE: dict[str, dict[str, str]] = {
    "bf16": {
        "label": "BF16",
        "legend": "BF16 dense",
        "long": "BF16 dense — HuggingFace generate",
        "color": "#E69F00",
        "dash": "solid",
        "pattern": "",
    },
    "nf4": {
        "label": "NF4",
        "legend": "NF4 driver",
        "long": "NF4 driver — CompressedLinear + TokenLoop",
        "color": "#0072B2",
        "dash": "longdash",
        "pattern": "/",
    },
}

# Panel B stack, innermost first. A neutral ramp: it prints as four distinct
# grays, so this panel needs no second encoding, and the codec colours go on
# meaning "codec" everywhere else on the plate.
BUDGET_SEGMENTS = (
    ("weight", "model weights", "#33383D", "#FFFFFF"),
    ("other", "rest of torch (KV cache, activations)", "#8A9099", "#FFFFFF"),
    ("outside", "outside torch (CUDA context, desktop)", "#C3C8CD", INK),
    ("free", "free headroom on the card", "#F2F3F4", INK_SOFT),
)

TITLE = "Qwen2.5-3B-Instruct on one RTX 3080 12 GB — NF4 driver against dense BF16"
SUBTITLE = (
    "One card, one chat script, three identical prompts per codec. "
    "Each codec runs in its own process so VRAM returns before the next load. "
    "VRAM graphs sit side by side on one 0–12288 MiB scale."
)
STACKS = (
    "BF16 dense = HuggingFace from_pretrained + generate.  "
    "NF4 driver = CompressedLinear + fused NF4 GEMM + TokenLoop."
)

CAPTION = (
    "Greedy decoding, max_new_tokens = 64, turns are independent — the KV cache is reset between "
    "prompts, so every time-to-first-token is a clean prefill. Time is session-local: each codec is "
    "loaded, warmed up, asked the three prompts and unloaded, and both sessions start at t = 0. "
    "nvidia-smi totals are reported as measured: they include the CUDA context and the Windows "
    "desktop compositor, and nothing is subtracted. Blank telemetry is a gap in the instrument, "
    "never a zero. Panel titles are generated from the tables below, not written by hand. "
    "Sources: timeline.csv, events.csv, messages.csv, summary.csv."
)

# --------------------------------------------------------------------------- #
# plate geometry, in pixels, measured down from the top of the plot area
# --------------------------------------------------------------------------- #

WIDTH = 1100
MARGIN = {"l": 66, "r": 30, "t": 146, "b": 78}

_CLAIM_BAND = 46  # room above a panel for its one-line claim plus the unit note
_WIDE_CLAIM_BAND = 54  # ... and for the half-width panels, where the note wraps

_A_TOP, _A_H, _A_AX = 46, 262, 52  # VRAM over time (axis band holds the x title)
_S_H, _S_AX = 220, 48  # CUDA working set / shared GPU memory, under VRAM
_B_H, _B_AX, _B_KEY = 88, 20, 30  # budget bars, tick band, swatch key
_C_H = 116  # replies
_D_H, _D_AX = 206, 34  # TTFT | tok/s

_B_TOP = _A_TOP + _A_H + _A_AX + _CLAIM_BAND
_C_TOP = _B_TOP + _B_H + _B_AX + _B_KEY + _CLAIM_BAND
_D_TOP = _C_TOP + _C_H + _WIDE_CLAIM_BAND

PLOT_H = _D_TOP + _D_H + _D_AX
PLOT_W = WIDTH - MARGIN["l"] - MARGIN["r"]
HEIGHT = PLOT_H + MARGIN["t"] + MARGIN["b"]
_ACTIVE_PLOT_H = float(PLOT_H)

_COL_LEFT = (0.0, 0.465)
_COL_RIGHT = (0.535, 1.0)

# Characters that fit on one line of a claim / note, per panel width. Arial at
# these sizes averages a shade over half the point size per glyph.
_WRAP_WIDE = (148, 186)
_WRAP_HALF = (66, 84)

# Panel C is a set-table drawn with annotations: column starts in axis units.
_TABLE_X = {"id": 0.0, "prompt": 0.026, "needle": 0.385, "bf16": 0.455, "nf4": 0.725}
_TABLE_ROW_Y = (0.60, 0.36, 0.12)
_TABLE_HEAD_Y = 0.87


def _set_plot_h(plot_h: float) -> None:
    """Switch the plate's vertical scale. Spill adds a row; the 3B plate does not."""
    global _ACTIVE_PLOT_H
    _ACTIVE_PLOT_H = float(plot_h)


def _py(px: float) -> float:
    """A pixel offset as a fraction of the plot area -- i.e. paper units."""
    return px / _ACTIVE_PLOT_H


def _domain(top: float, height: float) -> list[float]:
    """[y0, y1] in paper units for a band ``height`` px tall, ``top`` px down."""
    return [1.0 - (top + height) / _ACTIVE_PLOT_H, 1.0 - top / _ACTIVE_PLOT_H]


def _plate_geom(*, spill: bool) -> SimpleNamespace:
    """Vertical bands. Extra CUDA-working-set row only when a session left VRAM."""
    a_top, a_h, a_ax = _A_TOP, _A_H, _A_AX
    y = a_top + a_h + a_ax
    s_top: float | None = None
    if spill:
        y += _CLAIM_BAND
        s_top = y
        y += _S_H + _S_AX
    y += _CLAIM_BAND
    b_top = y
    y += _B_H + _B_AX + _B_KEY + _CLAIM_BAND
    c_top = y
    y += _C_H + _WIDE_CLAIM_BAND
    d_top = y
    plot_h = y + _D_H + _D_AX
    return SimpleNamespace(
        spill=spill,
        plot_h=plot_h,
        height=int(plot_h + MARGIN["t"] + MARGIN["b"]),
        a_top=a_top,
        a_h=a_h,
        s_top=s_top,
        s_h=_S_H,
        b_top=b_top,
        b_h=_B_H,
        c_top=c_top,
        c_h=_C_H,
        d_top=d_top,
        d_h=_D_H,
    )


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #


def _as_bundle(bundle: Any) -> LabBundle:
    """Accept a live bundle, a run directory, or the four tables as a mapping.

    The directory form is what lets the README and the notebook redraw a
    finished run with no GPU and no model on disk.
    """
    if isinstance(bundle, LabBundle):
        return bundle
    if isinstance(bundle, (str, Path)):
        return LabBundle.read(bundle)
    if isinstance(bundle, Mapping):
        return LabBundle.of(
            timeline=bundle.get("timeline", ()),
            events=bundle.get("events", ()),
            messages=bundle.get("messages", ()),
            summary=bundle.get("summary", ()),
            source=str(bundle.get("source", "live")),
        )
    for attribute in ("as_bundle", "bundle"):
        candidate = getattr(bundle, attribute, None)
        if callable(candidate):
            return _as_bundle(candidate())
        if isinstance(candidate, LabBundle):
            return candidate
    raise TypeError(f"cannot read a lab bundle out of {type(bundle).__name__}")


def _order(data: LabBundle) -> list[str]:
    return [codec for codec in CODECS if codec in data.codecs]


def _mib(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):,.0f}"


def _summary_value(data: LabBundle, codec: str, key: str) -> float | None:
    summary = data.summary_for(codec)
    if not summary:
        return None
    value = summary.get(key)
    return None if value is None else float(value)


def _event_time(data: LabBundle, codec: str, event: str) -> float | None:
    for row in data.rows_for("events", codec):
        if row.get("event") == event:
            return float(row["t_s"])
    return None


def _loaded_window(data: LabBundle, codec: str) -> tuple[float, float] | None:
    """``load_end`` -> ``unload_start``: the stretch where the model is resident."""
    start = _event_time(data, codec, "load_end")
    end = _event_time(data, codec, "unload_start")
    if start is None or end is None or end <= start:
        times, _ = data.series(codec, "used_mib")
        if not times:
            return None
        span = times[-1] - times[0]
        return (times[0] + 0.45 * span, times[0] + 0.9 * span)
    return (start, end)


def _plateau(data: LabBundle, codec: str) -> float | None:
    """Median nvidia-smi reading while the model is resident."""
    window = _loaded_window(data, codec)
    if window is None:
        return None
    low, high = window
    samples = [
        float(row["used_mib"])
        for row in data.rows_for("timeline", codec)
        if row.get("used_mib") is not None and low <= float(row["t_s"]) <= high
    ]
    return median(samples) if samples else None


def _after_load(data: LabBundle, codec: str) -> float | None:
    """What summary.csv reports after load, or the plateau if it is missing."""
    reported = _summary_value(data, codec, "vram_after_load_smi_mib")
    return reported if reported is not None else _plateau(data, codec)


def _after_load_torch(data: LabBundle, codec: str) -> float | None:
    reported = _summary_value(data, codec, "vram_after_load_torch_mib")
    if reported is not None:
        return reported
    return _peak_column(data, codec, "torch_alloc_mib")


def _working_series(data: LabBundle, codec: str) -> tuple[list[float], list[float | None]]:
    """What CUDA still holds: reserved pool first, else allocated.

    After decode, ``memory_allocated`` can fall while the WDDM working set
    (nvidia-smi + shared GPU memory) stays full; ``memory_reserved`` tracks that.
    """
    times, reserved = data.series(codec, "torch_reserved_mib")
    if times and any(value is not None and value > 0 for value in reserved):
        return times, reserved
    return data.series(codec, "torch_alloc_mib")


def _working_after_load(data: LabBundle, codec: str) -> float | None:
    reserved = _peak_column(data, codec, "torch_reserved_mib")
    alloc = _after_load_torch(data, codec)
    if reserved is None:
        return alloc
    if alloc is None:
        return reserved
    return max(float(reserved), float(alloc))


def _peak_column(data: LabBundle, codec: str, column: str) -> float | None:
    values = [float(value) for value in data.series(codec, column)[1] if value is not None]
    return max(values) if values else None


def _shared_after_load(data: LabBundle, codec: str) -> float | None:
    """CUDA bytes that do not fit in nvidia-smi dedicated VRAM.

    On Windows WDDM that remainder is Task Manager's Shared GPU memory
    (system RAM). Empty when CUDA sits inside the card.
    """
    torch_mib = _working_after_load(data, codec)
    used = _after_load(data, codec)
    if torch_mib is None or used is None:
        return None
    extra = float(torch_mib) - float(used)
    return extra if extra > 256.0 else 0.0


def _spills_vram(data: LabBundle) -> bool:
    """True when any session's CUDA working set is larger than the card."""
    total = data.total_mib()
    for codec in data.codecs:
        torch_mib = _working_after_load(data, codec)
        if torch_mib is not None and torch_mib > total * 1.05:
            return True
        peak = _peak_column(data, codec, "torch_reserved_mib") or _peak_column(
            data, codec, "torch_alloc_mib"
        )
        if peak is not None and peak > total * 1.05:
            return True
    return False


def _message_value(data: LabBundle, codec: str, key: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in data.rows_for("messages", codec):
        message_id = row.get("message_id")
        value = row.get(key)
        if message_id is not None and value is not None:
            out[int(message_id)] = float(value)
    return out


def _turn_ids(data: LabBundle) -> list[int]:
    ids = {
        int(row["message_id"])
        for row in data.messages
        if row.get("message_id") is not None
    }
    return sorted(ids)


# --------------------------------------------------------------------------- #
# typography helpers
# --------------------------------------------------------------------------- #


def _ann(fig: go.Figure, **kwargs: Any) -> None:
    """One annotation, never rotated, always in the plate's typeface."""
    font = dict(kwargs.pop("font", {}) or {})
    kwargs.setdefault("showarrow", False)
    kwargs.setdefault("align", "left")
    kwargs["textangle"] = 0
    kwargs["font"] = {"family": FONT, "size": SIZE_BODY, "color": INK} | font
    fig.add_annotation(**kwargs)


def _axis_ref(axis: int, letter: str) -> str:
    return f"{letter} domain" if axis == 1 else f"{letter}{axis} domain"


def _wrap(text: str, width: int) -> str:
    return "<br>".join(textwrap.wrap(text, width=width) or [""])


def _claim(
    fig: go.Figure,
    axis: int,
    letter: str,
    claim: str,
    note: str = "",
    *,
    color: str = INK,
    wrap: tuple[int, int] = _WRAP_WIDE,
    band: int = _CLAIM_BAND,
    paper: bool = False,
    y_paper: float | None = None,
) -> None:
    """The panel title: a bold letter and the sentence the reader should leave with.

    Anchored to the top of its band, not to the axis, so the letters of two
    panels in the same row stay on one line however the sentences wrap.
    ``paper=True`` spans the whole plate (VRAM's two columns share claim A).
    """
    text = f"<b>{letter}</b>   " + _wrap(claim, wrap[0])
    if note:
        text += (
            f'<br><span style="font-size:{SIZE_NOTE}px;color:{INK_SOFT}">'
            f"{_wrap(note, wrap[1])}</span>"
        )
    if paper:
        y = _domain(_A_TOP, _A_H)[1] if y_paper is None else y_paper
        xref, yref, x = "paper", "paper", 0.0
    else:
        xref, yref, x, y = _axis_ref(axis, "x"), _axis_ref(axis, "y"), 0, 1
    _ann(
        fig,
        xref=xref,
        yref=yref,
        x=x,
        y=y,
        xanchor="left",
        yanchor="top",
        yshift=band - 6,
        text=text,
        font={"size": SIZE_CLAIM, "color": color},
    )


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _swatch_key(
    fig: go.Figure,
    items: Sequence[tuple[str, str, float]],
    *,
    y: float,
    x: float = 0.0,
) -> None:
    """A hand-drawn key: filled rectangles plus labels, laid out left to right.

    Plotly's own legend belongs to the codec lines at the top of the plate. The
    stack in panel B needs its own key directly under the bars, where a floating
    legend box would either cover data or drift on export.
    """
    swatch_h, gap = 9.0, 22.0
    cursor = x
    for fill, label, swatch_w in items:
        x1 = cursor + swatch_w / PLOT_W
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=cursor,
            x1=x1,
            y0=y - _py(swatch_h / 2),
            y1=y + _py(swatch_h / 2),
            fillcolor=fill,
            line={"color": RULE if fill != INK else INK, "width": 0.6},
        )
        _ann(
            fig,
            xref="paper",
            yref="paper",
            x=x1 + 5.0 / PLOT_W,
            y=y,
            xanchor="left",
            yanchor="middle",
            text=label,
            font={"size": SIZE_SMALL, "color": INK_SOFT},
        )
        cursor = x1 + (5.0 + SIZE_SMALL * 0.545 * len(label) + gap) / PLOT_W


# --------------------------------------------------------------------------- #
# panel A -- VRAM over the session
# --------------------------------------------------------------------------- #


def _claim_vram(data: LabBundle, *, spill: bool = False) -> tuple[str, str, str]:
    total = data.total_mib()
    note = (
        f"nvidia-smi dedicated VRAM, MiB · left uncompressed, right compressed · "
        f"same 0–{_mib(total)} scale · session-local time"
        + (
            " · this meter stops at the card; shared GPU memory is the pair below"
            if spill
            else ""
        )
    )
    nf4, bf16 = _after_load(data, "nf4"), _after_load(data, "bf16")
    if nf4 is None or bf16 is None:
        present = [c for c in _order(data) if _after_load(data, c) is not None]
        if not present:
            return ("No session recorded an after-load VRAM reading.", note, INK_SOFT)
        codec = present[0]
        value = _after_load(data, codec)
        return (
            f"Only the {CODEC_STYLE[codec]['label']} session is in this run: "
            f"{_mib(value)} MiB after load. There is nothing to compare it against yet.",
            note,
            INK_SOFT,
        )
    if spill:
        return (
            f"nvidia-smi is capped at the card: BF16 fills {_mib(bf16)} of {_mib(total)} MiB "
            f"VRAM and spills the rest into shared GPU memory. NF4 holds {_mib(nf4)} MiB "
            "and stays on the card.",
            note,
            INK,
        )
    if nf4 >= 0.9 * bf16:
        return (
            f"The NF4 trace sits at {_mib(nf4)} MiB against {_mib(bf16)} MiB for BF16 — the two "
            "lines have met, so a layer was materialised. That is a bug, not a win.",
            note,
            LIMIT,
        )
    delta = bf16 - nf4
    return (
        f"VRAM is the product: after load NF4 holds the model in {_mib(nf4)} MiB where dense BF16 "
        f"needs {_mib(bf16)} MiB — {_mib(delta)} MiB, {100 * delta / total:.0f}% of the card, given back.",
        note,
        INK,
    )


def _panel_vram(fig: go.Figure, data: LabBundle) -> None:
    """Two VRAM traces side by side: uncompressed left, compressed right, same Y."""
    total = data.total_mib()
    columns = (("bf16", 1, "uncompressed BF16"), ("nf4", 2, "compressed NF4"))
    for codec, col, heading in columns:
        style = CODEC_STYLE[codec]
        times, used = data.series(codec, "used_mib")
        t_end = max(times) if times else 1.0
        if times:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=used,
                    name=style["legend"],
                    legendgroup=codec,
                    mode="lines",
                    line={"color": style["color"], "width": 2.1, "dash": style["dash"]},
                    connectgaps=False,
                    hovertemplate=(
                        f"<b>{style['label']}</b> %{{y:,.0f}} MiB"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        fig.add_hline(
            y=total,
            line={"color": LIMIT, "width": 1.1, "dash": "dot"},
            row=1,
            col=col,
        )
        axis = col  # yaxis / yaxis2
        if not times:
            _ann(
                fig,
                xref=_axis_ref(axis, "x"),
                yref=_axis_ref(axis, "y"),
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                text="no session (skipped or OOM)",
                font={"size": SIZE_BODY, "color": INK_SOFT},
            )
        _ann(
            fig,
            xref=_axis_ref(axis, "x"),
            yref=_axis_ref(axis, "y"),
            x=0.5,
            y=1,
            xanchor="center",
            yanchor="top",
            yshift=-8,
            text=f"<b>{heading}</b>",
            font={"size": SIZE_NOTE, "color": style["color"]},
            bgcolor="rgba(255,255,255,0.82)",
        )
        if times:
            window = _loaded_window(data, codec)
            level = _after_load(data, codec)
            if window is not None and level is not None:
                low, high = window
                above = level < 0.8 * total
                _ann(
                    fig,
                    xref=_axis_ref(axis, "x"),
                    yref=_axis_ref(axis, "y"),
                    x=low + 0.22 * (high - low),
                    y=level,
                    xanchor="center",
                    yanchor="bottom" if above else "top",
                    yshift=7 if above else -7,
                    text=f"{_mib(level)} MiB",
                    font={"size": SIZE_SMALL, "color": INK},
                    bgcolor="rgba(255,255,255,0.82)",
                )
        _ann(
            fig,
            xref=_axis_ref(axis, "x"),
            yref=_axis_ref(axis, "y"),
            x=t_end,
            y=total,
            xanchor="right",
            yanchor="bottom",
            yshift=2,
            text=f"card limit {_mib(total)} MiB",
            font={"size": SIZE_SMALL, "color": LIMIT},
        )
        fig.update_xaxes(
            range=[0, t_end * 1.015],
            title_text="session time, s",
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikedash="dot",
            spikecolor=RULE,
            row=1,
            col=col,
        )
        fig.update_yaxes(
            range=[0, total * 1.05],
            dtick=2048,
            tickformat=",",
            matches="y" if col == 2 else None,
            row=1,
            col=col,
        )


def _claim_working(data: LabBundle) -> tuple[str, str, str]:
    total = data.total_mib()
    note = (
        "CUDA working set, MiB · torch.cuda.memory_reserved · same Y on both · "
        "the part above the card limit is shared GPU memory (system RAM on WDDM)"
    )
    bf16 = _working_after_load(data, "bf16")
    nf4 = _working_after_load(data, "nf4")
    shared = _shared_after_load(data, "bf16")
    if bf16 is None and nf4 is None:
        return ("No CUDA working-set samples were recorded.", note, INK_SOFT)
    if bf16 is not None and bf16 > total * 1.05:
        nf4_text = (
            f"NF4 stays at {_mib(nf4)} MiB, inside the card."
            if nf4 is not None
            else "NF4 has no working-set sample."
        )
        shared_text = (
            f"{_mib(shared)} MiB of that sits in shared GPU memory"
            if shared
            else "the overflow is shared GPU memory"
        )
        return (
            f"GPU memory is not just VRAM: after load BF16 holds {_mib(bf16)} MiB "
            f"({shared_text}). {nf4_text}",
            note,
            INK,
        )
    return (
        f"CUDA working set after load: BF16 {_mib(bf16)} MiB, NF4 {_mib(nf4)} MiB.",
        note,
        INK,
    )


def _panel_working(fig: go.Figure, data: LabBundle, *, row: int = 2) -> None:
    """Two CUDA-working-set graphs under VRAM, one shared Y that can exceed the card."""
    total = data.total_mib()
    peaks = [_working_after_load(data, codec) or 0.0 for codec in ("bf16", "nf4")]
    ymax = max([total, *peaks]) * 1.05
    dtick = 4096.0 if ymax > 16384 else 2048.0
    columns = (
        ("bf16", 1, "uncompressed — VRAM + shared"),
        ("nf4", 2, "compressed — VRAM + shared"),
    )
    for codec, col, heading in columns:
        style = CODEC_STYLE[codec]
        times, alloc = _working_series(data, codec)
        t_end = max(times) if times else 1.0
        if times and any(value is not None for value in alloc):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=alloc,
                    name=f"{style['label']} working set",
                    legendgroup=codec,
                    showlegend=False,
                    mode="lines",
                    line={"color": style["color"], "width": 2.1, "dash": style["dash"]},
                    connectgaps=False,
                    hovertemplate=(
                        f"<b>{style['label']} CUDA</b> %{{y:,.0f}} MiB"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        fig.add_hline(
            y=total,
            line={"color": LIMIT, "width": 1.1, "dash": "dot"},
            row=row,
            col=col,
        )
        axis = 2 + col  # yaxis3 / yaxis4 when this is row 2 of a 5-row plate
        _ann(
            fig,
            xref=_axis_ref(axis, "x"),
            yref=_axis_ref(axis, "y"),
            x=0.5,
            y=1,
            xanchor="center",
            yanchor="top",
            yshift=-8,
            text=f"<b>{heading}</b>",
            font={"size": SIZE_NOTE, "color": style["color"]},
            bgcolor="rgba(255,255,255,0.82)",
        )
        level = _working_after_load(data, codec)
        shared = _shared_after_load(data, codec)
        if times and level is not None:
            _ann(
                fig,
                xref=_axis_ref(axis, "x"),
                yref=_axis_ref(axis, "y"),
                x=t_end * 0.55,
                y=level,
                xanchor="center",
                yanchor="bottom" if level < 0.85 * ymax else "top",
                yshift=7,
                text=(
                    f"{_mib(level)} MiB"
                    + (f"  ({_mib(shared)} shared)" if shared else "  (inside the card)")
                ),
                font={"size": SIZE_SMALL, "color": INK},
                bgcolor="rgba(255,255,255,0.82)",
            )
        _ann(
            fig,
            xref=_axis_ref(axis, "x"),
            yref=_axis_ref(axis, "y"),
            x=t_end,
            y=total,
            xanchor="right",
            yanchor="bottom",
            yshift=2,
            text=f"card limit {_mib(total)} MiB",
            font={"size": SIZE_SMALL, "color": LIMIT},
        )
        fig.update_xaxes(
            range=[0, t_end * 1.015],
            title_text="session time, s",
            row=row,
            col=col,
        )
        fig.update_yaxes(
            range=[0, ymax],
            dtick=dtick,
            tickformat=",",
            matches="y3" if col == 2 else None,
            row=row,
            col=col,
        )


def _plateau_labels(fig: go.Figure, data: LabBundle) -> None:
    """Name each line where it is flat, so the plate reads without the legend."""
    total = data.total_mib()
    for codec in _order(data):
        window = _loaded_window(data, codec)
        level = _after_load(data, codec)
        if window is None or level is None:
            continue
        low, high = window
        above = level < 0.8 * total
        _ann(
            fig,
            xref="x",
            yref="y",
            x=low + 0.22 * (high - low),
            y=level,
            xanchor="center",
            yanchor="bottom" if above else "top",
            yshift=7 if above else -7,
            # The codec name carries the colour; the number stays ink, so the
            # label is still readable off a black-and-white printer.
            text=(
                f'<span style="color:{CODEC_STYLE[codec]["color"]}">'
                f"<b>{CODEC_STYLE[codec]['label']}</b></span> {_mib(level)} MiB"
            ),
            font={"size": SIZE_SMALL, "color": INK},
            bgcolor="rgba(255,255,255,0.82)",
        )


def _delta_bracket(fig: go.Figure, data: LabBundle) -> None:
    """The one number the figure exists for, drawn between the two plateaus.

    Both ends are the after-load reading summary.csv carries, so the bracket and
    the panel title can never disagree by a sample of allocator wobble.
    """
    windows = {codec: _loaded_window(data, codec) for codec in ("bf16", "nf4")}
    levels = {codec: _after_load(data, codec) for codec in ("bf16", "nf4")}
    if any(value is None for value in (*windows.values(), *levels.values())):
        return
    low = max(window[0] for window in windows.values())  # type: ignore[index]
    high = min(window[1] for window in windows.values())  # type: ignore[index]
    if high <= low:
        return
    x = low + 0.62 * (high - low)
    top, bottom = max(levels.values()), min(levels.values())  # type: ignore[type-var]
    if top - bottom <= 0:
        return
    cap = 0.006 * max(float(row["t_s"]) for row in data.timeline)
    fig.add_shape(
        type="line", xref="x", yref="y", x0=x, x1=x, y0=bottom, y1=top,
        line={"color": INK_SOFT, "width": 1.0},
    )
    for y in (bottom, top):
        fig.add_shape(
            type="line", xref="x", yref="y", x0=x - cap, x1=x + cap, y0=y, y1=y,
            line={"color": INK_SOFT, "width": 1.0},
        )
    _ann(
        fig,
        xref="x",
        yref="y",
        x=x,
        y=(top + bottom) / 2,
        xanchor="left",
        yanchor="middle",
        xshift=7,
        text=f"<b>{_mib(top - bottom)} MiB</b><br>given back",
        font={"size": SIZE_SMALL, "color": INK},
        bgcolor="rgba(255,255,255,0.82)",
    )


# --------------------------------------------------------------------------- #
# panel B -- where the VRAM after load goes
# --------------------------------------------------------------------------- #


def _budget(data: LabBundle, codec: str) -> dict[str, float] | None:
    """Split the after-load nvidia-smi reading into the four stacked segments."""
    smi = _after_load(data, codec)
    if smi is None:
        return None
    total = data.total_mib()
    weight = _summary_value(data, codec, "weight_mib") or 0.0
    torch_mib = _summary_value(data, codec, "vram_after_load_torch_mib")
    torch_mib = weight if torch_mib is None else max(torch_mib, weight)
    return {
        "weight": weight,
        "other": max(0.0, torch_mib - weight),
        "outside": max(0.0, smi - torch_mib),
        "free": max(0.0, total - smi),
        "smi": smi,
        "torch": torch_mib,
    }


def _claim_budget(data: LabBundle) -> tuple[str, str, str]:
    note = (
        "VRAM after load, MiB, same scale as A · nvidia-smi is the whole card, torch is the "
        "allocator; the difference is context and desktop, and it is not subtracted"
    )
    nf4 = _budget(data, "nf4")
    dense_weight = _summary_value(data, "bf16", "weight_mib")
    if nf4 is None:
        return ("summary.csv has no after-load VRAM for the NF4 session.", note, INK_SOFT)
    packed, torch_mib = nf4["weight"], nf4["torch"]
    if packed <= 0:
        return (
            f"The NF4 allocator holds {_mib(torch_mib)} MiB after load; summary.csv reports no "
            "weight size, so the packed share cannot be shown.",
            note,
            INK_SOFT,
        )
    tail = (
        f" — the dense BF16 copy of those weights ({_mib(dense_weight)} MiB) is never materialised"
        if dense_weight
        else ""
    )
    return (
        f"Weights stay packed: torch holds {_mib(torch_mib)} MiB for {_mib(packed)} MiB "
        f"of NF4 weights{tail}.",
        note,
        INK,
    )


def _panel_budget(fig: go.Figure, data: LabBundle, axis: int = 2, *, row: int = 2) -> None:
    col = 1
    total = data.total_mib()
    codecs = [codec for codec in _order(data) if _budget(data, codec) is not None]
    if not codecs:
        _ann(
            fig,
            xref=_axis_ref(axis, "x"),
            yref=_axis_ref(axis, "y"),
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            text="no after-load VRAM recorded",
            font={"size": SIZE_BODY, "color": INK_SOFT},
        )
        return

    # Horizontal bars read top-down, so the first category has to be last here.
    bars = list(reversed(codecs))
    labels = [
        f'<span style="color:{CODEC_STYLE[c]["color"]}"><b>{CODEC_STYLE[c]["label"]}</b></span>'
        for c in bars
    ]
    budgets = [_budget(data, codec) for codec in bars]

    base = [0.0] * len(bars)
    for key, name, fill, text_color in BUDGET_SEGMENTS:
        values = [float(budget[key]) for budget in budgets]  # type: ignore[index]
        texts = [
            f"{value:,.0f}" if value / total >= 0.045 else ""
            for value in values
        ]
        fig.add_trace(
            go.Bar(
                x=values,
                y=labels,
                base=list(base),
                orientation="h",
                name=name,
                offsetgroup="budget",
                width=0.52,
                showlegend=False,
                marker={
                    "color": fill,
                    "line": {"color": RULE if key == "free" else fill, "width": 0.6},
                },
                text=texts,
                textposition="inside",
                insidetextanchor="middle",
                textfont={"family": FONT, "size": SIZE_SMALL, "color": text_color},
                cliponaxis=False,
                hovertemplate=f"<b>{name}</b> %{{x:,.0f}} MiB<extra></extra>",
            ),
            row=row,
            col=col,
        )
        base = [b + v for b, v in zip(base, values)]

    peaks = [_summary_value(data, codec, "vram_peak_smi_mib") for codec in bars]
    if any(peak is not None for peak in peaks):
        fig.add_trace(
            go.Scatter(
                x=[peak for peak in peaks if peak is not None],
                y=[label for label, peak in zip(labels, peaks) if peak is not None],
                mode="markers",
                name="peak nvidia-smi",
                marker={
                    "symbol": "line-ns-open",
                    "size": 17,
                    "color": INK,
                    "line": {"color": INK, "width": 1.6},
                },
                showlegend=False,
                hovertemplate="<b>peak nvidia-smi</b> %{x:,.0f} MiB<extra></extra>",
            ),
            row=row,
            col=col,
        )

    fig.add_vline(
        x=total,
        line={"color": LIMIT, "width": 1.1, "dash": "dot"},
        row=row,
        col=col,
    )
    fig.update_xaxes(
        range=[0, total * 1.05],
        dtick=2048,
        tickformat=",",
        showgrid=True,
        gridcolor=GRID,
        row=row,
        col=col,
    )
    fig.update_yaxes(showgrid=False, ticks="", row=row, col=col)


# --------------------------------------------------------------------------- #
# panel C -- the replies
# --------------------------------------------------------------------------- #


def _needle(message_id: int) -> str:
    needles = NEEDLES.get(int(message_id))
    if not needles:
        return "—"
    first = needles[0]
    return first if first.isdigit() else first.capitalize()


def _claim_replies(data: LabBundle) -> tuple[str, str, str]:
    note = (
        "identical prompts, greedy, max_new_tokens = 64, independent turns · "
        "replies are clipped for width; messages.csv has the full text"
    )
    rows = [row for row in data.messages if row.get("message_id") is not None]
    if not rows:
        return ("messages.csv is empty: this run recorded no replies.", note, INK_SOFT)
    misses = [row for row in rows if not row.get("quality_ok")]
    needles = " / ".join(_needle(mid) for mid in _turn_ids(data))
    both = (
        " — and both codecs answer the same prompts"
        if len(_order(data)) > 1
        else ""
    )
    if not misses:
        return (
            f"The model still talks: all {len(rows)} replies hit their needle "
            f"({needles}){both}.",
            note,
            INK,
        )
    return (
        f"{len(misses)} of {len(rows)} replies missed the needle ({needles}); "
        "the compressed path is not answering like the baseline.",
        note,
        LIMIT,
    )


def _panel_replies(fig: go.Figure, data: LabBundle, axis: int) -> None:
    xref, yref = f"x{axis}", f"y{axis}"
    codecs = _order(data)
    turns = _turn_ids(data)[: len(_TABLE_ROW_Y)]

    def rule(y: float, width: float, color: str) -> None:
        fig.add_shape(
            type="line",
            xref=xref,
            yref=yref,
            x0=0,
            x1=1,
            y0=y,
            y1=y,
            line={"color": color, "width": width},
        )

    def cell(x: float, y: float, text: str, *, color: str = INK, size: float = SIZE_BODY) -> None:
        _ann(
            fig,
            xref=xref,
            yref=yref,
            x=x,
            y=y,
            xanchor="left",
            yanchor="middle",
            text=text,
            font={"size": size, "color": color},
        )

    rule(1.0, 1.3, INK)
    rule(0.74, 0.8, RULE)
    rule(0.0, 1.3, INK)

    cell(_TABLE_X["id"], _TABLE_HEAD_Y, "#", color=INK_SOFT, size=SIZE_SMALL)
    cell(
        _TABLE_X["prompt"], _TABLE_HEAD_Y, "prompt (identical for both codecs)",
        color=INK_SOFT, size=SIZE_SMALL,
    )
    cell(_TABLE_X["needle"], _TABLE_HEAD_Y, "needle", color=INK_SOFT, size=SIZE_SMALL)
    for codec in codecs:
        cell(
            _TABLE_X[codec],
            _TABLE_HEAD_Y,
            f'<span style="color:{CODEC_STYLE[codec]["color"]}">'
            f"<b>{CODEC_STYLE[codec]['label']}</b></span> reply",
            color=INK_SOFT,
            size=SIZE_SMALL,
        )

    if not turns:
        cell(_TABLE_X["prompt"], _TABLE_ROW_Y[0], "no replies recorded", color=INK_SOFT)
        return

    for turn, y in zip(turns, _TABLE_ROW_Y):
        replies = {
            str(row["codec"]): row
            for row in data.messages
            if row.get("message_id") is not None and int(row["message_id"]) == turn
        }
        prompt = next(
            (str(row.get("prompt", "")) for row in replies.values() if row.get("prompt")), ""
        )
        cell(_TABLE_X["id"], y, str(turn), color=INK_SOFT, size=SIZE_SMALL)
        cell(_TABLE_X["prompt"], y, _clip(prompt, 62))
        cell(_TABLE_X["needle"], y, _needle(turn), color=INK_SOFT)
        for codec in codecs:
            row = replies.get(codec)
            if row is None:
                cell(_TABLE_X[codec], y, "—", color=INK_SOFT)
                continue
            text = _clip(str(row.get("response", "")), 44)
            if row.get("quality_ok"):
                cell(_TABLE_X[codec], y, text)
            else:
                cell(_TABLE_X[codec], y, f"<b>missed:</b> {text}", color=LIMIT)


# --------------------------------------------------------------------------- #
# panels D and E -- timing, per turn, with the caveat in the title
# --------------------------------------------------------------------------- #


def _claim_ttft(data: LabBundle) -> tuple[str, str, str]:
    note = "time to first token per turn, ms — prefill only (first_token − msg_send)"
    nf4 = _summary_value(data, "nf4", "mean_ttft_ms")
    bf16 = _summary_value(data, "bf16", "mean_ttft_ms")
    if nf4 is None or bf16 is None:
        return ("TTFT is prefill: first token per turn, no mean recorded.", note, INK_SOFT)
    if nf4 < bf16:
        return (
            f"TTFT is prefill: NF4 gets there in {nf4:.0f} ms, "
            f"{bf16 - nf4:.0f} ms ahead of BF16.",
            note,
            INK,
        )
    return (
        f"TTFT is prefill, and NF4 pays it: {nf4:.0f} ms against {bf16:.0f} ms for BF16.",
        note,
        INK,
    )


def _claim_decode(data: LabBundle) -> tuple[str, str, str]:
    note = (
        "decode throughput per turn, tokens/s — HF generate with dense cuBLAS GEMM "
        "against the fused NF4 loop; not a kernel benchmark"
    )
    nf4 = _summary_value(data, "nf4", "mean_decode_tok_s")
    bf16 = _summary_value(data, "bf16", "mean_decode_tok_s")
    if nf4 is None or bf16 is None:
        return ("Decode throughput per turn; the two paths are different stacks.", note, INK_SOFT)
    if nf4 > bf16:
        return (
            f"Decode: NF4 {nf4:.1f} tok/s against BF16 {bf16:.1f} — ahead, on another stack.",
            note,
            INK,
        )
    return (
        f"Decode: BF16 {bf16:.1f} tok/s, NF4 {nf4:.1f} — we are not faster here.",
        note,
        INK,
    )


def _panel_turns(
    fig: go.Figure,
    data: LabBundle,
    *,
    row: int,
    col: int,
    key: str,
    decimals: int,
    unit: str,
) -> None:
    turns = _turn_ids(data)
    if not turns:
        return
    peak = 0.0
    for codec in _order(data):
        style = CODEC_STYLE[codec]
        values = _message_value(data, codec, key)
        series = [values.get(turn) for turn in turns]
        peak = max([peak] + [value for value in series if value is not None])
        fig.add_trace(
            go.Bar(
                x=[f"turn {turn}" for turn in turns],
                y=series,
                name=style["long"],
                legendgroup=codec,
                offsetgroup=codec,
                showlegend=False,
                width=0.34,
                marker={
                    "color": style["color"],
                    "line": {"color": style["color"], "width": 0.8},
                    "pattern": {
                        "shape": style["pattern"],
                        "bgcolor": style["color"],
                        "fgcolor": "#FFFFFF",
                        "size": 5,
                        "solidity": 0.22,
                    },
                },
                text=[
                    "" if value is None else f"{value:,.{decimals}f}" for value in series
                ],
                textposition="outside",
                textfont={"family": FONT, "size": SIZE_SMALL, "color": INK_SOFT},
                cliponaxis=False,
                hovertemplate=f"<b>{style['label']}</b> %{{y:,.{decimals}f}} {unit}<extra></extra>",
            ),
            row=row,
            col=col,
        )
    fig.update_yaxes(range=[0, peak * 1.26 if peak else 1], row=row, col=col)
    fig.update_xaxes(ticks="", row=row, col=col)


# --------------------------------------------------------------------------- #
# the plate
# --------------------------------------------------------------------------- #


def _header(
    fig: go.Figure,
    data: LabBundle,
    *,
    title: str = TITLE,
    subtitle: str = SUBTITLE,
) -> None:
    synthetic = data.synthetic
    band_fill = "#FBECEA" if synthetic else "#F2F4F5"
    band_ink = LIMIT if synthetic else INK_SOFT
    band_text = (
        "<b>SYNTHETIC FIXTURE DATA</b> — shaped like a session so the harness can be tested. "
        "Nothing here was measured; do not quote a single number off this plate."
        if synthetic
        else "<b>MEASURED RUN</b> — nvidia-smi and torch.cuda sampled throughout both sessions; "
        "the numbers are as recorded, with nothing corrected or subtracted."
    )

    _ann(
        fig,
        xref="paper",
        yref="paper",
        x=0,
        y=1 + _py(96),
        xanchor="left",
        yanchor="bottom",
        text=title,
        font={"size": SIZE_TITLE, "color": INK},
    )
    _ann(
        fig,
        xref="paper",
        yref="paper",
        x=0,
        y=1 + _py(72),
        xanchor="left",
        yanchor="bottom",
        text=subtitle,
        font={"size": SIZE_SUB, "color": INK_SOFT},
    )
    _ann(
        fig,
        xref="paper",
        yref="paper",
        x=0,
        y=1 + _py(53),
        xanchor="left",
        yanchor="bottom",
        text=STACKS,
        font={"size": SIZE_NOTE, "color": INK_SOFT},
    )
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        x1=1,
        y0=1 + _py(8),
        y1=1 + _py(42),
        fillcolor=band_fill,
        line={"color": band_ink, "width": 0.7},
        layer="below",
    )
    _ann(
        fig,
        xref="paper",
        yref="paper",
        x=0.006,
        y=1 + _py(25),
        xanchor="left",
        yanchor="middle",
        text=band_text,
        font={"size": SIZE_NOTE, "color": band_ink},
    )


def _caption(fig: go.Figure, data: LabBundle) -> None:
    label = "SYNTHETIC FIXTURE" if data.synthetic else "Measured run"
    body = textwrap.fill(f"{label}. {CAPTION}", width=176)
    _ann(
        fig,
        xref="paper",
        yref="paper",
        x=0,
        y=-_py(16),
        xanchor="left",
        yanchor="top",
        text=body.replace("\n", "<br>"),
        font={"size": SIZE_CAPTION, "color": INK_SOFT},
    )


def _style_axes(fig: go.Figure) -> None:
    fig.update_xaxes(
        showline=True,
        linecolor=RULE,
        linewidth=1,
        ticks="outside",
        ticklen=4,
        tickcolor=RULE,
        tickfont={"family": FONT, "size": SIZE_TICK, "color": INK_SOFT},
        title_font={"family": FONT, "size": SIZE_NOTE, "color": INK_SOFT},
        showgrid=False,
        zeroline=False,
        automargin=False,
    )
    fig.update_yaxes(
        showline=False,
        ticks="outside",
        ticklen=4,
        tickcolor=RULE,
        tickfont={"family": FONT, "size": SIZE_TICK, "color": INK_SOFT},
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        automargin=False,
    )


def comparison_figure(
    bundle: Any,
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
    title: str | None = None,
    subtitle: str | None = None,
) -> go.Figure:
    """The plate: VRAM left/right, optional GPU-memory pair, then budget, replies, TTFT, decode.

    Panel A is two graphs side by side (uncompressed | compressed) on one
    0–card MiB scale. If CUDA's working set exceeds VRAM, a second pair sits
    directly underneath — same Y on both columns, scale high enough to show
    shared GPU memory. Accepts a :class:`~gpu.lab.bundle.LabBundle`, a finished
    run directory, or a mapping of the four tables. ``title`` / ``subtitle``
    override the 3B defaults so a 14B or InternLM run does not wear the wrong name.
    """
    data = _as_bundle(bundle)
    spill = _spills_vram(data)
    geom = _plate_geom(spill=spill)
    _set_plot_h(geom.plot_h)
    if height == HEIGHT:
        height = geom.height

    if spill:
        fig = make_subplots(
            rows=5,
            cols=2,
            specs=[
                [{}, {}],
                [{}, {}],
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
                [{}, {}],
            ],
        )
        budget_row, replies_row, turns_row = 3, 4, 5
        budget_axis, replies_axis = 5, 6
        ttft_axis, decode_axis = 7, 8
    else:
        fig = make_subplots(
            rows=4,
            cols=2,
            specs=[
                [{}, {}],
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
                [{}, {}],
            ],
        )
        budget_row, replies_row, turns_row = 2, 3, 4
        budget_axis, replies_axis = 3, 4
        ttft_axis, decode_axis = 5, 6

    _style_axes(fig)

    fig.update_yaxes(domain=_domain(geom.a_top, geom.a_h), row=1, col=1)
    fig.update_yaxes(domain=_domain(geom.a_top, geom.a_h), row=1, col=2)
    if spill:
        fig.update_yaxes(domain=_domain(geom.s_top, geom.s_h), row=2, col=1)
        fig.update_yaxes(domain=_domain(geom.s_top, geom.s_h), row=2, col=2)
        fig.update_xaxes(domain=list(_COL_LEFT), row=2, col=1)
        fig.update_xaxes(domain=list(_COL_RIGHT), row=2, col=2)
    fig.update_yaxes(domain=_domain(geom.b_top, geom.b_h), row=budget_row, col=1)
    fig.update_yaxes(domain=_domain(geom.c_top, geom.c_h), row=replies_row, col=1)
    fig.update_yaxes(domain=_domain(geom.d_top, geom.d_h), row=turns_row, col=1)
    fig.update_yaxes(domain=_domain(geom.d_top, geom.d_h), row=turns_row, col=2)
    fig.update_xaxes(domain=list(_COL_LEFT), row=1, col=1)
    fig.update_xaxes(domain=list(_COL_RIGHT), row=1, col=2)
    fig.update_xaxes(domain=[0.0, 1.0], row=budget_row, col=1)
    fig.update_xaxes(domain=[0.0, 1.0], row=replies_row, col=1)
    fig.update_xaxes(domain=list(_COL_LEFT), row=turns_row, col=1)
    fig.update_xaxes(domain=list(_COL_RIGHT), row=turns_row, col=2)

    _panel_vram(fig, data)
    if spill:
        _panel_working(fig, data, row=2)
    _panel_budget(fig, data, axis=budget_axis, row=budget_row)
    _panel_replies(fig, data, axis=replies_axis)
    _panel_turns(fig, data, row=turns_row, col=1, key="prefill_ms", decimals=0, unit="ms")
    _panel_turns(fig, data, row=turns_row, col=2, key="decode_tok_s", decimals=1, unit="tok/s")

    fig.update_xaxes(range=[0, 1], visible=False, row=replies_row, col=1)
    fig.update_yaxes(range=[-0.06, 1.06], visible=False, row=replies_row, col=1)

    claims = [
        ("A", 1, _WRAP_WIDE, _CLAIM_BAND, True, _domain(geom.a_top, geom.a_h)[1],
         _claim_vram(data, spill=spill)),
        ("B", budget_axis, _WRAP_WIDE, _CLAIM_BAND, False, None, _claim_budget(data)),
        ("C", replies_axis, _WRAP_WIDE, _CLAIM_BAND, False, None, _claim_replies(data)),
        ("D", ttft_axis, _WRAP_HALF, _WIDE_CLAIM_BAND, False, None, _claim_ttft(data)),
        ("E", decode_axis, _WRAP_HALF, _WIDE_CLAIM_BAND, False, None, _claim_decode(data)),
    ]
    if spill:
        claims.insert(
            1,
            ("F", 3, _WRAP_WIDE, _CLAIM_BAND, True, _domain(geom.s_top, geom.s_h)[1],
             _claim_working(data)),
        )
    for letter, axis, wrap, band, paper, y_paper, (claim, note, color) in claims:
        _claim(
            fig,
            axis,
            letter,
            claim,
            note,
            color=color,
            wrap=wrap,
            band=band,
            paper=paper,
            y_paper=y_paper,
        )

    _swatch_key(
        fig,
        [(fill, name, 11.0) for _key, name, fill, _text in BUDGET_SEGMENTS]
        + [(INK, "peak nvidia-smi", 2.5)],
        y=_domain(geom.b_top, geom.b_h)[0] - _py(_B_AX + 14),
    )

    _header(fig, data, title=title or TITLE, subtitle=subtitle or SUBTITLE)
    _caption(fig, data)

    fig.update_layout(
        width=width,
        height=height,
        margin={**MARGIN, "autoexpand": False},
        template="none",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": FONT, "size": SIZE_BODY, "color": INK},
        barmode="group",
        bargap=0.42,
        hovermode="x unified",
        hoverlabel={"font": {"family": FONT, "size": SIZE_NOTE}, "namelength": -1},
        legend={
            "orientation": "h",
            "x": 1,
            "xanchor": "right",
            "y": 1 + _py(118),
            "yanchor": "top",
            "font": {"family": FONT, "size": SIZE_NOTE, "color": INK},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "itemsizing": "constant",
            "tracegroupgap": 6,
        },
        showlegend=True,
    )
    return fig


# --------------------------------------------------------------------------- #
# the optional second sheet: the raw instrument traces
# --------------------------------------------------------------------------- #

_TELEMETRY_ROWS: tuple[tuple[str, str, str, str], ...] = (
    (
        "A",
        "allocator",
        "What the allocator holds against what the card reports.",
        "MiB — nvidia-smi total (solid) and torch.cuda allocated (dotted); the gap is the CUDA "
        "context and the Windows desktop, and it is not subtracted",
    ),
    (
        "B",
        "util_gpu",
        "Where each session spends the SMs.",
        "SM utilisation, %",
    ),
    (
        "C",
        "util_mem",
        "Where each session spends memory bandwidth.",
        "memory-controller utilisation, %",
    ),
    (
        "D",
        "power_w",
        "Board power over the session.",
        "board power, W",
    ),
)


def telemetry_figure(
    bundle: Any,
    *,
    width: int = WIDTH,
    height: int = 1180,
    title: str | None = None,
) -> go.Figure:
    """The second sheet: allocator, utilisation and power, both codecs overlaid.

    Not the plate. This is the instrument trace a reviewer asks for after the
    plate has convinced them, and it is written next to it as ``lab-telemetry``.
    """
    data = _as_bundle(bundle)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.075)
    _style_axes(fig)

    t_end = max([float(row["t_s"]) for row in data.timeline] or [1.0])

    for index, (letter, column, claim, note) in enumerate(_TELEMETRY_ROWS, start=1):
        for codec in _order(data):
            style = CODEC_STYLE[codec]
            if column == "allocator":
                for field, dash, width_px, tag in (
                    ("used_mib", style["dash"], 2.0, "nvidia-smi"),
                    ("torch_alloc_mib", "dot", 1.4, "torch allocated"),
                ):
                    times, values = data.series(codec, field)
                    if not times or all(value is None for value in values):
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=times,
                            y=values,
                            name=f"{style['label']} · {tag}",
                            mode="lines",
                            line={"color": style["color"], "width": width_px, "dash": dash},
                            connectgaps=False,
                            showlegend=index == 1,
                            hovertemplate=f"<b>{style['label']} {tag}</b> %{{y:,.0f}} MiB<extra></extra>",
                        ),
                        row=index,
                        col=1,
                    )
                continue
            times, values = data.series(codec, column)
            if not times or all(value is None for value in values):
                continue
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=values,
                    name=style["long"],
                    mode="lines",
                    line={"color": style["color"], "width": 1.8, "dash": style["dash"]},
                    connectgaps=False,
                    showlegend=False,
                    hovertemplate=f"<b>{style['label']}</b> %{{y:,.1f}}<extra></extra>",
                ),
                row=index,
                col=1,
            )
        _claim(fig, index, letter, claim, note)

    total = data.total_mib()
    fig.add_hline(y=total, line={"color": LIMIT, "width": 1.1, "dash": "dot"}, row=1, col=1)
    fig.update_yaxes(range=[0, total * 1.05], dtick=2048, tickformat=",", row=1, col=1)
    fig.update_yaxes(range=[0, 104], row=2, col=1)
    fig.update_yaxes(range=[0, 104], row=3, col=1)
    fig.update_yaxes(rangemode="tozero", row=4, col=1)
    fig.update_xaxes(range=[0, t_end * 1.015], row=4, col=1)
    fig.update_xaxes(title_text="session time, s", row=4, col=1)

    _ann(
        fig,
        xref="paper",
        yref="paper",
        x=0,
        y=1.10,
        xanchor="left",
        yanchor="bottom",
        text="Instrument traces — " + TITLE,
        font={"size": SIZE_TITLE - 2, "color": INK},
    )
    if data.synthetic:
        _ann(
            fig,
            xref="paper",
            yref="paper",
            x=0,
            y=1.055,
            xanchor="left",
            yanchor="bottom",
            text="<b>SYNTHETIC FIXTURE DATA</b> — not a measurement.",
            font={"size": SIZE_NOTE, "color": LIMIT},
        )
    fig.update_layout(
        width=width,
        height=height,
        margin={"l": 66, "r": 30, "t": 150, "b": 60, "autoexpand": False},
        template="none",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": FONT, "size": SIZE_BODY, "color": INK},
        hovermode="x unified",
        hoverlabel={"font": {"family": FONT, "size": SIZE_NOTE}, "namelength": -1},
        legend={
            "orientation": "h",
            "x": 1,
            "xanchor": "right",
            "y": 1.10,
            "yanchor": "top",
            "font": {"family": FONT, "size": SIZE_NOTE, "color": INK},
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    return fig


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #


def write_png(fig: go.Figure, path: str | Path, *, scale: float = 2.0) -> tuple[bool, str]:
    """Try the static export. Returns ``(ok, reason)``.

    A missing kaleido is a recorded skip, never a placeholder image: the README
    would rather have no picture than a fake one.
    """
    try:
        fig.write_image(str(path), format="png", scale=scale)
    except Exception as error:  # noqa: BLE001 -- kaleido/chrome missing is the common case
        return False, f"{type(error).__name__}: {error}"
    return True, str(path)


def write_artifacts(
    fig: go.Figure,
    out_dir: str | Path,
    *,
    png: bool = True,
    include_plotlyjs: Any = "cdn",
    name: str = "lab",
) -> dict[str, str]:
    """``lab.html`` (required) and ``lab.png`` (optional, needs kaleido).

    ``name`` is there for the optional second sheet, which is written next to
    the plate as ``lab-telemetry.*``; the plate itself keeps the frozen names.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    html_path = directory / f"{name}.html"
    fig.write_html(str(html_path), include_plotlyjs=include_plotlyjs, full_html=True)
    artifacts = {"html": str(html_path)}
    if png:
        ok, reason = write_png(fig, directory / f"{name}.png")
        artifacts["png"] = reason if ok else ""
        artifacts["png_skip_reason"] = "" if ok else reason
    else:
        artifacts["png"] = ""
        artifacts["png_skip_reason"] = "not requested"
    return artifacts

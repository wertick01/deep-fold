"""Progress plate: first lab graphs through now, one matplotlib sheet.

This is a *third* plate. It does not replace the per-model VRAM figures
(``docs/img/lab-qwen25-3b.png``, ``lab-qwen25-14b.png``, ``lab-internlm20b.png``)
or the hard-eval Q&A sheet (notebook 06, ``docs/img/hard-eval-qwen25.png``).
Those stay as they are. This sheet exists so a reader can see the *arc*
without mixing 17.0, 31.6 and 28.4 tok/s as if they were one comparison.

    python -m gpu.lab.progress_plate --redraw
    python -m gpu.lab.test_progress

Colours, hatch and the card-limit red are imported from :mod:`gpu.lab.hard_plate`
so ``python -m gpu.lab.test_hard`` cannot drift. Plotly is not imported.

Numbers are read from CSVs / documented live dirs. A missing optional
directory is drawn as "not on this disk", never as a remembered figure.
"""

from __future__ import annotations

import argparse
import csv
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bundle import LabBundle
from .competitor import read_competitor_csv, read_microbench_csv
from .hard_plate import (
    CARD_MIB,
    CODEC_STYLE,
    FONT,
    GRID,
    INK,
    INK_SOFT,
    LIMIT,
    RULE,
    SIZE_BODY,
    SIZE_CLAIM,
    SIZE_NOTE,
    SIZE_SMALL,
    SIZE_SUB,
    SIZE_TITLE,
    write_plate,
)

# 3B decode grids. Duplicated from gpu/nf4/plan.py / test_plan.py so this
# module never imports ``gpu.nf4`` (that package pulls torch).
STARVED_QO_CTAS = 16
STARVED_KV_CTAS = 2
FIX_QO_CTAS = 128
FIX_KV_CTAS = 64
SMS_3080 = 70
LIVE_MAX_N = 16
_STARVED_INK = "#33383D"

__all__ = [
    "GSM8K_TEST_N",
    "ProgressStory",
    "default_paths",
    "load_progress",
    "progress_plate",
    "write_plate",
]

_REPO = Path(__file__).resolve().parents[2]

#: openai/gsm8k main test split. The slice CSV is not this many rows.
GSM8K_TEST_N = 1319

# --------------------------------------------------------------------------- #
# documented locations (live dirs sit outside git)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProgressPaths:
    repo: Path
    committed_3b: Path
    occupancy_nf4: Path
    paired_3b: Path
    committed_14b: Path
    committed_20b: Path
    hard: Path
    hard_20b: Path
    gsm8k: Path
    ncu: Path
    ncu_n32: Path
    competitor: Path
    bnb_e2e: Path


def default_paths(repo: str | Path | None = None) -> ProgressPaths:
    """Committed plates under the repo; live dirs under ``DEEPFOLD_RUNS``."""
    root = Path(repo) if repo else _REPO
    runs = Path(os.environ.get("DEEPFOLD_RUNS", r"C:\dev\models\runs"))
    return ProgressPaths(
        repo=root,
        committed_3b=root / "docs" / "runs" / "qwen25-3b",
        occupancy_nf4=runs / "wave2-streamA-3b-nf4",
        paired_3b=runs / "qwen25-3b-paired-20260913",
        committed_14b=root / "docs" / "runs" / "qwen25-14b",
        committed_20b=root / "docs" / "runs" / "internlm20b",
        hard=root / "docs" / "runs" / "hard-qwen25",
        hard_20b=runs / "hard-internlm20b-nf4-20260914",
        gsm8k=runs / "eval-qwen25-3b-gsm8k-200-20260913",
        ncu=root / "docs" / "runs" / "ncu",
        ncu_n32=runs / "ncu-n32-vs-2xn16-20260913",
        competitor=root / "docs" / "runs" / "competitor-qwen25-3b",
        bnb_e2e=runs / "competitor-qwen25-3b-20260913-bnb-e2e",
    )


def default_png(repo: str | Path | None = None) -> Path:
    root = Path(repo) if repo else _REPO
    return root / "docs" / "img" / "progress-3080.png"


# --------------------------------------------------------------------------- #
# story records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CodecStats:
    tok_s: float | None = None
    ttft_ms: float | None = None
    weight_mib: float | None = None
    smi_after_mib: float | None = None
    smi_peak_mib: float | None = None
    working_set_mib: float | None = None
    n_messages: int | None = None
    quality_all_ok: bool | None = None
    notes: str = ""


@dataclass(frozen=True)
class SessionSlice:
    """One lab session, labeled with the plate / directory it came from."""

    key: str
    title: str
    plate: str
    path: Path | None
    present: bool
    codecs: dict[str, CodecStats] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True)
class HardSlice:
    label: str
    size: str
    codec: str
    n_ok: int | None
    n_items: int
    accuracy: float | None
    plate: str


@dataclass(frozen=True)
class Gsm8kSlice:
    present: bool
    path: Path | None
    plate: str
    n_items: int
    corpus_n: int
    codecs: dict[str, CodecStats] = field(default_factory=dict)
    n_ok: dict[str, int] = field(default_factory=dict)
    disagreements: int | None = None
    note: str = ""


@dataclass(frozen=True)
class NcuCase:
    case: str
    kernel: str
    dram_pct: float | None
    tensor_pct: float | None
    occupancy_pct: float | None


@dataclass(frozen=True)
class N32Gemm:
    """True n32 vs two LIVE_MAX_N launches. One linear, not TTFT."""

    present: bool
    path: Path | None
    one_n16_us: float | None = None
    two_n16_us: float | None = None
    true_n32_us: float | None = None
    ratio: float | None = None
    n32_kernel: str = ""
    two_kernel: str = ""
    n32_occ: float | None = None
    two_occ: float | None = None


@dataclass(frozen=True)
class CompetitorSlice:
    present: bool
    path: Path | None
    n_stacks: int
    n_skip: int
    n_measured: int
    stacks: tuple[str, ...]
    bnb_note: str
    note: str
    live_tok_s: float | None = None
    live_ttft_ms: float | None = None
    live_smi: float | None = None
    live_smoke: bool | None = None
    micro_skip: int | None = None
    micro_n: int | None = None


@dataclass(frozen=True)
class OccupancyGrid:
    sms: int
    starved_qo: int
    starved_kv: int
    fix_qo: int
    fix_kv: int
    live_max_n: int


@dataclass(frozen=True)
class ProgressStory:
    committed_3b: SessionSlice
    occupancy_nf4: SessionSlice
    paired_3b: SessionSlice
    fit_14b: SessionSlice
    internlm_20b: SessionSlice
    hard: tuple[HardSlice, ...]
    gsm8k: Gsm8kSlice
    ncu: tuple[NcuCase, ...]
    n32: N32Gemm
    internlm_hard: HardSlice | None
    competitor: CompetitorSlice
    occupancy: OccupancyGrid


# --------------------------------------------------------------------------- #
# loaders — CSV only, no GPU
# --------------------------------------------------------------------------- #


def _stats_from_row(row: Mapping[str, Any] | None) -> CodecStats | None:
    if not row:
        return None
    n_msg = row.get("n_messages")
    return CodecStats(
        tok_s=row.get("mean_decode_tok_s"),
        ttft_ms=row.get("mean_ttft_ms"),
        weight_mib=row.get("weight_mib"),
        smi_after_mib=row.get("vram_after_load_smi_mib"),
        smi_peak_mib=row.get("vram_peak_smi_mib"),
        working_set_mib=row.get("vram_after_load_torch_mib"),
        n_messages=None if n_msg is None else int(n_msg),
        quality_all_ok=row.get("quality_all_ok"),
        notes=str(row.get("notes") or ""),
    )


def _session(
    key: str,
    title: str,
    plate: str,
    path: Path,
    *,
    required: bool = False,
    note: str = "",
) -> SessionSlice:
    if not (path / "summary.csv").is_file():
        if required:
            raise FileNotFoundError(f"{key}: missing summary.csv at {path}")
        return SessionSlice(
            key=key,
            title=title,
            plate=plate,
            path=path,
            present=False,
            note=note or f"not on this disk: {path}",
        )
    bundle = LabBundle.read(path)
    codecs = {
        codec: stats
        for codec in ("bf16", "nf4")
        if (stats := _stats_from_row(bundle.summary_for(codec))) is not None
    }
    return SessionSlice(
        key=key,
        title=title,
        plate=plate,
        path=path,
        present=True,
        codecs=codecs,
        note=note,
    )


def _parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("", "n/a"):
        return None
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def _read_eval_scores(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, Any]] = []
        for raw in csv.DictReader(handle):
            row = dict(raw)
            row["correct"] = _parse_bool(row.get("correct"))
            row["pending_human"] = bool(_parse_bool(row.get("pending_human")))
            rows.append(row)
        return rows


def _collect_eval_scores(root: Path) -> list[dict[str, Any]]:
    """Prefer the merged ``eval_scores.csv``; else stitch per-codec files."""
    merged: list[dict[str, Any]] = []
    root_csv = root / "eval_scores.csv"
    if root_csv.is_file():
        merged = _read_eval_scores(root_csv)
        codecs = {str(row.get("codec")) for row in merged}
        if {"bf16", "nf4"} <= codecs:
            return merged
    for child in sorted(root.glob("*/eval_scores.csv")):
        merged.extend(_read_eval_scores(child))
    return merged


def _gsm8k_ok(rows: Sequence[Mapping[str, Any]], codec: str) -> tuple[int, int]:
    counted = [
        row
        for row in rows
        if row.get("codec") == codec
        and row.get("correct") is not None
        and not row.get("pending_human")
    ]
    hits = sum(1 for row in counted if row.get("correct"))
    return hits, len(counted)


def _disagreements(rows: Sequence[Mapping[str, Any]]) -> int | None:
    by_item: dict[str, dict[str, bool]] = {}
    for row in rows:
        codec = str(row.get("codec") or "")
        item = str(row.get("item_id") or "")
        if codec not in ("bf16", "nf4") or not item:
            continue
        if row.get("correct") is None or row.get("pending_human"):
            continue
        by_item.setdefault(item, {})[codec] = bool(row["correct"])
    paired = [verdict for verdict in by_item.values() if "bf16" in verdict and "nf4" in verdict]
    if not paired:
        return None
    return sum(1 for verdict in paired if verdict["bf16"] != verdict["nf4"])


def _load_gsm8k(path: Path) -> Gsm8kSlice:
    plate = "local GSM8K 200/1319 · greedy, max_new=256, first 200 of main test"
    if not path.is_dir():
        return Gsm8kSlice(
            present=False,
            path=path,
            plate=plate,
            n_items=0,
            corpus_n=GSM8K_TEST_N,
            note=f"not on this disk: {path}",
        )
    codecs: dict[str, CodecStats] = {}
    for codec in ("bf16", "nf4"):
        slot = path / f"qwen25-3b-{codec}"
        if (slot / "summary.csv").is_file():
            stats = _stats_from_row(LabBundle.read(slot).summary_for(codec))
            if stats is not None:
                codecs[codec] = stats
    rows = _collect_eval_scores(path)
    n_ok = {codec: _gsm8k_ok(rows, codec)[0] for codec in ("bf16", "nf4")}
    n_items = max((_gsm8k_ok(rows, codec)[1] for codec in ("bf16", "nf4")), default=0)
    return Gsm8kSlice(
        present=bool(codecs) or bool(rows),
        path=path,
        plate=plate,
        n_items=n_items,
        corpus_n=GSM8K_TEST_N,
        codecs=codecs,
        n_ok=n_ok,
        disagreements=_disagreements(rows),
        note=(
            "extractor-sensitive; quality_all_ok=false is expected; "
            "not a published GSM8K score"
        ),
    )


def _load_hard(path: Path) -> tuple[HardSlice, ...]:
    matrix = path / "hard_matrix.csv"
    plate = "hard-eval-qwen25.png · 12 independent items, not WikiText/GSM8K/MMLU"
    if not matrix.is_file():
        return ()
    slices: list[HardSlice] = []
    with matrix.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            acc_text = str(raw.get("accuracy") or "").strip()
            n_text = str(raw.get("n_messages") or "").strip()
            accuracy = float(acc_text) if acc_text else None
            n_items = int(float(n_text)) if n_text else 12
            n_ok = None if accuracy is None else int(round(accuracy * n_items))
            slices.append(
                HardSlice(
                    label=str(raw.get("label") or ""),
                    size=str(raw.get("size") or ""),
                    codec=str(raw.get("codec") or ""),
                    n_ok=n_ok,
                    n_items=n_items,
                    accuracy=accuracy,
                    plate=plate,
                )
            )
    return tuple(slices)


def _load_ncu(path: Path) -> tuple[NcuCase, ...]:
    csv_path = path / "summary.csv"
    if not csv_path.is_file():
        return ()
    cases: list[NcuCase] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            def num(name: str) -> float | None:
                text = str(raw.get(name) or "").strip()
                return float(text) if text else None

            cases.append(
                NcuCase(
                    case=str(raw.get("case") or ""),
                    kernel=str(raw.get("kernel") or ""),
                    dram_pct=num("dram_pct"),
                    tensor_pct=num("tensor_pct"),
                    occupancy_pct=num("occupancy_pct"),
                )
            )
    return tuple(cases)


def _load_internlm_hard(path: Path) -> HardSlice | None:
    scores = path / "hard_scores.csv"
    if not scores.is_file():
        return None
    rows: list[dict[str, str]] = []
    with scores.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    n_ok = sum(1 for raw in rows if str(raw.get("correct") or "").strip().lower() == "true")
    n_items = len(rows)
    return HardSlice(
        label="internlm20b-nf4",
        size="20B",
        codec="nf4",
        n_ok=n_ok,
        n_items=n_items,
        accuracy=n_ok / n_items,
        plate=(
            f"{path.name} · NF4 only, no BF16 pair, not on hard-eval-qwen25.png"
        ),
    )


def _load_n32(path: Path) -> N32Gemm:
    csv_path = path / "comparison.csv"
    if not csv_path.is_file():
        return N32Gemm(present=False, path=path)

    def num(row: Mapping[str, Any], name: str) -> float | None:
        text = str(row.get(name) or "").strip()
        return float(text) if text else None

    by_kind: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            by_kind[str(raw.get("kind") or "")] = raw
    two = by_kind.get("two_n16_chunked")
    true = by_kind.get("true_n32")
    one = by_kind.get("one_n16")
    if two is None or true is None:
        return N32Gemm(present=False, path=path)
    two_us = num(two, "us")
    n32_us = num(true, "us")
    ratio = None if two_us is None or n32_us is None or n32_us == 0 else two_us / n32_us
    return N32Gemm(
        present=True,
        path=path,
        one_n16_us=None if one is None else num(one, "us"),
        two_n16_us=two_us,
        true_n32_us=n32_us,
        ratio=ratio,
        n32_kernel=str(true.get("kernel") or ""),
        two_kernel=str(two.get("kernel") or ""),
        n32_occ=num(true, "occ_pct"),
        two_occ=num(two, "occ_pct"),
    )


def _load_competitor(committed: Path, bnb_e2e: Path) -> CompetitorSlice:
    csv_path = committed / "summary.csv"
    if not csv_path.is_file():
        return CompetitorSlice(
            present=False,
            path=committed,
            n_stacks=0,
            n_skip=0,
            n_measured=0,
            stacks=(),
            bnb_note="committed competitor CSV missing",
            note=f"not on this disk: {committed}",
        )
    rows = read_competitor_csv(csv_path)
    measured = [row for row in rows if row.get("mean_decode_tok_s") is not None]
    skipped = [row for row in rows if row.get("mean_decode_tok_s") is None]
    bnb_note = "live e2e CSV not on disk — still SKIP, no invented tok/s"
    live_tok_s = live_ttft_ms = live_smi = None
    live_smoke: bool | None = None
    micro_skip = micro_n = None
    bnb_csv = bnb_e2e / "summary.csv"
    if bnb_csv.is_file():
        live = read_competitor_csv(bnb_csv)
        bnb_rows = [row for row in live if str(row.get("stack")) == "bitsandbytes-nf4"]
        if not bnb_rows:
            bnb_note = f"live CSV at {bnb_e2e.name}: no bitsandbytes-nf4 row"
        else:
            row = bnb_rows[0]
            tok = row.get("mean_decode_tok_s")
            if tok is None:
                reason = str(row.get("skip_reason") or "empty tok/s")
                bnb_note = f"live {bnb_e2e.name}: still SKIP ({reason[:80]})"
            else:
                live_tok_s = float(tok)
                live_ttft_ms = row.get("mean_ttft_ms")
                live_smi = row.get("smi_after_mib")
                live_smoke = row.get("smoke_ok")
                smoke_txt = "smoke pass" if live_smoke else "smoke not ok"
                parts = [f"live measured {bnb_e2e.name}: bitsandbytes-nf4 {_tok(live_tok_s)} tok/s"]
                if live_ttft_ms is not None:
                    parts.append(f"{_ms(live_ttft_ms)} ms TTFT")
                if live_smi is not None:
                    parts.append(f"smi {_mib(live_smi)}")
                parts.append(smoke_txt)
                bnb_note = (
                    ", ".join(parts)
                    + ". Isolated venv, Linear4bit over the HF tree. Not a kernel ranking."
                )
        micro_csv = bnb_e2e / "microbench.csv"
        if micro_csv.is_file():
            cells = read_microbench_csv(micro_csv)
            micro_n = len(cells)
            micro_skip = sum(1 for cell in cells if cell.get("us") is None)
            if micro_n:
                bnb_note += f" Kernel microbench {micro_skip}/{micro_n} SKIP."
    return CompetitorSlice(
        present=True,
        path=committed,
        n_stacks=len(rows),
        n_skip=len(skipped),
        n_measured=len(measured),
        stacks=tuple(str(row.get("stack") or "") for row in rows),
        bnb_note=bnb_note,
        note=(
            f"{len(skipped)}/{len(rows)} SKIP in docs/runs/competitor-qwen25-3b/; "
            "live tok/s stay outside git"
        ),
        live_tok_s=live_tok_s,
        live_ttft_ms=None if live_ttft_ms is None else float(live_ttft_ms),
        live_smi=None if live_smi is None else float(live_smi),
        live_smoke=live_smoke,
        micro_skip=micro_skip,
        micro_n=micro_n,
    )


def _occupancy_grid() -> OccupancyGrid:
    return OccupancyGrid(
        sms=SMS_3080,
        starved_qo=STARVED_QO_CTAS,
        starved_kv=STARVED_KV_CTAS,
        fix_qo=FIX_QO_CTAS,
        fix_kv=FIX_KV_CTAS,
        live_max_n=LIVE_MAX_N,
    )


def load_progress(paths: ProgressPaths | None = None) -> ProgressStory:
    """Read the story from disk. Optional live dirs may be absent."""
    loc = paths or default_paths()
    return ProgressStory(
        committed_3b=_session(
            "committed_3b",
            "Committed 3B (starved kernel)",
            "lab-qwen25-3b.png · docs/runs/qwen25-3b/ — do not read decode speed off that picture",
            loc.committed_3b,
            required=True,
        ),
        occupancy_nf4=_session(
            "occupancy_nf4",
            "Occupancy fix, unpaired NF4",
            "WAVE 2 occupancy · wave2-streamA-3b-nf4 — not a same-session BF16 pair",
            loc.occupancy_nf4,
            note="64-row tile + split-K; NF4-only live re-measure",
        ),
        paired_3b=_session(
            "paired_3b",
            "Honest same-session pair",
            "qwen25-3b-paired-20260913 — isolated workers, not vs Marlin/AWQ/bnb",
            loc.paired_3b,
        ),
        fit_14b=_session(
            "fit_14b",
            "14B fit vs spill",
            "lab-qwen25-14b.png panel F · docs/runs/qwen25-14b/ — do not quote smi 11,955 vs 8,913",
            loc.committed_14b,
            required=True,
        ),
        internlm_20b=_session(
            "internlm_20b",
            "20B on the card",
            "lab-internlm20b.png · docs/runs/internlm20b/ — BF16 generate empty on purpose",
            loc.committed_20b,
            required=True,
        ),
        hard=_load_hard(loc.hard),
        gsm8k=_load_gsm8k(loc.gsm8k),
        ncu=_load_ncu(loc.ncu),
        n32=_load_n32(loc.ncu_n32),
        internlm_hard=_load_internlm_hard(loc.hard_20b),
        competitor=_load_competitor(loc.competitor, loc.bnb_e2e),
        occupancy=_occupancy_grid(),
    )


# --------------------------------------------------------------------------- #
# formatters
# --------------------------------------------------------------------------- #


def _tok(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}" if float(value) < 10 else f"{value:.1f}"


def _ms(value: float | None) -> str:
    if value is None:
        return "—"
    number = float(value)
    if abs(number - round(number)) < 0.05:
        return f"{round(number):,}"
    return f"{number:,.1f}"


def _mib(value: float | None) -> str:
    return "—" if value is None else f"{float(value):,.0f}"


def _frac(ok: int | None, n: int) -> str:
    return "—" if ok is None or n <= 0 else f"{ok}/{n}"


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #


FIG_W = 16.2
_M_LEFT, _M_RIGHT = 0.90, 0.30
_HEADER_H = 1.50
_ROW1_H = 3.55
_ROW2_H = 2.95
_ROW3_H = 2.72
_FOOT_H = 1.82
_CAPTION_H = 1.58
_GAP = 0.48
_BOTTOM = 0.18
_BAND = "#F2F4F5"
_CHIP = "#F4F5F6"
_SPILL_FILL = "#C9A7A4"


def _fig_h() -> float:
    return (
        _HEADER_H
        + _ROW1_H
        + _GAP
        + _ROW2_H
        + _GAP
        + _ROW3_H
        + _GAP
        + _FOOT_H
        + _CAPTION_H
        + _BOTTOM
    )


def _style(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=3, color=RULE, labelsize=SIZE_SMALL, labelcolor=INK_SOFT)
    ax.tick_params(axis="x", length=0, labelsize=SIZE_SMALL, labelcolor=INK)


def _letter(ax: Any, letter: str, heading: str, *, y: float = 0.99) -> None:
    """Bold letter + claim at the top of the axes, hanging down (not into the ribbon)."""
    ax.text(
        0.0,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=SIZE_CLAIM,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )
    ax.text(
        0.055,
        y,
        heading,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=SIZE_CLAIM,
        color=INK,
        clip_on=False,
    )


def _claim_note(ax: Any, note: str, *, y: float = 0.91, width: int = 70) -> None:
    """Caveat under the claim, inside the axes (never data-coords past the spine)."""
    ax.text(
        0.0,
        y,
        textwrap.fill(note, width=width),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=SIZE_NOTE,
        color=INK_SOFT,
        linespacing=1.25,
        clip_on=True,
    )


def _empty(ax: Any, message: str) -> None:
    ax.text(
        0.5,
        0.45,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=SIZE_BODY,
        color=INK_SOFT,
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _bar_codec(ax: Any, x: float, codec: str, height: float, *, width: float = 0.34) -> None:
    style = CODEC_STYLE[codec]
    ax.bar(
        x,
        height,
        width=width,
        color=style.color,
        edgecolor=style.color,
        linewidth=0.8,
        hatch=style.hatch,
        zorder=3,
    )


def _panel_a_decode(ax: Any, story: ProgressStory) -> None:
    """Three labeled clusters: committed / unpaired occupancy / honest pair."""
    _style(ax)
    _letter(ax, "A", "3B decode — three plates, not one comparison")
    _claim_note(
        ax,
        "Mixing 17.0 vs 31.6 vs 28.4 without these labels is the bug this plate retires. TTFT under each bar.",
        width=70,
    )

    clusters = (
        story.committed_3b,
        story.occupancy_nf4,
        story.paired_3b,
    )
    centers = (0.0, 2.35, 4.70)
    present_vals = [
        float(stats.tok_s)
        for session in clusters
        for stats in session.codecs.values()
        if stats.tok_s is not None
    ]
    peak = max(present_vals + [1.0])
    ceiling = peak * 2.12
    ax.set_ylim(0, ceiling)
    ax.set_xlim(-1.05, 5.75)
    top_tick = int(peak // 10) * 10
    ax.set_yticks(list(range(0, max(top_tick, 10) + 1, 10)))

    for center, session in zip(centers, clusters):
        ax.axvspan(
            center - 1.05,
            center + 1.05,
            color="#F7F7F7",
            zorder=0,
            lw=0,
        )
        ax.text(
            center,
            peak * 1.46,
            session.title,
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=INK,
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            center,
            peak * 1.31,
            textwrap.fill(session.plate.split(" · ")[0], width=34),
            ha="center",
            va="top",
            fontsize=6.7,
            color=INK_SOFT,
            zorder=4,
        )
        if not session.present:
            ax.text(
                center,
                ceiling * 0.45,
                "not on\nthis disk",
                ha="center",
                va="center",
                fontsize=SIZE_SMALL,
                color=RULE,
                linespacing=1.25,
            )
        else:
            for codec, dx in (("bf16", -0.22), ("nf4", 0.22)):
                stats = session.codecs.get(codec)
                x = center + dx
                if stats is None or stats.tok_s is None:
                    ax.text(
                        x,
                        ceiling * 0.06,
                        "no BF16\nin that\nsession" if codec == "bf16" else "no\ndata",
                        ha="center",
                        va="bottom",
                        fontsize=7.6,
                        color=RULE,
                        linespacing=1.15,
                    )
                    continue
                _bar_codec(ax, x, codec, float(stats.tok_s))
                ax.annotate(
                    _tok(stats.tok_s),
                    (x, float(stats.tok_s)),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=SIZE_SMALL,
                    color=INK,
                    fontweight="bold",
                )
                ax.text(
                    x,
                    -ceiling * 0.02,
                    f"{_ms(stats.ttft_ms)} ms",
                    ha="center",
                    va="top",
                    fontsize=7.4,
                    color=INK_SOFT,
                    clip_on=False,
                )
                tick_color = CODEC_STYLE[codec].color
                ax.text(
                    x,
                    -ceiling * 0.08,
                    codec.upper(),
                    ha="center",
                    va="top",
                    fontsize=7.6,
                    color=tick_color,
                    fontweight="bold",
                    clip_on=False,
                )

    ax.set_xticks([])
    ax.set_ylabel("tok/s", fontsize=SIZE_SMALL, color=INK_SOFT)


def _panel_b_occupancy(ax: Any, grid: OccupancyGrid) -> None:
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.9)
    _letter(ax, "B", "Why 17 became 31.6 — CTA grid vs 70 SMs")
    _claim_note(
        ax,
        "Host planner (gpu/nf4/plan.py), not tok/s. Starved: one 128-row block. Fix: BM=64 + split-K.",
        width=58,
        y=0.92,
    )
    labels = [
        "q/o starved",
        "q/o after fix",
        "k/v starved",
        "k/v after fix",
    ]
    values = [grid.starved_qo, grid.fix_qo, grid.starved_kv, grid.fix_kv]
    colors = [_STARVED_INK, CODEC_STYLE["nf4"].color, _STARVED_INK, CODEC_STYLE["nf4"].color]
    hatches = ["", "///", "", "///"]
    ys = [2.08, 1.32, 0.58, 0.0]
    ax.set_ylim(-0.42, 3.38)
    xmax = max(values + [grid.sms, 1]) * 1.22
    ax.set_xlim(0, xmax)
    for y, value, color, hatch in zip(ys, values, colors, hatches):
        ax.barh(
            y,
            value,
            height=0.56,
            color=color,
            edgecolor=color,
            linewidth=0.7,
            hatch=hatch,
        )
        ax.text(
            value + xmax * 0.02,
            y,
            str(value),
            va="center",
            ha="left",
            fontsize=SIZE_SMALL,
            color=INK,
            fontweight="bold",
        )
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=SIZE_SMALL, color=INK)
    ax.tick_params(axis="y", length=0, pad=4, labelcolor=INK)
    ax.axvline(grid.sms, color=LIMIT, linewidth=1.1, linestyle=(0, (2, 2)), zorder=4)
    ax.set_xlabel(
        f"CTAs launched (3B q_proj / k_proj, N=1)  ·  dashed = {grid.sms} SMs",
        fontsize=SIZE_SMALL,
        color=INK_SOFT,
    )


def _split_ws_bar(ax: Any, y: float, value: float, codec: str, *, height: float = 0.55) -> None:
    style = CODEC_STYLE.get(codec, CODEC_STYLE["bf16"])
    on_card = min(float(value), CARD_MIB)
    spill = max(0.0, float(value) - CARD_MIB)
    ax.barh(
        y,
        on_card,
        height=height,
        color=style.color,
        edgecolor=style.color,
        linewidth=0.7,
        hatch=style.hatch,
        zorder=3,
    )
    if spill:
        ax.barh(
            y,
            spill,
            left=CARD_MIB,
            height=height,
            color=_SPILL_FILL,
            edgecolor=LIMIT,
            linewidth=0.6,
            hatch="xxx",
            zorder=3,
        )


def _panel_c_working_set(ax: Any, story: ProgressStory) -> None:
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.9)
    _letter(ax, "C", "Fit vs spill — CUDA working set, not the capped nvidia-smi")
    _claim_note(
        ax,
        "torch.cuda.memory_reserved() after load, MiB. Past the red rule is Windows shared GPU memory (system RAM).",
        width=140,
    )
    rows: list[tuple[str, str, SessionSlice, str]] = [
        ("14B", "bf16", story.fit_14b, "lab-qwen25-14b"),
        ("14B", "nf4", story.fit_14b, "lab-qwen25-14b"),
        ("20B", "bf16", story.internlm_20b, "lab-internlm20b"),
        ("20B", "nf4", story.internlm_20b, "lab-internlm20b"),
    ]
    ys = [2.18, 1.55, 0.78, 0.22]
    values: list[float] = []
    ytick_labels: list[str] = []
    ytick_colors: list[str] = []
    ax.set_ylim(-0.78, 4.05)
    for y, (size, codec, session, _plate) in zip(ys, rows):
        stats = session.codecs.get(codec) if session.present else None
        value = None if stats is None else stats.working_set_mib
        ytick_labels.append(f"{size} {codec.upper()}")
        ytick_colors.append(CODEC_STYLE[codec].color if value is not None else RULE)
        if value is None:
            ax.text(200, y, "—", va="center", fontsize=SIZE_SMALL, color=RULE)
            continue
        values.append(float(value))
        _split_ws_bar(ax, y, float(value), codec, height=0.50)
        ax.text(
            float(value) + 400,
            y,
            f"{_mib(value)} MiB",
            va="center",
            ha="left",
            fontsize=SIZE_SMALL,
            color=INK,
        )
    ceiling = max(values + [CARD_MIB, 1.0]) * 1.18
    ax.set_xlim(0, ceiling)
    ax.set_yticks(ys)
    ax.set_yticklabels(ytick_labels, fontsize=SIZE_SMALL, fontweight="bold")
    for tick, color in zip(ax.get_yticklabels(), ytick_colors):
        tick.set_color(color)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.axvline(CARD_MIB, color=LIMIT, linewidth=1.15, linestyle=(0, (2, 2)), zorder=4)
    ax.text(
        CARD_MIB,
        0.78,
        f"  card {CARD_MIB:,.0f} MiB",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=SIZE_SMALL,
        color=LIMIT,
    )
    # Decode callouts, still labeled by plate.
    fourteen = story.fit_14b
    twenty = story.internlm_20b
    bf16_14 = fourteen.codecs.get("bf16") if fourteen.present else None
    nf4_14 = fourteen.codecs.get("nf4") if fourteen.present else None
    nf4_20 = twenty.codecs.get("nf4") if twenty.present else None
    bf16_20 = twenty.codecs.get("bf16") if twenty.present else None
    line_14 = (
        f"14B decode {_tok(None if bf16_14 is None else bf16_14.tok_s)} vs "
        f"{_tok(None if nf4_14 is None else nf4_14.tok_s)} tok/s — fit vs spill, "
        "not a kernel win. Do not quote smi 11,955 vs 8,913."
    )
    gen_empty = "empty"
    if bf16_20 is not None and not bf16_20.notes:
        gen_empty = "empty"
    elif bf16_20 is not None and "TypeError" in bf16_20.notes:
        gen_empty = "TypeError vs transformers 5, unpatched"
    line_20 = (
        f"20B NF4 {_tok(None if nf4_20 is None else nf4_20.tok_s)} tok/s, smoke pass. "
        f"BF16 generate {gen_empty}. Working set {_mib(None if bf16_20 is None else bf16_20.working_set_mib)}"
        f" vs {_mib(None if nf4_20 is None else nf4_20.working_set_mib)}."
    )
    ax.text(
        200,
        -0.10,
        textwrap.fill(line_14, width=128),
        ha="left",
        va="top",
        fontsize=SIZE_NOTE,
        color=INK,
    )
    ax.text(
        200,
        -0.46,
        textwrap.fill(line_20, width=128),
        ha="left",
        va="top",
        fontsize=SIZE_NOTE,
        color=INK_SOFT,
    )
    ax.set_xlabel("CUDA working set after load, MiB", fontsize=SIZE_SMALL, color=INK_SOFT)


def _panel_d_hard(
    ax: Any,
    runs: Sequence[HardSlice],
    internlm: HardSlice | None = None,
) -> None:
    _style(ax)
    _letter(ax, "D", "Hard eval 12 — regression, not a leaderboard")
    note = (
        "docs/img/hard-eval-qwen25.png · auto-scored items. "
        "NF4 matching BF16 here is not “quantization is lossless”."
    )
    if internlm is not None and internlm.n_ok is not None:
        note += (
            f" Live 20B NF4 {_frac(internlm.n_ok, internlm.n_items)}, no BF16 pair — "
            "not on that PNG, not a quality headline."
        )
    _claim_note(ax, note, width=62)
    if not runs:
        _empty(ax, "hard_matrix.csv not on this disk")
        return
    ax.set_ylim(0, 14.2)
    ax.set_yticks([0, 3, 6, 9, 12])
    for index, run in enumerate(runs):
        style = CODEC_STYLE.get(run.codec, CODEC_STYLE["bf16"])
        if run.n_ok is None:
            ax.text(index, 0.4, "no\ndata", ha="center", fontsize=SIZE_SMALL, color=RULE)
            continue
        ax.bar(
            index,
            run.n_ok,
            width=0.62,
            color=style.color,
            edgecolor=style.color,
            linewidth=0.8,
            hatch=style.hatch,
        )
        ax.annotate(
            _frac(run.n_ok, run.n_items),
            (index, float(run.n_ok)),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=SIZE_SMALL,
            color=INK,
            fontweight="bold",
        )
    ax.set_xlim(-0.62, len(runs) - 0.38)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"{run.size}\n{run.codec.upper()}" for run in runs])
    for tick, run in zip(ax.get_xticklabels(), runs):
        tick.set_color(CODEC_STYLE.get(run.codec, CODEC_STYLE["bf16"]).color)
    ax.set_ylabel("correct / 12", fontsize=SIZE_SMALL, color=INK_SOFT)


def _panel_e_gsm8k(ax: Any, slice_: Gsm8kSlice) -> None:
    _style(ax)
    _letter(ax, "E", "Local GSM8K slice — not a GSM8K quality headline")
    _claim_note(
        ax,
        "Greedy, max_new=256, first 200 of 1,319 main test. quality_all_ok=false is expected. Extractor-sensitive.",
        width=62,
    )
    if not slice_.present:
        _empty(ax, slice_.note or "GSM8K run not on this disk")
        return
    ax.set_ylim(0, max(slice_.n_items, 1) * 1.22)
    for index, codec in enumerate(("bf16", "nf4")):
        style = CODEC_STYLE[codec]
        ok = slice_.n_ok.get(codec)
        n = slice_.n_items
        stats = slice_.codecs.get(codec)
        if ok is None:
            ax.text(index, 4, "no data", ha="center", fontsize=SIZE_SMALL, color=RULE)
            continue
        ax.bar(
            index,
            ok,
            width=0.55,
            color=style.color,
            edgecolor=style.color,
            linewidth=0.8,
            hatch=style.hatch,
        )
        ax.annotate(
            _frac(ok, n),
            (index, float(ok)),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=SIZE_SMALL,
            color=INK,
            fontweight="bold",
        )
        extra = []
        if stats and stats.tok_s is not None:
            extra.append(f"{_tok(stats.tok_s)} tok/s")
        if stats and stats.ttft_ms is not None:
            extra.append(f"{_ms(stats.ttft_ms)} ms TTFT")
        if stats and stats.smi_peak_mib is not None:
            extra.append(f"peak {_mib(stats.smi_peak_mib)}")
        ax.text(
            index,
            -slice_.n_items * 0.04,
            "\n".join(extra),
            ha="center",
            va="top",
            fontsize=7.2,
            color=INK_SOFT,
            clip_on=False,
            linespacing=1.25,
        )
    ax.set_xlim(-0.7, 1.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BF16", "NF4"])
    for tick, codec in zip(ax.get_xticklabels(), ("bf16", "nf4")):
        tick.set_color(CODEC_STYLE[codec].color)
    disagree = (
        "—" if slice_.disagreements is None else str(slice_.disagreements)
    )
    ax.text(
        0.5,
        -0.28,
        f"{disagree} disagreements · {slice_.n_items}/{slice_.corpus_n} · {slice_.note}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color=INK_SOFT,
        clip_on=False,
    )
    ax.set_ylabel("correct / 200", fontsize=SIZE_SMALL, color=INK_SOFT)


def _footer_ncu(
    ax: Any,
    cases: Sequence[NcuCase],
    live_max_n: int,
    n32: N32Gemm | None = None,
) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.96, "F", fontsize=SIZE_CLAIM, fontweight="bold", color=INK, va="top")
    ax.text(
        0.06,
        0.96,
        "Nsight — our kernel only, not competitor tok/s",
        fontsize=SIZE_CLAIM,
        color=INK,
        va="top",
    )
    ax.text(
        0.0,
        0.78,
        f"docs/runs/ncu/ · L2-rotated weights · LIVE_MAX_N still {live_max_n}",
        fontsize=SIZE_NOTE,
        color=INK_SOFT,
        va="top",
    )
    wanted = ("qwen25-3b-q_proj-n1", "qwen25-3b-q_proj-n16")
    by_case = {case.case: case for case in cases}
    if not cases:
        ax.text(0.0, 0.45, "ncu summary.csv not on this disk", fontsize=SIZE_BODY, color=RULE)
        return
    y = 0.58
    for name, caption in (
        (wanted[0], "q_proj decode N=1"),
        (wanted[1], "q_proj prefill N=16"),
    ):
        case = by_case.get(name)
        if case is None:
            ax.text(0.0, y, f"{caption}: missing", fontsize=SIZE_SMALL, color=RULE, va="top")
            y -= 0.24
            continue
        bits: list[str] = [caption + ":"]
        if case.dram_pct is not None:
            bits.append(f"DRAM {case.dram_pct:.1f}%")
        if case.tensor_pct is not None:
            bits.append(f"tensor {case.tensor_pct:.1f}%")
        if case.occupancy_pct is not None:
            bits.append(f"occupancy {case.occupancy_pct:.0f}%")
        ax.text(0.0, y, "   ".join(bits), fontsize=SIZE_SMALL, color=INK, va="top")
        y -= 0.22
    if n32 is not None and n32.present and n32.true_n32_us is not None and n32.two_n16_us is not None:
        ratio = f"{n32.ratio:.2f}×" if n32.ratio is not None else "—"
        ax.text(
            0.0,
            0.08,
            (
                f"True n32 {n32.true_n32_us:.0f} µs vs two n16 {n32.two_n16_us:.0f} µs "
                f"({ratio}). TokenLoop still chunks at {live_max_n}."
            ),
            fontsize=7.2,
            color=INK_SOFT,
            va="top",
        )
    else:
        ax.text(
            0.0,
            0.08,
            f"Prefill occupancy ~29% is still this GEMM. TokenLoop still chunks at {live_max_n}.",
            fontsize=7.2,
            color=INK_SOFT,
            va="top",
        )


def _footer_competitor(
    ax: Any,
    slice_: CompetitorSlice,
    paired_nf4: CodecStats | None,
) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    title = (
        "4-bit competitors — one live smoke, rest SKIP"
        if slice_.live_tok_s is not None
        else "4-bit competitors — SKIP, nothing invented"
    )
    ax.text(0.0, 0.96, "G", fontsize=SIZE_CLAIM, fontweight="bold", color=INK, va="top")
    ax.text(0.06, 0.96, title, fontsize=SIZE_CLAIM, color=INK, va="top")
    ax.text(
        0.0,
        0.78,
        "Committed git is SKIP. Live tok/s stay outside git.",
        fontsize=SIZE_NOTE,
        color=INK_SOFT,
        va="top",
    )
    if not slice_.present:
        ax.text(0.0, 0.45, slice_.note, fontsize=SIZE_BODY, color=RULE, va="top")
        return
    lines: list[str] = [
        f"docs/runs/competitor-qwen25-3b/: {slice_.n_skip}/{slice_.n_stacks} e2e SKIP."
    ]
    if slice_.live_tok_s is not None:
        live = f"Live bitsandbytes-nf4: {_tok(slice_.live_tok_s)} tok/s"
        if slice_.live_ttft_ms is not None:
            live += f" / {_ms(slice_.live_ttft_ms)} ms"
        if slice_.live_smi is not None:
            live += f", smi {_mib(slice_.live_smi)}"
        if slice_.live_smoke:
            live += ", smoke pass"
        lines.append(live + ".")
        if paired_nf4 is not None and paired_nf4.tok_s is not None:
            lines.append(
                f"Our paired NF4: {_tok(paired_nf4.tok_s)} tok/s / "
                f"{_ms(paired_nf4.ttft_ms)} ms. Different stack "
                "(Linear4bit vs CompressedLinear). Not a kernel ranking."
            )
        if slice_.micro_n:
            lines.append(
                f"Kernel microbench {slice_.micro_skip}/{slice_.micro_n} SKIP."
            )
    else:
        lines.append(slice_.bnb_note)
    ax.text(
        0.0,
        0.64,
        textwrap.fill(" ".join(lines), width=50),
        fontsize=7.3,
        color=INK,
        va="top",
        linespacing=1.35,
    )


def _footer_stack(ax: Any) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.96, "H", fontsize=SIZE_CLAIM, fontweight="bold", color=INK, va="top")
    ax.text(
        0.06,
        0.96,
        "The claim is the stack",
        fontsize=SIZE_CLAIM,
        color=INK,
        va="top",
    )
    ax.text(
        0.0,
        0.72,
        textwrap.fill(
            "CPU compressor chr writes NF4 G=64 at 4.25 bit/weight. "
            "GPU keeps packed weights in VRAM. The kernel reconstructs tiles "
            "in registers. Activations stay BF16. NF4 codebook and packed "
            "residency are prior art. Not “faster than existing 4-bit engines”.",
            width=48,
        ),
        fontsize=SIZE_SMALL,
        color=INK,
        va="top",
        linespacing=1.4,
    )


def _caption_paragraphs(story: ProgressStory) -> tuple[str, str]:
    sources = (
        "Sources, one plate per number: docs/runs/qwen25-3b (committed 3B, figure lab-qwen25-3b.png — VRAM traces; "
        "CSV still has the pre-split-K NF4 row). Occupancy-fix NF4-only: C:\\dev\\models\\runs\\wave2-streamA-3b-nf4. "
        "Honest pair: C:\\dev\\models\\runs\\qwen25-3b-paired-20260913. 14B: docs/runs/qwen25-14b plus lab-qwen25-14b.png "
        "panel F. 20B: docs/runs/internlm20b. Hard 12: docs/runs/hard-qwen25 and docs/img/hard-eval-qwen25.png. "
        "GSM8K slice: C:\\dev\\models\\runs\\eval-qwen25-3b-gsm8k-200-20260913. "
        "20B hard NF4-only: C:\\dev\\models\\runs\\hard-internlm20b-nf4-20260914 (not docs/runs/internlm20b). "
        "Nsight n1/n16: docs/runs/ncu. True n32 vs 2×n16: C:\\dev\\models\\runs\\ncu-n32-vs-2xn16-20260913. "
        "Committed competitors: docs/runs/competitor-qwen25-3b (SKIP matrix). Live bitsandbytes-nf4 smoke: "
        "C:\\dev\\models\\runs\\competitor-qwen25-3b-20260913-bnb-e2e (not written into git)."
    )
    paired = story.paired_3b.codecs.get("nf4") if story.paired_3b.present else None
    bnb = story.competitor
    if bnb.live_tok_s is not None and paired is not None and paired.tok_s is not None:
        both = (
            f"Live bitsandbytes-nf4 {_tok(bnb.live_tok_s)} tok/s / {_ms(bnb.live_ttft_ms)} ms "
            f"and our paired NF4 {_tok(paired.tok_s)} / {_ms(paired.ttft_ms)} ms are different stacks, "
            "not a kernel ranking. "
        )
    elif bnb.live_tok_s is not None:
        both = (
            f"Live bitsandbytes-nf4 {_tok(bnb.live_tok_s)} tok/s is a different stack, "
            "not a kernel ranking. "
        )
    else:
        both = "No live bitsandbytes tok/s on disk; the committed competitor matrix stays SKIP. "
    internlm = story.internlm_hard
    if internlm is not None and internlm.n_ok is not None:
        internlm_txt = (
            f" Live 20B NF4 {_frac(internlm.n_ok, internlm.n_items)} has no BF16 pair "
            "and is not a quality headline."
        )
    else:
        internlm_txt = ""
    if story.n32.present and story.n32.ratio is not None:
        n32_txt = (
            f" True n32 vs two n16 is {story.n32.ratio:.2f}× on one q_proj GEMM, not TTFT; "
            "TokenLoop still chunks at 16. "
        )
    else:
        n32_txt = " TokenLoop still chunks at 16. "
    caveats = (
        "Do not mix 17.0, 31.6 and 28.4. 31.6 is unpaired NF4 after the occupancy fix; 28.4 is the same-session pair. "
        "Do not quote 14B nvidia-smi 11,955 vs 8,913. 20B BF16 generate is a recorded miss (unpatched). "
        "Hard 12 is a regression fixture."
        + internlm_txt
        + " GSM8K-200 is greedy, first 200 of 1,319, extractor-sensitive; "
        "quality_all_ok=false is expected; do not headline it as GSM8K quality. No WikiText PPL is published. "
        "CTA counts are gpu/nf4/plan.py. Nsight is occupancy and pipes of chr_nf4_gemm, not tok/s."
        + n32_txt
        + both
    )
    return sources, caveats


def _ribbon(fig: Any, x0: float, top: float, width: float, fig_h: float) -> None:
    steps = (
        "1  starved 3B",
        "2  occupancy",
        "3  honest pair",
        "4  14B spill",
        "5  20B",
        "6  hard 12",
        "7  GSM8K 200",
    )
    n = len(steps)
    gap = 0.012
    chip_w = (width - gap * (n - 1)) / n
    y = top
    h = 0.26 / fig_h
    import matplotlib.pyplot as plt

    for index, label in enumerate(steps):
        left = x0 + index * (chip_w + gap)
        fig.add_artist(
            plt.Rectangle(
                (left, y),
                chip_w,
                h,
                transform=fig.transFigure,
                facecolor=_CHIP,
                edgecolor=RULE,
                linewidth=0.6,
                zorder=1,
            )
        )
        fig.text(
            left + chip_w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            color=INK,
        )


def progress_plate(story: ProgressStory | None = None) -> Any:
    """Draw the progress sheet. Missing optional CSVs become labeled gaps."""
    import matplotlib.pyplot as plt

    story = story or load_progress()
    fig_h = _fig_h()
    plot_w = FIG_W - _M_LEFT - _M_RIGHT

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
        fig = plt.figure(figsize=(FIG_W, fig_h), dpi=110)
        x0 = _M_LEFT / FIG_W

        # --- header -------------------------------------------------------
        fig.text(
            x0,
            1.0 - 0.24 / fig_h,
            "What changed on this card — RTX 3080 12 GB, first lab graphs through now",
            ha="left",
            va="top",
            fontsize=SIZE_TITLE,
            color=INK,
        )
        fig.text(
            x0,
            1.0 - 0.50 / fig_h,
            "One stack, one consumer card. Every number names the plate it came from. "
            "Decode compares HuggingFace generate (BF16) against CompressedLinear + TokenLoop (NF4).",
            ha="left",
            va="top",
            fontsize=SIZE_SUB,
            color=INK_SOFT,
        )
        band_h = 0.28
        band_bottom = fig_h - _HEADER_H + 0.40
        fig.add_artist(
            plt.Rectangle(
                (x0, band_bottom / fig_h),
                plot_w / FIG_W,
                band_h / fig_h,
                transform=fig.transFigure,
                facecolor=_BAND,
                edgecolor=RULE,
                linewidth=0.7,
                zorder=0,
            )
        )
        fig.text(
            x0 + 0.006,
            (band_bottom + band_h / 2) / fig_h,
            "MEASURED FROM CSVs — committed docs/runs/ plus documented live dirs outside git. "
            "A missing live folder is a gap, not a remembered 31.6.",
            ha="left",
            va="center",
            fontsize=SIZE_NOTE,
            color=INK_SOFT,
        )
        _ribbon(fig, x0, (fig_h - _HEADER_H + 0.08) / fig_h, plot_w / FIG_W, fig_h)

        # --- row 1: decode + occupancy --------------------------------------
        y_row1 = fig_h - _HEADER_H - _ROW1_H
        gap_in = 0.95
        left_w = plot_w * 0.62
        right_w = plot_w - left_w - gap_in
        ax_a = fig.add_axes(
            [
                _M_LEFT / FIG_W,
                (y_row1 + 0.28) / fig_h,
                left_w / FIG_W,
                (_ROW1_H - 0.32) / fig_h,
            ]
        )
        _panel_a_decode(ax_a, story)
        ax_b = fig.add_axes(
            [
                (_M_LEFT + left_w + gap_in) / FIG_W,
                (y_row1 + 0.18) / fig_h,
                right_w / FIG_W,
                (_ROW1_H - 0.22) / fig_h,
            ]
        )
        _panel_b_occupancy(ax_b, story.occupancy)

        # --- row 2: working set ---------------------------------------------
        y_row2 = y_row1 - _GAP - _ROW2_H
        ax_c = fig.add_axes(
            [
                _M_LEFT / FIG_W,
                (y_row2 + 0.10) / fig_h,
                plot_w / FIG_W,
                (_ROW2_H - 0.14) / fig_h,
            ]
        )
        _panel_c_working_set(ax_c, story)

        # --- row 3: hard + gsm8k -----------------------------------------
        y_row3 = y_row2 - _GAP - _ROW3_H
        half = (plot_w - gap_in) / 2
        ax_d = fig.add_axes(
            [
                _M_LEFT / FIG_W,
                (y_row3 + 0.10) / fig_h,
                half / FIG_W,
                (_ROW3_H - 0.14) / fig_h,
            ]
        )
        _panel_d_hard(ax_d, story.hard, story.internlm_hard)
        ax_e = fig.add_axes(
            [
                (_M_LEFT + half + gap_in) / FIG_W,
                (y_row3 + 0.22) / fig_h,
                half / FIG_W,
                (_ROW3_H - 0.26) / fig_h,
            ]
        )
        _panel_e_gsm8k(ax_e, story.gsm8k)

        # --- footer cards --------------------------------------------------
        y_foot = _CAPTION_H + _BOTTOM
        card_gap = 0.28
        card_w = (plot_w - 2 * card_gap) / 3
        for index, drawer in enumerate((_footer_ncu, _footer_competitor, _footer_stack)):
            left = _M_LEFT + index * (card_w + card_gap)
            fig.add_artist(
                plt.Rectangle(
                    (left / FIG_W, y_foot / fig_h),
                    card_w / FIG_W,
                    _FOOT_H / fig_h,
                    transform=fig.transFigure,
                    facecolor=_CHIP,
                    edgecolor=RULE,
                    linewidth=0.6,
                    zorder=0,
                )
            )
            ax = fig.add_axes(
                [
                    (left + 0.10) / FIG_W,
                    (y_foot + 0.08) / fig_h,
                    (card_w - 0.16) / FIG_W,
                    (_FOOT_H - 0.14) / fig_h,
                ]
            )
            if drawer is _footer_ncu:
                drawer(ax, story.ncu, story.occupancy.live_max_n, story.n32)
            elif drawer is _footer_competitor:
                drawer(ax, story.competitor, story.paired_3b.codecs.get("nf4"))
            else:
                drawer(ax)

        caption = "\n".join(
            textwrap.fill(paragraph, width=168)
            for paragraph in _caption_paragraphs(story)
        )
        fig.text(
            x0,
            (_CAPTION_H - 0.10) / fig_h,
            caption,
            ha="left",
            va="top",
            fontsize=SIZE_NOTE,
            color=INK_SOFT,
            linespacing=1.45,
        )
    return fig


def redraw(*extra: str | Path, repo: str | Path | None = None) -> list[Path]:
    """Load CSVs and write the PNG. No GPU, no torch, no nvidia-smi."""
    loc = default_paths(repo)
    figure = progress_plate(load_progress(loc))
    targets: list[Path] = [default_png(loc.repo)]
    for path in extra:
        if path:
            dest = Path(path)
            if dest not in targets:
                targets.append(dest)
    written = write_plate(figure, *targets)
    import matplotlib.pyplot as plt

    plt.close(figure)
    return written


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gpu.lab.progress_plate",
        description="Redraw the 3080 progress infographic from CSVs. No GPU.",
    )
    parser.add_argument(
        "--redraw",
        action="store_true",
        help="write docs/img/progress-3080.png from the documented CSVs",
    )
    parser.add_argument(
        "--out",
        action="append",
        default=[],
        help="extra PNG path (repeatable); does not overwrite the lab VRAM plates",
    )
    parser.add_argument(
        "--repo",
        default="",
        help="repository root (default: this tree)",
    )
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

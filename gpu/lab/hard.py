"""Hard eval: reasoning quality + speed, not Paris / Berlin / 323.

Smoke lab (notebooks 03/04/05) only checks that the model still talks. This
module is a separate plate. The prompts are fixed in
``gpu/lab/data/hard_items.json`` so CI never downloads GSM8K. Override the
path with ``DEEPFOLD_HARD`` for a larger local set later.

Both codecs get the **same** prompts, greedy, the same ``max_new_tokens``.
Turns in the default protocol are independent (KV reset), matching smoke.
``--history`` packs the prior user/assistant pairs into each next prompt so
prefill grows; ``TokenLoop.generate`` still resets KV, so that protocol is
growing prefill, not incremental KV reuse.

Quality is task-specific (GSM8K-style number, yes/no). Speed is still TTFT
and decode tok/s plus nvidia-smi / CUDA working set from the existing lab
sessions. Isolated workers, same as smoke: this process must not hold the
weights.

    python -m gpu.lab.hard --list
    python -m gpu.lab.hard --lab qwen25-32b --codec nf4
    python -m gpu.lab.test_hard
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .catalog import SUPPORTED_NF4, LabModel, lab_by_slug
from .script import RUNS_DIR, quality_ok

__all__ = [
    "ANSWER_STATES",
    "AnswerCell",
    "FIXTURE_PATH",
    "HARD_MAX_NEW_TOKENS",
    "HARD_MAX_SEQ",
    "HardItem",
    "HardRun",
    "MATRIX_COLUMNS",
    "RUN_ORDER",
    "RunScript",
    "SCORE_COLUMNS",
    "Score",
    "accuracy",
    "answer_cell",
    "answer_rows",
    "expected_weight_mib",
    "extract_number",
    "extract_yesno",
    "hard_max_seq",
    "items_path",
    "load_fixture",
    "load_script",
    "pad_runs",
    "plate_items",
    "quality_fn_for",
    "read_matrix",
    "redraw_plate",
    "run_hard",
    "run_hard_one",
    "run_matrix",
    "runs_from_matrix",
    "score_bundle",
    "score_item",
    "size_label",
    "write_matrix_csv",
    "write_run_script",
    "write_scores",
]

_REPO = Path(__file__).resolve().parents[2]
FIXTURE_PATH = Path(__file__).resolve().parent / "data" / "hard_items.json"

HARD_MAX_NEW_TOKENS = 256
HARD_MAX_SEQ = 2048
HARD_MAX_SEQ_20B = 1024

MIB = 1024 * 1024

# The required order: uncompressed two models in sequence, then compressed two.
# One isolated worker per pair -- 12 GB cannot hold two copies.
RUN_ORDER: tuple[tuple[str, str], ...] = (
    ("qwen25-3b", "bf16"),
    ("qwen25-14b", "bf16"),
    ("qwen25-3b", "nf4"),
    ("qwen25-14b", "nf4"),
)

# One row per (lab, codec) run. `hard_scores.csv` keeps its frozen per-run
# schema; this table is the cross-run index the plate is drawn from.
MATRIX_COLUMNS = (
    "lab",
    "title",
    "size",
    "codec",
    "label",
    "weight_mib",
    "expected_weight_mib",
    "working_set_mib",
    "vram_after_load_smi_mib",
    "vram_peak_smi_mib",
    "mean_ttft_ms",
    "mean_decode_tok_s",
    "accuracy",
    "n_messages",
    "out_dir",
    "notes",
)

# Answer-table cell states. ASCII words, not ticks: the plate has to print in
# grayscale and Arial does not ship U+2713.
ANSWER_STATES = ("none", "ok", "miss", "human")

_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)

_NUM = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_HASH = re.compile(r"####\s*(" + _NUM.pattern + r")")
_ANSWER = re.compile(
    r"(?:final answer|the answer is|answer)\s*[:\s]*(" + _NUM.pattern + r")",
    re.IGNORECASE,
)
_YESNO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

SCORE_COLUMNS = (
    "codec",
    "message_id",
    "item_id",
    "kind",
    "gold",
    "extracted",
    "correct",
    "pending_human",
    "notes",
)


@dataclass(frozen=True)
class HardItem:
    id: str
    kind: str
    prompt: str
    gold: str = ""
    needles: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class Score:
    item_id: str
    kind: str
    gold: str
    extracted: str
    correct: bool
    pending_human: bool = False
    notes: str = ""


@dataclass(frozen=True)
class RunScript:
    conversation: str
    items: tuple[HardItem, ...]
    max_new_tokens: int = HARD_MAX_NEW_TOKENS

    @property
    def prompts(self) -> tuple[str, ...]:
        return tuple(item.prompt for item in self.items)

    @property
    def quality_fn(self) -> Callable[[int, str], bool]:
        return quality_fn_for(self.items)


def items_path() -> Path:
    """Committed fixture, or ``DEEPFOLD_HARD`` if that path exists."""
    override = os.environ.get("DEEPFOLD_HARD", "").strip()
    if override:
        return Path(override)
    return FIXTURE_PATH


def _item_from_dict(row: Mapping[str, Any]) -> HardItem:
    needles = row.get("needles") or ()
    if isinstance(needles, str):
        needles = (needles,)
    gold = row.get("gold")
    return HardItem(
        id=str(row["id"]),
        kind=str(row["kind"]),
        prompt=str(row["prompt"]),
        gold="" if gold is None else str(gold),
        needles=tuple(str(n) for n in needles),
        note=str(row.get("note") or ""),
    )


def load_fixture(path: str | Path | None = None) -> dict[str, tuple[HardItem, ...]]:
    """``{"independent": ..., "history": ...}`` from the JSON fixture."""
    source = Path(path) if path is not None else items_path()
    data = json.loads(source.read_text(encoding="utf-8"))
    out: dict[str, tuple[HardItem, ...]] = {}
    for key in ("independent", "history"):
        rows = data.get(key) or []
        out[key] = tuple(_item_from_dict(row) for row in rows)
    return out


def load_script(path: str | Path, *, history: bool | None = None) -> RunScript:
    """A flattened worker script, or the fixture (pick independent vs history)."""
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if "independent" in data or "history" in data:
        use_history = bool(history)
        key = "history" if use_history else "independent"
        items = tuple(_item_from_dict(row) for row in data.get(key) or [])
        conversation = "history" if use_history else "independent"
        max_new = int(data.get("max_new_tokens") or HARD_MAX_NEW_TOKENS)
        return RunScript(conversation=conversation, items=items, max_new_tokens=max_new)
    items = tuple(_item_from_dict(row) for row in data.get("items") or [])
    conversation = str(data.get("conversation") or "independent")
    if history is True:
        conversation = "history"
    elif history is False and "conversation" not in data:
        conversation = "independent"
    max_new = int(data.get("max_new_tokens") or HARD_MAX_NEW_TOKENS)
    return RunScript(conversation=conversation, items=items, max_new_tokens=max_new)


def write_run_script(
    path: str | Path,
    *,
    history: bool = False,
    fixture: str | Path | None = None,
) -> Path:
    """Flatten independent or history items for an isolated worker."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    script = load_script(fixture or items_path(), history=history)
    payload = {
        "conversation": script.conversation,
        "max_new_tokens": script.max_new_tokens,
        "items": [
            {
                "id": item.id,
                "kind": item.kind,
                "prompt": item.prompt,
                "gold": item.gold,
                "needles": list(item.needles),
                "note": item.note,
            }
            for item in script.items
        ],
    }
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dest


def hard_max_seq(lab: LabModel) -> int:
    """20B KV is the tight one; 3B/14B can hold a 2k window for the prefill item."""
    if lab.slug == "internlm20b":
        return HARD_MAX_SEQ_20B
    return HARD_MAX_SEQ


def normalize_number(text: str) -> str:
    cleaned = text.strip().replace(",", "").replace("$", "")
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    text_value = f"{value:.10f}".rstrip("0").rstrip(".")
    return text_value


def extract_number(text: str) -> str | None:
    """GSM8K-style: ``#### N``, then 'the answer is', then the last number."""
    if not text:
        return None
    hashed = _HASH.search(text)
    if hashed:
        return normalize_number(hashed.group(1))
    answered = _ANSWER.search(text)
    if answered:
        return normalize_number(answered.group(1))
    found = _NUM.findall(text)
    return normalize_number(found[-1]) if found else None


def extract_yesno(text: str) -> str | None:
    """Last Yes/No in the reply, so a restated question does not win by itself."""
    found = _YESNO.findall(text or "")
    return found[-1].lower() if found else None


def numbers_equal(left: str, right: str) -> bool:
    try:
        return abs(float(left.replace(",", "")) - float(right.replace(",", ""))) < 1e-6
    except ValueError:
        return False


def score_item(item: HardItem, response: str) -> Score:
    """Fail closed: unknown kind, empty reply, or missing gold is not a pass."""
    text = response or ""
    kind = item.kind.lower()
    if kind == "open":
        return Score(
            item_id=item.id,
            kind=item.kind,
            gold=item.gold,
            extracted="",
            correct=False,
            pending_human=True,
            notes="open item: raw text in messages.csv; two human ratings later",
        )
    if kind == "smoke":
        return Score(
            item_id=item.id,
            kind=item.kind,
            gold=item.gold,
            extracted=text[:80],
            correct=False,
            notes="smoke kind is scored by gpu.lab.script.quality_ok, not here",
        )
    if kind == "gsm8k":
        extracted = extract_number(text) or ""
        ok = bool(extracted) and bool(item.gold) and numbers_equal(extracted, item.gold)
        if not ok and item.needles:
            lowered = text.lower()
            ok = any(needle.lower() in lowered for needle in item.needles)
        return Score(
            item_id=item.id,
            kind=item.kind,
            gold=item.gold,
            extracted=extracted,
            correct=ok,
            notes="" if ok else "number miss",
        )
    if kind == "yesno":
        extracted = extract_yesno(text) or ""
        ok = bool(extracted) and extracted == item.gold.strip().lower()
        return Score(
            item_id=item.id,
            kind=item.kind,
            gold=item.gold,
            extracted=extracted,
            correct=ok,
            notes="" if ok else "yes/no miss",
        )
    if kind == "needle":
        lowered = text.lower()
        extracted = ""
        ok = any(needle.lower() in lowered for needle in item.needles)
        if ok:
            extracted = next(n for n in item.needles if n.lower() in lowered)
        return Score(
            item_id=item.id,
            kind=item.kind,
            gold=item.gold or ",".join(item.needles),
            extracted=extracted,
            correct=ok,
            notes="" if ok else "needle miss",
        )
    if kind == "exact":
        extracted = " ".join(text.split()).strip()
        ok = extracted.casefold() == item.gold.strip().casefold()
        return Score(
            item_id=item.id,
            kind=item.kind,
            gold=item.gold,
            extracted=extracted,
            correct=ok,
            notes="" if ok else "exact miss",
        )
    return Score(
        item_id=item.id,
        kind=item.kind,
        gold=item.gold,
        extracted="",
        correct=False,
        notes=f"unknown kind {item.kind!r}",
    )


def quality_fn_for(items: Sequence[HardItem]) -> Callable[[int, str], bool]:
    """``quality_ok(message_id, response)`` using 1-based message_id -> item."""

    def fn(message_id: int, response: str) -> bool:
        index = int(message_id) - 1
        if index < 0 or index >= len(items):
            return False
        item = items[index]
        if item.kind.lower() == "smoke":
            return quality_ok(int(message_id), response)
        return score_item(item, response).correct

    return fn


def score_messages(
    messages: Iterable[Mapping[str, Any]],
    items: Sequence[HardItem],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    items_list = list(items)
    for row in messages:
        message_id = int(row["message_id"])
        index = message_id - 1
        codec = str(row.get("codec") or "")
        response = str(row.get("response") or "")
        if index < 0 or index >= len(items_list):
            scored = Score(
                item_id=f"msg-{message_id}",
                kind="unknown",
                gold="",
                extracted="",
                correct=False,
                notes="message_id has no fixture item",
            )
        else:
            scored = score_item(items_list[index], response)
        rows.append(
            {
                "codec": codec,
                "message_id": message_id,
                "item_id": scored.item_id,
                "kind": scored.kind,
                "gold": scored.gold,
                "extracted": scored.extracted,
                "correct": scored.correct,
                "pending_human": scored.pending_human,
                "notes": scored.notes,
            }
        )
    return rows


def score_bundle(bundle: Any, items: Sequence[HardItem]) -> list[dict[str, Any]]:
    return score_messages(bundle.messages, items)


def accuracy(score_rows: Sequence[Mapping[str, Any]], codec: str) -> float | None:
    """Fraction correct among auto-scored items. Open/pending rows are skipped."""
    rows = [
        row
        for row in score_rows
        if row.get("codec") == codec and not row.get("pending_human")
    ]
    if not rows:
        return None
    return sum(1 for row in rows if row.get("correct")) / len(rows)


def write_scores(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SCORE_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    str(row.get("codec") or ""),
                    str(row.get("message_id") or ""),
                    str(row.get("item_id") or ""),
                    str(row.get("kind") or ""),
                    str(row.get("gold") or ""),
                    str(row.get("extracted") or ""),
                    "true" if row.get("correct") else "false",
                    "true" if row.get("pending_human") else "false",
                    str(row.get("notes") or ""),
                ]
            )
    return dest


def _nf4_ready(lab: LabModel) -> tuple[bool, str]:
    if lab.nf4_driver not in SUPPORTED_NF4:
        return False, f"no TokenLoop driver for nf4_driver={lab.nf4_driver!r}"
    if not Path(lab.chr_path).is_file():
        return False, f"NF4 file missing: {lab.chr_path}"
    return True, ""


def run_hard(
    lab: LabModel | str,
    out_dir: str | Path,
    *,
    history: bool = False,
    codec: str = "both",
    isolated: bool = True,
    graphs: bool = True,
    verbose: bool = True,
) -> tuple[Any, list[dict[str, Any]]]:
    """BF16 then NF4 on the hard set. Missing ``.chr`` skips NF4 without crashing."""
    from .sessions import run_both

    if isinstance(lab, str):
        lab = lab_by_slug(lab)
    if codec not in ("both", "bf16", "nf4"):
        raise ValueError(f"codec={codec!r}; expected 'both', 'bf16' or 'nf4'")

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    script_path = write_run_script(dest / "hard_script.json", history=history)
    script = load_script(script_path)

    run_codec = codec
    skip_reason = ""
    if codec in ("both", "nf4"):
        ready, skip_reason = _nf4_ready(lab)
        if not ready and codec == "both":
            run_codec = "bf16"
            if verbose:
                print(f"NF4 skipped: {skip_reason}", flush=True)
        elif not ready and codec == "nf4" and verbose:
            print(f"NF4 will record a miss: {skip_reason}", flush=True)

    bundle = run_both(
        dest,
        codec=run_codec,
        model_dir=lab.model_dir,
        chr_path=lab.chr_path,
        messages=script.prompts,
        max_new_tokens=script.max_new_tokens,
        max_seq=hard_max_seq(lab),
        graphs=graphs,
        verbose=verbose,
        trust_remote_code=lab.trust_remote_code,
        isolated=isolated,
        items_json=script_path,
        conversation=script.conversation,
    )
    scores = score_bundle(bundle, script.items)
    write_scores(dest / "hard_scores.csv", scores)
    if skip_reason and run_codec == "bf16":
        for row in bundle.summary:
            if row.get("codec") == "bf16":
                extra = f"NF4 skipped: {skip_reason}"
                notes = str(row.get("notes") or "")
                row["notes"] = f"{notes}; {extra}" if notes else extra
        bundle.write(dest)
    return bundle, scores


# --------------------------------------------------------------------------- #
# the 4-run matrix: 3B and 14B, uncompressed then compressed
# --------------------------------------------------------------------------- #


def size_label(lab: LabModel | str) -> str:
    """``3B`` / ``14B`` / ``20B`` out of the catalog title, for axis labels.

    The plate has to say which model each bar is, or a 3B run reads as a 14B
    one -- which is exactly how the first pass of notebook 06 got misread.
    """
    if isinstance(lab, str):
        lab = lab_by_slug(lab)
    found = _SIZE.findall(lab.title)
    return f"{found[-1]}B" if found else lab.slug


def expected_weight_mib(lab: LabModel | str) -> dict[str, float | None]:
    """Weight bytes **on disk**, so the notebook can pre-check the model size.

    ``bf16`` is the sum of the safetensor shards, ``nf4`` is the ``.chr``
    container. Measured from the filesystem, never a guess: 3B is ~5.9k /
    ~1.6k MiB and 14B is ~28k / ~7.5k MiB, and that is how a reader can see
    that a run labelled 14B really loaded 14B.
    """
    if isinstance(lab, str):
        lab = lab_by_slug(lab)
    model_dir = Path(lab.model_dir)
    shards = sorted(model_dir.glob("*.safetensors")) if model_dir.is_dir() else []
    chr_file = Path(lab.chr_path)
    return {
        "bf16": sum(f.stat().st_size for f in shards) / MIB if shards else None,
        "nf4": chr_file.stat().st_size / MIB if chr_file.is_file() else None,
    }


def _peak_column(bundle: Any, codec: str, column: str) -> float | None:
    values = [value for value in bundle.series(codec, column)[1] if value is not None]
    return max(values) if values else None


def _working_set_mib(bundle: Any, codec: str) -> float | None:
    """What CUDA actually holds: the largest of after-load, reserved and allocated.

    On 14B BF16 this is ~28 GiB on a 12 GiB card -- nvidia-smi cannot say that,
    it stops at 12288. This is the number to quote for the spill.
    """
    summary = bundle.summary_for(codec) or {}
    candidates = [
        summary.get("vram_after_load_torch_mib"),
        _peak_column(bundle, codec, "torch_reserved_mib"),
        _peak_column(bundle, codec, "torch_max_alloc_mib"),
        _peak_column(bundle, codec, "torch_alloc_mib"),
    ]
    present = [float(value) for value in candidates if value is not None]
    return max(present) if present else None


@dataclass(frozen=True)
class HardRun:
    """One (lab, codec) hard-eval session, reduced to what the plate draws."""

    slug: str
    title: str
    size: str
    codec: str
    out_dir: Path
    weight_mib: float | None = None
    expected_weight_mib: float | None = None
    working_set_mib: float | None = None
    smi_after_mib: float | None = None
    smi_peak_mib: float | None = None
    mean_ttft_ms: float | None = None
    mean_tok_s: float | None = None
    accuracy: float | None = None
    n_messages: int = 0
    notes: str = ""
    scores: tuple[Mapping[str, Any], ...] = ()
    # True only for a slot :func:`pad_runs` invented so the plate can show the
    # whole 2x2 matrix while the queue is still working through it. A pending
    # slot has no measurement of any kind and is never written to a CSV.
    pending: bool = False

    @property
    def label(self) -> str:
        """``14B BF16`` -- size first, so 3B and 14B never look alike."""
        return f"{self.size} {self.codec.upper()}"

    @property
    def ran(self) -> bool:
        return self.n_messages > 0

    @property
    def accuracy_pct(self) -> float | None:
        return None if self.accuracy is None else 100.0 * self.accuracy

    def score_by_item(self) -> dict[str, Mapping[str, Any]]:
        return {str(row.get("item_id")): row for row in self.scores}

    @classmethod
    def of(
        cls,
        lab: LabModel,
        codec: str,
        out_dir: str | Path,
        bundle: Any,
        scores: Sequence[Mapping[str, Any]],
    ) -> "HardRun":
        summary = bundle.summary_for(codec) or {}
        expected = expected_weight_mib(lab).get(codec)
        return cls(
            slug=lab.slug,
            title=lab.title,
            size=size_label(lab),
            codec=codec,
            out_dir=Path(out_dir),
            weight_mib=summary.get("weight_mib"),
            expected_weight_mib=expected,
            working_set_mib=_working_set_mib(bundle, codec),
            smi_after_mib=summary.get("vram_after_load_smi_mib"),
            smi_peak_mib=summary.get("vram_peak_smi_mib"),
            mean_ttft_ms=summary.get("mean_ttft_ms"),
            mean_tok_s=summary.get("mean_decode_tok_s"),
            accuracy=accuracy(scores, codec),
            n_messages=int(summary.get("n_messages") or 0),
            notes=str(summary.get("notes") or ""),
            scores=tuple(row for row in scores if row.get("codec") == codec),
        )

    def as_row(self) -> dict[str, Any]:
        return {
            "lab": self.slug,
            "title": self.title,
            "size": self.size,
            "codec": self.codec,
            "label": self.label,
            "weight_mib": self.weight_mib,
            "expected_weight_mib": self.expected_weight_mib,
            "working_set_mib": self.working_set_mib,
            "vram_after_load_smi_mib": self.smi_after_mib,
            "vram_peak_smi_mib": self.smi_peak_mib,
            "mean_ttft_ms": self.mean_ttft_ms,
            "mean_decode_tok_s": self.mean_tok_s,
            "accuracy": self.accuracy,
            "n_messages": self.n_messages,
            "out_dir": str(self.out_dir),
            "notes": self.notes,
        }


def write_matrix_csv(path: str | Path, runs: Sequence[HardRun]) -> Path:
    """One row per run: which model, which codec, memory, speed, accuracy."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MATRIX_COLUMNS)
        for run in runs:
            row = run.as_row()
            writer.writerow(
                [
                    "" if row[name] is None else
                    f"{row[name]:.6g}" if isinstance(row[name], float) else str(row[name])
                    for name in MATRIX_COLUMNS
                ]
            )
    return dest


def read_matrix(path: str | Path) -> list[dict[str, Any]]:
    """``hard_matrix.csv`` back as dicts, so the plate can be redrawn with no GPU."""
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for name in MATRIX_COLUMNS:
                text = str(row.get(name) or "").strip()
                if name in ("lab", "title", "size", "codec", "label", "out_dir", "notes"):
                    row[name] = text
                    continue
                try:
                    row[name] = float(text) if text else None
                except ValueError:
                    row[name] = None
            rows.append(row)
    return rows


def runs_from_matrix(path: str | Path) -> list[HardRun]:
    """Rebuild :class:`HardRun` objects (with their scores) from a finished root."""
    root = Path(path)
    matrix = root / "hard_matrix.csv" if root.is_dir() else root
    runs: list[HardRun] = []
    for row in read_matrix(matrix):
        out_dir = Path(str(row["out_dir"]))
        scores: tuple[Mapping[str, Any], ...] = ()
        scores_csv = out_dir / "hard_scores.csv"
        if scores_csv.is_file():
            with scores_csv.open("r", encoding="utf-8", newline="") as handle:
                scores = tuple(
                    {**score_row, "correct": str(score_row.get("correct")) == "true",
                     "pending_human": str(score_row.get("pending_human")) == "true"}
                    for score_row in csv.DictReader(handle)
                )
        runs.append(
            HardRun(
                slug=str(row["lab"]),
                title=str(row["title"]),
                size=str(row["size"]),
                codec=str(row["codec"]),
                out_dir=out_dir,
                weight_mib=row["weight_mib"],
                expected_weight_mib=row["expected_weight_mib"],
                working_set_mib=row["working_set_mib"],
                smi_after_mib=row["vram_after_load_smi_mib"],
                smi_peak_mib=row["vram_peak_smi_mib"],
                mean_ttft_ms=row["mean_ttft_ms"],
                mean_tok_s=row["mean_decode_tok_s"],
                accuracy=row["accuracy"],
                n_messages=int(row["n_messages"] or 0),
                notes=str(row["notes"]),
                scores=scores,
            )
        )
    return runs


def pad_runs(
    runs: Sequence[HardRun],
    *,
    order: Sequence[tuple[str, str]] = RUN_ORDER,
) -> list[HardRun]:
    """``order`` as complete slots, using the live runs and marking the rest pending.

    The matrix takes hours and the 14B BF16 spill takes most of them, so the
    plate is usually drawn while some pairs have CSVs and some do not. Padding
    keeps all four ticks on every panel -- ``3B BF16``, ``14B BF16``,
    ``3B NF4``, ``14B NF4`` -- so a half-finished matrix cannot be misread as
    a two-run comparison. A padded slot carries only the weight expected on
    disk; every measured field stays ``None`` and draws as *not run yet*,
    never as a zero. Live runs outside ``order`` (a 20B control) keep their
    place at the end.
    """
    by_pair = {(run.slug, run.codec): run for run in runs}
    root = runs[0].out_dir.parent if runs else Path()
    padded: list[HardRun] = []
    for slug, codec in order:
        live = by_pair.get((slug, codec))
        if live is not None:
            padded.append(live)
            continue
        try:
            lab = lab_by_slug(slug)
        except KeyError:
            continue
        padded.append(
            HardRun(
                slug=slug,
                title=lab.title,
                size=size_label(lab),
                codec=codec,
                out_dir=root / f"{slug}-{codec}",
                expected_weight_mib=expected_weight_mib(lab).get(codec),
                notes="not run yet: no summary.csv for this pair",
                pending=True,
            )
        )
    seen = {(run.slug, run.codec) for run in padded}
    padded.extend(run for run in runs if (run.slug, run.codec) not in seen)
    return padded


def _recorded_miss(
    lab: LabModel,
    codec: str,
    dest: Path,
    items: Sequence[HardItem],
    note: str,
    *,
    verbose: bool = True,
) -> HardRun:
    """A skip or a dead worker is a session row with notes, never a crashed cell."""
    from .bundle import LabBundle

    if verbose:
        print(f"[{lab.slug} {codec}] {note}", flush=True)
    bundle = LabBundle.of(summary=[{"codec": codec, "n_messages": 0,
                                   "quality_all_ok": False, "notes": note}])
    dest.mkdir(parents=True, exist_ok=True)
    bundle.write(dest)
    scores = score_bundle(bundle, items)
    write_scores(dest / "hard_scores.csv", scores)
    return HardRun.of(lab, codec, dest, bundle, scores)


def run_hard_one(
    lab: LabModel | str,
    out_root: str | Path,
    *,
    codec: str,
    script_path: str | Path | None = None,
    history: bool = False,
    items: Sequence[HardItem] | None = None,
    isolated: bool = True,
    graphs: bool = True,
    verbose: bool = True,
) -> HardRun:
    """One codec of one model, in its own child process, into ``<slug>-<codec>/``.

    Never loads a model in this process and never raises for a missing model,
    a missing ``.chr``, a CUDA OOM or a worker that died before writing
    ``summary.csv``: all of those come back as a :class:`HardRun` whose notes
    say what happened. The child's stdout/stderr is echoed by
    :mod:`gpu.lab.sessions` and kept in ``<slug>-<codec>/<codec>/worker.log``.
    """
    if isinstance(lab, str):
        lab = lab_by_slug(lab)
    if codec not in ("bf16", "nf4"):
        raise ValueError(f"codec={codec!r}; expected 'bf16' or 'nf4'")

    root = Path(out_root)
    dest = root / f"{lab.slug}-{codec}"
    dest.mkdir(parents=True, exist_ok=True)
    if script_path is None:
        script_path = write_run_script(root / "hard_script.json", history=history)
    script = load_script(script_path, history=history)
    run_items = tuple(items) if items is not None else script.items

    if not Path(lab.model_dir).is_dir():
        return _recorded_miss(
            lab, codec, dest, run_items,
            f"{codec} skipped: model dir missing: {lab.model_dir}. "
            "Recorded miss, not a crashed cell.",
            verbose=verbose,
        )
    if codec == "nf4":
        ready, why = _nf4_ready(lab)
        if not ready:
            return _recorded_miss(
                lab, codec, dest, run_items,
                f"nf4 skipped: {why}. Compress on CPU first: "
                f"{lab.compress_cmd or '(no compress command in the catalog)'}. "
                "Recorded miss, not a crashed cell.",
                verbose=verbose,
            )

    from .sessions import run_bf16, run_nf4

    common = dict(
        out_dir=dest,
        model_dir=lab.model_dir,
        messages=script.prompts,
        max_new_tokens=script.max_new_tokens,
        trust_remote_code=lab.trust_remote_code,
        isolated=isolated,
        items_json=script_path,
        conversation=script.conversation,
        verbose=verbose,
    )
    try:
        if codec == "bf16":
            session = run_bf16(**common)
        else:
            session = run_nf4(
                chr_path=lab.chr_path,
                max_seq=hard_max_seq(lab),
                graphs=graphs,
                **common,
            )
    except Exception as exc:  # noqa: BLE001 -- a dead worker is data, not a traceback
        return _recorded_miss(
            lab, codec, dest, run_items,
            f"{type(exc).__name__} from the isolated {codec} worker: {exc}",
            verbose=verbose,
        )

    bundle = session.as_bundle()
    bundle.write(dest)
    scores = score_bundle(bundle, run_items)
    write_scores(dest / "hard_scores.csv", scores)
    return HardRun.of(lab, codec, dest, bundle, scores)


def run_matrix(
    out_root: str | Path,
    *,
    pairs: Sequence[tuple[str, str]] = RUN_ORDER,
    existing: Sequence[HardRun] = (),
    history: bool = False,
    isolated: bool = True,
    graphs: bool = True,
    verbose: bool = True,
    on_run: Callable[[HardRun], None] | None = None,
    wait_s: float = 45.0,
) -> list[HardRun]:
    """Run ``pairs`` sequentially, one isolated worker each, and index them.

    Default order is :data:`RUN_ORDER`: BF16 3B, BF16 14B, NF4 3B, NF4 14B --
    uncompressed both models, then compressed both models. Between runs it
    waits for nvidia-smi to come back down, so the next load does not start on
    top of the previous CUDA pool.

    ``existing`` carries runs from an earlier call (the notebook does phase A
    and phase B in separate cells), so ``hard_matrix.csv`` always holds the
    whole matrix. Returns ``existing`` plus the new runs, in order.
    """
    from .sampler import smi_used_mib
    from .sessions import _wait_released

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    script_path = write_run_script(root / "hard_script.json", history=history)
    script = load_script(script_path, history=history)
    idle = smi_used_mib()

    runs: list[HardRun] = list(existing)
    for index, (slug, codec) in enumerate(pairs, start=1):
        lab = lab_by_slug(slug)
        if runs and isolated:
            released, used = _wait_released(runs[-1].smi_peak_mib, idle, seconds=wait_s)
            if verbose:
                state = "clear" if released else "STILL HIGH"
                print(
                    f"[card] {state} at {'n/a' if used is None else f'{used:.0f}'} MiB "
                    f"before {slug} {codec}",
                    flush=True,
                )
        if verbose:
            print(
                f"\n=== run {index}/{len(pairs)}  {codec.upper()}  {lab.title} "
                f"({size_label(lab)})  ===",
                flush=True,
            )
        run = run_hard_one(
            lab,
            root,
            codec=codec,
            script_path=script_path,
            history=history,
            items=script.items,
            isolated=isolated,
            graphs=graphs,
            verbose=verbose,
        )
        runs.append(run)
        write_matrix_csv(root / "hard_matrix.csv", runs)
        if on_run is not None:
            on_run(run)
    return runs


# --------------------------------------------------------------------------- #
# the answer table: the fixture's questions and golds, plus live cells
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AnswerCell:
    """One (item, run) cell of the answer table: what the model said, and a verdict."""

    text: str
    state: str  # one of ANSWER_STATES

    @property
    def correct(self) -> bool:
        return self.state == "ok"


def answer_cell(score: Mapping[str, Any] | None, *, limit: int = 16) -> AnswerCell:
    """``ok 164`` / ``miss 0.10`` / ``human`` / ``-`` -- words, not ticks.

    ``None`` means this run has no score row for the item: it was skipped, it
    OOMed, or it has not been run yet. That is drawn as a dash, never as a
    zero and never as a pass.
    """
    if score is None:
        return AnswerCell("-", "none")
    if score.get("pending_human"):
        return AnswerCell("human", "human")
    extracted = " ".join(str(score.get("extracted") or "").split())
    if len(extracted) > limit:
        extracted = extracted[: limit - 1].rstrip() + "\u2026"
    if score.get("correct"):
        return AnswerCell(f"ok {extracted}".strip(), "ok")
    # Empty `extracted` means nothing could be pulled out of the reply -- an
    # empty session, or prose with no number in it. Both are misses.
    return AnswerCell(f"miss {extracted}".strip() if extracted else "miss (no answer)", "miss")


def answer_rows(
    items: Sequence[HardItem],
    runs: Sequence[HardRun] = (),
    *,
    question_limit: int = 74,
) -> list[dict[str, Any]]:
    """The fixture Q&A table, with one cell per run when live scores exist.

    Renders from the fixture alone: the questions and their golds are on the
    plate before any GPU has been touched, and the run columns fill in as the
    CSVs appear.
    """
    by_run = [(run.label, run.score_by_item()) for run in runs]
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        question = " ".join(item.prompt.split())
        if len(question) > question_limit:
            question = question[: question_limit - 1].rstrip() + "\u2026"
        gold = item.gold or ", ".join(item.needles)
        rows.append(
            {
                "n": index,
                "item": item.id,
                "kind": item.kind,
                "question": question,
                "gold": gold or "(human)",
                "cells": {
                    label: answer_cell(scores.get(item.id)) for label, scores in by_run
                },
            }
        )
    return rows


def plate_items(run_root: str | Path, *, history: bool = False) -> tuple[HardItem, ...]:
    """The script a run was actually asked, falling back to the committed fixture.

    ``hard_script.json`` is frozen into the run root at launch, so a redraw
    shows the questions that run saw even if the fixture has moved on since.
    """
    script_path = Path(run_root) / "hard_script.json"
    if script_path.is_file():
        return load_script(script_path).items
    return load_fixture()["history" if history else "independent"]


def redraw_plate(
    run_root: str | Path,
    *extra_paths: str | Path | None,
    history: bool = False,
) -> list[Path]:
    """Redraw the summary plate from a finished run root. No GPU, no re-run.

    Reads ``hard_matrix.csv`` plus each run's ``hard_scores.csv``, so a matrix
    started without ``--plate`` (or one still filling in) can be drawn at any
    time. Always writes ``hard-eval.png`` next to the CSVs; ``extra_paths``
    adds copies, typically ``docs/img/hard-eval-qwen25.png``.
    """
    from .hard_plate import hard_plate, write_plate

    root = Path(run_root)
    runs = runs_from_matrix(root)
    figure = hard_plate(plate_items(root, history=history), runs)

    targets: list[Path] = [root / "hard-eval.png"]
    for path in extra_paths:
        if not path:
            continue
        dest = Path(path)
        if dest not in targets:
            targets.append(dest)
    return write_plate(figure, *targets)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gpu.lab.hard",
        description="Hard eval (GSM8K-style / multi-step) for BF16 vs NF4. Not smoke.",
    )
    parser.add_argument(
        "--lab",
        default="",
        help="catalog slug: qwen25-3b, qwen25-14b, internlm20b, qwen25-32b",
    )
    parser.add_argument(
        "--out",
        default="",
        help="output directory; default $DEEPFOLD_RUNS/hard-<slug>-<time>",
    )
    parser.add_argument(
        "--codec",
        choices=("both", "bf16", "nf4"),
        default="both",
        help="which session(s) to run (default: both, NF4 skipped if .chr is missing)",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="4-turn growing-prefill protocol instead of independent items",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the fixture and exit (no GPU)",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="run RUN_ORDER (BF16 3B, BF16 14B, NF4 3B, NF4 14B) instead of one --lab",
    )
    parser.add_argument(
        "--plate",
        default="",
        help="also write the summary infographic PNG to this path",
    )
    parser.add_argument(
        "--redraw",
        default="",
        help=(
            "redraw the plate from a finished run root (its hard_matrix.csv and "
            "hard_script.json) and exit; no GPU, no model, no re-run"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _report_matrix(runs: Sequence[HardRun]) -> None:
    print(
        "\nrun        weights MiB (disk)   working set   smi peak   TTFT ms   tok/s   accuracy"
    )
    for run in runs:
        def cell(value: float | None, decimals: int = 0, suffix: str = "") -> str:
            return "n/a" if value is None else f"{float(value):,.{decimals}f}{suffix}"

        print(
            f"{run.label:<10} {cell(run.weight_mib):>8} ({cell(run.expected_weight_mib):>7})  "
            f"{cell(run.working_set_mib):>11}  {cell(run.smi_peak_mib):>9}  "
            f"{cell(run.mean_ttft_ms):>7}  {cell(run.mean_tok_s, 1):>6}  "
            f"{'n/a' if run.accuracy is None else f'{100 * run.accuracy:.0f}%':>8}"
        )
        if run.notes:
            print(f"  notes: {run.notes[:240]}")
    print(
        "\nnvidia-smi stops at 12288 MiB. The working set is the honest 14B BF16 number; "
        "do not quote the two smi peaks as the win."
    )


def _report(bundle: Any, scores: Sequence[Mapping[str, Any]]) -> None:
    print("\ncodec  accuracy  mean_ttft_ms  mean_tok_s  after_load_smi  torch_mib")
    for row in bundle.summary:
        codec = str(row["codec"])
        acc = accuracy(scores, codec)
        acc_text = "n/a" if acc is None else f"{100 * acc:.0f}%"

        def cell(key: str, decimals: int = 0) -> str:
            value = row.get(key)
            return "n/a" if value is None else f"{float(value):.{decimals}f}"

        print(
            f"{codec:<6} {acc_text:>8}  {cell('mean_ttft_ms'):>12}  "
            f"{cell('mean_decode_tok_s', 1):>10}  {cell('vram_after_load_smi_mib'):>14}  "
            f"{cell('vram_after_load_torch_mib'):>9}"
        )
        print(f"  notes: {row.get('notes', '')}")
    print("\nDo not quote Paris/Berlin/323 as quality. hard_scores.csv is the quality table.")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    if args.list:
        fixture = load_fixture()
        for key, items in fixture.items():
            print(f"{key}: {len(items)} items")
            for item in items:
                print(f"  {item.id:22} {item.kind:8} gold={item.gold!r}")
        print(f"\nfixture: {items_path()}")
        print("override with DEEPFOLD_HARD if you have a larger local JSON.")
        return 0

    if args.redraw:
        for path in redraw_plate(args.redraw, args.plate or None, history=args.history):
            print(f"wrote {path}")
        return 0

    import time

    if args.matrix:
        out_dir = Path(args.out) if args.out else Path(RUNS_DIR) / time.strftime(
            "hard-matrix-%Y%m%d-%H%M%S"
        )
        order = ", ".join(f"{codec} {slug}" for slug, codec in RUN_ORDER)
        print(f"matrix [{order}]  history={args.history}  out={out_dir}", flush=True)
        runs = run_matrix(
            out_dir,
            history=args.history,
            verbose=not args.quiet,
        )
        _report_matrix(runs)
        print(f"\nwrote {out_dir / 'hard_matrix.csv'}")
        # The plate always lands next to the CSVs it was drawn from; --plate
        # adds the docs/img copy.
        for path in redraw_plate(out_dir, args.plate, history=args.history):
            print(f"wrote {path}")
        return 0

    if not args.lab:
        parser.error("--lab is required for a live run (or pass --list / --matrix)")

    lab = lab_by_slug(args.lab)

    out_dir = Path(args.out) if args.out else Path(RUNS_DIR) / time.strftime(
        f"hard-{lab.slug}-%Y%m%d-%H%M%S"
    )
    print(f"lab {lab.slug}  history={args.history}  out={out_dir}", flush=True)
    bundle, scores = run_hard(
        lab,
        out_dir,
        history=args.history,
        codec=args.codec,
        verbose=not args.quiet,
    )
    write_scores(out_dir / "hard_scores.csv", scores)
    print(f"wrote {out_dir / 'hard_scores.csv'}")
    _report(bundle, scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

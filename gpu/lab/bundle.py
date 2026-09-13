"""The four frozen tables of a lab run, and the bundle that carries them.

Stdlib only, on purpose: the notebook and the README must be able to load a
finished run and draw the figure on a machine with no CUDA, no torch and no
model on disk. ``comparison_figure`` therefore accepts a bundle that came
straight off these CSVs.

The column names come from ``docs/lab.md`` and are frozen. Do not add,
rename or reorder them here -- write a new artifact instead. ``None`` means "the
GPU did not report this", and it is written as an empty cell, never as ``0``.
"""

from __future__ import annotations

import csv
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "CODECS",
    "EVENTS_COLUMNS",
    "MESSAGES_COLUMNS",
    "REQUIRED_EVENTS",
    "SUMMARY_COLUMNS",
    "TABLES",
    "TIMELINE_COLUMNS",
    "LabBundle",
    "MessageWindow",
]

# Plot and table order. `bf16` is the baseline, so it comes first everywhere.
CODECS = ("bf16", "nf4")

TIMELINE_COLUMNS = (
    "t_s",
    "codec",
    "used_mib",
    "total_mib",
    "util_gpu",
    "util_mem",
    "power_w",
    "temp_c",
    "clock_sm_mhz",
    "clock_mem_mhz",
    "torch_alloc_mib",
    "torch_reserved_mib",
    "torch_max_alloc_mib",
    "message_id",
)

EVENTS_COLUMNS = ("t_s", "codec", "event", "message_id", "detail")

MESSAGES_COLUMNS = (
    "codec",
    "message_id",
    "prompt",
    "response",
    "prompt_tokens",
    "new_tokens",
    "prefill_ms",
    "decode_ms",
    "decode_tok_s",
    "stop_reason",
    "quality_ok",
)

SUMMARY_COLUMNS = (
    "codec",
    "load_s",
    "vram_before_mib",
    "vram_after_load_smi_mib",
    "vram_after_load_torch_mib",
    "vram_peak_smi_mib",
    "weight_mib",
    "kv_mib",
    "mean_ttft_ms",
    "mean_decode_tok_s",
    "n_messages",
    "quality_all_ok",
    "notes",
)

# Every session must emit all of these; `msg_*` and `first_token` once per turn.
REQUIRED_EVENTS = (
    "start",
    "load_start",
    "load_end",
    "warmup_start",
    "warmup_end",
    "msg_send",
    "first_token",
    "msg_done",
    "unload_start",
    "unload_end",
    "stop",
)

# Cell kinds: "time" keeps millisecond resolution, "num" is a float that may be
# missing, "text" is always a string (empty, never None), "bool" is true/false.
_KINDS: dict[str, dict[str, str]] = {
    "timeline": {
        "t_s": "time",
        "codec": "text",
        "message_id": "text",
    },
    "events": {
        "t_s": "time",
        "codec": "text",
        "event": "text",
        "message_id": "text",
        "detail": "text",
    },
    "messages": {
        "codec": "text",
        "message_id": "int",
        "prompt": "text",
        "response": "text",
        "prompt_tokens": "int",
        "new_tokens": "int",
        "stop_reason": "text",
        "quality_ok": "bool",
    },
    "summary": {
        "codec": "text",
        "n_messages": "int",
        "quality_all_ok": "bool",
        "notes": "text",
    },
}

TABLES: dict[str, tuple[str, ...]] = {
    "timeline": TIMELINE_COLUMNS,
    "events": EVENTS_COLUMNS,
    "messages": MESSAGES_COLUMNS,
    "summary": SUMMARY_COLUMNS,
}

# What each table is written to.
FILENAMES: dict[str, str] = {name: f"{name}.csv" for name in TABLES}

_MISSING = ("", "n/a", "[n/a]", "[not supported]", "none", "nan")


def _kind(table: str, column: str) -> str:
    return _KINDS[table].get(column, "num")


# --------------------------------------------------------------------------- #
# cells
# --------------------------------------------------------------------------- #


def _parse_num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.lower() in _MISSING:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value: Any) -> int | None:
    number = _parse_num(value)
    return None if number is None else int(round(number))


def _parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _MISSING:
        return None
    if text in ("true", "1", "yes", "ok", "pass"):
        return True
    if text in ("false", "0", "no", "fail"):
        return False
    return None


def _parse_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse_cell(table: str, column: str, value: Any) -> Any:
    kind = _kind(table, column)
    if kind == "text":
        return _parse_text(value)
    if kind == "int":
        return _parse_int(value)
    if kind == "bool":
        return _parse_bool(value)
    return _parse_num(value)


def format_cell(table: str, column: str, value: Any) -> str:
    """One CSV cell. Missing stays empty so nobody reads a gap as zero."""
    if value is None:
        return ""
    kind = _kind(table, column)
    if kind == "text":
        return _parse_text(value)
    if kind == "bool":
        parsed = _parse_bool(value)
        return "" if parsed is None else ("true" if parsed else "false")
    if kind == "int":
        parsed_int = _parse_int(value)
        return "" if parsed_int is None else str(parsed_int)
    number = _parse_num(value)
    if number is None:
        return ""
    return f"{number:.4f}" if kind == "time" else f"{number:.6g}"


def normalize_row(table: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """One row with exactly the frozen columns, values in their final types."""
    unknown = set(row) - set(TABLES[table])
    if unknown:
        raise ValueError(f"{table}.csv has no column(s) {sorted(unknown)}")
    return {name: parse_cell(table, name, row.get(name)) for name in TABLES[table]}


def normalize_rows(table: str, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_row(table, row) for row in rows]


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MessageWindow:
    """One user turn, as the plot needs it: a span plus the hover text."""

    codec: str
    message_id: int
    t_send: float
    t_done: float
    t_first_token: float | None = None
    detail: str = ""

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t_done - self.t_send)


@dataclass
class LabBundle:
    """One lab run: both sessions' timelines, events, replies and summaries.

    Times are seconds from **that session's** ``start``, which is what makes the
    two codecs comparable on one x axis.
    """

    timeline: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    summary: list[dict[str, Any]] = field(default_factory=list)
    source: str = "live"

    # --- construction -----------------------------------------------------
    @classmethod
    def of(
        cls,
        timeline: Iterable[Mapping[str, Any]] = (),
        events: Iterable[Mapping[str, Any]] = (),
        messages: Iterable[Mapping[str, Any]] = (),
        summary: Iterable[Mapping[str, Any]] = (),
        *,
        source: str = "live",
    ) -> "LabBundle":
        """Normalize whatever the sessions produced into the frozen schema."""
        return cls(
            timeline=normalize_rows("timeline", timeline),
            events=normalize_rows("events", events),
            messages=normalize_rows("messages", messages),
            summary=normalize_rows("summary", summary),
            source=source,
        )

    def merge(self, other: "LabBundle") -> "LabBundle":
        """Append another session's tables. Session-local times are untouched."""
        return LabBundle(
            timeline=self.timeline + other.timeline,
            events=self.events + other.events,
            messages=self.messages + other.messages,
            summary=self.summary + other.summary,
            source=self.source,
        )

    def table(self, name: str) -> list[dict[str, Any]]:
        return getattr(self, name)  # type: ignore[no-any-return]

    # --- queries the plot and the notebook need ---------------------------
    @property
    def codecs(self) -> list[str]:
        """Codecs that actually have samples, in plot order."""
        seen = {str(row["codec"]) for row in self.timeline}
        seen |= {str(row["codec"]) for row in self.summary}
        ordered = [codec for codec in CODECS if codec in seen]
        return ordered + sorted(seen - set(CODECS))

    @property
    def synthetic(self) -> bool:
        """True when the numbers are a fixture, so the figure can say so."""
        if self.source == "fixture":
            return True
        return any("fixture" in str(row.get("notes", "")).lower() for row in self.summary)

    def rows_for(self, table: str, codec: str) -> list[dict[str, Any]]:
        return [row for row in self.table(table) if row.get("codec") == codec]

    def series(self, codec: str, column: str) -> tuple[list[float], list[float | None]]:
        """``(t_s, values)`` for one codec, gaps kept as ``None``."""
        rows = self.rows_for("timeline", codec)
        return [float(r["t_s"]) for r in rows], [r.get(column) for r in rows]

    def summary_for(self, codec: str) -> dict[str, Any] | None:
        rows = self.rows_for("summary", codec)
        return rows[0] if rows else None

    def total_mib(self, default: float = 12288.0) -> float:
        """Card size as reported by nvidia-smi, for the red limit line."""
        for row in self.timeline:
            value = row.get("total_mib")
            if value:
                return float(value)
        return default

    def message_windows(self, codec: str) -> list[MessageWindow]:
        """``msg_send`` -> ``msg_done`` spans, the source for the plot's vrects."""
        sends: dict[int, float] = {}
        firsts: dict[int, float] = {}
        windows: list[MessageWindow] = []
        for row in self.rows_for("events", codec):
            message_id = _parse_int(row.get("message_id"))
            if message_id is None:
                continue
            t_s = float(row["t_s"])
            event = row.get("event")
            if event == "msg_send":
                sends[message_id] = t_s
            elif event == "first_token":
                firsts[message_id] = t_s
            elif event == "msg_done" and message_id in sends:
                windows.append(
                    MessageWindow(
                        codec=codec,
                        message_id=message_id,
                        t_send=sends.pop(message_id),
                        t_done=t_s,
                        t_first_token=firsts.get(message_id),
                        detail=str(row.get("detail", "")),
                    )
                )
        return sorted(windows, key=lambda w: w.t_send)

    def value_at(self, codec: str, column: str, t_s: float) -> float | None:
        """Nearest sample of ``column`` at ``t_s`` -- used to sit event markers
        on the VRAM line instead of on a rotated label."""
        times, values = self.series(codec, column)
        if not times:
            return None
        index = min(bisect_left(times, t_s), len(times) - 1)
        for candidate in (index, index - 1, index + 1):
            if 0 <= candidate < len(values) and values[candidate] is not None:
                return float(values[candidate])
        return None

    # --- csv --------------------------------------------------------------
    def write(self, out_dir: str | Path) -> dict[str, Path]:
        """Write the four CSVs. Returns ``{table: path}``."""
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for table, columns in TABLES.items():
            path = directory / FILENAMES[table]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(columns)
                for row in self.table(table):
                    writer.writerow([format_cell(table, name, row.get(name)) for name in columns])
            written[table] = path
        return written

    @classmethod
    def read(cls, path: str | Path) -> "LabBundle":
        """Load a finished run from its directory. No GPU, no torch, no model."""
        directory = Path(path)
        tables: dict[str, list[dict[str, Any]]] = {}
        for table, columns in TABLES.items():
            csv_path = directory / FILENAMES[table]
            if not csv_path.is_file():
                tables[table] = []
                continue
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                header = tuple(reader.fieldnames or ())
                if header != columns:
                    raise ValueError(
                        f"{csv_path} header is {header}, expected the frozen {columns}"
                    )
                tables[table] = normalize_rows(table, reader)
        return cls(source="csv", **tables)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return (
            f"LabBundle(source={self.source!r}, codecs={self.codecs}, "
            f"timeline={len(self.timeline)}, events={len(self.events)}, "
            f"messages={len(self.messages)}, summary={len(self.summary)})"
        )


def mean(values: Sequence[float | None]) -> float | None:
    """Mean of the samples that exist. Empty stays empty, not ``0``."""
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None

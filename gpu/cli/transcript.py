"""Saved ``deepfold chat`` transcripts under ``$DEEPFOLD_HOME/chats``.

History is JSON on disk. Resume from `/chats` is a full prefill: the GPU
KV cache is not stored, only the messages.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import chats_root, slug

__all__ = [
    "Transcript",
    "chats_root",
    "list_transcripts",
    "load_transcript",
    "new_transcript",
    "save_transcript",
    "title_from",
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"


def title_from(messages: list[dict[str, Any]], *, limit: int = 72) -> str:
    """First user line, collapsed, for the picker."""
    for row in messages:
        if row.get("role") != "user":
            continue
        text = re.sub(r"\s+", " ", str(row.get("content") or "")).strip()
        if text:
            return text if len(text) <= limit else text[: limit - 1] + "…"
    return "(empty)"


@dataclass
class Transcript:
    id: str
    model: str
    model_slug: str
    created: str
    updated: str
    title: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    path: Path | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "model_slug": self.model_slug,
            "created": self.created,
            "updated": self.updated,
            "title": self.title,
            "messages": self.messages,
        }


def new_transcript(model: str) -> Transcript:
    stamp = _now()
    ident = _new_id()
    path = chats_root() / f"{ident}.json"
    return Transcript(
        id=ident,
        model=str(model),
        model_slug=slug(model),
        created=stamp,
        updated=stamp,
        path=path,
    )


def save_transcript(row: Transcript) -> Path:
    """Write the file. Empty conversations are still written so `/clear` sticks."""
    if not row.title:
        row.title = title_from(row.messages)
    row.updated = _now()
    dest = row.path or (chats_root() / f"{row.id}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    row.path = dest
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(row.to_json(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(dest)
    return dest


def _from_payload(payload: dict[str, Any], path: Path) -> Transcript | None:
    ident = str(payload.get("id") or path.stem)
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None
    clean: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        if role not in ("user", "assistant", "tool", "system"):
            continue
        content = str(item.get("content") or "")
        row: dict[str, Any] = {"role": role, "content": content}
        calls = item.get("tool_calls")
        if isinstance(calls, list) and calls:
            row["tool_calls"] = calls
        name = item.get("name")
        if isinstance(name, str) and name:
            row["name"] = name
        if content or row.get("tool_calls") or role in ("tool", "system"):
            clean.append(row)
    model = str(payload.get("model") or "")
    return Transcript(
        id=ident,
        model=model,
        model_slug=str(payload.get("model_slug") or (slug(model) if model else "")),
        created=str(payload.get("created") or ""),
        updated=str(payload.get("updated") or ""),
        title=str(payload.get("title") or title_from(clean)),
        messages=clean,
        path=path,
    )


def load_transcript(ident: str) -> Transcript | None:
    """``ident`` is a stem, a filename, or an absolute path."""
    raw = Path(ident)
    candidates = [raw] if raw.is_file() else [chats_root() / f"{raw.stem}.json", raw]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            return _from_payload(payload, path)
    return None


def list_transcripts(model: str | None = None) -> list[Transcript]:
    """Newest first. ``model`` restricts to that tree's slug."""
    root = chats_root()
    if not root.is_dir():
        return []
    wanted = slug(model) if model else None
    rows: list[Transcript] = []
    for path in root.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        row = _from_payload(payload, path)
        if row is None:
            continue
        if wanted and row.model_slug != wanted:
            continue
        if not row.messages:
            continue
        rows.append(row)
    rows.sort(key=lambda r: r.updated or r.created, reverse=True)
    return rows

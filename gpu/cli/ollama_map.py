"""Allowlisted Ollama library tags → HuggingFace ids. Never ~/.ollama.

WAVE 10 P2: ``from-ollama`` is a **name service**. The table is exact tags
(and the documented aliases below). No fuzzy match, no ``ollama show``
scrape for an id, no GGUF. ``llama3.1:8b`` is not a row until a measured
3080 Llama generate exists; walker classification is not a license to add it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .paths import looks_like_gguf

__all__ = [
    "ALLOWLIST",
    "HF_ONLY",
    "AllowlistRow",
    "Mapped",
    "ResolveError",
    "canonical_tag",
    "hf_ids",
    "known_tags",
    "resolve",
    "row_for_hf",
]


@dataclass(frozen=True)
class AllowlistRow:
    """One HuggingFace repo the command is allowed to name."""

    hf_id: str
    disk_gb: float
    #: Local trees we already keep on the machine of record (WAVE 7 §6.2).
    local_hints: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


#: Exact Ollama library tags this command will look up. Case-sensitive.
ALLOWLIST: dict[str, AllowlistRow] = {}

_QWEN_3B = AllowlistRow(
    hf_id="Qwen/Qwen2.5-3B-Instruct",
    disk_gb=6.2,
    local_hints=(r"C:\dev\models\Qwen2.5-3B-Instruct",),
    tags=("qwen2.5:3b", "qwen2.5:3b-instruct"),
)
_QWEN_14B = AllowlistRow(
    hf_id="Qwen/Qwen2.5-14B-Instruct",
    disk_gb=28.0,
    local_hints=(r"C:\dev\models\Qwen2.5-14B-Instruct",),
    tags=("qwen2.5:14b", "qwen2.5:14b-instruct"),
)

#: Table HF ids that have no short Ollama library tag. ``--hf`` only.
HF_ONLY: dict[str, AllowlistRow] = {
    "internlm/internlm2_5-20b-chat": AllowlistRow(
        hf_id="internlm/internlm2_5-20b-chat",
        disk_gb=37.0,
        local_hints=(r"C:\dev\models\internlm2_5-20b-chat",),
    ),
}

for _row in (_QWEN_3B, _QWEN_14B):
    for _tag in _row.tags:
        ALLOWLIST[_tag] = _row


def known_tags() -> tuple[str, ...]:
    """Tags named in the refuse copy, in table order."""
    out: list[str] = []
    seen: set[str] = set()
    for row in (_QWEN_3B, _QWEN_14B):
        for tag in row.tags:
            if tag not in seen:
                seen.add(tag)
                out.append(tag)
    return tuple(out)


def hf_ids() -> frozenset[str]:
    """Every HuggingFace id the table (including ``--hf``-only rows) names."""
    return frozenset({row.hf_id for row in ALLOWLIST.values()} | set(HF_ONLY))


def row_for_hf(hf_id: str) -> AllowlistRow | None:
    if hf_id in HF_ONLY:
        return HF_ONLY[hf_id]
    for row in ALLOWLIST.values():
        if row.hf_id == hf_id:
            return row
    return None


def canonical_tag(tag: str) -> str:
    """Strip a leading ``library/`` (any case). The rest is case-sensitive."""
    text = tag.strip()
    if text.lower().startswith("library/"):
        return text.split("/", 1)[1]
    return text


class ResolveError(ValueError):
    """Closed lookup: GGUF path, unknown tag, or ``--hf`` that does not match."""

    def __init__(self, kind: str, tag: str, detail: str = "") -> None:
        self.kind = kind
        self.tag = tag
        self.detail = detail
        super().__init__(detail or kind)


@dataclass(frozen=True)
class Mapped:
    """What ``from-ollama`` will name (and maybe download)."""

    tag: str
    row: AllowlistRow
    via_hf_flag: bool = False

    @property
    def hf_id(self) -> str:
        return self.row.hf_id

    @property
    def disk_gb(self) -> float:
        return self.row.disk_gb


def resolve(tag: str, hf: str | None = None) -> Mapped:
    """Look up ``tag`` / ``--hf``. Never opens a file.

    1. A GGUF / ``.ollama`` / ``sha256-`` **argument** is refused (WAVE 7 §8).
    2. ``--hf`` on a known tag must be that tag's id.
    3. Unknown tag + ``--hf`` is allowed only when the id is on the table.
    4. Else the tag must be an exact allowlisted name.
    """
    if looks_like_gguf(tag):
        raise ResolveError("gguf", tag)

    key = canonical_tag(tag)
    row = ALLOWLIST.get(key)
    hf_id = (hf or "").strip() or None

    if hf_id is not None:
        if row is not None:
            if hf_id != row.hf_id:
                raise ResolveError(
                    "hf_mismatch",
                    key,
                    f"--hf {hf_id} does not match tag {key!r} (expected {row.hf_id}).",
                )
            return Mapped(tag=key, row=row, via_hf_flag=True)
        table = row_for_hf(hf_id)
        if table is None:
            raise ResolveError(
                "unknown",
                key,
                f"--hf {hf_id} is not a row in the from-ollama table.",
            )
        return Mapped(tag=key, row=table, via_hf_flag=True)

    if row is None:
        raise ResolveError("unknown", key)
    return Mapped(tag=key, row=row)

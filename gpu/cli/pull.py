"""``deepfold pull <hf_id>``: allowlisted HuggingFace BF16 tree. Never GGUF."""

from __future__ import annotations

import os
import sys
from argparse import Namespace

from . import hub, messages
from .ollama_map import ResolveError, hf_id_list, row_for_hf
from .paths import looks_like_gguf

__all__ = ["pull", "resolve_hf"]


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def resolve_hf(hf_id: str) -> object:
    """Look up a Hub id. Never opens a file."""
    if looks_like_gguf(hf_id):
        raise ResolveError("gguf", hf_id)
    key = hf_id.strip()
    row = row_for_hf(key)
    if row is None:
        raise ResolveError("unknown", key)
    return row


def pull(args: Namespace) -> int:
    """CLI body. Exit 1 on refuse; 0 after a resolved tree."""
    try:
        row = resolve_hf(args.hf_id)
    except ResolveError as exc:
        if exc.kind == "gguf":
            _err(messages.GGUF)
            return 1
        _err(messages.unknown_hf_id(exc.tag, hf_id_list()))
        return 1

    dest = hub.pull_destination(row, getattr(args, "dir", None))
    already = dest.is_dir() and (dest / "config.json").is_file()
    disk = row.disk_gb
    extra = f"{disk:g}" if disk != int(disk) else f"{int(disk)}"
    _err(
        f"HuggingFace id we will download (BF16 safetensors): {row.hf_id}\n"
        f"Approximate extra disk: ~{extra} GB  (then NF4 .chr; shards optional to delete)"
    )
    if already:
        _err(f"Already on disk: {dest}  (no Hub round-trip)")
    else:
        if hub.hub_missing():
            _err(messages.NEED_HUB)
            return 1
        if not hub.confirmed(yes=bool(getattr(args, "yes", False))):
            return 1
        dest.mkdir(parents=True, exist_ok=True)
        try:
            hub.snapshot_download(repo_id=row.hf_id, local_dir=os.fspath(dest))
        except Exception as exc:  # noqa: BLE001 -- Hub errors are a user-facing refuse
            _err(f"pull: HuggingFace download failed ({type(exc).__name__}: {exc}).")
            return 1

    print(f"deepfold chat --model {dest}")
    print(f"deepfold run --model {dest}")
    return 0

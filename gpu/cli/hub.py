"""HuggingFace snapshot_download, stubbed in tests. Never called from CI."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from .ollama_map import AllowlistRow
from .paths import deepfold_home, models_root, slug

__all__ = [
    "confirmed",
    "existing_tree",
    "hub_missing",
    "leaf_name",
    "pull_destination",
    "snapshot_download",
    "source_complete",
]

_TOKENIZER_MARKERS = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def hub_missing() -> bool:
    return importlib.util.find_spec("huggingface_hub") is None


def snapshot_download(*, repo_id: str, local_dir: str) -> str:
    """The only Hub call. Tests replace this function; CI must not hit the network."""
    from huggingface_hub import snapshot_download as _sd

    return _sd(repo_id=repo_id, local_dir=local_dir)


def source_complete(path: Path | str) -> bool:
    """True only when a Hub tree can actually be compressed, not when config.json exists.

    A directory that only has ``config.json`` (or an index whose shards are
    still missing) is an interrupted download. ``pull`` must resume, not skip.
    """
    root = Path(path)
    if not (root / "config.json").is_file():
        return False
    if not any((root / name).is_file() for name in _TOKENIZER_MARKERS):
        return False
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if indexes:
        needed: set[str] = set()
        for index in indexes:
            try:
                data = json.loads(index.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            weight_map = data.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                return False
            needed.update(str(name) for name in weight_map.values() if name)
        return bool(needed) and all((root / name).is_file() for name in needed)
    return any(root.glob("*.safetensors"))


def confirmed(*, yes: bool) -> bool:
    if yes:
        return True
    stdin = sys.stdin
    if stdin is None or not stdin.isatty():
        print("No TTY and no --yes: nothing downloaded.", file=sys.stderr)
        return False
    print("Proceed with HuggingFace download? [y/N] ", file=sys.stderr, end="", flush=True)
    line = stdin.readline()
    return line.strip().lower() in ("y", "yes")


def leaf_name(row: AllowlistRow) -> str:
    return row.hf_id.rsplit("/", 1)[-1]


def existing_tree(row: AllowlistRow) -> Path | None:
    """A complete local tree (config + tokenizer + weight shards), not a stub."""
    leaf = leaf_name(row)
    candidates = [Path(h) for h in row.local_hints]
    candidates.append(models_root() / leaf)
    candidates.append(deepfold_home() / "hf" / slug(row.hf_id))
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if source_complete(path):
            return path
    return None


def pull_destination(row: AllowlistRow, explicit: str | None) -> Path:
    """``--dir``, else ``$DEEPFOLD_MODELS/<leaf>`` when that root is in play, else cache."""
    if explicit:
        return Path(explicit)
    found = existing_tree(row)
    if found is not None:
        return found
    import os

    from .paths import ENV_MODELS, _LEGACY_MODELS

    if os.environ.get(ENV_MODELS) or _LEGACY_MODELS.is_dir():
        return models_root() / leaf_name(row)
    return deepfold_home() / "hf" / slug(row.hf_id)

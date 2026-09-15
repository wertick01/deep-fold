"""HuggingFace snapshot_download, stubbed in tests. Never called from CI."""

from __future__ import annotations

import importlib.util
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
]


def hub_missing() -> bool:
    return importlib.util.find_spec("huggingface_hub") is None


def snapshot_download(*, repo_id: str, local_dir: str) -> str:
    """The only Hub call. Tests replace this function; CI must not hit the network."""
    from huggingface_hub import snapshot_download as _sd

    return _sd(repo_id=repo_id, local_dir=local_dir)


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
    """A complete local tree. ``config.json`` is the marker."""
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
        if (path / "config.json").is_file():
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

"""``deepfold from-ollama <tag>``: allowlisted name → HuggingFace BF16 tree.

Never reads ``~/.ollama``, never loads GGUF, never converts a blob to CHR.
``snapshot_download`` is behind :func:`_snapshot_download` so tests stub Hub
without a network. A live pull is Pavel's machine, not CI.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from . import messages, run as run_mod
from .ollama_map import Mapped, ResolveError, resolve
from .paths import deepfold_home, slug

__all__ = ["from_ollama"]


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def _hub_missing() -> bool:
    return importlib.util.find_spec("huggingface_hub") is None


def _snapshot_download(*, repo_id: str, local_dir: str) -> str:
    """The only Hub call. Tests replace this function; CI must not hit the network."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, local_dir=local_dir)


def existing_hf_dir(mapped: Mapped) -> Path | None:
    """A complete local tree we already have. ``config.json`` is the marker."""
    for hint in mapped.row.local_hints:
        path = Path(hint)
        if (path / "config.json").is_file():
            return path
    return None


def destination(mapped: Mapped, explicit: str | None) -> Path:
    """``--dir``, else an existing hint, else ``$DEEPFOLD_HOME/hf/<slug>/``."""
    if explicit:
        return Path(explicit)
    found = existing_hf_dir(mapped)
    if found is not None:
        return found
    return deepfold_home() / "hf" / slug(mapped.hf_id)


def _announce(mapped: Mapped) -> None:
    disk = mapped.disk_gb
    extra = f"{disk:g}" if disk != int(disk) else f"{int(disk)}"
    _err(
        "Ollama already has a GGUF copy. Deepfold cannot load GGUF.\n"
        f"HuggingFace id we will download (BF16 safetensors): {mapped.hf_id}\n"
        f"Approximate extra disk: ~{extra} GB  (then NF4 .chr; shards optional to delete)"
    )


def _confirmed(*, yes: bool) -> bool:
    if yes:
        return True
    stdin = sys.stdin
    if stdin is None or not stdin.isatty():
        _err("No TTY and no --yes: nothing downloaded.")
        return False
    print("Proceed with HuggingFace download? [y/N] ", file=sys.stderr, end="", flush=True)
    line = stdin.readline()
    return line.strip().lower() in ("y", "yes")


def _run_model(model_dir: Path) -> int:
    """Hand the tree to ``deepfold run``. Compress stays first-run of run, not here."""
    return run_mod.run(
        SimpleNamespace(
            model=str(model_dir),
            chr=None,
            chr_bin=None,
            prompt=None,
            max_new_tokens=64,
            max_seq=512,
            raw=False,
            warmup=True,
            no_compress=False,
            quiet=False,
            debug=False,
        )
    )


def from_ollama(args: Namespace) -> int:
    """CLI body. Exit 1 on refuse; 0 after a resolved tree (and after ``--run``)."""
    try:
        mapped = resolve(args.tag, hf=getattr(args, "hf", None))
    except ResolveError as exc:
        if exc.kind == "gguf":
            _err(messages.GGUF)
            return 1
        if exc.kind == "hf_mismatch":
            _err(f"from-ollama: {exc.detail}")
            return 1
        _err(messages.unknown_ollama_tag(exc.tag))
        return 1

    dest = destination(mapped, getattr(args, "dir", None))
    already = dest.is_dir() and (dest / "config.json").is_file()

    _announce(mapped)
    if already:
        _err(f"Already on disk: {dest}  (no Hub round-trip)")
    else:
        if _hub_missing():
            _err(messages.NEED_HUB)
            return 1
        if not _confirmed(yes=bool(getattr(args, "yes", False))):
            return 1
        dest.mkdir(parents=True, exist_ok=True)
        try:
            _snapshot_download(repo_id=mapped.hf_id, local_dir=os.fspath(dest))
        except Exception as exc:  # noqa: BLE001 -- Hub errors are a user-facing refuse
            _err(f"from-ollama: HuggingFace download failed ({type(exc).__name__}: {exc}).")
            return 1

    print(f"deepfold run --model {dest}")
    if getattr(args, "run", False):
        return _run_model(dest)
    return 0

"""Where things live: the ``chr`` binary, the cache root, and the ``.chr`` file.

Env vars are ``DEEPFOLD_MODEL``, ``DEEPFOLD_CHR``, ``DEEPFOLD_CHR_BIN``,
``DEEPFOLD_HOME``. :func:`looks_like_gguf` inspects the *string* the user
typed: a GGUF path is refused without being read. :func:`find_chr_file`
picks a sibling ``.chr`` only when ``accept`` (CHR0 header vs ``config.json``)
says it belongs to this model.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

ENV_MODEL = "DEEPFOLD_MODEL"
ENV_CHR = "DEEPFOLD_CHR"
ENV_CHR_BIN = "DEEPFOLD_CHR_BIN"
ENV_HOME = "DEEPFOLD_HOME"

# Substrings that mean "this is Ollama's or llama.cpp's copy, not a HF tree".
# Matched against the user's argument, never against a directory listing.
_GGUF_MARKERS = (".gguf", ".ollama", "sha256-")


def deepfold_home() -> Path:
    """Cache root: ``$DEEPFOLD_HOME``, else ``%LOCALAPPDATA%\\deepfold``."""
    env = os.environ.get(ENV_HOME)
    if env:
        return Path(env)
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    if local:
        return Path(local) / "deepfold"
    return Path.home() / ".cache" / "deepfold"


def chr_exe_names() -> tuple[str, ...]:
    return ("chr.exe",) if os.name == "nt" else ("chr",)


def find_chr_bin(explicit: str | None = None) -> Path | None:
    """wave8-install.md §6.3, in order. Returns the path that won.

    1. ``--chr-bin``; 2. ``DEEPFOLD_CHR_BIN``; 3. ``PATH``; 4. the checkout
    root; 5. ``$DEEPFOLD_HOME/bin``.
    """
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None

    env = os.environ.get(ENV_CHR_BIN)
    if env:
        p = Path(env)
        if p.is_file():
            return p

    for name in chr_exe_names():
        hit = shutil.which(name)
        if hit:
            return Path(hit)

    for name in chr_exe_names():
        p = REPO / name
        if p.is_file():
            return p

    for name in chr_exe_names():
        p = deepfold_home() / "bin" / name
        if p.is_file():
            return p

    return None


def slug(model: str | os.PathLike[str]) -> str:
    """Cache-safe name: a directory contributes its leaf, a hub id both halves.

    ``C:\\dev\\models\\Qwen2.5-3B-Instruct`` -> ``Qwen2.5-3B-Instruct``;
    ``Qwen/Qwen2.5-3B-Instruct`` -> ``Qwen_Qwen2.5-3B-Instruct``.
    """
    path = Path(str(model))
    text = str(model).replace("\\", "/").strip("/")
    if not path.is_absolute() and not path.exists() and text.count("/") == 1:
        name = text.replace("/", "_")
    else:
        name = path.name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "model"


def cached_chr(model_dir: str | os.PathLike[str], codec: str = "nf4") -> Path:
    """Where ``run`` writes a first-run compress (wave7-ux.md §6.2).

    ``codec`` is ``nf4`` or ``vq``; the files are ``<slug>.nf4.chr`` and
    ``<slug>.vq2.chr`` so both can sit in the cache without clobbering.
    """
    from .codec import suffix

    return deepfold_home() / "chr" / f"{slug(model_dir)}.{suffix(codec)}"


def chr_candidates(model_dir: Path, explicit: str | None = None) -> list[Path]:
    """Every ``.chr`` that could belong to ``model_dir``, best guess first.

    wave7-ux.md §2.1 step 3: the flag and ``DEEPFOLD_CHR`` are definite answers;
    after them come a ``.chr`` inside the model directory, then its siblings,
    then the cache. ``DEEPFOLD_CHR`` is honoured only when it points at a file
    that exists -- the lab default names a 3B file, and ``--model`` may be the
    14B tree.
    """
    if explicit:
        p = Path(explicit)
        return [p] if p.is_file() else []

    out: list[Path] = []
    env = os.environ.get(ENV_CHR)
    if env and Path(env).is_file():
        out.append(Path(env))

    for directory in (model_dir, model_dir.parent):
        try:
            matches = sorted(directory.glob("*.nf4.chr")) + sorted(
                directory.glob("*.vq2.chr")
            )
        except OSError:
            matches = []
        out.extend(p for p in matches if p not in out)

    for codec in ("nf4", "vq"):
        cached = cached_chr(model_dir, codec)
        if cached.is_file() and cached not in out:
            out.append(cached)
    return out


def find_chr_file(
    model_dir: Path,
    explicit: str | None = None,
    *,
    accept: object = None,
) -> Path | None:
    """Pick one ``.chr``. Ambiguity is never resolved by sort order.

    A flat ``C:\\dev\\models`` holds a 3B, a 14B and a 20B ``.chr`` next to
    three model directories, so "first sibling wins" would happily hand the 20B
    file to a 3B skeleton. ``accept`` (a predicate over a path, in practice a
    CHR0 header comparison) decides between candidates -- including the one
    ``DEEPFOLD_CHR`` names, which the design admits only "if it matches this
    model". Without an ``accept``, an ambiguous directory returns None and the
    caller compresses rather than guesses.

    An explicit ``--chr`` is never second-guessed: the user named the file.
    """
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None

    candidates = chr_candidates(model_dir)
    if not candidates:
        return None
    if accept is not None:
        for path in candidates:
            if accept(path):  # type: ignore[operator]
                return path
        return None
    return candidates[0] if len(candidates) == 1 else None


def looks_like_gguf(text: str | os.PathLike[str]) -> bool:
    """True for a GGUF file or anything inside Ollama's blob store."""
    low = str(text).replace("\\", "/").lower()
    if any(marker in low for marker in _GGUF_MARKERS):
        return True
    return "/blobs/" in low or low.endswith("/blobs")

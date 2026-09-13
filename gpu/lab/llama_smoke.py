"""3080 Llama generate, or an honest skip. Never a Hub pull, never an Ollama tag.

P1 (walker + TokenLoop ``llama_swiglu``) is already in the tree. This module
does not add ``llama3.1:8b`` to ``from-ollama``: walker classification is not a
license to name that tag. A measured generate on this 3080 is.

What it will do:

1. Look for a HuggingFace Llama tree **already on disk** (``DEEPFOLD_LLAMA``,
   well-known ``C:\\dev\\models\\…`` names, then ``model_type=llama`` under
   ``C:\\dev\\models``). Never ``~/.ollama``, never GGUF, never ``snapshot_download``.
2. Refuse SWA and qk-norm from ``config.json`` before touching the GPU.
3. ``--generate`` is the only path that loads weights. Default is discover +
   write ``SOURCE.txt``.

    python -m gpu.lab.llama_smoke
    python -m gpu.lab.test_llama_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from gpu.cli.paths import looks_like_gguf
from gpu.graphs import config_refusal, effective_sliding_window

__all__ = [
    "CANDIDATE_NAMES",
    "ENV_LLAMA",
    "LlamaTree",
    "SKIP",
    "discover",
    "read_config",
    "write_source",
]

ENV_LLAMA = "DEEPFOLD_LLAMA"
SKIP = "SKIP:"
_MODELS_ROOT = Path(r"C:\dev\models")

#: Leaf names we look for first. None of these have to exist.
CANDIDATE_NAMES = (
    "Meta-Llama-3.1-8B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-8B",
    "Meta-Llama-3-8B-Instruct",
    "Llama-3-8B-Instruct",
)


@dataclass(frozen=True)
class LlamaTree:
    """A directory that looks like a Llama HuggingFace tree."""

    path: Path
    model_type: str
    sliding_window: object
    refusal: str
    note: str = ""

    @property
    def attachable(self) -> bool:
        return not self.refusal


def read_config(model_dir: Path) -> dict[str, object] | None:
    """``config.json`` or None. Missing/broken is not a Llama tree."""
    cfg_path = model_dir / "config.json"
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _from_dir(model_dir: Path) -> LlamaTree | None:
    cfg = read_config(model_dir)
    if cfg is None:
        return None
    model_type = str(cfg.get("model_type") or "")
    if model_type != "llama":
        return None
    refusal = config_refusal(cfg) or ""
    window = effective_sliding_window(cfg)
    note = ""
    if not refusal:
        note = "config looks like llama_swiglu (no SWA in config.json); walker still decides at load"
    return LlamaTree(
        path=model_dir,
        model_type=model_type,
        sliding_window=window,
        refusal=refusal,
        note=note,
    )


def discover(
    *,
    explicit: str | None = None,
    models_root: Path | None = _MODELS_ROOT,
) -> list[LlamaTree]:
    """HuggingFace Llama dirs on disk, in attempt order. Never Hub, never Ollama."""
    found: list[LlamaTree] = []
    seen: set[Path] = set()

    def add(path: Path | None) -> None:
        if path is None:
            return
        resolved = path
        if looks_like_gguf(path):
            return
        if not path.is_dir():
            return
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        tree = _from_dir(path)
        if tree is None:
            return
        seen.add(resolved)
        found.append(tree)

    if explicit:
        add(Path(explicit))
        return found

    env = os.environ.get(ENV_LLAMA)
    if env:
        add(Path(env))

    root = models_root
    if root is not None and root.is_dir():
        for name in CANDIDATE_NAMES:
            add(root / name)
        try:
            children = sorted(root.iterdir())
        except OSError:
            children = []
        for child in children:
            if child.is_dir() and child.name != "runs":
                add(child)
    return found


def skip_reason(trees: list[LlamaTree]) -> str:
    """The CSV/SOURCE sentence. Empty numeric cells belong with this string."""
    if not trees:
        return (
            f"{SKIP} no HuggingFace Llama directory on disk. "
            "Looked at $DEEPFOLD_LLAMA and C:\\dev\\models (model_type=llama). "
            "Did not download meta-llama/Meta-Llama-3.1-8B-Instruct "
            "(gated, ~16 GB) and did not add llama3.1:8b to from-ollama."
        )
    blocked = [tree for tree in trees if not tree.attachable]
    if blocked and not any(tree.attachable for tree in trees):
        first = blocked[0]
        return (
            f"{SKIP} Llama tree at {first.path} is not attachable: "
            f"{first.refusal.splitlines()[0] if first.refusal else 'refused'}"
        )
    return ""


def write_source(out_dir: Path, trees: list[LlamaTree]) -> Path:
    """Honest skip/discover note. No tok/s."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "WAVE 10 leftover 7 — Llama smoke on the 3080.",
        "Walker (gpu.host.attach) already classifies Llama 3.x without qk-norm.",
        "from-ollama does not grow a llama3.1:8b row from this file.",
        "No Hub download. No ~/.ollama. No GGUF.",
        "",
    ]
    reason = skip_reason(trees)
    if reason:
        lines.append(reason)
        lines.append("A skip row is the result. Do not invent a Llama tok/s.")
    else:
        lines.append("Trees on disk:")
        for tree in trees:
            lines.append(f"  {tree.path}  model_type={tree.model_type}  {tree.note or tree.refusal}")
        lines.append("Generate was not started from discover-only.")
    path = out_dir / "SOURCE.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gpu.lab.llama_smoke",
        description=(
            "Find a local HuggingFace Llama tree. Do not download. "
            "--generate is the only path that would load the 3080."
        ),
    )
    parser.add_argument("--model", default="", help="explicit HuggingFace directory")
    parser.add_argument(
        "--out",
        default="",
        help="where to write SOURCE.txt (default C:\\dev\\models\\runs\\llama-smoke)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="load packed NF4 / compress+run; refused unless a tree is on disk",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser


def _generate(tree: LlamaTree, *, max_new_tokens: int) -> int:
    """Real 3080 path. Not reached when nothing is on disk."""
    from gpu.cli import run as run_mod
    from types import SimpleNamespace

    if tree.refusal:
        print(tree.refusal, file=sys.stderr)
        return 1
    return run_mod.run(
        SimpleNamespace(
            model=str(tree.path),
            chr=None,
            chr_bin=None,
            prompt="Say hello in one sentence.",
            max_new_tokens=max_new_tokens,
            max_seq=512,
            raw=False,
            warmup=True,
            no_compress=False,
            quiet=False,
            debug=False,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    if args.model and looks_like_gguf(args.model):
        print(
            "Llama smoke will not open GGUF or ~/.ollama. "
            "Point --model at a HuggingFace directory.",
            file=sys.stderr,
        )
        return 1

    trees = discover(explicit=args.model or None)
    out = Path(args.out) if args.out else Path(r"C:\dev\models\runs\llama-smoke")
    source = write_source(out, trees)
    print(f"wrote {source}")

    if not trees:
        print(skip_reason(trees), file=sys.stderr)
        return 1

    for tree in trees:
        state = "attachable-config" if tree.attachable else "refused"
        print(f"{tree.path}  {state}")
        if tree.refusal:
            print(f"  {tree.refusal.splitlines()[0]}")
        elif tree.note:
            print(f"  {tree.note}")

    if not args.generate:
        print(
            "Discover only: no generate. Pass --generate once a Llama HF tree is on disk.",
            file=sys.stderr,
        )
        return 0 if any(tree.attachable for tree in trees) else 1

    attachable = [tree for tree in trees if tree.attachable]
    if not attachable:
        print(skip_reason(trees), file=sys.stderr)
        return 1
    return _generate(attachable[0], max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    raise SystemExit(main())

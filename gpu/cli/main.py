"""Argument parsing for ``deepfold``. Thin on purpose.

    python -m gpu.cli doctor
    python -m gpu.cli run --model C:\\dev\\models\\Qwen2.5-3B-Instruct

After ``pip install -e .`` the same thing is on PATH as ``deepfold``. The lab
stays at ``python -m gpu.lab.run``. The parser builds no model and imports
neither torch nor transformers, so ``doctor`` can report a broken torch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from . import doctor as doctor_mod  # noqa: E402
from . import from_ollama as from_ollama_mod  # noqa: E402
from . import run as run_mod  # noqa: E402

PROG = "deepfold"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Packed CHR0 driver (NF4 or VQ 2-bit): weights stay packed in "
            "VRAM for the whole run. Ampere sm_86 (RTX 3080 class) only."
        ),
    )
    sub = ap.add_subparsers(dest="command", metavar="<command>")

    doc = sub.add_parser(
        "doctor",
        help="can deepfold run succeed on this machine? exit 0 if yes",
        description=(
            "Exit 0 run is possible; 2 broken install on a card that could "
            "run; 3 generate refused by this machine's class but chr compress "
            "works; 1 neither."
        ),
    )
    doc.add_argument("--model", help="also check this HuggingFace dir's config.json")
    doc.add_argument("--chr-bin", help="path to the Go chr binary")
    doc.add_argument(
        "--compress-only",
        action="store_true",
        help="ask only whether chr compress can run (macOS / CPU boxes)",
    )
    doc.set_defaults(func=doctor_mod.doctor)

    comp = sub.add_parser(
        "compress",
        help="wrap chr compress (NF4 or VQ 2-bit)",
        description=(
            "Packs a HuggingFace BF16/FP16 tree into one .chr. CPU only. "
            "Default --codec auto: NF4 if it fits the card, else NF4 overflow (H2). "
            "VQ 2-bit is --codec vq only (3B greedy canary failed)."
        ),
    )
    comp.add_argument("--in", dest="inp", required=True, help="HuggingFace directory")
    comp.add_argument("--out", help="output .chr (default: $DEEPFOLD_HOME/chr/<slug>)")
    comp.add_argument("--chr-bin", help="path to the Go chr binary")
    comp.add_argument(
        "--codec",
        choices=("auto", "nf4", "vq"),
        default="auto",
        help="packed format; auto = NF4 if it fits, else NF4 overflow (H2); VQ is --codec vq only",
    )
    comp.add_argument(
        "--vram-mib",
        type=int,
        default=None,
        help="card size for --codec auto (default: this GPU, else 12288)",
    )
    comp.add_argument("--force", action="store_true", help="repack over an existing .chr")
    comp.add_argument("--quiet", action="store_true", help="no chr progress output")
    comp.set_defaults(func=run_mod.compress)

    gen = sub.add_parser(
        "run",
        help="load packed NF4 or VQ weights and generate",
        description=(
            "Needs two things: a HuggingFace directory (config.json, tokenizer) "
            "and one .chr of packed weights (NF4 or VQ 2-bit). Compresses once "
            "if the .chr is missing. --codec auto (default) packs NF4 when it "
            "fits this card, else NF4 overflow (H2). VQ 2-bit is --codec vq only."
        ),
    )
    gen.add_argument("--model", help="HuggingFace directory (or $DEEPFOLD_MODEL)")
    gen.add_argument("--chr", help="packed weights (or $DEEPFOLD_CHR, or a sibling)")
    gen.add_argument(
        "--codec",
        choices=("auto", "nf4", "vq"),
        default="auto",
        help="when packing: NF4 if it fits, else NF4 overflow (H2); --codec vq is oracle-only",
    )
    gen.add_argument("--chr-bin", help="path to the Go chr binary")
    gen.add_argument("--prompt", help="one-shot prompt instead of the stdin REPL")
    gen.add_argument("--max-new-tokens", type=int, default=64)
    gen.add_argument("--max-seq", type=int, default=512, help="preallocated KV length")
    gen.add_argument(
        "--max-resident-mib",
        type=int,
        default=None,
        help=(
            "HBM cap for NF4 weights in MiB; overflow streams the rest (H2). "
            "Default: fully resident if NF4 fits, else auto from VRAM and --max-seq. "
            "Canary: fake a small cap on 3B without a 32B file. --codec vq ignores this."
        ),
    )
    gen.add_argument(
        "--raw",
        action="store_true",
        help="tokenize the prompt as-is, without the model's chat template",
    )
    gen.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="skip the warmup pass (first token then pays for kernel setup)",
    )
    gen.add_argument(
        "--no-compress",
        action="store_true",
        help="fail instead of packing when no .chr is found",
    )
    gen.add_argument("--quiet", action="store_true", help="no chr progress output")
    gen.add_argument("--debug", action="store_true", help="traceback after the report")
    gen.set_defaults(func=run_mod.run)

    ollama = sub.add_parser(
        "from-ollama",
        help="map an allowlisted Ollama tag to a HuggingFace id (never GGUF)",
        description=(
            "Allowlisted Ollama library names become HuggingFace BF16 trees. "
            "The command never reads ~/.ollama and never loads GGUF."
        ),
    )
    ollama.add_argument("tag", help="library tag, e.g. qwen2.5:3b")
    ollama.add_argument(
        "--hf",
        help="HuggingFace id; must match this tag, or be a table id if the tag is unknown",
    )
    ollama.add_argument("--dir", help="download destination (default: $DEEPFOLD_HOME/hf/<slug>)")
    ollama.add_argument(
        "--run",
        action="store_true",
        help="after resolving the tree, invoke deepfold run (compress is still first-run of run)",
    )
    ollama.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="do not ask before snapshot_download (required when stdin is not a TTY)",
    )
    ollama.set_defaults(func=from_ollama_mod.from_ollama)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "command", None):
        ap.print_help(sys.stderr)
        return 1
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

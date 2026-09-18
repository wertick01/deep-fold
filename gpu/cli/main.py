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

from . import chat as chat_mod  # noqa: E402
from . import doctor as doctor_mod  # noqa: E402
from . import from_ollama as from_ollama_mod  # noqa: E402
from . import pull as pull_mod  # noqa: E402
from . import run as run_mod  # noqa: E402
from . import selftest as selftest_mod  # noqa: E402
from . import setup_env as setup_mod  # noqa: E402

PROG = "deepfold"


def _add_runtime_flags(
    p: argparse.ArgumentParser,
    *,
    with_prompt: bool,
    max_new_tokens: int = 64,
    max_seq: int | None = 512,
) -> None:
    p.add_argument("--model", help="HuggingFace directory (or $DEEPFOLD_MODEL)")
    p.add_argument("--chr", help="packed weights (or $DEEPFOLD_CHR, or a sibling)")
    p.add_argument(
        "--codec",
        choices=("auto", "nf4", "vq"),
        default="auto",
        help="when packing: NF4 if it fits, else NF4 overflow (H2); --codec vq is oracle-only",
    )
    p.add_argument("--chr-bin", help="path to the Go chr binary")
    if with_prompt:
        p.add_argument("--prompt", help="one-shot prompt instead of the stdin REPL")
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=max_new_tokens,
        help=f"tokens to generate per turn (default: {max_new_tokens})",
    )
    p.add_argument(
        "--max-seq",
        type=int,
        default=max_seq,
        help=(
            f"preallocated KV length (default: {max_seq})"
            if max_seq is not None
            else "preallocated KV length (default: 2048; 4096 with --agent on 12 GB)"
        ),
    )
    p.add_argument(
        "--max-resident-mib",
        type=int,
        default=None,
        help=(
            "HBM cap for NF4 weights in MiB; overflow streams the rest (H2). "
            "Default: fully resident if NF4 fits, else auto from VRAM and --max-seq. "
            "Canary: fake a small cap on 3B without a 32B file. --codec vq ignores this."
        ),
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="tokenize the prompt as-is, without the model's chat template",
    )
    p.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="skip the warmup pass (first token then pays for kernel setup)",
    )
    p.add_argument(
        "--no-compress",
        action="store_true",
        help="fail instead of packing when no .chr is found",
    )
    p.add_argument("--quiet", action="store_true", help="no chr progress output")
    p.add_argument("--debug", action="store_true", help="traceback after the report")
    p.add_argument(
        "--residency",
        default="D",
        help="overflow residency policy (default D)",
    )
    p.add_argument(
        "--executor",
        choices=("auto", "tokenloop", "decodev2"),
        default="auto",
        help=(
            "decode engine. auto (default): Decode V2 on resident NF4 "
            "(3B/14B/20B), TokenLoop on overflow/VQ. tokenloop: MMA, CopyRing "
            "on 32B. decodev2: refuse unless resident NF4"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Packed CHR0 driver (NF4 or VQ 2-bit): weights stay packed in "
            "VRAM for the whole run. Ampere-family CUDA (sm_86 measured; "
            "sm_80/sm_89 experimental). SM120 (RTX 50, first remote SKU "
            "RTX 5070 Ti) is experimental. Turing / Hopper / SM100 refuse."
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
            "fits this card, else NF4 overflow (H2). VQ 2-bit is --codec vq only. "
            "TTY one-liners: prefer deepfold chat."
        ),
    )
    _add_runtime_flags(gen, with_prompt=True)
    gen.set_defaults(func=run_mod.run)

    talk = sub.add_parser(
        "chat",
        help="TTY chat session (history + streamed tokens)",
        description=(
            "Same load path as run, then a prompt_toolkit session. "
            "Enter sends, Ctrl+J newline, Ctrl+C stops a reply. "
            "Later turns prefill only the new suffix when the template prefix "
            "matches. --agent adds workspace tools and web_search (free Tavily) "
            "unless --no-agent-web. Persist with /agent default or DEEPFOLD_AGENT. "
            "Writes and tests ask first unless --agent-trust. "
            "Needs a TTY; scripts use run --prompt."
        ),
    )
    _add_runtime_flags(talk, with_prompt=False, max_new_tokens=256, max_seq=None)
    talk.add_argument(
        "--new",
        action="store_true",
        help="start a new conversation (skip the saved-chat picker)",
    )
    talk.add_argument("--session", help="resume this saved chat id from $DEEPFOLD_HOME/chats")
    talk.add_argument(
        "--agent",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable workspace tools (grep, patch, pytest, allowlisted argv). "
        "Also turns on web_search unless --no-agent-web. "
        "Default: DEEPFOLD_AGENT or $DEEPFOLD_HOME/prefs.env, else off",
    )
    talk.add_argument(
        "--workspace",
        default=None,
        help="sandbox root for --agent (default: current directory)",
    )
    talk.add_argument(
        "--agent-trust",
        choices=("ask", "write", "workspace"),
        default="ask",
        help="ask (default): confirm edits and commands; write: auto-edit; "
        "workspace: auto-edit and allowlisted commands (git writes still ask)",
    )
    talk.add_argument(
        "--max-tool-rounds",
        type=int,
        default=24,
        help="max generate+tool cycles per user turn in --agent (default: 24)",
    )
    talk.add_argument(
        "--agent-web",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="web_search on/off (free Tavily; no key). Default: on when --agent "
        "or DEEPFOLD_AGENT is on, unless DEEPFOLD_AGENT_WEB=0. "
        "See docs/web-search.md",
    )
    talk.set_defaults(func=chat_mod.chat)

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

    get = sub.add_parser(
        "pull",
        help="download an allowlisted HuggingFace BF16 tree (never GGUF)",
        description=(
            "Allowlisted Hub ids from docs/models.md. Arbitrary repos are refused. "
            "Confirm disk (--yes or a TTY). Extra: pip install \"deepfold[hub]\"."
        ),
    )
    get.add_argument("hf_id", help="HuggingFace id, e.g. Qwen/Qwen2.5-3B-Instruct")
    get.add_argument("--dir", help="download destination")
    get.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="do not ask before snapshot_download (required when stdin is not a TTY)",
    )
    get.set_defaults(func=pull_mod.pull)

    boot = sub.add_parser(
        "setup",
        help="install CUDA torch + build chr in this interpreter (not torch-gpu)",
        description=(
            "Catch-up inside an existing venv. Refuses conda env torch-gpu. "
            "A neighbor PC should run scripts/setup.ps1 or scripts/setup.sh first. "
            "If chr is missing, fetches portable Go 1.22 from go.dev and builds it. "
            "If the NF4 kernel is missing, installs VS Build Tools + CUDA 12.4 "
            "(12.8 on RTX 50) via winget when needed and compiles gpu/nf4."
        ),
    )
    boot.add_argument("--chr-bin", help="path to the Go chr binary")
    boot.add_argument(
        "--chr-only",
        action="store_true",
        help="build chr (fetch portable Go if needed); do not pip install",
    )
    boot.add_argument(
        "--kernel-only",
        action="store_true",
        help="build gpu/nf4 (install VS Build Tools + CUDA 12.4, or 12.8 on RTX 50, if needed)",
    )
    boot.add_argument(
        "--no-install-tools",
        action="store_true",
        help="do not winget-install VS Build Tools or CUDA Toolkit",
    )
    boot.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands; never pip install",
    )
    boot.set_defaults(func=setup_mod.setup, model=None, compress_only=False)

    tests = sub.add_parser(
        "test",
        help="run CLI acceptance (no Hub). --live skips unless 3B is on disk",
    )
    tests.add_argument(
        "--live",
        action="store_true",
        help="check doctor + 3B tree; does not generate and does not download",
    )
    tests.add_argument("--chr-bin", help="path to the Go chr binary")
    tests.set_defaults(func=selftest_mod.selftest)

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

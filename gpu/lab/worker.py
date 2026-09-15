"""Run one codec session in a fresh process so VRAM actually returns.

The notebook kernel must not hold the model. A child loads it, records the
session, exits; Windows then drops that process's CUDA allocations and
nvidia-smi comes back before the other codec starts.

    python -m gpu.lab.worker --codec bf16 --out DIR --model-dir ...
    python -m gpu.lab.worker --codec nf4 --out DIR --model-dir ... --chr ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab.script import (  # noqa: E402
    CHR_PATH,
    MAX_NEW_TOKENS,
    MAX_SEQ,
    MODEL_DIR,
    POLL_INTERVAL_S,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gpu.lab.worker")
    parser.add_argument("--codec", choices=("bf16", "nf4"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--chr", dest="chr_path", default=CHR_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-seq", type=int, default=MAX_SEQ)
    parser.add_argument("--interval", type=float, default=POLL_INTERVAL_S)
    parser.add_argument("--no-graphs", dest="graphs", action="store_false")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--items-json",
        default="",
        help="hard-eval (or custom) JSON script; default is the smoke MESSAGES",
    )
    parser.add_argument(
        "--conversation",
        choices=("independent", "history"),
        default="independent",
        help="independent (KV reset, smoke) or history (growing prefill)",
    )
    parser.add_argument(
        "--plate",
        choices=("hard", "eval"),
        default="hard",
        help=(
            "which scorer owns --items-json: the hard plate (gsm8k / yesno / needle) "
            "or the eval stub (mcq / truthful / ppl). The child must see this so "
            "kind=ppl runs teacher-forced NLL instead of generate."
        ),
    )
    parser.add_argument(
        "--residency",
        default="D",
        help="NF4 overflow residency policy (default D)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 -- a closed pipe still has to load the model
        pass
    args = _parser().parse_args(argv)
    from gpu.lab.sessions import run_bf16, run_nf4

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra: dict = {}
    if args.items_json:
        extra["items_json"] = args.items_json
        extra["plate"] = args.plate
        if args.plate == "eval":
            from gpu.lab.eval import load_eval_script

            eval_script = load_eval_script(args.items_json)
            extra["messages"] = eval_script.prompts
            extra["quality"] = eval_script.quality_fn
            extra["conversation"] = "independent"
        else:
            from gpu.lab.hard import load_script

            script = load_script(args.items_json, history=(args.conversation == "history"))
            extra["messages"] = script.prompts
            extra["quality"] = script.quality_fn
            extra["conversation"] = script.conversation
    common = dict(
        out_dir=out_dir,
        model_dir=args.model_dir,
        max_new_tokens=args.max_new_tokens,
        max_seq=args.max_seq,
        interval_s=args.interval,
        verbose=not args.quiet,
        trust_remote_code=args.trust_remote_code,
        isolated=False,
        **extra,
    )
    if args.codec == "bf16":
        run_bf16(**common)
    else:
        run_nf4(
            chr_path=args.chr_path,
            graphs=args.graphs,
            residency_policy=getattr(args, "residency", "D"),
            **common,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

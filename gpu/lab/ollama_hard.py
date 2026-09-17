"""Ollama independent hard-12. HTTP only; no torch, no TokenLoop.

Default is the 3B library tag. 32B hard-12 is a long GPU job; pass
``--model qwen2.5:32b`` only when the card is free and you mean it.

    python -m gpu.lab.ollama_hard --plan-only
    python -m gpu.lab.ollama_hard --model qwen2.5:3b
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab.hard import HARD_MAX_NEW_TOKENS, HARD_MAX_SEQ, load_fixture, score_item
from gpu.lab.ollama_h2 import _base, _chat, _reply_text, _tags, _timings
from gpu.lab.script import RUNS_DIR

__all__ = ["main"]

SCHEMA = "deepfold.ollama_hard.v1"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gpu.lab.ollama_hard")
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11434)
    p.add_argument("--out", default="")
    p.add_argument("--ctx", type=int, default=HARD_MAX_SEQ)
    p.add_argument("--n-predict", type=int, default=HARD_MAX_NEW_TOKENS)
    p.add_argument("--limit", type=int, default=0, help="first N independent items; 0 = all")
    p.add_argument("--plan-only", action="store_true")
    return p


def _out_dir(explicit: str) -> Path:
    if explicit:
        dest = Path(explicit)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = Path(RUNS_DIR) / f"ollama-hard-{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def run_live(args: argparse.Namespace, dest: Path) -> dict[str, Any]:
    base = _base(args)
    tags = _tags(base)
    items = list(load_fixture()["independent"])
    if int(args.limit) > 0:
        items = items[: int(args.limit)]
    if args.model not in tags and not any(t.startswith(args.model) for t in tags):
        raise FileNotFoundError(f"ollama model {args.model!r} not in {tags}")
    plate: dict[str, Any] = {
        "schema": SCHEMA,
        "plan_only": False,
        "model": args.model,
        "ctx": int(args.ctx),
        "n_predict": int(args.n_predict),
        "tags": tags,
        "items": [],
    }
    scored = 0
    correct = 0
    pending = 0
    for item in items:
        payload = _chat(base, args.model, item.prompt, int(args.n_predict), int(args.ctx))
        text = _reply_text(payload)
        times = _timings(payload)
        score = score_item(item, text)
        row = {
            "id": item.id,
            "kind": item.kind,
            "gold": item.gold,
            "extracted": score.extracted,
            "correct": score.correct,
            "pending_human": score.pending_human,
            "notes": score.notes,
            "reply": text,
            **times,
        }
        (dest / f"{item.id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        plate["items"].append(row)
        if not score.pending_human:
            scored += 1
            if score.correct:
                correct += 1
        else:
            pending += 1
        print(
            f"{item.id} correct={score.correct} pending={score.pending_human} "
            f"extracted={score.extracted!r} tok/s={times['decode_tok_s']:.3f}",
            flush=True,
        )
    plate["n_items"] = len(items)
    plate["n_scored"] = scored
    plate["n_correct"] = correct
    plate["n_pending_human"] = pending
    plate["accuracy"] = (correct / scored) if scored else None
    return plate


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dest = _out_dir(args.out)
    items = load_fixture()["independent"]
    if args.plan_only:
        plate = {
            "schema": SCHEMA,
            "plan_only": True,
            "model": args.model,
            "n_items": len(items),
            "item_ids": [item.id for item in items],
            "ctx": int(args.ctx),
            "n_predict": int(args.n_predict),
        }
        print(f"plan-only out={dest} n_items={len(items)}", flush=True)
    else:
        plate = run_live(args, dest)
        print(
            f"accuracy={plate.get('accuracy')} "
            f"{plate.get('n_correct')}/{plate.get('n_scored')} "
            f"pending={plate.get('n_pending_human')}",
            flush=True,
        )
    (dest / "plate.json").write_text(
        json.dumps(plate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (dest / "SUMMARY.txt").write_text(
        "\n".join(
            [
                f"schema {SCHEMA}",
                f"model {args.model}",
                f"n_items {plate.get('n_items')}",
                f"n_correct {plate.get('n_correct')}",
                f"n_scored {plate.get('n_scored')}",
                f"accuracy {plate.get('accuracy')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

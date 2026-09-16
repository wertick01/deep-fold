"""Ollama plate on the same Instruct smoke as H2 / llama.cpp.

Does not import torch. Does not pip into conda ``torch-gpu``.
Ollama must already be installed and the model pulled.

    python -m gpu.lab.ollama_h2 --plan-only
    python -m gpu.lab.ollama_h2 --model qwen2.5:32b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab.compare import upsert
from gpu.lab.script import MESSAGES, NEEDLES, RUNS_DIR, quality_ok

__all__ = ["main"]

SCHEMA = "deepfold.ollama_h2.v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
LONG_PROMPT = (
    "Write a long travelogue about rivers, forests, and cities. "
    "Keep going with more sentences."
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gpu.lab.ollama_h2")
    p.add_argument("--model", default="qwen2.5:32b")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--out", default="")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--n-predict", type=int, default=64)
    p.add_argument("--long-n", type=int, default=64)
    p.add_argument("--size", default="", help="3B or 32B label for compare.json")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    return p


def _smi_used_mib() -> float | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    try:
        return float(line.split(",")[0].strip())
    except ValueError:
        return None


def _out_dir(explicit: str) -> Path:
    if explicit:
        dest = Path(explicit)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = Path(RUNS_DIR) / f"ollama-h2-{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _base(args: argparse.Namespace) -> str:
    return f"http://{args.host}:{int(args.port)}"


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 900.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _tags(base: str) -> list[str]:
    try:
        payload = _http_json(f"{base}/api/tags", timeout=8)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []
    names = []
    for row in payload.get("models") or []:
        name = str(row.get("name") or "")
        if name:
            names.append(name)
    return names


def _chat(base: str, model: str, prompt: str, n_predict: int, ctx: int) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 1,
            "num_ctx": int(ctx),
            "num_predict": int(n_predict),
        },
    }
    t0 = time.perf_counter()
    payload = _http_json(f"{base}/api/chat", body, timeout=1200)
    payload["_wall_ms"] = (time.perf_counter() - t0) * 1000.0
    return payload


def _reply_text(payload: dict[str, Any]) -> str:
    msg = payload.get("message") or {}
    return str(msg.get("content") or "")


def _timings(payload: dict[str, Any]) -> dict[str, Any]:
    prompt_n = int(payload.get("prompt_eval_count") or 0)
    prompt_ns = float(payload.get("prompt_eval_duration") or 0.0)
    pred_n = int(payload.get("eval_count") or 0)
    pred_ns = float(payload.get("eval_duration") or 0.0)
    prompt_ms = prompt_ns / 1e6
    pred_ms = pred_ns / 1e6
    tok_s = (pred_n / (pred_ns / 1e9)) if pred_ns > 0 and pred_n > 0 else 0.0
    return {
        "prompt_n": prompt_n,
        "prompt_ms": prompt_ms,
        "predicted_n": pred_n,
        "predicted_ms": pred_ms,
        "decode_tok_s": tok_s,
        "wall_ms": float(payload.get("_wall_ms") or 0.0),
    }


def _size_label(args: argparse.Namespace) -> str:
    if args.size:
        return str(args.size)
    name = str(args.model).lower()
    if "32b" in name:
        return "32B"
    if "3b" in name:
        return "3B"
    return "unknown"


def run_live(args: argparse.Namespace, dest: Path) -> dict[str, Any]:
    base = _base(args)
    tags = _tags(base)
    plate: dict[str, Any] = {
        "schema": SCHEMA,
        "plan_only": False,
        "model": args.model,
        "ctx": int(args.ctx),
        "n_predict": int(args.n_predict),
        "tags": tags,
        "smi_before_mib": _smi_used_mib(),
        "messages": [],
    }
    if args.model not in tags and not any(t.startswith(args.model) for t in tags):
        raise FileNotFoundError(f"ollama model {args.model!r} not in {tags}")
    if args.warmup:
        warm = _chat(base, args.model, "Hello.", 8, int(args.ctx))
        plate["warmup"] = {"reply": _reply_text(warm), **_timings(warm)}
        plate["smi_after_load_mib"] = _smi_used_mib()
        print(
            f"warmup tok/s={plate['warmup']['decode_tok_s']:.3f} "
            f"smi={plate['smi_after_load_mib']}",
            flush=True,
        )
    tok_s: list[float] = []
    ttft: list[float] = []
    smoke: list[bool] = []
    for index, prompt in enumerate(MESSAGES, start=1):
        payload = _chat(base, args.model, prompt, int(args.n_predict), int(args.ctx))
        text = _reply_text(payload)
        times = _timings(payload)
        needle = quality_ok(index, text)
        row = {
            "id": index,
            "prompt": prompt,
            "reply": text,
            "needles": NEEDLES.get(index),
            "quality_ok": needle,
            **times,
        }
        (dest / f"msg-{index}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        plate["messages"].append(row)
        if times["decode_tok_s"] > 0:
            tok_s.append(times["decode_tok_s"])
        if times["prompt_ms"] > 0:
            ttft.append(times["prompt_ms"])
        smoke.append(bool(needle))
        print(
            f"msg {index} tok/s={times['decode_tok_s']:.3f} "
            f"ttft_ms={times['prompt_ms']:.0f} smoke={needle} "
            f"reply={text[:80]!r}",
            flush=True,
        )
    if int(args.long_n) > 0:
        payload = _chat(base, args.model, LONG_PROMPT, int(args.long_n), int(args.ctx))
        times = _timings(payload)
        plate["long"] = {
            "prompt": LONG_PROMPT,
            "reply": _reply_text(payload),
            **times,
        }
        (dest / "long.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"long tok/s={times['decode_tok_s']:.3f} n={times['predicted_n']}",
            flush=True,
        )
    plate["mean_decode_tok_s"] = sum(tok_s) / len(tok_s) if tok_s else None
    plate["mean_ttft_ms"] = sum(ttft) / len(ttft) if ttft else None
    plate["smoke_ok"] = all(smoke) if smoke else False
    plate["smi_after_generate_mib"] = _smi_used_mib()
    return plate


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dest = _out_dir(args.out)
    base = _base(args)
    if args.plan_only:
        tags = _tags(base)
        plate = {
            "schema": SCHEMA,
            "plan_only": True,
            "model": args.model,
            "server_ok": bool(tags) or True,
            "tags": tags,
            "ctx": int(args.ctx),
        }
        print(f"plan-only out={dest} tags={tags}", flush=True)
    else:
        plate = run_live(args, dest)
        long = plate.get("long") or {}
        print(
            f"mean tok/s={plate.get('mean_decode_tok_s')} "
            f"long={long.get('decode_tok_s')} smoke={plate.get('smoke_ok')}",
            flush=True,
        )
        size = _size_label(args)
        upsert(
            {
                "id": f"ollama-{size}-{args.model.replace(':', '-')}",
                "stack": "ollama",
                "engine": "Ollama 0.34.0",
                "model": "Qwen2.5-Instruct" if "coder" not in str(args.model) else str(args.model),
                "size": size,
                "quant": str(args.model),
                "mean_decode_tok_s": plate.get("mean_decode_tok_s"),
                "long_decode_tok_s": long.get("decode_tok_s"),
                "mean_ttft_ms": plate.get("mean_ttft_ms"),
                "smi_after_load_mib": plate.get("smi_after_load_mib")
                or plate.get("smi_after_generate_mib"),
                "smoke_ok": plate.get("smoke_ok"),
                "source": str(dest),
            }
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
                f"mean_decode_tok_s {plate.get('mean_decode_tok_s')}",
                f"long_decode_tok_s {(plate.get('long') or {}).get('decode_tok_s')}",
                f"mean_ttft_ms {plate.get('mean_ttft_ms')}",
                f"smoke_ok {plate.get('smoke_ok')}",
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

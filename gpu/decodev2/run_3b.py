"""Qwen2.5-3B on Decode V2: greedy ids vs TokenLoop, then ignore-EOS plateau.

    python gpu/decodev2/run_3b.py
    python gpu/decodev2/run_3b.py --max-seq 512 --long-n 64 --backend mma
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.decodev2.graph import capture_greedy  # noqa: E402
from gpu.decodev2.linear import set_linear_backend  # noqa: E402
from gpu.decodev2.load import load_chr  # noqa: E402
from gpu.decodev2.runner import consume_prompt  # noqa: E402
from gpu.decodev2.step import greedy_decode  # noqa: E402
from gpu.lab.catalog import lab_by_slug  # noqa: E402
from gpu.lab.script import LONG_PROMPT, chat_text  # noqa: E402
from gpu.lab.sessions import _load_tokenizer  # noqa: E402
from gpu.loop import TokenLoop  # noqa: E402
from gpu.loop.generate import generated_tok_s  # noqa: E402
from gpu.tests.skips import cuda_reason  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python gpu/decodev2/run_3b.py")
    p.add_argument("--max-seq", type=int, default=512)
    p.add_argument("--match-n", type=int, default=8)
    p.add_argument("--long-n", type=int, default=64)
    p.add_argument("--backend", choices=("mma", "gemv"), default="mma")
    p.add_argument("--out", default="")
    p.add_argument("--no-long", action="store_true")
    p.add_argument("--no-graph", action="store_true")
    return p


def _prompt_ids(tokenizer, text: str) -> list[int]:
    packed = chat_text(tokenizer, text)
    ids = tokenizer(packed, add_special_tokens=False).input_ids
    return [int(x) for x in ids]


def _tokenloop_n1(loop: TokenLoop, prompt: list[int], n_new: int) -> list[int]:
    """Sequential N=1 prefill + greedy, same accounting as Decode V2 runner."""
    loop.reset()
    logits = None
    for i, tok in enumerate(prompt):
        logits = loop.forward(torch.tensor([tok], device=loop.device), i)
    assert logits is not None
    out = [int(logits.argmax())]
    token = out[0]
    for _ in range(max(0, n_new - 1)):
        logits = loop.step(token)
        token = int(logits.argmax())
        out.append(token)
    return out


def _decodev2_n1(state, weights, prompt: list[int], n_new: int) -> list[int]:
    state.reset()
    consume_prompt(state, weights, prompt)
    out = [int(state.next_token.item())]
    state.token.copy_(state.next_token)
    for _ in range(max(0, n_new - 1)):
        greedy_decode(state, weights)
        out.append(int(state.token.item()))
    return out


def _time_decodev2(
    state,
    weights,
    prompt: list[int],
    n_new: int,
    *,
    step,
    host_read: bool,
) -> dict:
    n_new = int(n_new)
    state.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    consume_prompt(state, weights, prompt)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000.0
    tokens = [int(state.next_token.item())]
    state.token.copy_(state.next_token)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(max(0, n_new - 1)):
        step(state, weights)
        if host_read:
            tokens.append(int(state.token.item()))
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - t1) * 1000.0
    steps = max(0, n_new - 1)
    if not host_read:
        # One host copy after the window. token is the last id.
        last = int(state.token.item())
        tokens = tokens[:1] + [last] * steps
    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "decode_steps": steps,
        "n_tokens": n_new,
        "decode_tok_s": steps / (decode_ms / 1000.0) if decode_ms > 0 else 0.0,
        "eval_tok_s": generated_tok_s(n_new, decode_ms),
        "host_read": host_read,
        "last_token": int(state.token.item()),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reason = cuda_reason()
    if reason is not None:
        print(f"SKIP {reason}")
        return 0
    lab = lab_by_slug("qwen25-3b")
    torch.cuda.empty_cache()
    set_linear_backend(args.backend)
    print(
        f"load {lab.chr_path} max_seq={args.max_seq} backend={args.backend}",
        flush=True,
    )
    loaded = load_chr(lab.model_dir, lab.chr_path, max_seq=args.max_seq)
    print(f"loaded {loaded.report}", flush=True)
    tokenizer = _load_tokenizer(lab.model_dir, lab.trust_remote_code)
    prompt = _prompt_ids(tokenizer, LONG_PROMPT)
    if len(prompt) + args.long_n > args.max_seq:
        raise SystemExit(
            f"prompt {len(prompt)} + long_n {args.long_n} exceeds max_seq {args.max_seq}"
        )
    loop = TokenLoop(loaded.model, max_seq=args.max_seq, overlap=True)
    plate: dict = {
        "schema": "decodev2.3b.v1",
        "backend": args.backend,
        "max_seq": args.max_seq,
        "prompt_len": len(prompt),
        "report": str(loaded.report),
        "match_n": args.match_n,
    }
    if not args.no_long:
        warm = loop.warmup(prompt=8, tokens=8)
        graph_mode = "off" if args.no_graph else loop.capture_graphs()
        ids = torch.tensor(prompt, dtype=torch.long, device=loop.device)
        run = loop.generate(ids, args.long_n, stop=())
        plate["tokenloop"] = {
            "warmup_ms": warm,
            "graph": graph_mode,
            "n_tokens": len(run.tokens),
            "decode_steps": run.decode_steps,
            "prefill_ms": run.prefill_ms,
            "decode_ms": run.decode_ms,
            "decode_tok_s": run.decode_tok_s,
            "eval_tok_s": run.eval_tok_s,
        }
        print(
            f"tokenloop long tok/s={run.decode_tok_s:.3f} eval={run.eval_tok_s:.3f} "
            f"steps={run.decode_steps} prefill_ms={run.prefill_ms:.0f} graph={graph_mode}",
            flush=True,
        )
        try:
            loop.drop_graphs()
        except Exception:
            pass
    print(f"match sequential N=1 n={args.match_n} prompt_len={len(prompt)}", flush=True)
    want = _tokenloop_n1(loop, prompt, args.match_n)
    got = _decodev2_n1(loaded.state, loaded.weights, prompt, args.match_n)
    match = got == want
    print(f"tokenloop {want}", flush=True)
    print(f"decodev2  {got}", flush=True)
    print(f"greedy_match={match}", flush=True)
    plate["greedy_match"] = match
    plate["tokenloop_ids"] = want
    plate["decodev2_ids"] = got
    if not args.no_long:
        loaded.state.reset()
        consume_prompt(loaded.state, loaded.weights, prompt)
        loaded.state.token.copy_(loaded.state.next_token)
        captured = None
        if not args.no_graph:
            try:
                loop.drop_graphs()
            except Exception:
                pass
            try:
                captured = capture_greedy(loaded.state, loaded.weights, warmup=2)
            except Exception as exc:  # noqa: BLE001
                print(f"decodev2 graph capture failed: {type(exc).__name__}: {exc}", flush=True)
                captured = None
        step = captured if captured is not None else greedy_decode
        host = _time_decodev2(
            loaded.state,
            loaded.weights,
            prompt,
            args.long_n,
            step=step,
            host_read=True,
        )
        device = _time_decodev2(
            loaded.state,
            loaded.weights,
            prompt,
            args.long_n,
            step=step,
            host_read=False,
        )
        plate["decodev2_host"] = host
        plate["decodev2_device"] = device
        plate["decodev2_graph"] = captured is not None
        print(
            f"decodev2 host tok/s={host['decode_tok_s']:.3f} eval={host['eval_tok_s']:.3f} "
            f"prefill_ms={host['prefill_ms']:.0f} graph={captured is not None}",
            flush=True,
        )
        print(
            f"decodev2 device-window tok/s={device['decode_tok_s']:.3f} "
            f"(no per-token .item())",
            flush=True,
        )
    dest = Path(args.out) if args.out else (
        Path(r"C:\dev\models\runs") / f"decodev2-3b-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "plate.json").write_text(
        json.dumps(plate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {dest / 'plate.json'}", flush=True)
    set_linear_backend("mma")
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())

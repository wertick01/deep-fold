"""Lab: T_verify-bound tok/s when the draft is the target's own greedy.

    python -m gpu.lab.h2_draft --max-seq 512 --max-new-tokens 32 --speculate 8

Runs greedy ``step()``, then ``draft="oracle"`` with teacher = prompt + greedy.
Does not change product defaults. No second GPU model. Writes
``$DEEPFOLD_RUNS/h2-draft-<stamp>/``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab.h2_metrics import CHR_32B, MODEL_32B, dump_json
from gpu.lab.script import MESSAGES, RUNS_DIR, chat_text, stop_token_ids
from gpu.nf4.plan import LIVE_MAX_N

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gpu.lab.h2_draft")
    p.add_argument("--model", default=MODEL_32B)
    p.add_argument("--chr", dest="chr_path", default=CHR_32B)
    p.add_argument("--out", default="")
    p.add_argument("--max-seq", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--speculate", type=int, default=8)
    p.add_argument(
        "--cpu-draft",
        default="",
        help="local HF dir for CPU draft (Qwen2.5-3B). Empty = oracle only",
    )
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--no-graphs", dest="graphs", action="store_false")
    return p


def _warmup_nk(loop, k: int) -> None:
    """Pay the first eager N=k tape so oracle generate is not the JIT wall."""
    import torch

    from gpu.loop.speculate import verify_block

    k = int(k)
    loop.reset()
    ids = torch.arange(1, 9, device=loop.device, dtype=torch.long)
    loop.prefill(ids)
    block = torch.arange(1, k + 1, device=loop.device, dtype=torch.long)
    verify_block(loop, block, int(loop.kv.seq_len))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    loop.reset()


def _out_dir(explicit: str) -> Path:
    if explicit:
        dest = Path(explicit)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%HHmmss")
        dest = Path(RUNS_DIR) / f"h2-draft-{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _row(gen, *, tokens: list[int] | None = None) -> dict[str, Any]:
    ids = list(gen.tokens) if tokens is None else list(tokens)
    n = len(ids)
    verifies = int(getattr(gen, "spec_verifies", 0) or 0)
    accepted = int(getattr(gen, "spec_draft_accepted", 0) or 0)
    return {
        "n_tokens": n,
        "decode_steps": int(gen.decode_steps),
        "prefill_ms": float(gen.prefill_ms),
        "decode_ms": float(gen.decode_ms),
        "decode_tok_s": float(gen.decode_tok_s),
        "h2d_copies": int(gen.h2d_copies),
        "h2d_bytes": int(gen.h2d_bytes),
        "h2d_forwards": int(gen.h2d_forwards),
        "spec_verifies": verifies,
        "spec_skips": int(getattr(gen, "spec_skips", 0) or 0),
        "spec_draft_accepted": accepted,
        "spec_draft_ms": float(getattr(gen, "spec_draft_ms", 0.0) or 0.0),
        "tokens_per_verify": (accepted / verifies) if verifies else None,
        "stop_token": gen.stop_token,
    }


def run_live(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    from gpu.host import load_model
    from gpu.lab.h2_metrics import auto_max_resident_bytes
    from gpu.loop import TokenLoop
    from transformers import AutoTokenizer

    k = int(args.speculate)
    if k < 2 or k > int(LIVE_MAX_N):
        raise ValueError(f"--speculate {k} not in 2..{LIVE_MAX_N}")
    cap, cap_info = auto_max_resident_bytes(args.model, args.chr_path, args.max_seq)
    if cap is None:
        raise RuntimeError(f"expected overflow cap: {cap_info}")
    print(f"load cap_mib={cap / (1024**2):.1f} {cap_info.get('reason', '')[:80]}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    model, report = load_model(
        args.model,
        args.chr_path,
        strict=True,
        max_resident_bytes=int(cap),
        residency_policy="D",
    )
    loop = TokenLoop(model, max_seq=int(args.max_seq), overlap=True)
    print(
        f"loaded streamed={report.streamed} host_tape={len(loop._host_tape)} "
        f"prefill_chunk={loop.prefill_chunk}",
        flush=True,
    )
    loop.warmup(prompt=8, tokens=8)
    if args.graphs:
        loop.capture_graphs()
        print(f"graphs={loop.graph_mode}", flush=True)

    packed = chat_text(tokenizer, MESSAGES[0])
    prompt_ids = tokenizer(packed, return_tensors="pt", add_special_tokens=False).input_ids[0]
    stop = stop_token_ids(tokenizer)
    n_new = int(args.max_new_tokens)
    print(f"warmup N={k} verify (discard first eager k-forward)...", flush=True)
    _warmup_nk(loop, k)
    print("discard cold greedy...", flush=True)
    loop.generate(prompt_ids, min(8, n_new), stop=stop)
    print("generate greedy...", flush=True)
    greedy_run = loop.generate(prompt_ids, n_new, stop=stop)
    greedy = list(greedy_run.tokens)
    print(
        f"greedy n={len(greedy)} tok/s={greedy_run.decode_tok_s:.3f} "
        f"prefill_ms={greedy_run.prefill_ms:.0f} copies={greedy_run.h2d_copies}",
        flush=True,
    )

    teacher = prompt_ids.detach().cpu().tolist() + greedy
    print(f"generate oracle k={k}...", flush=True)
    oracle_run = loop.generate(
        prompt_ids,
        n_new,
        stop=stop,
        speculate=k,
        draft="oracle",
        oracle_ids=teacher,
    )
    match = list(oracle_run.tokens) == greedy
    print(
        f"oracle n={len(oracle_run.tokens)} tok/s={oracle_run.decode_tok_s:.3f} "
        f"verifies={oracle_run.spec_verifies} skips={oracle_run.spec_skips} "
        f"accepted={oracle_run.spec_draft_accepted} match={match} "
        f"copies={oracle_run.h2d_copies}",
        flush=True,
    )

    cpu_row: dict[str, Any] | None = None
    cpu_match = None
    if args.cpu_draft:
        from gpu.loop.draft_cpu import CpuHfDraft

        print(f"load CPU draft {args.cpu_draft}...", flush=True)
        drafter = CpuHfDraft(args.cpu_draft)
        print(
            f"CPU draft loaded propose warmup...",
            flush=True,
        )
        drafter.reset()
        _ = drafter(prompt_ids.detach().cpu().tolist(), k)
        drafter.reset()
        print(f"generate cpu k={k}...", flush=True)
        cpu_run = loop.generate(
            prompt_ids,
            n_new,
            stop=stop,
            speculate=k,
            draft="cpu",
            drafter=drafter,
        )
        cpu_match = list(cpu_run.tokens) == greedy
        cpu_row = _row(cpu_run)
        cpu_row["drafter_ms"] = float(drafter.propose_ms)
        cpu_row["drafter_calls"] = int(drafter.propose_calls)
        print(
            f"cpu n={len(cpu_run.tokens)} tok/s={cpu_run.decode_tok_s:.3f} "
            f"verifies={cpu_run.spec_verifies} skips={cpu_run.spec_skips} "
            f"accepted={cpu_run.spec_draft_accepted} draft_ms={cpu_run.spec_draft_ms:.0f} "
            f"match={cpu_match} copies={cpu_run.h2d_copies}",
            flush=True,
        )

    payload = {
        "schema": "deepfold.h2_draft.v1",
        "model": args.model,
        "chr": args.chr_path,
        "max_seq": int(args.max_seq),
        "max_new_tokens": n_new,
        "speculate": k,
        "cap": cap_info,
        "n_host": int(report.streamed),
        "graph_mode": loop.graph_mode,
        "greedy": _row(greedy_run),
        "oracle": _row(oracle_run),
        "cpu": cpu_row,
        "greedy_match": match,
        "cpu_match": cpu_match,
        "cpu_draft": args.cpu_draft or None,
    }
    dump_json(out / "draft.json", payload)
    g, o = payload["greedy"], payload["oracle"]
    lines = [
        f"schema deepfold.h2_draft.v1  graphs={loop.graph_mode} n_host={report.streamed}",
        f"greedy  tok/s={g['decode_tok_s']:.3f} n={g['n_tokens']} "
        f"copies={g['h2d_copies']} fwds={g['h2d_forwards']}",
        f"oracle  tok/s={o['decode_tok_s']:.3f} n={o['n_tokens']} k={k} "
        f"verifies={o['spec_verifies']} skips={o['spec_skips']} "
        f"accepted={o['spec_draft_accepted']} tok/verify={o['tokens_per_verify']} "
        f"copies={o['h2d_copies']} fwds={o['h2d_forwards']} match={match}",
    ]
    if cpu_row is not None:
        lines.append(
            f"cpu     tok/s={cpu_row['decode_tok_s']:.3f} n={cpu_row['n_tokens']} k={k} "
            f"verifies={cpu_row['spec_verifies']} skips={cpu_row['spec_skips']} "
            f"accepted={cpu_row['spec_draft_accepted']} tok/verify={cpu_row['tokens_per_verify']} "
            f"draft_ms={cpu_row.get('spec_draft_ms')} match={cpu_match} "
            f"copies={cpu_row['h2d_copies']} fwds={cpu_row['h2d_forwards']}"
        )
    (out / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = _parser().parse_args(argv)
    out = _out_dir(args.out)
    if args.plan_only:
        print(f"plan-only out={out} (no CUDA)", flush=True)
        dump_json(
            out / "draft.json",
            {
                "schema": "deepfold.h2_draft.v1",
                "plan_only": True,
                "speculate": int(args.speculate),
            },
        )
        return 0
    run_live(args, out)
    print(f"artifacts: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

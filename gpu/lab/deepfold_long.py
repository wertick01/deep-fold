"""Ignore-EOS 64-token plateau on our NF4 TokenLoop.

Same travelogue as ``llamacpp_h2`` / ``ollama_h2``. Does not rerun smoke.
Merges ``long_decode_tok_s`` into the existing compare.json row.

    python -m gpu.lab.deepfold_long --size 3B
    python -m gpu.lab.deepfold_long --size 32B
    python -m gpu.lab.deepfold_long --size 3B --plan-only
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

from gpu.lab.compare import load, upsert
from gpu.lab.h2_metrics import CHR_32B, MODEL_32B
from gpu.lab.script import CHR_PATH, LONG_PROMPT, MODEL_DIR, RUNS_DIR, chat_text

__all__ = ["PRESETS", "main"]

SCHEMA = "deepfold.long.v1"
PRESETS = {
    "3B": {
        "model": MODEL_DIR,
        "chr": CHR_PATH,
        "compare_id": "deepfold-nf4-3B-resident",
    },
    "32B": {
        "model": MODEL_32B,
        "chr": CHR_32B,
        "compare_id": "deepfold-nf4-32B-overflow",
    },
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gpu.lab.deepfold_long")
    p.add_argument("--size", choices=sorted(PRESETS), default="32B")
    p.add_argument("--model", default="")
    p.add_argument("--chr", dest="chr_path", default="")
    p.add_argument("--out", default="")
    p.add_argument("--max-seq", type=int, default=2048)
    p.add_argument("--long-n", type=int, default=64)
    p.add_argument("--compare-id", default="")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--no-graphs", dest="graphs", action="store_false")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--compute",
        choices=("gpu", "cpu-suffix", "hybrid"),
        default="gpu",
        help="Overflow compute policy (default gpu = CopyRing). Not a tok/s claim.",
    )
    p.add_argument(
        "--gpu-layers",
        type=int,
        default=None,
        help="Repeating GPU layers; requires --compute cpu-suffix|hybrid.",
    )
    return p


def _out_dir(explicit: str, size: str) -> Path:
    if explicit:
        dest = Path(explicit)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = Path(RUNS_DIR) / f"deepfold-long-{size}-{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _merge_long(
    compare_id: str,
    *,
    tok_s: float,
    source: str,
    predicted_n: int,
    decode_steps: int,
) -> None:
    payload = load()
    old = next((row for row in (payload.get("rows") or []) if row.get("id") == compare_id), None)
    if old is None:
        raise KeyError(f"compare.json has no row {compare_id}")
    merged = dict(old)
    merged["long_decode_tok_s"] = float(tok_s)
    merged["source"] = source
    extra = (
        f"Long ignore-EOS {predicted_n} tokens / {decode_steps} decode steps. "
        f"Live {source}."
    )
    notes = str(merged.get("notes") or "")
    if "Long ignore_eos plateau not recorded" in notes:
        notes = notes.replace(
            "Long ignore_eos plateau not recorded.",
            extra,
        ).strip()
    elif extra not in notes:
        notes = f"{notes} {extra}".strip() if notes else extra
    merged["notes"] = notes
    upsert(merged)


def run_live(args: argparse.Namespace, dest: Path, preset: dict[str, str]) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from gpu.host import load_model
    from gpu.lab.h2_metrics import auto_max_resident_bytes
    from gpu.lab.sampler import smi_used_mib
    from gpu.loop import TokenLoop

    from gpu.cli.codec import load_config
    from gpu.host.compute import format_compute_stderr, plan_compute
    from gpu.host.residency import descs_from_header, descs_from_qwen

    model_dir = str(args.model or preset["model"])
    chr_path = str(args.chr_path or preset["chr"])
    cfg = load_config(model_dir)
    try:
        from gpu.chr0 import load_header

        descs = descs_from_header(load_header(chr_path))
    except Exception:
        descs = descs_from_qwen(cfg)
    compute_plan = plan_compute(
        descs,
        cfg,
        compute=getattr(args, "compute", "gpu") or "gpu",
        gpu_layers=getattr(args, "gpu_layers", None),
        max_seq=int(args.max_seq),
    )
    cap, cap_info = auto_max_resident_bytes(model_dir, chr_path, int(args.max_seq))
    if compute_plan.n_cpu > 0:
        cap = None
        cap_info = {"skipped": "cpu suffix, no CopyRing"}
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=args.trust_remote_code
    )
    print(format_compute_stderr(compute_plan), flush=True)
    model, report = load_model(
        model_dir,
        chr_path,
        trust_remote_code=args.trust_remote_code,
        strict=True,
        max_resident_bytes=cap,
        residency_policy="D",
        compute_plan=compute_plan,
    )
    torch.cuda.synchronize()
    smi_load = smi_used_mib()
    loop = TokenLoop(
        model, max_seq=int(args.max_seq), norm="exact", overlap=True, ring_timing=False
    )
    warm_ms = loop.warmup(prompt=8, tokens=8)
    graph_mode = loop.capture_graphs() if args.graphs else "off"
    packed = chat_text(tokenizer, LONG_PROMPT)
    ids = tokenizer(packed, return_tensors="pt", add_special_tokens=False).input_ids[0]
    # Empty stop set = ignore EOS, same as llama.cpp ignore_eos.
    run = loop.generate(ids, int(args.long_n), stop=())
    text = tokenizer.decode(run.tokens, skip_special_tokens=True)
    plate = {
        "schema": SCHEMA,
        "size": args.size,
        "model": model_dir,
        "chr": chr_path,
        "overflow": bool(report.overflow),
        "device_mib": report.device_mib,
        "cap": cap_info,
        "smi_after_load_mib": smi_load,
        "warmup_ms": warm_ms,
        "graph": graph_mode,
        "prompt": LONG_PROMPT,
        "reply": text,
        "n_tokens": len(run.tokens),
        "decode_steps": run.decode_steps,
        "decode_ms": run.decode_ms,
        "prefill_ms": run.prefill_ms,
        "decode_tok_s": run.decode_tok_s,
        "stop_token": run.stop_token,
        "report": str(report),
        "compute": compute_plan.compute,
        "n_gpu": compute_plan.n_gpu,
        "n_cpu": compute_plan.n_cpu,
        "ring": compute_plan.ring,
    }
    print(
        f"long tok/s={run.decode_tok_s:.3f} n={len(run.tokens)} "
        f"steps={run.decode_steps} prefill_ms={run.prefill_ms:.0f} "
        f"overflow={report.overflow} graph={graph_mode}",
        flush=True,
    )
    try:
        loop.kv = None
        model.to("cpu")
    except Exception:
        pass
    del loop, model, tokenizer
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return plate


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    preset = PRESETS[args.size]
    dest = _out_dir(args.out, args.size)
    model_dir = str(args.model or preset["model"])
    chr_path = str(args.chr_path or preset["chr"])
    compare_id = str(args.compare_id or preset["compare_id"])
    if args.plan_only:
        from gpu.host.compute import format_compute_stderr, plan_compute
        from gpu.host.residency import descs_from_qwen

        cfg_path = Path(model_dir) / "config.json"
        if cfg_path.is_file():
            from gpu.cli.codec import load_config

            cfg = load_config(model_dir)
        else:
            cfg = {
                "hidden_size": 5120,
                "intermediate_size": 27648,
                "num_hidden_layers": 64,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "vocab_size": 152064,
                "tie_word_embeddings": False,
            }
            if args.size == "3B":
                cfg = {
                    "hidden_size": 2048,
                    "intermediate_size": 11008,
                    "num_hidden_layers": 36,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "vocab_size": 151936,
                    "tie_word_embeddings": True,
                }
        compute_plan = plan_compute(
            descs_from_qwen(cfg),
            cfg,
            compute=getattr(args, "compute", "gpu") or "gpu",
            gpu_layers=getattr(args, "gpu_layers", None),
            max_seq=int(args.max_seq),
        )
        plate = {
            "schema": SCHEMA,
            "plan_only": True,
            "size": args.size,
            "model": model_dir,
            "chr": chr_path,
            "chr_exists": Path(chr_path).is_file(),
            "compare_id": compare_id,
            "long_n": int(args.long_n),
            "prompt": LONG_PROMPT,
            "compute_plan": compute_plan.to_dict(),
        }
        print(format_compute_stderr(compute_plan), flush=True)
        print(f"plan-only out={dest} chr_exists={plate['chr_exists']}", flush=True)
    else:
        plate = run_live(args, dest, preset)
        if str(getattr(args, "compute", "gpu") or "gpu") == "gpu":
            _merge_long(
                compare_id,
                tok_s=float(plate["decode_tok_s"]),
                source=str(dest),
                predicted_n=int(plate["n_tokens"]),
                decode_steps=int(plate["decode_steps"]),
            )
        else:
            print(
                "skip compare.json merge for cpu-suffix/hybrid "
                "(do not overwrite deepfold-nf4-32B-overflow)",
                flush=True,
            )
    (dest / "plate.json").write_text(
        json.dumps(plate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (dest / "SUMMARY.txt").write_text(
        "\n".join(
            [
                f"schema {SCHEMA}",
                f"size {args.size}",
                f"compare_id {compare_id}",
                f"long_decode_tok_s {plate.get('decode_tok_s')}",
                f"n_tokens {plate.get('n_tokens')}",
                f"decode_steps {plate.get('decode_steps')}",
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

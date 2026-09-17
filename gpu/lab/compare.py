"""Matched 3080 comparison rows for a later infographic.

One JSON: same greedy smoke (Paris / Berlin / 323), ctx 2048, n_predict 64
where the stack can do it. Empty numeric cells stay null. A skip is a row.

    python -m gpu.lab.compare
    python -m gpu.lab.compare --seed-known

Does not import torch. Does not pip into conda torch-gpu.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_REPO))

from gpu.lab.script import RUNS_DIR

__all__ = ["SCHEMA", "COMPARE_DIR", "load", "upsert", "seed_known"]

SCHEMA = "deepfold.compare.v1"
COMPARE_DIR = Path(RUNS_DIR) / "compare-3080"
COMPARE_JSON = COMPARE_DIR / "compare.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_row(**fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema": SCHEMA,
        "id": "",
        "stack": "",
        "engine": "",
        "model": "",
        "size": "",
        "quant": "",
        "protocol": "smoke-greedy-ctx2048-n64",
        "mean_decode_tok_s": None,
        "long_decode_tok_s": None,
        "long_eval_tok_s": None,
        "mean_ttft_ms": None,
        "mean_client_ttft_ms": None,
        "smi_after_load_mib": None,
        "smoke_ok": None,
        "bench_pp512_tok_s": None,
        "bench_tg_tok_s": None,
        "source": "",
        "skip_reason": None,
        "notes": "",
        "recorded_at": _now(),
    }
    row.update(fields)
    if not row["id"]:
        row["id"] = f"{row.get('stack')}-{row.get('size')}-{row.get('quant')}".strip("-")
    return row


def load(path: Path | None = None) -> dict[str, Any]:
    dest = path or COMPARE_JSON
    if not dest.is_file():
        return {"schema": SCHEMA, "machine": "RTX 3080 12 GB WDDM", "rows": []}
    return json.loads(dest.read_text(encoding="utf-8"))


def save(payload: dict[str, Any], path: Path | None = None) -> Path:
    dest = path or COMPARE_JSON
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"schema {SCHEMA}",
        f"rows {len(payload.get('rows') or [])}",
    ]
    for row in payload.get("rows") or []:
        tok = row.get("mean_decode_tok_s")
        skip = row.get("skip_reason")
        if skip:
            lines.append(f"{row.get('id')} SKIP {skip}")
        else:
            lines.append(
                f"{row.get('id')} decode={tok} long={row.get('long_decode_tok_s')} "
                f"ttft_ms={row.get('mean_ttft_ms')} smoke={row.get('smoke_ok')}"
            )
    (dest.parent / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def upsert(row: dict[str, Any], path: Path | None = None) -> Path:
    payload = load(path)
    rows = list(payload.get("rows") or [])
    key = row.get("id")
    rows = [old for old in rows if old.get("id") != key]
    rows.append(empty_row(**row))
    rows.sort(key=lambda item: str(item.get("id") or ""))
    payload["schema"] = SCHEMA
    payload["machine"] = payload.get("machine") or "RTX 3080 12 GB WDDM"
    payload["rows"] = rows
    payload["updated_at"] = _now()
    dest = save(payload, path)
    print(f"upsert {key} -> {dest}", flush=True)
    return dest


def seed_known() -> Path:
    """Rows already measured; do not invent new tok/s."""
    upsert(
        {
            "id": "deepfold-nf4-32B-overflow",
            "stack": "deepfold-nf4",
            "engine": "TokenLoop + CopyRing",
            "model": "Qwen2.5-32B-Instruct",
            "size": "32B",
            "quant": "NF4 overflow",
            "mean_decode_tok_s": 2.3128940562710195,
            "long_decode_tok_s": 2.488212340879585,
            "long_eval_tok_s": 2.5277077748618004,
            "mean_ttft_ms": 1006.0023333353456,
            "smi_after_load_mib": 11926.0,
            "smoke_ok": True,
            "source": "docs/runs/h2-qwen25-32b/",
            "notes": (
                "Product H2 smoke 2026-09-14. Long ignore-EOS 64 tokens / 63 decode "
                "steps (decode_tok_s=2.49). eval_tok_s=2.53 is n_tokens/decode_ms, "
                "same counting as Ollama eval_count."
            ),
            "recorded_at": "2026-09-14T16:40:48Z",
        }
    )
    upsert(
        {
            "id": "llamacpp-q4-32B-Q4_K_M",
            "stack": "llamacpp-q4",
            "engine": "llama.cpp b10964 llama-server CUDA 12.4",
            "model": "Qwen2.5-32B-Instruct",
            "size": "32B",
            "quant": "Q4_K_M",
            "mean_decode_tok_s": 1.5536354958047693,
            "long_decode_tok_s": 1.5240900577557557,
            "mean_ttft_ms": 1010.0813333333332,
            "smi_after_load_mib": 11520.0,
            "smoke_ok": True,
            "bench_pp512_tok_s": 69.77799,
            "bench_tg_tok_s": 1.47332,
            "source": "docs/runs/llamacpp-h2/SUMMARY.txt",
            "notes": "parallel 1, ngl 99 (fit abort), ignore_eos 64. Live C:\\dev\\models\\runs\\llamacpp-h2-20260915-224614. Runner default now omits --n-gpu-layers.",
            "recorded_at": "2026-09-15T15:46:14Z",
        }
    )
    upsert(
        {
            "id": "llamacpp-q4-32B-Q4_K_M-autofit",
            "stack": "llamacpp-q4",
            "engine": "llama.cpp b10964 llama-server CUDA 12.4",
            "model": "Qwen2.5-32B-Instruct",
            "size": "32B",
            "quant": "Q4_K_M auto-fit",
            "mean_decode_tok_s": 2.622731852031036,
            "long_decode_tok_s": 2.5418356903345325,
            "mean_ttft_ms": 1444.1003333333335,
            "smi_after_load_mib": 11636.0,
            "smoke_ok": True,
            "source": "docs/runs/llamacpp-h2-autofit/",
            "notes": (
                "ngl omitted (llama-server auto-fit), parallel 1, ignore_eos 64. "
                "Long 2.54 matches Ollama 2.54. Live C:\\dev\\models\\runs\\llamacpp-h2-32b-autofit-20260917. "
                "ngl 99 row stays llamacpp-q4-32B-Q4_K_M (1.52)."
            ),
            "recorded_at": "2026-09-17T04:05:28Z",
        }
    )
    upsert(
        {
            "id": "ollama-3B-qwen2.5-3b",
            "stack": "ollama",
            "engine": "Ollama 0.34.0",
            "model": "Qwen2.5-Instruct",
            "size": "3B",
            "quant": "qwen2.5:3b",
            "mean_decode_tok_s": 189.58064823274708,
            "long_decode_tok_s": 187.2724931821108,
            "mean_ttft_ms": None,
            "smi_after_load_mib": 3837.0,
            "smoke_ok": True,
            "source": "docs/runs/ollama-h2-3b/",
            "notes": (
                "Library tag qwen2.5:3b, greedy, ctx 2048, n_predict 64. "
                "Long = 64-token travelogue. Historical mean_ttft_ms null: "
                "prompt_eval_cached_count=24, stream=False. Runner now stream=True "
                "and records client_ttft_ms."
            ),
            "recorded_at": "2026-09-15T17:06:46Z",
        }
    )
    upsert(
        {
            "id": "ollama-32B-qwen2.5-32b",
            "stack": "ollama",
            "engine": "Ollama 0.34.0",
            "model": "Qwen2.5-Instruct",
            "size": "32B",
            "quant": "qwen2.5:32b",
            "mean_decode_tok_s": 3.178478727957889,
            "long_decode_tok_s": 2.543018637028632,
            "mean_ttft_ms": None,
            "smi_after_load_mib": 11559.0,
            "smoke_ok": True,
            "source": "docs/runs/ollama-h2-32b/",
            "notes": (
                "Library tag qwen2.5:32b (~19 GB), greedy, ctx 2048. "
                "Smoke mean is short EOS (8/8/4 tokens); quote long 64-token plateau for decode. "
                "llama-server auto-fit 33/65 layers; --load-mode none --flash-attn auto. "
                "Historical mean_ttft_ms null: prompt_eval_cached_count=24, stream=False "
                "(server prompt_eval ~901 ms is not TokenLoop TTFT). Runner now stream=True."
            ),
            "recorded_at": "2026-09-15T17:07:55Z",
        }
    )
    upsert(
        {
            "id": "deepfold-nf4-3B-resident",
            "stack": "deepfold-nf4",
            "engine": "TokenLoop CompressedLinear",
            "model": "Qwen2.5-3B-Instruct",
            "size": "3B",
            "quant": "NF4 resident",
            "mean_decode_tok_s": 29.5427,
            "long_decode_tok_s": 35.2244285325278,
            "long_eval_tok_s": 35.78354644574253,
            "mean_ttft_ms": 93.9284,
            "smi_after_load_mib": 3444.0,
            "smoke_ok": True,
            "source": "docs/runs/deepfold-long-3b/",
            "notes": "Isolated lab worker, graph=linears, max_seq=2048. Long ignore-EOS 64 tokens / 63 steps (decode_tok_s=35.2); eval_tok_s=35.8 is n/decode_ms.",
            "recorded_at": "2026-09-15T16:29:46Z",
        }
    )
    upsert(
        {
            "id": "llamacpp-q4-3B-Q4_K_M",
            "stack": "llamacpp-q4",
            "engine": "llama.cpp b10964 llama-server CUDA 12.4",
            "model": "Qwen2.5-3B-Instruct",
            "size": "3B",
            "quant": "Q4_K_M",
            "mean_decode_tok_s": 148.89397123579184,
            "long_decode_tok_s": 186.99466916784405,
            "mean_ttft_ms": 17.023333333333337,
            "smi_after_load_mib": 3710.0,
            "smoke_ok": True,
            "bench_pp512_tok_s": 8270.247216,
            "bench_tg_tok_s": 206.384133,
            "source": "docs/runs/compare-3080/",
            "recorded_at": "2026-09-15T16:27:50Z",
        }
    )
    upsert(
        {
            "id": "bitsandbytes-nf4-3B",
            "stack": "bitsandbytes-nf4",
            "engine": "HF generate Linear4bit",
            "model": "Qwen2.5-3B-Instruct",
            "size": "3B",
            "quant": "NF4",
            "mean_decode_tok_s": 22.204636335182755,
            "mean_ttft_ms": 60.45566666095207,
            "smi_after_load_mib": 3842.0,
            "smoke_ok": True,
            "source": "docs/runs/competitor-qwen25-3b/",
            "notes": "Smoke only; no long plateau. Isolated venv, never torch-gpu.",
            "recorded_at": "2026-09-15T16:27:22Z",
        }
    )
    return COMPARE_JSON


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m gpu.lab.compare")
    p.add_argument("--seed-known", action="store_true")
    args = p.parse_args(argv)
    if args.seed_known:
        dest = seed_known()
    else:
        dest = save(load())
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

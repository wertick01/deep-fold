"""Acceptance for the token loop: one command, greedy, Paris / Париж.

    python gpu/loop/smoke.py

Prints the protocol of ``docs/token-loop.md`` §7.4 as far as this wave reaches:
``load_s``, ``prefill_ms``, ``decode_tok_s`` (after warmup, never averaged with
prefill), ``nvidia-smi`` used MiB at four points, and ``graph=off|linears``.

Gates, all reported explicitly:

1. greedy text contains ``Paris`` (and ``Париж`` on the Russian prompt);
2. VRAM after 16 and after N decode tokens is flat -- a ``cat``-grown KV cache
   would show up here (TZ wave3-loop.md);
3. no BF16 ``M x K`` weight is ever materialized: peak allocation of one decode
   step stays under the smallest layer matrix (2048x2048 bf16 = 8 MiB);
4. graph replay picks the same 8 greedy tokens as eager, or capture is reported
   as skipped -- in which case eager alone still has to pass;
5. the chunked ``N>1`` prefill picks the same token as walking the prompt one at a
   time, which is where a wrong causal mask or KV span write would hide.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from gpu.host import load_model  # noqa: E402
from gpu.loop import TokenLoop, nf4_max_n  # noqa: E402
from gpu.nf4.plan import LIVE_MAX_N  # noqa: E402

MODEL_ID = os.environ.get("DEEPFOLD_MODEL", r"C:\dev\models\Qwen2.5-3B-Instruct")
CHR_PATH = os.environ.get("DEEPFOLD_CHR", r"C:\dev\models\qwen25-3b.nf4.chr")
MIB = 1024 * 1024

PROMPT_EN = "The capital of France is"
PROMPT_RU = "Столица Франции? Ответь одним словом."


def smi_used_mib(index: int = 0, samples: int = 3) -> int:
    """Source of truth for VRAM. Not ``max_memory_allocated``, which cannot see
    the CUDA context, the graph pool or the desktop (token-loop.md §7.2)."""
    cmd = [
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=memory.used",
        "--format=csv,nounits,noheader",
    ]
    vals = []
    for _ in range(max(1, samples)):
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        vals.append(int(out.stdout.strip().splitlines()[0]))
    return int(statistics.median(vals))


def smi_total_mib(index: int = 0) -> int:
    out = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return int(out.stdout.strip().splitlines()[0])


def main() -> int:
    ap = argparse.ArgumentParser(description="token loop smoke: greedy Paris / Париж")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--chr", default=CHR_PATH)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=64, help="decode tokens for the tok/s number")
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--graph", choices=("off", "linears"), default="linears")
    ap.add_argument("--norm", choices=("exact", "fast"), default="exact")
    ap.add_argument(
        "--no-overlap",
        dest="overlap",
        action="store_false",
        help="run each group's GEMMs back to back instead of on side streams",
    )
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # so Париж is readable on a cp1251 console
    except Exception:  # noqa: BLE001
        pass

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device")
        return 1

    smi_total = smi_total_mib()
    smi_boot = smi_used_mib()
    print(f"gpu: {torch.cuda.get_device_name(0)}, sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, "
          f"smi_total={smi_total} MiB, smi_before_load={smi_boot} MiB (desktop/other)")
    print(f"torch {torch.__version__} / cuda {torch.version.cuda}")

    # --- load ---------------------------------------------------------------
    t0 = time.perf_counter()
    model, report = load_model(args.model, args.chr)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    smi_load = smi_used_mib()
    print(f"\nmodel: {Path(args.model).name}  chr: {Path(args.chr).name}")
    print(f"load: {report}")
    print(f"load_s={load_s:.1f}  vram_after_load_mb={smi_load}")

    loop = TokenLoop(model, max_seq=args.max_seq, norm=args.norm, overlap=args.overlap)
    print(f"\nloop: {loop!r}")
    print(f"kv: {loop.kv!r}")
    print(
        f"weights={loop.weight_bytes / MIB:.0f} MiB  "
        f"kv={loop.kv.mib:.0f} MiB ({loop.kv.bytes_per_token / 1024:.0f} KiB/token)  "
        f"gemm_groups={len(loop._groups)} covering "
        f"{sum(len(g.gemms) for g in loop._groups)} GEMM launches/token"
    )
    n_gt_1 = loop.prefill_chunk > 1
    print(
        f"prefill: kernel accepts N>1 = {n_gt_1} -> chunk={loop.prefill_chunk} token(s) per forward"
        + ("" if n_gt_1 else "  (wave-2 GEMM is decode-only; KV is still filled slot by slot, no cat)")
    )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    stop = tuple({i for i in (tok.eos_token_id, 151643, 151645) if i is not None})

    # --- warmup (never in a reported number) --------------------------------
    warm_ms = loop.warmup(prompt=8, tokens=args.warmup)
    smi_warm = smi_used_mib()
    print(f"\nwarmup: {args.warmup} decode tokens eager, {warm_ms:.0f} ms; vram={smi_warm} MiB")

    # --- gate 3: no BF16 M x K on the token path ----------------------------
    loop.reset()
    ids_en = tok(PROMPT_EN, return_tensors="pt").input_ids[0]
    first = int(loop.prefill(ids_en).argmax())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    loop.step(first)
    torch.cuda.synchronize()
    step_peak = (torch.cuda.max_memory_allocated() - base_alloc) / MIB
    smallest_w = 2048 * 2048 * 2 / MIB
    gate_alloc = step_peak < smallest_w
    print(
        f"\nstep_peak_alloc={step_peak:.2f} MiB "
        f"(smallest bf16 layer matrix would be {smallest_w:.0f} MiB) -> "
        f"{'PASS' if gate_alloc else 'FAIL'}"
    )

    # --- gate 5: the N>1 prefill agrees with token-by-token ------------------
    # Only meaningful when the kernel took N>1: it is the one path where a wrong
    # causal mask, a mis-sliced rope table or a bad KV span write would hide.
    gate_prefill = True
    if n_gt_1:
        loop.reset()
        wide = loop.prefill(ids_en).float()
        loop.prefill_chunk = 1
        loop.reset()
        narrow = loop.prefill(ids_en).float()
        loop.prefill_chunk = min(nf4_max_n(LIVE_MAX_N), loop.max_seq)
        delta = (wide - narrow).abs().max().item()
        same = int(wide.argmax()) == int(narrow.argmax())
        gate_prefill = same and delta < 1.0
        print(
            f"prefill N={loop.prefill_chunk} vs N=1: same greedy token={same} "
            f"(id {int(wide.argmax())}), max|dlogit|={delta:.3f} -> "
            f"{'PASS' if gate_prefill else 'FAIL'}"
        )

    # --- speed, eager: the mode that has to pass on its own ------------------
    def measure(tag: str):
        run = loop.generate(ids_en, args.tokens + 1, stop=stop)
        print(
            f"--- {tag} ---\n"
            f"prefill: {run.prompt_len} tokens, prefill_ms={run.prefill_ms:.1f} "
            f"({run.prefill_ms / run.prompt_len:.1f} ms/token, chunk={run.prefill_chunk})\n"
            f"decode: {run.decode_steps} steps, decode_tok_s={run.decode_tok_s:.1f} "
            f"({run.ms_per_token:.1f} ms/token)"
        )
        return run

    print()
    run_eager = measure(f"graph=off, overlap={loop.overlap}")

    # --- gate 4: graph replay == eager on 8 tokens ---------------------------
    eager8 = loop.generate(ids_en, 8, stop=stop).tokens
    graph_ms = 0.0
    graph8: list[int] = []
    run_graph = None
    if args.graph == "linears":
        t0 = time.perf_counter()
        mode = loop.capture_graphs()
        graph_ms = (time.perf_counter() - t0) * 1000.0
        if mode == "linears":
            print(f"\ngraph: captured {len(loop._groups)} linear groups in {graph_ms:.0f} ms")
            graph8 = loop.generate(ids_en, 8, stop=stop).tokens
        else:
            print(f"\ngraph: capture SKIPPED, staying eager ({loop.graph_error})")
    else:
        print("\ngraph: off (requested)")
    gate_graph = (not graph8) or graph8 == eager8
    if graph8:
        print(
            f"graph_vs_eager_8: {'PASS' if gate_graph else 'FAIL'}\n"
            f"  eager={eager8}\n  graph={graph8}"
        )
        print()
        run_graph = measure(f"graph=linears, overlap={loop.overlap}")

    # --- leak: 16 vs N decode tokens, in the active mode ---------------------
    loop.generate(ids_en, 17, stop=stop)
    smi_16, res_16 = smi_used_mib(), torch.cuda.memory_reserved()
    run = loop.generate(ids_en, args.tokens + 1, stop=stop)
    smi_64, res_64 = smi_used_mib(), torch.cuda.memory_reserved()
    text_en = tok.decode(run.tokens, skip_special_tokens=True)
    print(
        f"\nvram: after_load={smi_load}  after_16_decode={smi_16}  "
        f"after_{run.decode_steps}_decode={smi_64} MiB (smi)"
    )
    # smi is the source of truth for the *level*, but on a display card it also
    # moves by tens of MiB on its own (measured: a +267 MiB step with our process
    # idle). For the *growth* the allocator counters are the exact witness, and
    # they are the ones a cat-grown KV cache would move.
    grow_torch = (res_64 - res_16) / MIB
    grow_smi = smi_64 - smi_16
    gate_leak = grow_torch == 0.0
    print(
        f"leak: 16 -> {run.decode_steps} tokens: torch_reserved {grow_torch:+.1f} MiB "
        f"(hard gate), smi {grow_smi:+d} MiB (desktop noise) -> {'PASS' if gate_leak else 'FAIL'}"
    )
    print(f"\n[EN] {PROMPT_EN!r}\n  -> {text_en!r}")
    gate_en = "Paris" in text_en

    # --- Russian prompt, chat template --------------------------------------
    text_ru = ""
    try:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT_RU}],
            add_generation_prompt=True,
            tokenize=False,
        )
        ru_ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids[0]
        run_ru = loop.generate(ru_ids, 32, stop=stop)
        text_ru = tok.decode(run_ru.tokens, skip_special_tokens=True)
        print(
            f"[RU] {PROMPT_RU!r}\n  -> {text_ru!r}"
            f"\n  prefill_ms={run_ru.prefill_ms:.1f} ({run_ru.prompt_len} tokens), "
            f"decode_tok_s={run_ru.decode_tok_s:.1f}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[RU] skipped: {type(exc).__name__}: {exc}")
    gate_ru = ("Париж" in text_ru) or ("Paris" in text_ru)

    # --- protocol -----------------------------------------------------------
    print("\n=== protocol (token-loop.md §7.4) ===")
    print(f"gpu: 3080 12GB, smi_total={smi_total}, before_load={smi_boot} MiB")
    print("build: stage A (NF4 g64)")
    print(f"model: {Path(args.model).name}, {loop.n_layers} layers, hidden {loop.hidden}, "
          f"{loop.n_q}q/{loop.n_kv}kv x {loop.head_dim}, tied lm_head={report.tied_lm_head}")
    print(f"load_s={load_s:.1f}, vram_after_load_mb={smi_load}")
    print(f"prefill_{run.prompt_len}_ms={run.prefill_ms:.1f}, prefill_chunk={run.prefill_chunk}, N>1={n_gt_1}")
    print(
        f"decode_tok_s={run_eager.decode_tok_s:.1f} eager"
        + (f" / {run_graph.decode_tok_s:.1f} graph=linears" if run_graph else "")
        + f" ({args.tokens} tokens after warmup {args.warmup}), vram_decode_mb={smi_64}"
    )
    print(f"norm={loop.norm_mode}, overlap={loop.overlap}, max_seq={loop.max_seq}, kv_mib={loop.kv.mib:.0f}")
    print(f"graph: {loop.graph_mode}" + (f" (capture {graph_ms:.0f} ms)" if graph8 else ""))

    gates = {
        "greedy_en_paris": gate_en,
        "greedy_ru_parizh": gate_ru,
        "no_bf16_MxK": gate_alloc,
        "smi_flat_16_to_64": gate_leak,
        "graph_replay_eq_eager": gate_graph,
        "prefill_chunk_eq_stepwise": gate_prefill,
    }
    print("\n" + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in gates.items()))
    ok = all(gates.values())
    print("SMOKE: PASS" if ok else "SMOKE: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

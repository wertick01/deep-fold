"""Named GPU time of one Decode V2 greedy step. Production ``step.py`` stays cold.

Nsight Systems is not installed. CUDA events around the production graph give
the step total. Graph-captured events cannot ``elapsed_time`` on this WDDM
stack, so categories are L2-hot live kernel windows scaled by ``n_layers``.

    python gpu/decodev2/profile_step.py
    python gpu/decodev2/profile_step.py --slug qwen25-3b --backend gemv --max-seq 2048
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.cli.paths import runs_root  # noqa: E402
from gpu.decodev2.embed import PackedEmbed, gather_embed  # noqa: E402
from gpu.decodev2.graph import GreedyGraph, capture_greedy  # noqa: E402
from gpu.decodev2.linear import nf4_linear, set_linear_backend  # noqa: E402
from gpu.decodev2.load import load_chr  # noqa: E402
from gpu.decodev2.prefill import prefill_chunk_width  # noqa: E402
from gpu.decodev2.runner import consume_prompt  # noqa: E402
from gpu.decodev2.step import (  # noqa: E402
    _attend_into,
    _gemv_accum,
    _into,
    _mlp_down_in,
    _qkv_into_arena,
    _rms_into,
    _rope_and_cache,
    greedy_decode,
)
from gpu.lab.catalog import lab_by_slug  # noqa: E402
from gpu.lab.script import LONG_PROMPT, chat_text  # noqa: E402
from gpu.lab.sessions import _load_tokenizer  # noqa: E402
from gpu.tests.skips import cuda_reason  # noqa: E402

__all__ = [
    "CATEGORIES",
    "GpuSpans",
    "NullSpans",
    "capture_profiled",
    "profiled_greedy",
    "reconstruct_kernels",
    "spans_per_step",
]

CATEGORIES = (
    "embed",
    "rms",
    "qkv",
    "attn",
    "o_proj",
    "swiglu",
    "down",
    "lm_head",
    "commit",
)


def spans_per_step(n_layers: int) -> int:
    """embed + (rms, qkv, attn, o, rms, swiglu, down) * L + final rms + lm_head + commit."""
    return 1 + 7 * int(n_layers) + 3


class NullSpans:
    """CPU / correctness path. No CUDA events."""

    def reset(self) -> None:
        return

    @contextmanager
    def span(self, name: str):
        del name
        yield


class GpuSpans:
    """Preallocated CUDA events. ``elapsed_time`` is GPU-timeline, not host wall."""

    def __init__(self, n_spans: int) -> None:
        n = max(1, int(n_spans))
        self._starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        self._ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        self._names = [""] * n
        self._n = 0

    def reset(self) -> None:
        self._n = 0

    @contextmanager
    def span(self, name: str):
        i = self._n
        if i >= len(self._starts):
            raise RuntimeError(f"span pool exhausted at {i}/{len(self._starts)}")
        self._names[i] = name
        self._starts[i].record()
        try:
            yield
        finally:
            self._ends[i].record()
            self._n = i + 1

    def totals_ms(self) -> dict[str, float]:
        torch.cuda.synchronize()
        out = {key: 0.0 for key in CATEGORIES}
        for i in range(self._n):
            name = self._names[i]
            if name not in out:
                out[name] = 0.0
            out[name] += float(self._starts[i].elapsed_time(self._ends[i]))
        return out


def profiled_greedy(state, weights, spans) -> None:
    """Same ops as ``greedy_decode``. Events are the only extra work."""
    spec = state.spec
    ar = state.arena
    with spans.span("embed"):
        x = _into(ar.x, gather_embed(weights.embed, state.token.reshape(1), dtype=ar.x.dtype))
    for li, layer in enumerate(weights.layers):
        with spans.span("rms"):
            h = _rms_into(x, layer.norm1, spec.rms_eps, ar.h)
        with spans.span("qkv"):
            _qkv_into_arena(state, layer, h)
        with spans.span("attn"):
            if ar.q.device.type != "cuda":
                _rope_and_cache(state, li)
            if li == 0:
                state.kv.mark_written(state.position)
            _attend_into(state, li)
        with spans.span("o_proj"):
            _gemv_accum(layer.o, ar.attn, ar.x)
        with spans.span("rms"):
            h = _rms_into(ar.x, layer.norm2, spec.rms_eps, ar.h)
        with spans.span("swiglu"):
            _mlp_down_in(state, layer, h)
        with spans.span("down"):
            _gemv_accum(layer.down, ar.down_in, ar.x)
        x = ar.x
    with spans.span("rms"):
        hidden = _rms_into(x, weights.final_norm, spec.rms_eps, ar.h)
    with spans.span("lm_head"):
        logits = nf4_linear(weights.lm_head, hidden)
        _into(ar.logits, logits)
        state.next_token.copy_(ar.logits.argmax(dim=-1).reshape(()).to(dtype=torch.int64))
    with spans.span("commit"):
        state.commit_step()


def capture_profiled(
    state,
    weights,
    spans: GpuSpans,
    *,
    warmup: int = 2,
) -> GreedyGraph:
    """Capture ``profiled_greedy``. Do not call ``GpuSpans.totals_ms`` after replay.

    WDDM rejects ``elapsed_time`` on events recorded inside a CUDA graph
    (``invalid argument``). Live kernel windows in ``reconstruct_kernels`` name
    the mix; production ``capture_greedy`` stays the step total.
    """
    if state.device.type != "cuda":
        raise RuntimeError("profile graph is CUDA-only")
    stream = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(stream)
    with torch.cuda.stream(side):
        for _ in range(max(1, warmup)):
            spans.reset()
            profiled_greedy(state, weights, spans)
        side.synchronize()
        spans.reset()
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin()
        try:
            profiled_greedy(state, weights, spans)
        finally:
            graph.capture_end()
    stream.wait_stream(side)
    return GreedyGraph(graph, state, weights)


def _median_kernel_ms(fn, *, warmup: int = 16, iters: int = 32, repeats: int = 3) -> float:
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end) / iters))
    return statistics.median(samples)


def reconstruct_kernels(state, weights) -> dict:
    """Median ms of each live kernel on this state's tensors. L2-hot tight loop."""
    spec = state.spec
    ar = state.arena
    layer = weights.layers[0]
    n_layers = int(spec.n_layers)

    def embed():
        _into(ar.x, gather_embed(weights.embed, state.token.reshape(1), dtype=ar.x.dtype))

    def rms():
        _rms_into(ar.x, layer.norm1, spec.rms_eps, ar.h)

    def qkv():
        _qkv_into_arena(state, layer, ar.h)

    def attn():
        _attend_into(state, 0)

    def o_proj():
        _gemv_accum(layer.o, ar.attn, ar.x)

    def swiglu():
        _mlp_down_in(state, layer, ar.h)

    def down():
        _gemv_accum(layer.down, ar.down_in, ar.x)

    def lm_head():
        logits = nf4_linear(weights.lm_head, ar.h)
        _into(ar.logits, logits)
        state.next_token.copy_(ar.logits.argmax(dim=-1).reshape(()).to(dtype=torch.int64))

    tok = state.token
    nxt = state.next_token

    def commit():
        tok.copy_(nxt)

    once = {
        "embed": _median_kernel_ms(embed),
        "rms_once": _median_kernel_ms(rms),
        "qkv": _median_kernel_ms(qkv),
        "attn": _median_kernel_ms(attn),
        "o_proj": _median_kernel_ms(o_proj),
        "swiglu": _median_kernel_ms(swiglu),
        "down": _median_kernel_ms(down),
        "lm_head": _median_kernel_ms(lm_head),
        "commit": _median_kernel_ms(commit),
    }
    scaled = {
        "embed": once["embed"],
        "rms": once["rms_once"] * (2 * n_layers + 1),
        "qkv": once["qkv"] * n_layers,
        "attn": once["attn"] * n_layers,
        "o_proj": once["o_proj"] * n_layers,
        "swiglu": once["swiglu"] * n_layers,
        "down": once["down"] * n_layers,
        "lm_head": once["lm_head"],
        "commit": once["commit"],
    }
    return {
        "once_ms": {key: round(val, 4) for key, val in once.items()},
        "scaled_ms": {key: round(val, 4) for key, val in scaled.items()},
        "reconstructed_ms": round(sum(scaled.values()), 4),
        "n_layers": n_layers,
        "valid_len": int(state.valid_len.item()),
        "note": (
            "Tight-loop on live tensors (L2-hot). Graph replay can be slower "
            "(weight cache thrash + inter-node idle) or similar."
        ),
    }


def _vram(tag: str) -> dict:
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(
        f"vram {tag} allocated={alloc:.0f} MiB reserved={reserved:.0f} MiB "
        f"peak={peak:.0f} MiB",
        flush=True,
    )
    return {
        "tag": tag,
        "allocated_mib": round(alloc, 1),
        "reserved_mib": round(reserved, 1),
        "peak_allocated_mib": round(peak, 1),
    }


def _prompt_ids(tokenizer, text: str) -> list[int]:
    packed = chat_text(tokenizer, text)
    ids = tokenizer(packed, add_special_tokens=False).input_ids
    return [int(x) for x in ids]


def _event_ms(body, repeats: int) -> list[float]:
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        body()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _pct(part: float, whole: float) -> float:
    if whole <= 0:
        return 0.0
    return round(100.0 * part / whole, 1)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python gpu/decodev2/profile_step.py")
    p.add_argument("--slug", default="qwen25-3b")
    p.add_argument("--max-seq", type=int, default=2048)
    p.add_argument("--backend", choices=("mma", "gemv"), default="gemv")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--out", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reason = cuda_reason()
    if reason is not None:
        print(f"SKIP {reason}")
        return 0
    nsys = shutil.which("nsys")
    ncu = shutil.which("ncu")
    print(
        f"nsys={'yes ' + nsys if nsys else 'missing'} "
        f"ncu={'yes ' + ncu if ncu else 'missing (Nsight Compute may still exist off PATH)'}",
        flush=True,
    )
    lab = lab_by_slug(args.slug)
    dest = Path(args.out) if args.out else (
        runs_root() / f"decodev2-{lab.slug}-step-profile-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    dest.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    set_linear_backend(args.backend)
    print(
        f"load slug={lab.slug} {lab.chr_path} max_seq={args.max_seq} "
        f"backend={args.backend} out={dest}",
        flush=True,
    )
    loaded = load_chr(
        lab.model_dir,
        lab.chr_path,
        max_seq=args.max_seq,
        trust_remote_code=lab.trust_remote_code,
    )
    emb = loaded.weights.embed
    embed_kind = "nf4-rows" if isinstance(emb, PackedEmbed) else "dense"
    print(
        f"spec family={loaded.spec.family} layers={loaded.spec.n_layers} "
        f"hidden={loaded.spec.hidden} q/kv/hd={loaded.spec.n_q}/"
        f"{loaded.spec.n_kv}/{loaded.spec.head_dim} vocab={loaded.spec.vocab} "
        f"embed={embed_kind} {tuple(emb.shape)} "
        f"{emb.nbytes / (1024 ** 2):.0f} MiB "
        f"prefill_chunk={prefill_chunk_width(loaded.state)}",
        flush=True,
    )
    vram_load = _vram("after_decodev2_load")
    tokenizer = _load_tokenizer(lab.model_dir, lab.trust_remote_code)
    prompt = _prompt_ids(tokenizer, LONG_PROMPT)
    if len(prompt) + args.steps * args.windows + args.steps * 3 > args.max_seq:
        raise SystemExit(
            f"prompt {len(prompt)} + decode windows exceed max_seq={args.max_seq}"
        )
    loaded.state.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    consume_prompt(loaded.state, loaded.weights, prompt)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000.0
    loaded.state.token.copy_(loaded.state.next_token)
    valid0 = int(loaded.state.valid_len.item())
    print(f"prefill_ms={prefill_ms:.1f} prompt_len={len(prompt)} valid_len={valid0}", flush=True)
    captured = capture_greedy(loaded.state, loaded.weights, warmup=2)
    print("graph captured", flush=True)
    for _ in range(max(0, args.warmup)):
        captured.replay()
    torch.cuda.synchronize()

    def graph_window():
        for _ in range(args.steps):
            captured.replay()

    graph_windows = []
    for i in range(args.windows):
        pos = int(loaded.state.position.item())
        ms = _event_ms(graph_window, 1)[0]
        per = ms / args.steps
        graph_windows.append(
            {
                "i": i,
                "position": pos,
                "window_ms": round(ms, 3),
                "ms_per_step": round(per, 4),
                "tok_s": round(1000.0 / per, 2) if per > 0 else 0.0,
            }
        )
        print(
            f"graph window {i} pos={pos} {per:.3f} ms/step "
            f"({graph_windows[-1]['tok_s']:.1f} tok/s)",
            flush=True,
        )

    def eager_window():
        for _ in range(args.steps):
            greedy_decode(loaded.state, loaded.weights)

    eager_ms = _event_ms(eager_window, 1)[0]
    eager_per = eager_ms / args.steps
    print(f"eager {eager_per:.3f} ms/step ({1000.0 / eager_per:.1f} tok/s)", flush=True)

    recon = reconstruct_kernels(loaded.state, loaded.weights)
    recon_ms = float(recon["reconstructed_ms"])
    print(
        f"reconstructed {recon_ms:.3f} ms/step valid_len={recon['valid_len']} "
        f"(L2-hot live kernels x layers)",
        flush=True,
    )
    for key, val in recon["scaled_ms"].items():
        print(
            f"  {key:<8} {val:7.3f} ms  {_pct(val, recon_ms):5.1f}% recon",
            flush=True,
        )
    once = recon["once_ms"]
    print(
        f"  once rms={once['rms_once']*1000:.1f}µs qkv={once['qkv']*1000:.1f}µs "
        f"attn={once['attn']*1000:.1f}µs o={once['o_proj']*1000:.1f}µs "
        f"swiglu={once['swiglu']*1000:.1f}µs down={once['down']*1000:.1f}µs "
        f"lm_head={once['lm_head']*1000:.1f}µs",
        flush=True,
    )

    cupti = {
        "ok": False,
        "reason": (
            "nsys missing. CUDA graph events cannot elapsed_time on this WDDM stack. "
            "Kineto does not list kernels inside graph replay. Categories are live "
            "kernel windows on the loaded 3B tensors, scaled by n_layers."
        ),
    }
    print(f"CUPTI/nsys: {cupti['reason']}", flush=True)

    torch.cuda.synchronize()
    host_s = time.perf_counter()
    for _ in range(args.steps):
        captured.replay()
        _ = int(loaded.state.token.item())
    torch.cuda.synchronize()
    host_ms = (time.perf_counter() - host_s) * 1000.0
    host_per = host_ms / args.steps
    print(
        f"host-read graph {host_per:.3f} ms/step ({1000.0 / host_per:.1f} tok/s)",
        flush=True,
    )
    vram_end = _vram("after_profile")
    graph_ms = statistics.median(w["ms_per_step"] for w in graph_windows)
    gap = graph_ms - recon_ms
    print(
        f"graph {graph_ms:.3f} ms vs reconstructed {recon_ms:.3f} ms "
        f"gap {gap:+.3f} ms ({_pct(gap, graph_ms):.1f}% of graph)",
        flush=True,
    )
    plate = {
        "schema": "decodev2.step_profile.v1",
        "slug": lab.slug,
        "backend": args.backend,
        "max_seq": args.max_seq,
        "prompt_len": len(prompt),
        "valid_len_after_prefill": valid0,
        "prefill_ms": round(prefill_ms, 2),
        "nsys": nsys or "",
        "ncu_on_path": ncu or "",
        "graph_windows": graph_windows,
        "graph_ms_per_step": round(graph_ms, 4),
        "graph_tok_s": round(1000.0 / graph_ms, 2) if graph_ms > 0 else 0.0,
        "eager_ms_per_step": round(eager_per, 4),
        "eager_tok_s": round(1000.0 / eager_per, 2) if eager_per > 0 else 0.0,
        "reconstruct": recon,
        "graph_minus_reconstructed_ms": round(gap, 4),
        "host_read_ms_per_step": round(host_per, 4),
        "host_read_tok_s": round(1000.0 / host_per, 2) if host_per > 0 else 0.0,
        "cupti": cupti,
        "vram": [vram_load, vram_end],
        "spec": {
            "family": loaded.spec.family,
            "n_layers": loaded.spec.n_layers,
            "hidden": loaded.spec.hidden,
            "n_q": loaded.spec.n_q,
            "n_kv": loaded.spec.n_kv,
            "head_dim": loaded.spec.head_dim,
            "intermediate": loaded.spec.intermediate,
            "vocab": loaded.spec.vocab,
        },
        "embed_kind": embed_kind,
        "note": (
            "nsys missing. Graph-captured cudaEvent elapsed_time is invalid on this WDDM stack. "
            "Categories are L2-hot live kernel windows scaled by n_layers. Production "
            "greedy_decode is untimed."
        ),
    }
    (dest / "plate.json").write_text(
        json.dumps(plate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {dest / 'plate.json'}", flush=True)
    set_linear_backend("mma")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

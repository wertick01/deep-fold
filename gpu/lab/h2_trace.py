"""H2-6: instrumented Qwen2.5-32B NF4 overflow smoke (Paris / Berlin / 323).

Writes system snapshots, residency tape, H2D bytes, nvidia-smi timeline and
the data-path dump under ``C:\\dev\\models\\runs\\h2-qwen25-32b-*``. Does not
start the hard-12 plate.

    python -m gpu.lab.h2_trace
    python -m gpu.lab.h2_trace --plan-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.host.residency import (  # noqa: E402
    MIB,
    descs_from_header,
    plan_residency,
    summarize_residency,
)
from gpu.lab.h2_metrics import (  # noqa: E402
    CHR_32B,
    MODEL_32B,
    auto_max_resident_bytes,
    dump_json,
    expected_h2d_bytes,
    host_pin_snapshot,
    render_data_path,
    smi_snapshot,
    system_snapshot,
    torch_snapshot,
)
from gpu.lab.script import (  # noqa: E402
    MESSAGES,
    NEEDLES,
    RUNS_DIR,
    chat_text,
    quality_ok,
    stop_token_ids,
)
from gpu.lab.sampler import Sampler, smi_used_mib  # noqa: E402

__all__ = ["main", "run_h2_trace"]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gpu.lab.h2_trace")
    p.add_argument("--model", default=MODEL_32B)
    p.add_argument("--chr", dest="chr_path", default=CHR_32B)
    p.add_argument("--out", default="")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-seq", type=int, default=2048)
    p.add_argument("--warmup-tokens", type=int, default=8)
    p.add_argument("--interval", type=float, default=0.12)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--no-graphs", dest="graphs", action="store_false")
    p.add_argument("--no-timing", dest="ring_timing", action="store_false")
    p.add_argument("--trust-remote-code", action="store_true")
    return p


def _out_dir(explicit: str) -> Path:
    if explicit:
        dest = Path(explicit)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = Path(RUNS_DIR) / f"h2-qwen25-32b-{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _plan_from_chr(chr_path: str, cap_bytes: int) -> tuple[dict, dict]:
    from gpu.chr0 import load_header, quantized_codec

    hdr = load_header(chr_path)
    descs = descs_from_header(hdr)
    plan = plan_residency(descs, int(cap_bytes))
    summary = summarize_residency(descs, plan)
    header = {
        "codec": quantized_codec(hdr),
        "n_tensors": len(hdr.tensors),
        "arch": hdr.arch,
        "num_layers": hdr.num_layers,
        "hidden_size": hdr.hidden_size,
        "file_size_mib": hdr.file_size / MIB,
        "packed_nf4_mib": sum(
            int(info.blobs["data"].nbytes) + int(info.blobs["scale"].nbytes)
            for info in hdr.tensors.values()
            if info.codec == "nf4"
        )
        / MIB,
    }
    return summary, header


def _write_data_path(out: Path, snap: dict) -> None:
    (out / "data_path.md").write_text(render_data_path(snap), encoding="utf-8")


def _prefill_chunks(prompt_len: int, chunk: int) -> int:
    step = max(1, int(chunk))
    return int(math.ceil(int(prompt_len) / step))


def run_h2_trace(args: argparse.Namespace) -> int:
    out = _out_dir(args.out)
    notes: list[str] = []
    snap: dict[str, Any] = {
        "out_dir": str(out),
        "model": args.model,
        "chr": args.chr_path,
        "max_seq": args.max_seq,
        "max_new_tokens": args.max_new_tokens,
        "messages": [],
        "notes": notes,
    }
    dump_json(out / "system_before.json", system_snapshot(label="before"))

    cap, cap_info = auto_max_resident_bytes(
        args.model, args.chr_path, args.max_seq
    )
    snap["cap"] = cap_info
    dump_json(out / "cap.json", cap_info)
    if cap is None:
        notes.append("decide() did not request overflow; 32B/12 GB should.")
        dump_json(out / "summary.json", snap)
        _write_data_path(out, snap)
        print("FAIL: no overflow cap", cap_info, flush=True)
        return 1

    try:
        residency, header = _plan_from_chr(args.chr_path, cap)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"plan failed: {type(exc).__name__}: {exc}")
        dump_json(out / "summary.json", snap)
        print(traceback.format_exc(), flush=True)
        return 1
    snap["residency"] = residency
    snap["chr_header"] = header
    dump_json(out / "residency.json", {"header": header, **residency})
    _write_data_path(out, snap)
    print(
        f"plan: streamed={residency['n_streamed']} "
        f"({residency['streamed_bytes'] / MIB:.1f} MiB) "
        f"resident={residency['resident_bytes'] / MIB:.1f} MiB "
        f"slot={residency['slot_nbytes'] / MIB:.2f} MiB",
        flush=True,
    )
    if args.plan_only:
        print(f"plan-only wrote {out}", flush=True)
        return 0

    import gc

    import torch
    from transformers import AutoTokenizer

    from gpu.host import load_model
    from gpu.loop import TokenLoop

    sampler = Sampler("nf4", interval_s=min(0.15, float(args.interval)), verbose=True)
    sampler.start()
    model = None
    loop = None
    tokenizer = None
    gate_ok = False
    try:
        sampler.mark("load_start", detail=str(args.chr_path))
        t_load = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=args.trust_remote_code
        )
        model, report = load_model(
            args.model,
            args.chr_path,
            trust_remote_code=args.trust_remote_code,
            strict=True,
            max_resident_bytes=cap,
        )
        torch.cuda.synchronize()
        load_s = time.perf_counter() - t_load
        snap["load_s"] = load_s
        snap["device_mib"] = report.device_mib
        snap["report"] = str(report)
        snap["report_overflow"] = bool(report.overflow)
        snap["report_streamed"] = int(report.streamed)
        snap["report_streamed_bytes"] = int(report.streamed_bytes)
        snap["report_resident_bytes"] = int(report.resident_bytes)
        snap["report_slot_nbytes"] = int(report.slot_nbytes)
        pin = host_pin_snapshot(model)
        snap["pin"] = pin
        after = system_snapshot(label="after_load")
        dump_json(out / "system_after_load.json", after)
        snap["smi_after_load"] = after["smi"].get("memory.used")
        snap["torch_after_load"] = after["torch"].get("allocated_mib")
        sampler.mark("load_end", detail=str(report))
        print(f"load {load_s:.1f}s {report}", flush=True)
        print(
            f"pin all_pinned={pin['all_pinned']} "
            f"{pin['pinned_mib']:.1f}/{pin['host_mib']:.1f} MiB",
            flush=True,
        )
        if not pin["all_pinned"]:
            notes.append(
                f"host arenas not pinned ({pin['pinned_mib']:.0f}/{pin['host_mib']:.0f} MiB); "
                "pageable H2D."
            )
        if not report.overflow:
            notes.append("LoadReport.overflow is False; expected True on 32B/12 GB.")
        if abs(report.device_mib - 16601) < 50:
            notes.append(
                f"device_mib={report.device_mib:.0f} looks like full packed 16601; "
                "resident cap did not apply."
            )

        loop = TokenLoop(
            model, max_seq=int(args.max_seq), overlap=True, ring_timing=bool(args.ring_timing)
        )
        dump_json(out / "loop_init.json", loop.h2_snapshot())
        sampler.mark("warmup_start", detail=repr(loop))
        warm_ms = loop.warmup(prompt=8, tokens=int(args.warmup_tokens))
        graph_mode = "off"
        if args.graphs:
            graph_mode = loop.capture_graphs()
        torch.cuda.synchronize()
        sampler.mark("warmup_end", detail=f"warmup_ms={warm_ms:.0f} graph={graph_mode}")
        snap["loop"] = loop.h2_snapshot()
        dump_json(out / "loop.json", snap["loop"])
        print(
            f"warmup {warm_ms:.0f} ms graph={graph_mode} "
            f"graphed={snap['loop']['n_graphed_groups']}/"
            f"{snap['loop']['n_groups']}",
            flush=True,
        )
        _write_data_path(out, snap)

        stop = stop_token_ids(tokenizer)
        hits = 0
        smi_series: list[float | None] = []
        for index, prompt in enumerate(MESSAGES, start=1):
            sampler.mark("msg_send", index, detail=prompt)
            packed = chat_text(tokenizer, prompt)
            ids = tokenizer(packed, return_tensors="pt", add_special_tokens=False).input_ids[0]
            t_msg = time.perf_counter()
            run = loop.generate(ids, int(args.max_new_tokens), stop=stop)
            text = tokenizer.decode(run.tokens, skip_special_tokens=True)
            quality = quality_ok(index, text)
            hits += int(quality)
            smi_now = smi_used_mib()
            smi_series.append(smi_now)
            torch_now = torch_snapshot()
            row = {
                "index": index,
                "prompt": prompt,
                "response": text,
                "quality": quality,
                "needles": NEEDLES.get(index),
                "prompt_len": run.prompt_len,
                "new_tokens": len(run.tokens),
                "prefill_ms": run.prefill_ms,
                "prefill_chunks": _prefill_chunks(run.prompt_len, run.prefill_chunk),
                "decode_ms": run.decode_ms,
                "decode_steps": run.decode_steps,
                "decode_tok_s": run.decode_tok_s,
                "ms_per_token": run.ms_per_token,
                "stop_reason": "eos" if run.stop_token is not None else "max_new_tokens",
                "h2d_bytes": run.h2d_bytes,
                "h2d_copies": run.h2d_copies,
                "h2d_copy_ms": run.h2d_copy_ms if args.ring_timing else None,
                "h2d_forwards": run.h2d_forwards,
                "h2d_expect_bytes": expected_h2d_bytes(
                    streamed_bytes=int(report.streamed_bytes),
                    forwards=int(run.h2d_forwards),
                ),
                "smi_used_mib": smi_now,
                "torch_alloc_mib": torch_now.get("allocated_mib"),
                "wall_s": time.perf_counter() - t_msg,
                "graph": run.graph,
            }
            snap["messages"].append(row)
            dump_json(out / "messages.json", snap["messages"])
            sampler.mark(
                "msg_done",
                index,
                detail=(
                    f"tok/s={run.decode_tok_s:.2f} prefill={run.prefill_ms:.0f}ms "
                    f"quality={quality} h2d={run.h2d_bytes / MIB:.1f}MiB"
                ),
            )
            preview = text.replace("\n", " ")[:120]
            print(
                f"[{index}/3] quality={quality} {run.decode_tok_s:.2f} tok/s "
                f"prefill {run.prefill_ms:.0f} ms h2d {run.h2d_bytes / MIB:.1f} MiB | {preview}",
                flush=True,
            )
            if run.prefill_ms > 180_000:
                notes.append(f"TTFT {run.prefill_ms:.0f} ms on message {index}; stopping.")
                break
            _write_data_path(out, snap)

        used = [x for x in smi_series if x is not None]
        if used:
            snap["smi_decode_min"] = min(used)
            snap["smi_decode_max"] = max(used)
            snap["smi_decode_span_mib"] = max(used) - min(used)
            if snap["smi_decode_span_mib"] > 256:
                notes.append(
                    f"nvidia-smi used span {snap['smi_decode_span_mib']:.0f} MiB "
                    "across smoke turns (want ~flat)."
                )
        tok = [m["decode_tok_s"] for m in snap["messages"] if m.get("decode_steps")]
        snap["mean_decode_tok_s"] = sum(tok) / len(tok) if tok else 0.0
        snap["quality_hits"] = f"{hits}/{len(snap['messages'])}"
        floor_ok = bool(tok) and min(tok) > 1.5
        overflow_ok = bool(report.overflow) and report.device_mib < 14000
        gate_ok = hits == 3 and overflow_ok and floor_ok
        snap["gate"] = {
            "quality_3_of_3": hits == 3,
            "overflow": overflow_ok,
            "decode_above_floor": floor_ok,
            "pass": gate_ok,
        }
        if not floor_ok:
            notes.append(
                f"decode tok/s {snap['mean_decode_tok_s']:.2f} did not beat the 1–2 floor."
            )
        (out / "gate.txt").write_text(
            "PASS\n" if gate_ok else "FAIL\n", encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"{type(exc).__name__}: {exc}")
        (out / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(traceback.format_exc(), flush=True)
        gate_ok = False
        (out / "gate.txt").write_text("FAIL\n", encoding="utf-8")
    finally:
        try:
            sampler.stop()
        except Exception:
            pass
        dump_json(out / "timeline.json", {"rows": sampler.rows, "events": sampler.events})
        try:
            if loop is not None:
                loop.kv = None
        except Exception:
            pass
        try:
            if model is not None:
                model.to("cpu")
        except Exception:
            pass
        del loop, model, tokenizer
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        dump_json(out / "system_after.json", system_snapshot(label="after_unload"))
        dump_json(out / "summary.json", snap)
        _write_data_path(out, snap)
        print(f"artifacts: {out}", flush=True)
        print(json.dumps(snap.get("gate") or {"pass": False}, ensure_ascii=False), flush=True)
    return 0 if gate_ok else 1


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    return run_h2_trace(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

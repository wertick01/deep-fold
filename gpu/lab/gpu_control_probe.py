"""Isolate 32B gpu overflow: eager vs forced DEVICE graphs vs serial GEMMs.

One load. Does not merge compare.json. Not a tok/s claim.

Default skip leaves graph=off on 177 DEVICE groups. ``--ladder`` force-captures
a prefix (streams empty during capture) to see if CopyRing recovers toward 2.49.

    python -m gpu.lab.gpu_control_probe --steps 8 --ladder 0,24,177
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from gpu.cli.codec import load_config
from gpu.host import load_model
from gpu.host.compute import format_compute_stderr, plan_compute
from gpu.host.host_image import cpu_is_pinned
from gpu.host.residency import descs_from_header
from gpu.lab.h2_metrics import CHR_32B, MODEL_32B, auto_max_resident_bytes
from gpu.lab.script import RUNS_DIR
from gpu.loop import TokenLoop


def _pin_ok(loop: TokenLoop) -> dict:
    n_host = n_pin = n_page = 0
    for grp in loop._groups:
        for g in grp.gemms:
            if g.home != "host":
                continue
            n_host += 1
            arena = g.host_image.arena
            if cpu_is_pinned(arena):
                n_pin += 1
            else:
                n_page += 1
    return {"n_host": n_host, "pinned": n_pin, "pageable": n_page}


def _time_steps(loop: TokenLoop, n_prefill: int, n_decode: int) -> dict:
    loop.reset()
    ids = torch.arange(1, n_prefill + 1, device=loop.device, dtype=torch.long)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = loop.prefill(ids)
    token = int(logits.argmax())
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ring = loop._ring
    c0 = 0 if ring is None else ring.total_copies
    b0 = 0 if ring is None else ring.total_bytes
    for _ in range(n_decode):
        logits = loop.step(token)
        token = int(logits.argmax())
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    copies = 0 if ring is None else int(ring.total_copies - c0)
    nbytes = 0 if ring is None else int(ring.total_bytes - b0)
    decode_s = t2 - t1
    snap = loop.h2_snapshot()
    return {
        "prefill_ms": (t1 - t0) * 1000.0,
        "decode_ms": decode_s * 1000.0,
        "decode_steps": n_decode,
        "tok_s": n_decode / decode_s if decode_s > 0 else 0.0,
        "h2d_copies": copies,
        "h2d_mib": nbytes / (1024 * 1024),
        "graph": loop.graph_mode,
        "pin": _pin_ok(loop),
        "n_graphed": snap.get("n_graphed_groups"),
        "n_eager": snap.get("n_eager_groups"),
    }


def _parse_ladder(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or [0]


def _set_device_streams(loop: TokenLoop, enabled: bool) -> None:
    from gpu.loop.graph import group_is_resident

    streams = loop._streams if enabled else ()
    for grp in loop._groups:
        if not group_is_resident(grp):
            grp.streams = ()
            continue
        n_fork = max(0, len(grp.gemms) - 1)
        grp.streams = tuple(streams)[:n_fork] if n_fork else ()
        if grp.streams and len(grp.streams) != n_fork:
            grp.streams = ()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m gpu.lab.gpu_control_probe")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--prefill", type=int, default=32)
    p.add_argument("--out", default="")
    p.add_argument(
        "--ladder",
        default="0,24,177",
        help="Comma list of max DEVICE graphs. 0 = skip/eager (product 32B path).",
    )
    p.add_argument(
        "--serial-device",
        action="store_true",
        help="Also time eager DEVICE GEMMs with overlap streams cleared.",
    )
    args = p.parse_args(argv)
    dest = Path(args.out) if args.out else Path(RUNS_DIR) / (
        "gpu-control-probe-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    dest.mkdir(parents=True, exist_ok=True)

    cfg = load_config(MODEL_32B)
    from gpu.chr0 import load_header
    from gpu.loop.graph import group_is_resident

    compute_plan = plan_compute(
        descs_from_header(load_header(CHR_32B)),
        cfg,
        compute="gpu",
        max_seq=2048,
    )
    cap, _ = auto_max_resident_bytes(MODEL_32B, CHR_32B, 2048)
    print(format_compute_stderr(compute_plan), flush=True)
    model, report = load_model(
        MODEL_32B,
        CHR_32B,
        strict=True,
        max_resident_bytes=cap,
        residency_policy="D",
        compute_plan=compute_plan,
    )
    loop = TokenLoop(model, max_seq=2048, norm="exact", overlap=True, ring_timing=True)
    warm_ms = loop.warmup(prompt=8, tokens=8)
    n_dev = sum(1 for g in loop._groups if group_is_resident(g))
    print(
        f"warmup_ms={warm_ms:.0f} overflow={report.overflow} n_dev={n_dev} "
        f"pin={_pin_ok(loop)}",
        flush=True,
    )

    phases: list[dict] = []
    n_prefill = int(args.prefill)
    n_decode = int(args.steps)
    wedge_floor = 0.5

    if args.serial_device:
        _set_device_streams(loop, False)
        serial = _time_steps(loop, n_prefill, n_decode)
        serial["phase"] = "eager_serial_device"
        phases.append(serial)
        print(
            f"eager_serial_device tok/s={serial['tok_s']:.3f} copies={serial['h2d_copies']}",
            flush=True,
        )
        _set_device_streams(loop, True)

    for cap_n in _parse_ladder(args.ladder):
        loop.drop_graphs()
        t_cap0 = time.perf_counter()
        if cap_n <= 0:
            mode = loop.capture_graphs(force=False)
            cap_ms = (time.perf_counter() - t_cap0) * 1000.0
            label = "eager_skip"
        else:
            mode = loop.capture_graphs(force=True, max_graphs=cap_n, fork=False)
            cap_ms = (time.perf_counter() - t_cap0) * 1000.0
            label = f"force_{cap_n}"
        run = _time_steps(loop, n_prefill, n_decode)
        run["phase"] = label
        run["capture_ms"] = cap_ms
        run["max_graphs"] = cap_n
        run["capture_mode"] = mode
        run["graph_error"] = loop.graph_error
        phases.append(run)
        print(
            f"{label} tok/s={run['tok_s']:.3f} graph={mode} graphed={run['n_graphed']} "
            f"eager={run['n_eager']} copies={run['h2d_copies']} capture_ms={cap_ms:.0f}",
            flush=True,
        )
        if cap_n > 0 and run["tok_s"] < wedge_floor:
            print(f"wedge tok/s={run['tok_s']:.3f} < {wedge_floor}; stop ladder", flush=True)
            break

    payload = {
        "warmup_ms": warm_ms,
        "overflow": bool(report.overflow),
        "n_dev": n_dev,
        "report": str(report),
        "ladder": args.ladder,
        "phases": phases,
        "ref_overflow_2_49": 2.49,
        "ref_ollama_2_54": 2.54,
    }
    (dest / "probe.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

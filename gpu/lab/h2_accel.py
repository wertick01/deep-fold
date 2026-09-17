"""H2-accel comparison runner: switchable overflow bets on one TokenLoop.

    python -m gpu.lab.h2_accel --variant baseline,verify-k --plan-only
    python -m gpu.lab.h2_accel --force-overflow --variant baseline --max-seq 512

``--plan-only`` is CPU: ``plan_residency`` + expected H2D bytes. No CUDA.
Missing Accel-1/2 APIs become ``status=blocked`` instead of a crash.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.host.residency import (  # noqa: E402
    DEFAULT_POLICY,
    HOST_EMBED_POLICY,
    MIB,
    WeightDesc,
    descs_from_header,
    descs_from_qwen,
    overflow_resident_cap,
    pin_nbytes,
    plan_residency,
    summarize_residency,
)
from gpu.lab.h2_metrics import (  # noqa: E402
    CHR_32B,
    MODEL_32B,
    auto_max_resident_bytes,
    canary_overflow_bytes,
    dump_json,
    h2d_ms,
)
from gpu.lab.h2_place import QWEN_32B  # noqa: E402
from gpu.lab.script import RUNS_DIR  # noqa: E402

__all__ = [
    "SCHEMA",
    "VARIANT_IDS",
    "VARIANT_FIELDS",
    "main",
    "parse_k",
    "parse_variants",
    "plan_variant",
    "run_h2_accel",
    "variant_config",
    "verify_token_budget",
    "write_summary",
]

SCHEMA = "deepfold.h2_accel.v1"
VARIANT_IDS = (
    "baseline",
    "profile",
    "verify-k",
    "spec-lookup",
    "prefill-hold",
    "pairs-stride",
    "host-embed",
)
VARIANT_FIELDS = (
    "id",
    "status",
    "prompt_len",
    "prefill_ms",
    "decode_tok_s",
    "decode_ms_per_tok",
    "h2d_bytes",
    "h2d_copies",
    "h2d_mib_per_fwd",
    "h2d_forwards",
    "copy_floor_ms",
    "copy_ms",
    "smi_mib",
    "residency_policy",
    "prefill_mode",
    "speculate",
    "n_host",
    "resident_mib",
    "smoke",
    "greedy_match_baseline",
    "notes",
)


def parse_variants(text: str) -> list[str]:
    """Comma-separated variant ids. ``all`` is the full matrix."""
    raw = str(text or "").strip()
    if not raw or raw.lower() in ("all", "*"):
        return list(VARIANT_IDS)
    out: list[str] = []
    unknown: list[str] = []
    for part in raw.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in VARIANT_IDS:
            unknown.append(name)
            continue
        if name not in out:
            out.append(name)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown variant(s) {unknown}; expected one of {list(VARIANT_IDS)}"
        )
    if not out:
        raise argparse.ArgumentTypeError("empty --variant")
    return out


def parse_k(text: str) -> list[int]:
    """Comma-separated verify block sizes, default ``2,4,8``."""
    raw = str(text or "").strip()
    if not raw:
        return [2, 4, 8]
    out: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        k = int(token)
        if k < 1:
            raise argparse.ArgumentTypeError(f"k must be >= 1, got {k}")
        if k not in out:
            out.append(k)
    if not out:
        raise argparse.ArgumentTypeError("empty --k")
    return out


def verify_token_budget(k_values: Sequence[int] | None) -> int:
    """Ignore-EOS tokens so the last block of ``max(k)`` is full width.

    The 2026-09-15 plate replayed a 4-token smoke reply, then divided the k=8
    wall by 8. That is not T_verify(8).
    """
    need = max(int(k) for k in (k_values or [8]))
    return max(need * 2, 32)


def variant_config(vid: str) -> dict[str, Any]:
    """Flags one lab variant feeds into load / generate. Default is baseline D."""
    cfg: dict[str, Any] = {
        "id": vid,
        "residency_policy": DEFAULT_POLICY,
        "pin_embed": True,
        "prefill_mode": "chunk",
        "speculate": 1,
        "draft": "none",
        "ring_timing": False,
    }
    if vid == "pairs-stride":
        cfg["residency_policy"] = "pairs_stride"
    elif vid == "host-embed":
        cfg["residency_policy"] = HOST_EMBED_POLICY
        cfg["pin_embed"] = False
    elif vid == "prefill-hold":
        cfg["prefill_mode"] = "hold"
    elif vid == "spec-lookup":
        cfg["speculate"] = 4
        cfg["draft"] = "lookup"
    elif vid == "profile":
        cfg["ring_timing"] = True
    return cfg


def _empty_row(vid: str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or variant_config(vid)
    return {
        "id": vid,
        "status": "ok",
        "prompt_len": None,
        "prefill_ms": None,
        "decode_tok_s": None,
        "decode_ms_per_tok": None,
        "h2d_bytes": None,
        "h2d_copies": None,
        "h2d_mib_per_fwd": None,
        "h2d_forwards": None,
        "copy_floor_ms": None,
        "copy_ms": None,
        "smi_mib": None,
        "residency_policy": cfg["residency_policy"],
        "prefill_mode": cfg["prefill_mode"],
        "speculate": cfg["speculate"],
        "n_host": None,
        "resident_mib": None,
        "smoke": None,
        "greedy_match_baseline": None,
        "notes": "",
        "streamed_bytes": None,
        "host_layers": None,
    }


def _accel1_block() -> str | None:
    try:
        from gpu.loop.ring import CopyRing
    except Exception as exc:  # noqa: BLE001
        return f"CopyRing import failed: {type(exc).__name__}: {exc}"
    if not hasattr(CopyRing, "bind_hold"):
        return "CopyRing.bind_hold missing (Accel-1)"
    try:
        from gpu.loop.generate import TokenLoop

        init_p = inspect.signature(TokenLoop.__init__).parameters
        gen_p = inspect.signature(TokenLoop.generate).parameters
        if "prefill_mode" not in init_p and "prefill_mode" not in gen_p:
            return "TokenLoop prefill_mode missing (Accel-1)"
    except Exception as exc:  # noqa: BLE001
        return f"TokenLoop inspect failed: {type(exc).__name__}: {exc}"
    return None


def _accel2_block(*, spec_lookup: bool = False) -> str | None:
    try:
        import gpu.loop.speculate as spec
    except ImportError:
        return "gpu.loop.speculate missing (Accel-2)"
    if spec_lookup:
        draft = getattr(spec, "draft_ngram", None) or getattr(spec, "spec_lookup", None)
        gen_ok = False
        try:
            from gpu.loop.generate import TokenLoop

            gen_ok = "speculate" in inspect.signature(TokenLoop.generate).parameters
        except Exception:
            gen_ok = False
        if draft is None and not gen_ok:
            return "spec-lookup API missing (Accel-2 generate speculate= / draft)"
    elif not hasattr(spec, "measure_verify") and not hasattr(spec, "verify_block"):
        return "verify_block/measure_verify missing (Accel-2)"
    return None


def _block_reason(vid: str) -> str | None:
    if vid == "prefill-hold":
        return _accel1_block()
    if vid == "verify-k":
        return _accel2_block(spec_lookup=False)
    if vid == "spec-lookup":
        return _accel2_block(spec_lookup=True)
    return None


def _load_descs(chr_path: str, model_dir: str) -> tuple[tuple[WeightDesc, ...], str]:
    chr_file = Path(chr_path) if chr_path else None
    if chr_file is not None and chr_file.is_file():
        from gpu.chr0 import load_header

        return descs_from_header(load_header(str(chr_file))), "chr"
    cfg_path = Path(model_dir) / "config.json" if model_dir else None
    if cfg_path is not None and cfg_path.is_file():
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        return descs_from_qwen(payload), "config"
    return descs_from_qwen(QWEN_32B), "qwen32b-shapes"


def _resolve_cap(
    args: argparse.Namespace,
    descs: Sequence[WeightDesc],
    source: str,
) -> tuple[int, dict[str, Any]]:
    if args.max_resident_mib is not None:
        cap = int(float(args.max_resident_mib) * MIB)
        return cap, {
            "overflow": True,
            "cap_bytes": cap,
            "cap_mib": float(args.max_resident_mib),
            "forced": True,
            "reason": f"forced --max-resident-mib={args.max_resident_mib}",
        }
    if args.force_overflow:
        chr_file = Path(args.chr_path)
        if chr_file.is_file():
            cap = canary_overflow_bytes(args.chr_path)
            reason = "forced --force-overflow (pin + 4*(2*gate))"
        else:
            pin = pin_nbytes(descs)
            gate_n = next((int(d.nbytes) for d in descs if d.kind == "gate"), 0)
            cap = int(pin + 4 * (2 * gate_n))
            reason = "forced --force-overflow on shapes (no .chr); pin + 4*(2*gate)"
        return cap, {
            "overflow": True,
            "cap_bytes": cap,
            "cap_mib": cap / MIB,
            "forced": True,
            "reason": reason,
            "descs": source,
        }
    chr_file = Path(args.chr_path)
    model_dir = Path(args.model)
    if chr_file.is_file() and model_dir.is_dir():
        cap, info = auto_max_resident_bytes(args.model, args.chr_path, args.max_seq)
        if cap is not None:
            info["descs"] = source
            return int(cap), info
    cap = overflow_resident_cap(12288, int(args.max_seq), descs)
    return cap, {
        "overflow": True,
        "cap_bytes": cap,
        "cap_mib": cap / MIB,
        "forced": False,
        "reason": "overflow_resident_cap(12288, max_seq, descs) (no decide/.chr)",
        "descs": source,
    }


def plan_variant(
    vid: str,
    descs: tuple[WeightDesc, ...],
    cap: int,
    *,
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """CPU plan for one variant. Never loads CUDA or a ``.chr`` body."""
    cfg = variant_config(vid)
    row = _empty_row(vid, cfg)
    blocked = _block_reason(vid)
    if blocked:
        row["status"] = "blocked"
        row["notes"] = blocked
    try:
        plan = plan_residency(
            descs,
            int(cap),
            policy=cfg["residency_policy"],
            pin_embed=cfg["pin_embed"],
        )
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error" if row["status"] != "blocked" else row["status"]
        extra = f"{type(exc).__name__}: {exc}"
        row["notes"] = f"{row['notes']}; {extra}".strip("; ") if row["notes"] else extra
        return row
    summary = summarize_residency(descs, plan)
    row["n_host"] = len(plan.streamed)
    row["resident_mib"] = plan.resident_bytes / MIB
    row["streamed_bytes"] = int(plan.streamed_bytes)
    row["h2d_bytes"] = int(plan.streamed_bytes)
    row["h2d_forwards"] = 1
    row["copy_floor_ms"] = h2d_ms(plan.streamed_bytes)
    row["host_layers"] = summary["host_layers"]
    row["cpu"] = sorted(plan.cpu)
    if vid == "profile" and not row["notes"]:
        row["notes"] = "copy_ms=null until a timed generate; plan has floor only"
    if vid == "verify-k":
        row["k"] = list(k_values or [2, 4, 8])
    row["h2d_mib_per_fwd"] = _h2d_mib_per_fwd(row)
    if row["status"] == "ok":
        row["status"] = "ok"
    return row


def write_summary(path: Path, plate: dict[str, Any]) -> None:
    """Human table: variant × prefill / tok/s / H2D / smi / smoke / notes."""
    lines = [
        f"schema {plate.get('schema')}",
        f"plan_only={plate.get('plan_only')} cap_mib={((plate.get('cap') or {}).get('cap_mib'))}",
        f"descs={plate.get('descs_source')} chr={plate.get('chr')}",
        "",
        f"{'variant':<14} {'status':<9} {'pol':<12} {'n_host':>6} "
        f"{'res_MiB':>8} {'h2d/fwd':>10} {'copies':>7} {'floor_ms':>8} {'tok/s':>7} "
        f"{'prefill':>8} {'smoke':<6} notes",
    ]
    for row in plate.get("variants") or []:
        n_host = row.get("n_host")
        res = row.get("resident_mib")
        h2d = _h2d_mib_per_fwd(row)
        copies = row.get("h2d_copies")
        floor = row.get("copy_floor_ms")
        tok = row.get("decode_tok_s")
        pre = row.get("prefill_ms")
        smoke = row.get("smoke")
        lines.append(
            f"{str(row.get('id', '')):<14} {str(row.get('status', '')):<9} "
            f"{str(row.get('residency_policy', '')):<12} "
            f"{'-' if n_host is None else n_host:>6} "
            f"{'-' if res is None else f'{res:.0f}':>8} "
            f"{'-' if h2d is None else f'{h2d:.1f}':>10} "
            f"{'-' if copies is None else copies:>7} "
            f"{'-' if floor is None else f'{floor:.0f}':>8} "
            f"{'-' if tok is None else f'{tok:.2f}':>7} "
            f"{'-' if pre is None else f'{pre:.0f}':>8} "
            f"{'-' if smoke is None else str(smoke):<6} "
            f"{row.get('notes') or ''}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gpu.lab.h2_accel")
    p.add_argument("--model", default=MODEL_32B)
    p.add_argument("--chr", dest="chr_path", default=CHR_32B)
    p.add_argument("--out", default="")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-seq", type=int, default=2048)
    p.add_argument(
        "--variant",
        default="baseline",
        help="comma-separated ids, or all",
    )
    p.add_argument("--k", default="2,4,8", help="verify-k block sizes")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--no-graphs", dest="graphs", action="store_false")
    p.add_argument(
        "--max-resident-mib",
        type=float,
        default=None,
        help="force overflow cap in MiB",
    )
    p.add_argument(
        "--force-overflow",
        action="store_true",
        help="3B canary cap: pin + 4*(2*gate), even if packed NF4 fits",
    )
    return p


def _out_dir(explicit: str) -> Path:
    if explicit:
        dest = Path(explicit)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = Path(RUNS_DIR) / f"h2-accel-{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _h2d_mib_per_fwd(row: dict[str, Any]) -> float | None:
    """MiB copied per TokenLoop forward. Live uses ring totals; plan uses tape size."""
    nbytes = row.get("h2d_bytes")
    if nbytes is None:
        return None
    nfwd = row.get("h2d_forwards")
    if nfwd:
        return float(nbytes) / float(nfwd) / MIB
    return float(nbytes) / MIB


def _apply_plan_fields(
    row: dict[str, Any], plan_row: dict[str, Any], *, live: bool = False
) -> None:
    """Overlay WHO / floor from the plan. Live copy counters stay measured."""
    for key in (
        "n_host",
        "resident_mib",
        "streamed_bytes",
        "copy_floor_ms",
        "host_layers",
        "cpu",
        "k",
    ):
        if plan_row.get(key) is not None:
            row[key] = plan_row[key]
    if not live:
        for key in ("h2d_bytes", "h2d_forwards"):
            if plan_row.get(key) is not None:
                row[key] = plan_row[key]
    if plan_row.get("status") in ("blocked", "error"):
        row["status"] = plan_row["status"]
        row["notes"] = plan_row.get("notes") or row.get("notes") or ""
    row["h2d_mib_per_fwd"] = _h2d_mib_per_fwd(row)


def _run_live_variant(
    vid: str,
    args: argparse.Namespace,
    cap: int,
    baseline_tokens: dict[int, list[int]] | None,
) -> dict[str, Any]:
    from gpu.host import load_model
    from gpu.lab.sampler import smi_used_mib
    from gpu.lab.script import MESSAGES, chat_text, quality_ok, stop_token_ids
    from gpu.loop import TokenLoop
    from transformers import AutoTokenizer

    import torch

    cfg = variant_config(vid)
    row = _empty_row(vid, cfg)
    blocked = _block_reason(vid)
    if blocked:
        row["status"] = "blocked"
        row["notes"] = blocked
        return row

    generate_kw: dict[str, Any] = {}
    loop_kw: dict[str, Any] = {
        "max_seq": int(args.max_seq),
        "overlap": True,
        "ring_timing": bool(cfg["ring_timing"]),
    }
    init_p = inspect.signature(TokenLoop.__init__).parameters
    gen_p = inspect.signature(TokenLoop.generate).parameters
    if cfg["prefill_mode"] != "chunk":
        if "prefill_mode" in init_p:
            loop_kw["prefill_mode"] = cfg["prefill_mode"]
        elif "prefill_mode" in gen_p:
            generate_kw["prefill_mode"] = cfg["prefill_mode"]
        else:
            row["status"] = "blocked"
            row["notes"] = "TokenLoop prefill_mode missing (Accel-1)"
            return row
    if cfg["speculate"] != 1 or cfg["draft"] != "none":
        if "speculate" in gen_p:
            generate_kw["speculate"] = cfg["speculate"]
        if "draft" in gen_p:
            generate_kw["draft"] = cfg["draft"]
        if "speculate" not in gen_p and "draft" not in gen_p:
            row["status"] = "blocked"
            row["notes"] = "TokenLoop.generate speculate/draft missing (Accel-2)"
            return row

    model = None
    loop = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=args.trust_remote_code
        )
        model, report = load_model(
            args.model,
            args.chr_path,
            trust_remote_code=args.trust_remote_code,
            strict=True,
            max_resident_bytes=cap,
            residency_policy=cfg["residency_policy"],
            pin_embed=cfg["pin_embed"],
        )
        row["n_host"] = int(report.streamed)
        row["resident_mib"] = report.resident_bytes / MIB
        row["streamed_bytes"] = int(report.streamed_bytes)
        row["copy_floor_ms"] = h2d_ms(report.streamed_bytes)
        loop = TokenLoop(model, **loop_kw)
        loop.warmup(prompt=8, tokens=min(8, int(args.max_new_tokens)))
        if args.graphs:
            loop.capture_graphs()
        stop = stop_token_ids(tokenizer)
        hits = 0
        tok_s: list[float] = []
        prefill: list[float] = []
        smi_series: list[float | None] = []
        match_ok: list[bool] = []
        h2d_b = 0
        h2d_c = 0
        h2d_f = 0
        copy_ms = 0.0
        prompt_len = None
        for index, prompt in enumerate(MESSAGES, start=1):
            packed = chat_text(tokenizer, prompt)
            ids = tokenizer(packed, return_tensors="pt", add_special_tokens=False).input_ids[0]
            run = loop.generate(ids, int(args.max_new_tokens), stop=stop, **generate_kw)
            text = tokenizer.decode(run.tokens, skip_special_tokens=True)
            quality = quality_ok(index, text)
            hits += int(quality)
            prompt_len = run.prompt_len
            prefill.append(float(run.prefill_ms))
            if run.decode_steps:
                tok_s.append(float(run.decode_tok_s))
            smi_series.append(smi_used_mib())
            h2d_b += int(run.h2d_bytes)
            h2d_c += int(run.h2d_copies)
            h2d_f += int(run.h2d_forwards)
            copy_ms += float(run.h2d_copy_ms)
            if baseline_tokens is not None and index in baseline_tokens:
                match_ok.append(list(run.tokens) == list(baseline_tokens[index]))
            if baseline_tokens is not None and vid == "baseline":
                baseline_tokens[index] = list(run.tokens)
        if vid == "verify-k":
            try:
                from gpu.lab.script import LONG_PROMPT
                from gpu.loop.speculate import (
                    format_verify_note,
                    measure_verify,
                    verify_stats,
                )

                n_verify = verify_token_budget(parse_k(args.k))
                packed = chat_text(tokenizer, LONG_PROMPT)
                ids = tokenizer(
                    packed, return_tensors="pt", add_special_tokens=False
                ).input_ids[0]
                long_run = loop.generate(ids, n_verify, stop=(), **generate_kw)
                verify_tokens = list(long_run.tokens)
                verifies = []
                for k in parse_k(args.k):
                    loop.reset()
                    loop.prefill(ids)
                    item = measure_verify(loop, verify_tokens, k)
                    stats = verify_stats(item)
                    item["ms_per_token"] = stats["ms_per_token"]
                    item["avg_block_ms"] = stats["avg_block_ms"]
                    item["measured_width"] = stats["measured_width"]
                    item["full_k"] = stats["full_k"]
                    verifies.append(item)
                row["verify"] = verifies
                row["verify_n_tokens"] = len(verify_tokens)
                bits = [format_verify_note(item) for item in verifies]
                if bits:
                    extra = "; ".join(bits)
                    row["notes"] = (
                        f"{row['notes']}; {extra}".strip("; ") if row["notes"] else extra
                    )
            except Exception as exc:  # noqa: BLE001
                extra = f"measure_verify: {type(exc).__name__}: {exc}"
                row["notes"] = (
                    f"{row['notes']}; {extra}".strip("; ") if row["notes"] else extra
                )
        used = [x for x in smi_series if x is not None]
        row["prompt_len"] = prompt_len
        row["prefill_ms"] = sum(prefill) / len(prefill) if prefill else None
        row["decode_tok_s"] = sum(tok_s) / len(tok_s) if tok_s else None
        row["decode_ms_per_tok"] = (
            (1000.0 / row["decode_tok_s"]) if row["decode_tok_s"] else None
        )
        row["h2d_bytes"] = h2d_b
        row["h2d_copies"] = h2d_c
        row["h2d_forwards"] = h2d_f
        row["copy_ms"] = copy_ms if cfg["ring_timing"] else None
        row["smi_mib"] = max(used) if used else None
        row["smoke"] = f"{hits}/{len(MESSAGES)}"
        if match_ok:
            row["greedy_match_baseline"] = all(match_ok)
        elif vid == "baseline":
            row["greedy_match_baseline"] = True
        if vid == "profile" and row["copy_ms"] is None:
            row["notes"] = "copy_ms=null (ring_timing off or no copies)"
        return row
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["notes"] = f"{type(exc).__name__}: {exc}"
        return row
    finally:
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
        del loop, model
        try:
            import gc

            gc.collect()
            import torch as _torch

            _torch.cuda.empty_cache()
        except Exception:
            pass


def run_h2_accel(args: argparse.Namespace) -> int:
    try:
        variants = parse_variants(args.variant)
        k_values = parse_k(args.k)
    except argparse.ArgumentTypeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = _out_dir(args.out)
    descs, source = _load_descs(args.chr_path, args.model)
    cap, cap_info = _resolve_cap(args, descs, source)
    plate: dict[str, Any] = {
        "schema": SCHEMA,
        "out_dir": str(out),
        "model": args.model,
        "chr": args.chr_path,
        "plan_only": bool(args.plan_only),
        "max_seq": int(args.max_seq),
        "max_new_tokens": int(args.max_new_tokens),
        "force_overflow": bool(args.force_overflow),
        "descs_source": source,
        "cap": cap_info,
        "k": k_values,
        "variants": [],
    }
    rows: list[dict[str, Any]] = []
    if args.plan_only:
        for vid in variants:
            rows.append(plan_variant(vid, descs, cap, k_values=k_values))
        plate["variants"] = rows
        dump_json(out / "plate.json", plate)
        write_summary(out / "SUMMARY.txt", plate)
        print(f"plan-only wrote {out}", flush=True)
        errors = [r for r in rows if r.get("status") == "error"]
        return 1 if errors else 0

    baseline_tokens: dict[int, list[int]] = {}
    ordered = list(variants)
    if "baseline" in ordered:
        ordered = ["baseline"] + [v for v in ordered if v != "baseline"]
    for vid in ordered:
        plan_row = plan_variant(vid, descs, cap, k_values=k_values)
        if plan_row.get("status") in ("blocked", "error"):
            rows.append(plan_row)
            print(f"{vid}: {plan_row['status']} {plan_row.get('notes')}", flush=True)
            continue
        print(f"{vid}: load + generate...", flush=True)
        live = _run_live_variant(vid, args, cap, baseline_tokens)
        _apply_plan_fields(live, plan_row, live=True)
        rows.append(live)
        print(
            f"{vid}: status={live.get('status')} tok/s={live.get('decode_tok_s')} "
            f"notes={live.get('notes')}",
            flush=True,
        )
    plate["variants"] = rows
    dump_json(out / "plate.json", plate)
    write_summary(out / "SUMMARY.txt", plate)
    print(f"artifacts: {out}", flush=True)
    if any(r.get("status") == "error" for r in rows):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = _parser().parse_args(argv)
    return run_h2_accel(args)


if __name__ == "__main__":
    raise SystemExit(main())

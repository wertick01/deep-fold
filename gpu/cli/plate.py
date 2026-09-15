"""Neighbor measurement plate: one model in, machine + smoke metrics out.

Not a benchmark and not the BF16 lab. NF4 only (VQ is the kernel oracle).
Hard-12 is not in this runner.

    python -m gpu.cli.plate 3b
    python -m gpu.cli.plate Qwen/Qwen2.5-3B-Instruct
    python -m gpu.cli.plate D:\\weights\\Qwen2.5-3B-Instruct --dry-run

Wrappers: ``scripts/plate.ps1`` / ``scripts/plate.sh``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from . import hub, messages, run as run_mod  # noqa: E402
from .arch import gate, missing_internlm_extras  # noqa: E402
from .codec import CodecFitError, decide, detect_vram_mib, load_config  # noqa: E402
from .doctor import checks, exit_code, probe, verdict  # noqa: E402
from .ollama_map import ResolveError, all_rows, hf_id_list, resolve, row_for_hf  # noqa: E402
from .paths import (  # noqa: E402
    REPO,
    cached_chr,
    find_chr_bin,
    find_chr_file,
    looks_like_gguf,
    runs_root,
    slug,
)

_JOIN_ENV = "DEEPFOLD_COPY_JOIN"

SCHEMA = "deepfold.plate.v1"
NF4_FAIL_RMSE = 0.12
NF4_FAIL_MAXABS = 2.0

SHORT = {
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "14b": "Qwen/Qwen2.5-14B-Instruct",
    "20b": "internlm/internlm2_5-20b-chat",
    "32b": "Qwen/Qwen2.5-32B-Instruct",
}

__all__ = [
    "SCHEMA",
    "SHORT",
    "ModelSpec",
    "main",
    "resolve_spec",
]


@dataclass(frozen=True)
class ModelSpec:
    """What the user named, before any download."""

    raw: str
    hf_id: str | None
    path: Path | None
    disk_gb: float | None


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def _say(text: str) -> None:
    print(text, flush=True)


def resolve_spec(text: str) -> ModelSpec:
    """HF id, short alias, Ollama tag, or a local HuggingFace directory.

    Never opens GGUF. Unknown Hub ids (not on the allowlist) are refused unless
    the argument is an existing directory with ``config.json``.
    """
    raw = (text or "").strip()
    if not raw:
        raise ResolveError("unknown", raw)
    if looks_like_gguf(raw):
        raise ResolveError("gguf", raw)

    path = Path(raw)
    if path.is_dir() and (path / "config.json").is_file():
        leaf = path.name
        row = next(
            (candidate for candidate in all_rows() if candidate.hf_id.rsplit("/", 1)[-1] == leaf),
            None,
        )
        return ModelSpec(
            raw=raw,
            hf_id=None if row is None else row.hf_id,
            path=path,
            disk_gb=None if row is None else row.disk_gb,
        )

    key = raw.lower() if raw.lower() in SHORT else raw
    if key in SHORT:
        hf_id = SHORT[key]
        row = row_for_hf(hf_id)
        assert row is not None
        return ModelSpec(raw=raw, hf_id=row.hf_id, path=None, disk_gb=row.disk_gb)

    row = row_for_hf(raw)
    if row is not None:
        return ModelSpec(raw=raw, hf_id=row.hf_id, path=None, disk_gb=row.disk_gb)

    try:
        mapped = resolve(raw)
    except ResolveError:
        raise ResolveError("unknown", raw) from None
    return ModelSpec(
        raw=raw, hf_id=mapped.hf_id, path=None, disk_gb=mapped.disk_gb
    )


def _git() -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        return out.stdout.strip()

    commit = _run("rev-parse", "HEAD")
    porcelain = _run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(porcelain) if porcelain is not None else None,
        "describe": _run("describe", "--always", "--dirty"),
    }


def _ram() -> dict[str, Any]:
    if os.name == "nt":
        from gpu.lab.h2_metrics import ram_snapshot

        return ram_snapshot()
    out: dict[str, Any] = {}
    try:
        text = Path("/proc/meminfo").read_text(encoding="ascii")
    except OSError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    fields: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        num = rest.strip().split()[0]
        try:
            fields[name] = int(num)
        except ValueError:
            continue
    if "MemTotal" in fields:
        out["ram_total_mib"] = fields["MemTotal"] / 1024.0
    if "MemAvailable" in fields:
        out["ram_avail_mib"] = fields["MemAvailable"] / 1024.0
    return out


def _smi() -> dict[str, Any]:
    from gpu.lab.h2_metrics import smi_snapshot

    return smi_snapshot()


def _machine_blob() -> dict[str, Any]:
    m = probe()
    v = verdict(m)
    report = checks(m, v)
    return {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "git": _git(),
        "doctor": {
            "generate": v.generate,
            "arch": v.arch,
            "line": v.line,
            "exit_code": exit_code(m, v, report),
            "sm": m.sm,
            "device_name": m.device_name,
            "vram_total_mib": m.vram_total_mib,
            "smi_used_mib": m.smi_used_mib,
            "torch": m.torch,
            "torch_cuda": m.torch_cuda,
            "cuda_available": m.cuda_available,
            "chr_bin": None if m.chr_bin is None else str(m.chr_bin),
            "chr_runs": m.chr_runs,
            "nf4_ext": None if m.nf4_ext is None else str(m.nf4_ext),
            "host_cc": m.host_cc,
            "nvcc": m.nvcc,
            "checks": [str(c) for c in report],
        },
        "smi": _smi(),
        "ram": _ram(),
        "env": {
            "DEEPFOLD_MODELS": os.environ.get("DEEPFOLD_MODELS"),
            "DEEPFOLD_HOME": os.environ.get("DEEPFOLD_HOME"),
            "DEEPFOLD_RUNS": os.environ.get("DEEPFOLD_RUNS"),
            "DEEPFOLD_COPY_JOIN": os.environ.get(_JOIN_ENV),
            "DEEPFOLD_CHR_BIN": os.environ.get("DEEPFOLD_CHR_BIN"),
        },
    }


def _has_shards(model_dir: Path) -> bool:
    try:
        return any(model_dir.glob("*.safetensors"))
    except OSError:
        return False


def _find_nf4(model_dir: Path, explicit: str | None) -> Path | None:
    matcher = run_mod._header_matcher(model_dir)

    def accept(path: Path) -> bool:
        if run_mod._file_codec(path) != "nf4":
            return False
        if matcher is None:
            return False
        return bool(matcher(path))

    if explicit:
        p = Path(explicit)
        if not p.is_file():
            return None
        if run_mod._file_codec(p) == "vq":
            return None
        return p
    return find_chr_file(model_dir, None, accept=accept)


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _write_summary(out_dir: Path, payload: dict[str, Any]) -> None:
    doc = payload.get("doctor") or payload.get("machine", {}).get("doctor") or {}
    smoke = payload.get("smoke") or {}
    verify = payload.get("verify") or {}
    lines = [
        f"schema: {SCHEMA}",
        f"utc: {payload.get('utc')}",
        f"git: {payload.get('machine', {}).get('git', {}).get('describe')}",
        f"gpu: {doc.get('device_name')}  {doc.get('sm')}  {doc.get('arch')}  "
        f"{doc.get('vram_total_mib')} MiB",
        f"os: {payload.get('machine', {}).get('os', {}).get('platform')}",
        f"python: {payload.get('machine', {}).get('python', {}).get('version')}  "
        f"{payload.get('machine', {}).get('python', {}).get('executable')}",
        f"torch: {doc.get('torch')}  cuda {doc.get('torch_cuda')}",
        f"model: {payload.get('hf_id') or payload.get('model_dir')}",
        f"chr: {payload.get('chr')}",
        f"cpu_tests: {payload.get('cpu_tests', {}).get('status')}",
        f"verify: {verify.get('status')}  summary={verify.get('summary')}",
        f"smoke_quality: {smoke.get('quality_hits')}/{smoke.get('quality_n')}",
        f"mean_ttft_ms: {smoke.get('mean_ttft_ms')}",
        f"mean_decode_tok_s: {smoke.get('mean_decode_tok_s')}",
        f"vram_after_load_smi_mib: {smoke.get('vram_after_load_smi_mib')}",
        f"weight_mib: {smoke.get('weight_mib')}",
        f"notes: {smoke.get('notes') or payload.get('error') or ''}",
        "",
        "This is greedy smoke (Paris / Berlin / 323), not a quality benchmark.",
        "Do not compare tok/s to the author's RTX 3080 plate.",
        "Send this folder back: SUMMARY.txt, plate.json, and nf4/*.csv if present.",
    ]
    (out_dir / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m gpu.cli.plate",
        description=(
            "Run CPU tests, pack NF4, verify, and greedy smoke on one allowlisted "
            "model. Writes plate.json for a neighbor PC. Not BF16, not VQ, not hard-12."
        ),
    )
    ap.add_argument(
        "model",
        help="3b/14b/20b/32b, HuggingFace id, Ollama tag, or a local HF directory",
    )
    ap.add_argument("--dir", help="download destination (same as deepfold pull --dir)")
    ap.add_argument("--chr", help="existing NF4 .chr (skip compress when it matches)")
    ap.add_argument("--chr-bin", help="path to chr / chr.exe")
    ap.add_argument(
        "--out",
        help="run directory (default: $DEEPFOLD_RUNS/plate-<slug>-<timestamp>)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=64, help="greedy cap per smoke turn (default: 64)")
    ap.add_argument("--max-seq", type=int, default=512, help="KV length (default: 512)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and write plate.json; no pull/compress/generate",
    )
    ap.add_argument("--skip-test", action="store_true", help="skip gpu/cli/test_cli.py")
    ap.add_argument("--skip-verify", action="store_true", help="skip chr verify")
    ap.add_argument("--skip-generate", action="store_true", help="stop after verify")
    ap.add_argument(
        "--no-download",
        action="store_true",
        help="do not Hub-pull; fail if the tree is missing",
    )
    ap.add_argument("--force", action="store_true", help="repack .chr even if one exists")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        spec = resolve_spec(args.model)
    except ResolveError as exc:
        if exc.kind == "gguf":
            _err(messages.GGUF)
            return 1
        _err(messages.unknown_hf_id(exc.tag, hf_id_list()))
        _err("Aliases: 3b, 14b, 20b, 32b. Or pass a local directory with config.json.")
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    leaf = (spec.hf_id or spec.raw).rsplit("/", 1)[-1]
    out_dir = (
        Path(args.out)
        if args.out
        else runs_root() / f"plate-{slug(leaf)}-{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plate: dict[str, Any] = {
        "schema": SCHEMA,
        "utc": datetime.now(timezone.utc).isoformat(),
        "raw": spec.raw,
        "hf_id": spec.hf_id,
        "out": str(out_dir),
        "flags": {
            "dry_run": bool(args.dry_run),
            "skip_test": bool(args.skip_test),
            "skip_verify": bool(args.skip_verify),
            "skip_generate": bool(args.skip_generate),
            "no_download": bool(args.no_download),
            "force": bool(args.force),
            "max_new_tokens": int(args.max_new_tokens),
            "max_seq": int(args.max_seq),
        },
        "steps": [],
    }

    def save() -> None:
        _dump(out_dir / "plate.json", plate)
        _write_summary(out_dir, {**plate, "doctor": plate.get("machine", {}).get("doctor")})

    def step(name: str, status: str, **extra: Any) -> None:
        row = {"name": name, "status": status, **extra}
        plate["steps"].append(row)
        _say(f"[{status}] {name}" + (f"  {extra.get('detail', '')}" if extra.get("detail") else ""))
        save()

    _say(f"plate out: {out_dir}")
    _say(f"model: {spec.hf_id or spec.path or spec.raw}")

    try:
        plate["machine"] = _machine_blob()
        step("machine", "ok", detail=plate["machine"]["doctor"].get("line", ""))
    except Exception as exc:  # noqa: BLE001
        plate["machine"] = {"error": f"{type(exc).__name__}: {exc}"}
        step("machine", "error", detail=str(exc))

    if args.dry_run:
        dest = spec.path
        if dest is None and spec.hf_id:
            row = row_for_hf(spec.hf_id)
            if row is not None:
                dest = hub.pull_destination(row, args.dir)
        plate["model_dir"] = None if dest is None else str(dest)
        plate["plan"] = [
            "cpu tests (gpu/cli/test_cli.py)" if not args.skip_test else "skip cpu tests",
            "pull HuggingFace BF16 if missing" if not args.no_download else "no download",
            "chr compress NF4 if no matching .chr",
            "chr verify orig vs .chr" if not args.skip_verify else "skip verify",
            "NF4 greedy smoke Paris/Berlin/323" if not args.skip_generate else "skip generate",
        ]
        step("dry-run", "ok", detail="; ".join(plate["plan"]))
        save()
        _say("dry-run: nothing downloaded or generated.")
        return 0

    if not args.skip_test:
        t0 = time.perf_counter()
        script = REPO / "gpu" / "cli" / "test_cli.py"
        code = int(subprocess.call([sys.executable, str(script)]))
        plate["cpu_tests"] = {
            "status": "PASS" if code == 0 else "FAIL",
            "exit_code": code,
            "seconds": round(time.perf_counter() - t0, 2),
        }
        step("cpu_tests", "ok" if code == 0 else "fail", detail=plate["cpu_tests"]["status"])
        if code != 0:
            save()
            return 1
    else:
        plate["cpu_tests"] = {"status": "SKIP"}
        step("cpu_tests", "skip")

    model_dir = spec.path
    if model_dir is None:
        if spec.hf_id is None:
            _err("plate: no HuggingFace id and no local directory.")
            step("resolve", "fail")
            save()
            return 1
        row = row_for_hf(spec.hf_id)
        if row is None:
            _err(messages.unknown_hf_id(spec.hf_id, hf_id_list()))
            step("resolve", "fail")
            save()
            return 1
        dest = hub.pull_destination(row, args.dir)
        already = hub.source_complete(dest)
        if already:
            model_dir = dest
            step("pull", "skip", detail=f"already at {dest}")
        elif args.no_download:
            _err(f"plate: no tree at {dest} and --no-download was set.")
            step("pull", "fail", detail="missing tree")
            save()
            return 1
        else:
            if hub.hub_missing():
                _err(messages.NEED_HUB)
                step("pull", "fail", detail="huggingface_hub missing")
                save()
                return 1
            _say(
                f"Downloading {row.hf_id} (~{row.disk_gb:g} GB BF16) to {dest}"
            )
            dest.mkdir(parents=True, exist_ok=True)
            try:
                hub.snapshot_download(repo_id=row.hf_id, local_dir=os.fspath(dest))
            except Exception as exc:  # noqa: BLE001
                _err(f"pull failed ({type(exc).__name__}: {exc}).")
                step("pull", "fail", detail=str(exc))
                save()
                return 1
            model_dir = dest
            step("pull", "ok", detail=str(dest))
    else:
        step("pull", "skip", detail=f"local {model_dir}")

    plate["model_dir"] = str(model_dir)
    checked = gate(str(model_dir))
    if not checked.ok:
        _err(checked.reason)
        step("gate", "fail", detail=checked.reason.splitlines()[0])
        save()
        return 1
    if checked.note:
        _err(checked.note)
    if checked.model_type == "internlm2":
        missing = missing_internlm_extras()
        if missing:
            _err(messages.INTERNLM_EXTRAS)
            step("internlm_extra", "fail", detail=", ".join(missing))
            save()
            return 1
    step("gate", "ok", detail=checked.model_type or "")

    chr_bin = find_chr_bin(args.chr_bin)
    if chr_bin is None:
        _err(messages.missing_chr())
        step("chr_bin", "fail")
        save()
        return 1

    chr_path = None if args.force else _find_nf4(model_dir, args.chr)
    if chr_path is None:
        vram = int(detect_vram_mib())
        try:
            decision = decide(load_config(str(model_dir)), vram, requested="auto")
        except CodecFitError as exc:
            _err(str(exc))
            step("compress", "fail", detail=str(exc))
            save()
            return 1
        codec = decision.codec
        _err(decision.reason)
        out_chr = cached_chr(str(model_dir), codec)
        t0 = time.perf_counter()
        code = run_mod.compress_to(
            str(model_dir), out_chr, chr_bin, quiet=args.quiet, codec=codec
        )
        plate["compress"] = {
            "exit_code": code,
            "codec": codec,
            "overflow": bool(decision.overflow),
            "seconds": round(time.perf_counter() - t0, 1),
            "path": str(out_chr),
        }
        if code != 0:
            step("compress", "fail", detail=f"chr exit {code}")
            save()
            return 1
        chr_path = out_chr
        step("compress", "ok", detail=str(chr_path))
    else:
        plate["compress"] = {"status": "skip", "path": str(chr_path)}
        step("compress", "skip", detail=str(chr_path))

    plate["chr"] = str(chr_path)

    if args.skip_verify:
        plate["verify"] = {"status": "SKIP"}
        step("verify", "skip")
    elif not _has_shards(model_dir):
        plate["verify"] = {
            "status": "SKIP",
            "reason": "no *.safetensors left next to config.json",
        }
        step("verify", "skip", detail="no shards")
    else:
        verify_json = out_dir / "verify.json"
        cmd = [
            str(chr_bin),
            "verify",
            "--orig",
            str(model_dir),
            "--chr",
            str(chr_path),
            "--fail-rmse",
            str(NF4_FAIL_RMSE),
            "--fail-maxabs",
            str(NF4_FAIL_MAXABS),
            "--json",
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        raw = proc.stdout or ""
        try:
            parsed = json.loads(raw) if raw.strip().startswith("{") else {}
        except json.JSONDecodeError:
            parsed = {}
        if parsed:
            verify_json.write_text(
                json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        summary = parsed.get("summary") if isinstance(parsed, dict) else None
        plate["verify"] = {
            "status": "PASS" if proc.returncode == 0 else ("FAIL" if proc.returncode == 2 else "ERROR"),
            "exit_code": proc.returncode,
            "seconds": round(time.perf_counter() - t0, 1),
            "ok": parsed.get("ok") if isinstance(parsed, dict) else None,
            "worst": parsed.get("worst") if isinstance(parsed, dict) else None,
            "summary": summary,
            "stderr": (proc.stderr or "")[-2000:],
        }
        step("verify", "ok" if proc.returncode == 0 else "fail", detail=plate["verify"]["status"])
        if proc.returncode not in (0, 2):
            save()
            return 1
        if proc.returncode == 2:
            save()
            return 2

    if args.skip_generate:
        plate["smoke"] = {"status": "SKIP"}
        step("generate", "skip")
        save()
        return 0

    doc = plate.get("machine", {}).get("doctor") or {}
    if doc.get("exit_code") != 0:
        plate["smoke"] = {
            "status": "SKIP",
            "reason": doc.get("line") or "doctor is not 0; generate skipped",
        }
        step("generate", "skip", detail=plate["smoke"]["reason"])
        save()
        return 0

    from gpu.lab.script import MESSAGES, quality_ok
    from gpu.lab.sessions import run_nf4

    t0 = time.perf_counter()
    try:
        session = run_nf4(
            out_dir=out_dir,
            model_dir=str(model_dir),
            chr_path=str(chr_path),
            max_new_tokens=int(args.max_new_tokens),
            max_seq=int(args.max_seq),
            trust_remote_code=bool(checked.trust_remote_code),
            isolated=True,
            verbose=not args.quiet,
        )
    except Exception as exc:  # noqa: BLE001
        plate["smoke"] = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.perf_counter() - t0, 1),
        }
        plate["error"] = plate["smoke"]["error"]
        step("generate", "fail", detail=str(exc)[:200])
        save()
        return 1

    replies = []
    hits = 0
    for row in session.messages:
        mid = int(row.get("message_id") or 0)
        text = str(row.get("response") or "")
        ok = bool(row.get("quality_ok")) if "quality_ok" in row else quality_ok(mid, text)
        if ok:
            hits += 1
        replies.append(
            {
                "message_id": mid,
                "prompt": MESSAGES[mid - 1] if 1 <= mid <= len(MESSAGES) else row.get("prompt"),
                "response": text,
                "quality_ok": ok,
                "ttft_ms": row.get("ttft_ms"),
                "decode_tok_s": row.get("decode_tok_s"),
            }
        )
    summary = dict(session.summary or {})
    plate["smoke"] = {
        "status": "PASS" if hits == len(MESSAGES) else "FAIL",
        "seconds": round(time.perf_counter() - t0, 1),
        "quality_hits": hits,
        "quality_n": len(MESSAGES),
        "quality_all_ok": summary.get("quality_all_ok"),
        "mean_ttft_ms": summary.get("mean_ttft_ms"),
        "mean_decode_tok_s": summary.get("mean_decode_tok_s"),
        "vram_peak_smi_mib": summary.get("vram_peak_smi_mib"),
        "vram_after_load_smi_mib": summary.get("vram_after_load_smi_mib"),
        "weight_mib": summary.get("weight_mib"),
        "notes": summary.get("notes"),
        "replies": replies,
    }
    step(
        "generate",
        "ok" if hits == len(MESSAGES) else "fail",
        detail=f"smoke {hits}/{len(MESSAGES)}  "
        f"{summary.get('mean_decode_tok_s')} tok/s  "
        f"ttft {summary.get('mean_ttft_ms')} ms",
    )
    save()
    _say(f"\nDone. Send {out_dir} (SUMMARY.txt + plate.json).")
    return 0 if hits == len(MESSAGES) else 2


if __name__ == "__main__":
    raise SystemExit(main())

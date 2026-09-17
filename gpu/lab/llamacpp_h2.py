"""llama.cpp CUDA Q4_K_M plate on the same 32B Instruct smoke as H2.

Uses the official Windows CUDA ``llama-server`` binary, not ``llama-cpp-python``
(that wheel fails Windows long paths here). Does not import torch or
``gpu.nf4``. Does not pip into conda ``torch-gpu``.

    python -m gpu.lab.llamacpp_h2 --plan-only
    python -m gpu.lab.llamacpp_h2
    python -m gpu.lab.llamacpp_h2 --bench

    Default artifact: bartowski ``Qwen2.5-32B-Instruct-Q4_K_M.gguf`` (~18.5 GiB).
    Live default is one server slot (not llama-server's auto 4) plus an
    ``ignore_eos`` decode plateau so tok/s is not a 4-token EOS sample.
    Product default generate is unchanged. ``--ngl`` default is ``-1`` (omit
    ``--n-gpu-layers``, llama-server auto-fit). ``--ngl 99`` reproduces the
    old fill-card launch that aborted auto-fit (32B long **1.52**).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab.compare import upsert
from gpu.lab.script import MESSAGES, NEEDLES, RUNS_DIR, quality_ok

__all__ = ["EXPECTED_GGUF_BYTES", "main"]

DEFAULT_CLI = Path(r"C:\dev\models\llama.cpp\llama-server.exe")
DEFAULT_GGUF = Path(r"C:\dev\models\gguf\Qwen2.5-32B-Instruct-Q4_K_M.gguf")
EXPECTED_GGUF_BYTES = 19_851_336_576
SCHEMA = "deepfold.llamacpp_h2.v1"
LONG_PROMPT = (
    "Write a long travelogue about rivers, forests, and cities. "
    "Keep going with more sentences."
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gpu.lab.llamacpp_h2")
    p.add_argument("--server", default=str(DEFAULT_CLI), help="llama-server.exe")
    p.add_argument("--gguf", default=str(DEFAULT_GGUF))
    p.add_argument("--out", default="")
    p.add_argument("--ctx", type=int, default=2048, help="match H2 32B smoke max_seq")
    p.add_argument("--n-predict", type=int, default=64)
    p.add_argument("--ngl", type=int, default=-1, help="-1 omit --n-gpu-layers (auto-fit); 99 = old fill-card launch")
    p.add_argument("--parallel", type=int, default=1, help="llama-server slots; 1 = one KV")
    p.add_argument("--long-n", type=int, default=64, help="ignore_eos plateau tokens; 0 disables")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--load-timeout", type=int, default=1200, help="seconds waiting /health")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--bench", action="store_true", help="llama-bench pp512/tg after the server plate")
    p.add_argument("--bench-reps", type=int, default=3)
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    return p


def _smi_used_mib() -> float | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    try:
        return float(line.split(",")[0].strip())
    except ValueError:
        return None


def _gguf_ok(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing {path}"
    size = path.stat().st_size
    if size < 1_000_000_000:
        return False, f"{path} is {size} bytes (HTML/error page, not a GGUF)"
    if path.name == DEFAULT_GGUF.name and size != EXPECTED_GGUF_BYTES:
        return False, f"incomplete download: {size} / {EXPECTED_GGUF_BYTES} bytes"
    if size != EXPECTED_GGUF_BYTES:
        return True, f"{path} size={size} (not the bartowski Q4_K_M length {EXPECTED_GGUF_BYTES})"
    return True, f"{path} size={size} bartowski Q4_K_M"


def _out_dir(explicit: str) -> Path:
    if explicit:
        dest = Path(explicit)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = Path(RUNS_DIR) / f"llamacpp-h2-{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _size_and_model(gguf_name: str) -> tuple[str, str]:
    """Label from the filename. Check 32B before 3B: '32B' contains '3B'."""
    name = gguf_name.upper().replace("_", "-")
    if "32B" in name:
        return "32B", "Qwen2.5-32B-Instruct"
    if "20B" in name:
        return "20B", "internlm2_5-20b-chat"
    if "3B" in name:
        return "3B", "Qwen2.5-3B-Instruct"
    return "unknown", Path(gguf_name).stem


def _compare_id(size: str, ngl: int) -> str:
    """Keep the ngl-99 row id stable; auto-fit is a separate compare.json line."""
    if int(ngl) < 0:
        return f"llamacpp-q4-{size}-Q4_K_M-autofit"
    if int(ngl) == 99:
        return f"llamacpp-q4-{size}-Q4_K_M"
    return f"llamacpp-q4-{size}-Q4_K_M-ngl{int(ngl)}"


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    gguf = Path(args.gguf)
    server = Path(args.server)
    ok, note = _gguf_ok(gguf)
    return {
        "schema": SCHEMA,
        "plan_only": True,
        "server": str(server),
        "server_exists": server.is_file(),
        "gguf": str(gguf),
        "gguf_ok": ok,
        "gguf_note": note,
        "ctx": int(args.ctx),
        "n_predict": int(args.n_predict),
        "ngl": int(args.ngl),
        "ngl_omitted": int(args.ngl) < 0,
        "parallel": int(args.parallel),
        "long_n": int(args.long_n),
        "bench": bool(args.bench),
        "prompts": list(MESSAGES),
        "quant": "Q4_K_M",
        "model": "Qwen2.5-32B-Instruct",
        "source": "bartowski/Qwen2.5-32B-Instruct-GGUF",
        "llama_cpp": "ggml-org/llama.cpp b10964 win-cuda-12.4",
    }


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 600.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _server_cmd(
    server: str | Path,
    gguf: str | Path,
    *,
    ctx: int,
    ngl: int,
    host: str,
    port: int,
    parallel: int,
) -> list[str]:
    """llama-server argv. Negative ``ngl`` omits ``--n-gpu-layers`` (Ollama-style auto-fit)."""
    cmd = [
        str(server),
        "--model",
        str(gguf),
        "--ctx-size",
        str(int(ctx)),
        "--host",
        str(host),
        "--port",
        str(int(port)),
        "--parallel",
        str(int(parallel)),
        "--jinja",
        "--no-webui",
        "--perf",
        "-lv",
        "1",
    ]
    if int(ngl) >= 0:
        cmd.extend(["--n-gpu-layers", str(int(ngl))])
    return cmd


def _wait_health(base: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            payload = _http_json(f"{base}/health", timeout=5)
            status = str(payload.get("status") or payload.get("error") or payload)
            if payload.get("status") in ("ok", "healthy") or int(payload.get("code") or 0) == 200:
                return
            last = status
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code == 200:
                return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last = str(exc)
        time.sleep(2)
    raise RuntimeError(f"llama-server /health not ok after {timeout_s}s: {last}")


def _chat(
    base: str,
    prompt: str,
    n_predict: int,
    *,
    ignore_eos: bool = False,
    cache_prompt: bool = False,
) -> dict[str, Any]:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": int(n_predict),
        "cache_prompt": bool(cache_prompt),
        "stream": False,
    }
    if ignore_eos:
        body["ignore_eos"] = True
    t0 = time.perf_counter()
    payload = _http_json(f"{base}/v1/chat/completions", body, timeout=900)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    payload["_wall_ms"] = wall_ms
    return payload


def _reply_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    return str(msg.get("content") or "")


def _timings(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("timings") or {}
    prompt_ms = float(raw.get("prompt_ms") or 0.0)
    pred_ms = float(raw.get("predicted_ms") or 0.0)
    pred_n = int(raw.get("predicted_n") or 0)
    tok_s = float(raw.get("predicted_per_second") or 0.0)
    if tok_s <= 0 and pred_ms > 0 and pred_n > 0:
        tok_s = pred_n / (pred_ms / 1000.0)
    return {
        "prompt_n": int(raw.get("prompt_n") or 0),
        "prompt_ms": prompt_ms,
        "predicted_n": pred_n,
        "predicted_ms": pred_ms,
        "decode_tok_s": tok_s,
        "wall_ms": float(payload.get("_wall_ms") or 0.0),
        "raw": raw,
    }


def run_live(args: argparse.Namespace, dest: Path) -> dict[str, Any]:
    gguf = Path(args.gguf)
    server = Path(args.server)
    ok, note = _gguf_ok(gguf)
    if not server.is_file():
        raise FileNotFoundError(server)
    if not ok:
        raise FileNotFoundError(note)

    base = f"http://{args.host}:{int(args.port)}"
    cmd = _server_cmd(
        server,
        gguf,
        ctx=int(args.ctx),
        ngl=int(args.ngl),
        host=str(args.host),
        port=int(args.port),
        parallel=int(args.parallel),
    )
    log_path = dest / "server.log"
    print("spawn:", " ".join(cmd), flush=True)
    log_f = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(server.parent),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    plate: dict[str, Any] = {
        "schema": SCHEMA,
        "plan_only": False,
        "cmd": cmd,
        "gguf_note": note,
        "ctx": int(args.ctx),
        "n_predict": int(args.n_predict),
        "ngl": int(args.ngl),
        "ngl_omitted": int(args.ngl) < 0,
        "parallel": int(args.parallel),
        "long_n": int(args.long_n),
        "quant": "Q4_K_M",
        "model": "Qwen2.5-32B-Instruct",
        "source": "bartowski/Qwen2.5-32B-Instruct-GGUF",
        "llama_cpp": "ggml-org/llama.cpp b10964 win-cuda-12.4",
        "smi_before_mib": _smi_used_mib(),
        "messages": [],
    }
    try:
        _wait_health(base, int(args.load_timeout))
        plate["smi_after_load_mib"] = _smi_used_mib()
        print(f"health ok smi={plate['smi_after_load_mib']}", flush=True)
        if args.warmup:
            warm = _chat(base, "Hello.", 8)
            plate["warmup"] = {
                "reply": _reply_text(warm),
                **_timings(warm),
            }
            print(f"warmup tok/s={plate['warmup']['decode_tok_s']:.3f}", flush=True)
        tok_s: list[float] = []
        ttft: list[float] = []
        smoke: list[bool] = []
        for index, prompt in enumerate(MESSAGES, start=1):
            payload = _chat(base, prompt, int(args.n_predict))
            text = _reply_text(payload)
            times = _timings(payload)
            needle = quality_ok(index, text)
            row = {
                "id": index,
                "prompt": prompt,
                "reply": text,
                "needles": NEEDLES.get(index),
                "quality_ok": needle,
                **times,
            }
            (dest / f"msg-{index}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            plate["messages"].append(row)
            if times["decode_tok_s"] > 0:
                tok_s.append(times["decode_tok_s"])
            if times["prompt_ms"] > 0:
                ttft.append(times["prompt_ms"])
            smoke.append(bool(needle))
            print(
                f"msg {index} tok/s={times['decode_tok_s']:.3f} "
                f"ttft_ms={times['prompt_ms']:.0f} smoke={needle} "
                f"reply={text[:80]!r}",
                flush=True,
            )
        if int(args.long_n) > 0:
            payload = _chat(base, LONG_PROMPT, int(args.long_n), ignore_eos=True)
            text = _reply_text(payload)
            times = _timings(payload)
            plate["long"] = {
                "prompt": LONG_PROMPT,
                "reply": text,
                "ignore_eos": True,
                **times,
            }
            (dest / "long.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(
                f"long tok/s={times['decode_tok_s']:.3f} "
                f"n={times['predicted_n']} ttft_ms={times['prompt_ms']:.0f}",
                flush=True,
            )
        plate["mean_decode_tok_s"] = sum(tok_s) / len(tok_s) if tok_s else None
        plate["mean_ttft_ms"] = sum(ttft) / len(ttft) if ttft else None
        plate["smoke_ok"] = all(smoke) if smoke else False
        plate["smi_after_generate_mib"] = _smi_used_mib()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log_f.close()
    return plate


def run_bench(args: argparse.Namespace, dest: Path) -> dict[str, Any]:
    bench = Path(args.server).with_name("llama-bench.exe")
    if not bench.is_file():
        raise FileNotFoundError(bench)
    cmd = [
        str(bench),
        "-m",
        str(Path(args.gguf)),
    ]
    if int(args.ngl) >= 0:
        cmd.extend(["-ngl", str(int(args.ngl))])
    cmd.extend(
        [
            "-p",
            "512",
            "-n",
            str(int(args.n_predict)),
            "-r",
            str(int(args.bench_reps)),
            "-o",
            "json",
            "--progress",
        ]
    )
    print("bench:", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(bench.parent),
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    (dest / "bench.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    stdout = proc.stdout or ""
    (dest / "bench.stdout.txt").write_text(stdout, encoding="utf-8")
    parsed: Any = None
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        parsed = None
    rows: list[dict[str, Any]] = []
    if isinstance(parsed, list):
        rows = [row for row in parsed if isinstance(row, dict)]
    elif isinstance(parsed, dict):
        rows = [parsed]
    tg = [float(row["avg_ts"]) for row in rows if row.get("n_gen") and row.get("avg_ts")]
    pp = [float(row["avg_ts"]) for row in rows if row.get("n_prompt") and not row.get("n_gen") and row.get("avg_ts")]
    summary = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "pp512_tok_s": sum(pp) / len(pp) if pp else None,
        "tg_tok_s": sum(tg) / len(tg) if tg else None,
        "rows": rows,
    }
    (dest / "bench.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"bench pp512={summary['pp512_tok_s']} tg={summary['tg_tok_s']} rc={proc.returncode}",
        flush=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench exited {proc.returncode}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dest = _out_dir(args.out)
    if args.plan_only:
        plate = _plan(args)
        print(f"plan-only out={dest} gguf_ok={plate['gguf_ok']}", flush=True)
    else:
        plate = run_live(args, dest)
        long = plate.get("long") or {}
        print(
            f"mean tok/s={plate.get('mean_decode_tok_s')} "
            f"long={long.get('decode_tok_s')} n={long.get('predicted_n')} "
            f"ttft_ms={plate.get('mean_ttft_ms')} smoke={plate.get('smoke_ok')}",
            flush=True,
        )
        if args.bench:
            plate["bench"] = run_bench(args, dest)
        gguf_name = Path(args.gguf).name
        size, model = _size_and_model(gguf_name)
        bench = plate.get("bench") or {}
        ngl_note = (
            "ngl omitted (llama-server auto-fit)"
            if int(args.ngl) < 0
            else f"ngl {int(args.ngl)}"
        )
        upsert(
            {
                "id": _compare_id(size, int(args.ngl)),
                "stack": "llamacpp-q4",
                "engine": "llama.cpp b10964 llama-server CUDA 12.4",
                "model": model,
                "size": size,
                "quant": "Q4_K_M",
                "mean_decode_tok_s": plate.get("mean_decode_tok_s"),
                "long_decode_tok_s": long.get("decode_tok_s"),
                "mean_ttft_ms": plate.get("mean_ttft_ms"),
                "smi_after_load_mib": plate.get("smi_after_load_mib"),
                "smoke_ok": plate.get("smoke_ok"),
                "bench_pp512_tok_s": bench.get("pp512_tok_s"),
                "bench_tg_tok_s": bench.get("tg_tok_s"),
                "source": str(dest),
                "notes": ngl_note,
            }
        )
    (dest / "plate.json").write_text(
        json.dumps(plate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    long = plate.get("long") or {}
    bench = plate.get("bench") or {}
    lines = [
        f"schema {SCHEMA}",
        f"quant Q4_K_M  model {Path(args.gguf).name}",
        f"ctx {plate.get('ctx')} n_predict {plate.get('n_predict')} "
        f"ngl {plate.get('ngl')} parallel {plate.get('parallel')}",
        f"gguf {plate.get('gguf_note') or plate.get('gguf')}",
        f"mean_decode_tok_s {plate.get('mean_decode_tok_s')}",
        f"long_decode_tok_s {long.get('decode_tok_s')} n={long.get('predicted_n')}",
        f"mean_ttft_ms {plate.get('mean_ttft_ms')}",
        f"smoke_ok {plate.get('smoke_ok')}",
        f"smi_after_load_mib {plate.get('smi_after_load_mib')}",
        f"bench_pp512_tok_s {bench.get('pp512_tok_s')}",
        f"bench_tg_tok_s {bench.get('tg_tok_s')}",
    ]
    (dest / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

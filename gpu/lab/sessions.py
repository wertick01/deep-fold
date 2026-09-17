"""The two codec sessions, run one after the other on a single 12 GB card.

``run_bf16`` is the baseline people already trust: HuggingFace
``from_pretrained`` + ``generate``. ``run_nf4`` is our driver:
``gpu.host.load_model`` + ``gpu.loop.TokenLoop``, no ``from_pretrained`` on the
weight shards and no ``transformers.generate`` anywhere near it.

Hard constraints, enforced by the shape of this module:

* **Never both models resident.** ``run_both`` finishes the BF16 session --
  including unload -- before the NF4 load starts, and refuses to start NF4 on a
  card that is still full.
* ``torch.cuda.empty_cache()`` lives in :func:`unload`, which runs *between*
  sessions. It is never called inside a token loop.
* **Independent turns.** Every message is a fresh single-user conversation with
  the KV cache reset, so every TTFT is a clean prefill and the two codecs stay
  comparable. Recorded in ``summary.notes``.

Timing is split the same way on both paths: ``prefill_ms`` is the time to the
first generated token, ``decode_tok_s`` counts only the tokens after it. Prefill
is never averaged into tok/s.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .bundle import LabBundle, mean
from .sampler import MIB, Sampler, smi_used_mib
from .script import (
    CHR_PATH,
    MAX_NEW_TOKENS,
    MAX_SEQ,
    MESSAGES,
    MODEL_DIR,
    POLL_INTERVAL_S,
    TURNS_NOTE,
    chat_text,
    quality_ok,
    stop_token_ids,
)

__all__ = ["LabSession", "run_both", "run_bf16", "run_nf4", "unload"]

_REPO = Path(__file__).resolve().parents[2]

# After an isolated worker exits, nvidia-smi must fall this far below that
# session's peak before the other codec starts. The kernel itself must not
# still be holding a model.
RELEASED_BELOW_PEAK_MIB = 1500.0
IDLE_HEADROOM_MIB = 600.0


@dataclass
class LabSession:
    """One codec's tables. ``summary`` is a single row; the rest are lists."""

    codec: str
    timeline: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    # Sidecar: frozen messages.csv cannot grow nll columns (bundle.py).
    loglikelihood: list[dict[str, Any]] = field(default_factory=list)

    def as_bundle(self) -> LabBundle:
        return LabBundle.of(
            timeline=self.timeline,
            events=self.events,
            messages=self.messages,
            summary=[self.summary] if self.summary else [],
        )

    def add_note(self, note: str) -> None:
        existing = str(self.summary.get("notes", "")).strip()
        self.summary["notes"] = f"{existing}; {note}" if existing else note


def session_from_bundle(
    bundle: LabBundle, codec: str, *, source: str | Path | None = None
) -> LabSession:
    """Rebuild a session from CSVs a worker wrote."""
    session = LabSession(codec)
    session.timeline = bundle.rows_for("timeline", codec)
    session.events = bundle.rows_for("events", codec)
    session.messages = bundle.rows_for("messages", codec)
    summary = bundle.summary_for(codec)
    session.summary = dict(summary) if summary else {"codec": codec}
    if source is not None:
        from .nll import read_loglikelihood

        session.loglikelihood = read_loglikelihood(Path(source) / "loglikelihood.csv")
    return session


def _turns_note(conversation: str) -> str:
    if conversation == "history":
        return (
            "history turns: each user prompt is packed with prior replies "
            "(full chat re-encoded); TokenLoop.generate still resets KV, so this "
            "is growing prefill, not incremental KV reuse"
        )
    return TURNS_NOTE


def _write_worker_script(
    dest: Path,
    *,
    messages: Sequence[str],
    conversation: str,
    items_json: str | Path | None,
) -> Path | None:
    """Point the child at a JSON script when this is not the smoke MESSAGES."""
    if items_json is not None:
        return Path(items_json)
    if conversation != "independent" or list(messages) != list(MESSAGES):
        path = dest / "worker_script.json"
        payload = {
            "conversation": conversation,
            "items": [
                {"id": f"msg-{index}", "kind": "smoke", "prompt": prompt}
                for index, prompt in enumerate(messages, start=1)
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path
    return None


def _spawn_session(
    codec: str,
    *,
    out_dir: str | Path | None,
    model_dir: str | Path,
    chr_path: str | Path = CHR_PATH,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_seq: int = MAX_SEQ,
    interval_s: float = POLL_INTERVAL_S,
    graphs: bool = True,
    verbose: bool = True,
    trust_remote_code: bool = False,
    messages: Sequence[str] = MESSAGES,
    items_json: str | Path | None = None,
    conversation: str = "independent",
    plate: str = "hard",
    residency_policy: str = "D",
    executor: str = "tokenloop",
) -> LabSession:
    """Child process loads the model, records, exits; this process never holds it."""
    parent = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="lab-iso-"))
    dest = parent / codec
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "gpu.lab.worker",
        "--codec",
        codec,
        "--out",
        str(dest),
        "--model-dir",
        str(model_dir),
        "--chr",
        str(chr_path),
        "--max-new-tokens",
        str(max_new_tokens),
        "--max-seq",
        str(max_seq),
        "--interval",
        str(interval_s),
    ]
    script_path = _write_worker_script(
        dest, messages=messages, conversation=conversation, items_json=items_json
    )
    if script_path is not None:
        cmd.extend(["--items-json", str(script_path)])
        cmd.extend(["--conversation", conversation])
        # A `quality` callable cannot cross a process boundary; the child rebuilds
        # it from the script, and this says which scorer owns those kinds.
        cmd.extend(["--plate", plate])
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if not graphs:
        cmd.append("--no-graphs")
    if not verbose:
        cmd.append("--quiet")
    if residency_policy and residency_policy != "D":
        cmd.extend(["--residency", str(residency_policy)])
    if executor and executor != "tokenloop":
        cmd.extend(["--executor", str(executor)])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"[isolated {codec}] {' '.join(cmd)}", flush=True)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # Jupyter does not show a child's inherited stderr. Echo it so a missing
    # dep or a CUDA OOM is in the cell, not only in the Jupyter server log,
    # and keep it next to the CSVs so the run directory explains itself later.
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", flush=True)
    try:
        (dest / "worker.log").write_text(
            f"$ {' '.join(cmd)}\nreturncode={completed.returncode}\n\n"
            f"--- stdout ---\n{completed.stdout or ''}\n--- stderr ---\n{completed.stderr or ''}",
            encoding="utf-8",
        )
    except OSError:
        pass
    # Windows keeps nvidia-smi high until the process is fully gone.
    time.sleep(1.5)
    if not (dest / "summary.csv").is_file():
        tail = (completed.stderr or completed.stdout or "").strip()
        extra = f"\n{tail[-4000:]}" if tail else ""
        raise RuntimeError(
            f"isolated {codec} exited {completed.returncode} and wrote no "
            f"summary.csv in {dest}{extra}"
        )
    session = session_from_bundle(LabBundle.read(dest), codec, source=dest)
    if completed.returncode != 0:
        session.add_note(f"worker exited {completed.returncode}")
    return session


def _wait_released(peak_mib: float | None, idle_mib: float | None, *, seconds: float = 45.0) -> tuple[bool, float | None]:
    """Wait until nvidia-smi drops after a worker process has exited."""
    deadline = time.time() + seconds
    used = smi_used_mib()
    while time.time() < deadline:
        used = smi_used_mib()
        if used is None:
            return True, used
        if peak_mib is not None and used < float(peak_mib) - RELEASED_BELOW_PEAK_MIB:
            return True, used
        if idle_mib is not None and used < float(idle_mib) + IDLE_HEADROOM_MIB:
            return True, used
        time.sleep(1.0)
    return False, used


# --------------------------------------------------------------------------- #
# shared plumbing
# --------------------------------------------------------------------------- #


def unload(sampler: Sampler | None = None, detail: str = "") -> float | None:
    """Collect and hand unused CUDA blocks back to the driver.

    Called **between** sessions and at the end of one, never on a token path.
    Returns nvidia-smi used MiB after the release. On Windows that number often
    stays near the previous peak: the process keeps the CUDA pool, and the
    next session reuses it. ``torch.cuda.memory_allocated()`` is the signal
    that the dense weights are actually gone.
    """
    import torch

    if sampler is not None:
        sampler.mark("unload_start", detail=detail)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        ipc = getattr(torch.cuda, "ipc_collect", None)
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
            if callable(ipc):
                ipc()
        gc.collect()
        torch.cuda.empty_cache()
    else:
        gc.collect()
    after = smi_used_mib()
    if sampler is not None:
        alloc = (
            torch.cuda.memory_allocated() / MIB if torch.cuda.is_available() else None
        )
        after_text = "unknown" if after is None else f"{after:.0f} MiB"
        alloc_text = "n/a" if alloc is None else f"{alloc:.0f} MiB"
        sampler.mark(
            "unload_end",
            detail=f"smi_after_unload={after_text}, torch_alloc={alloc_text}",
        )
    return after


def _cpu_then_drop(*objects: Any) -> None:
    """Move leftover modules off the GPU, then drop the Python refs."""
    for obj in objects:
        if obj is None:
            continue
        try:
            if hasattr(obj, "to"):
                obj.to("cpu")
            elif hasattr(obj, "cpu"):
                obj.cpu()
        except Exception:
            pass
    gc.collect()


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "the live lab needs a CUDA device (conda env torch-gpu). "
            "For a GPU-free artifact use `python -m gpu.lab.run --dry-plot`."
        )


def _is_cuda_oom(exc: BaseException) -> bool:
    """HuggingFace sometimes wraps the allocator error in a plain RuntimeError."""
    import torch

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, torch.cuda.OutOfMemoryError):
            return True
        if isinstance(current, RuntimeError) and "out of memory" in str(current).lower():
            return True
        current = current.__cause__ or current.__context__
    return False


def _recorded_miss_notes(exc: BaseException, name: str) -> str:
    """One summary.notes line: OOM or any other load failure, never a crashed cell."""
    if _is_cuda_oom(exc):
        return f"CUDA OOM on {name}: {exc}. Recorded miss, not a crashed cell."
    stale = _stale_remote_code_note(exc, name)
    if stale is not None:
        return f"{stale} Recorded miss, not a crashed cell."
    return (
        f"{type(exc).__name__} on {name}: {exc}. Recorded miss, not a crashed cell."
    )


def _stale_remote_code_note(exc: BaseException, model_dir: str) -> str | None:
    """Name the transformers-5 / remote-code mismatch instead of leaving a bare TypeError.

    InternLM2's ``trust_remote_code`` module was written against transformers
    4.41. On 5.17 ``Cache.get_max_cache_shape()`` answers a shape tuple where
    its ``prepare_inputs_for_generation`` expects an ``int``, so the BF16
    baseline dies with ``can only concatenate tuple (not "int") to tuple``
    before the first token.

    Swapping in the stock ``GenerationMixin`` method makes ``generate`` *run*,
    and that is the trap: 5.17 no longer passes ``cache_position``, the remote
    forward then places RoPE at the wrong positions, and the model emits fluent
    repetition ("and wine, and wine") instead of an answer. Measured here on
    20B: our NF4 loop answered all three prompts from the same weights while
    the patched HF path failed two needles. A baseline that quietly decodes
    garbage is worse than one that does not run, so it is not patched -- the
    load-time VRAM is still recorded, and that is what the lab claims.
    """
    if not isinstance(exc, TypeError):
        return None
    text = str(exc)
    if "concatenate tuple" not in text and "get_max_cache_shape" not in text:
        return None
    return (
        f"{Path(model_dir).name} ships transformers-4.41 remote code and its "
        "prepare_inputs_for_generation cannot run on transformers 5: "
        f"{type(exc).__name__}: {text}. No BF16 decode numbers from this env; "
        "load-time VRAM above is the measurement. Not patched on purpose -- "
        "forcing the stock method drops cache_position and the model decodes "
        "fluent repetition instead of an answer, which would be a fake baseline."
    )


def _load_tokenizer(model_dir: str, trust_remote_code: bool):
    """Load the tokenizer the model shipped, not the one transformers 5 prefers.

    5.17 always picks ``auto_map['AutoTokenizer'][1]`` (the fast class) and
    ignores ``use_fast=False``. InternLM2's fast converter then wants protobuf
    and tiktoken. The slow SentencePiece class at index 0 loads with
    ``sentencepiece==0.1.99``.
    """
    from transformers import AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    kwargs = dict(local_files_only=True, trust_remote_code=trust_remote_code)
    config_path = Path(model_dir) / "tokenizer_config.json"
    auto_map = None
    if trust_remote_code and config_path.is_file():
        try:
            auto_map = json.loads(config_path.read_text(encoding="utf-8")).get(
                "auto_map", {}
            ).get("AutoTokenizer")
        except (OSError, ValueError):
            auto_map = None
    slow_ref = None
    if isinstance(auto_map, (list, tuple)) and auto_map and auto_map[0]:
        slow_ref = auto_map[0]
    if slow_ref:
        cls = get_class_from_dynamic_module(slow_ref, model_dir, **kwargs)
        return cls.from_pretrained(model_dir, **kwargs)
    return AutoTokenizer.from_pretrained(model_dir, **kwargs)


def _message_row(
    codec: str,
    message_id: int,
    prompt: str,
    response: str,
    *,
    prompt_tokens: int,
    new_tokens: int,
    prefill_ms: float,
    decode_ms: float,
    decode_tok_s: float,
    stop_reason: str,
    quality: Callable[[int, str], bool] | None = None,
) -> dict[str, Any]:
    checker = quality or quality_ok
    return {
        "codec": codec,
        "message_id": message_id,
        "prompt": prompt,
        "response": response,
        "prompt_tokens": prompt_tokens,
        "new_tokens": new_tokens,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "decode_tok_s": decode_tok_s,
        "stop_reason": stop_reason,
        "quality_ok": checker(message_id, response),
    }


def _done_detail(row: dict[str, Any]) -> str:
    """``msg_done`` hover text. tok/s lives here and in the table, not on a
    fourth crowded plot row."""
    return (
        f"{row['new_tokens']} new tokens, ttft={row['prefill_ms']:.0f} ms, "
        f"{row['decode_tok_s']:.1f} tok/s, stop={row['stop_reason']}, "
        f"quality_ok={str(bool(row['quality_ok'])).lower()}"
    )


def _finish(
    session: LabSession,
    sampler: Sampler,
    *,
    load_s: float | None,
    vram_before: float | None,
    vram_after_load_smi: float | None,
    vram_after_load_torch: float | None,
    weight_mib: float | None,
    kv_mib: float | None,
    notes: str,
) -> LabSession:
    """Close the sampler and fold everything into the frozen summary row."""
    sampler.stop()
    session.timeline = sampler.timeline_rows()
    session.events = sampler.event_rows()
    rows = session.messages
    all_ok = bool(rows) and all(bool(row["quality_ok"]) for row in rows)
    if sampler.errors:
        notes = f"{notes}; nvidia-smi errors: {sampler.errors[0]}"
    session.summary = {
        "codec": session.codec,
        "load_s": load_s,
        "vram_before_mib": vram_before,
        "vram_after_load_smi_mib": vram_after_load_smi,
        "vram_after_load_torch_mib": vram_after_load_torch,
        "vram_peak_smi_mib": sampler.peak_used_mib,
        "weight_mib": weight_mib,
        "kv_mib": kv_mib,
        "mean_ttft_ms": mean([row["prefill_ms"] for row in rows]),
        "mean_decode_tok_s": mean([row["decode_tok_s"] for row in rows]),
        "n_messages": len(rows),
        "quality_all_ok": all_ok,
        "notes": notes,
    }
    return session


# --------------------------------------------------------------------------- #
# BF16: HuggingFace, the baseline
# --------------------------------------------------------------------------- #


def _step_clock(on_step: Callable[[], None]):
    """A ``StoppingCriteria`` that never stops -- it only timestamps each token.

    ``generate`` calls this once per generated token, the first time right after
    prefill, which is exactly the TTFT boundary we need. It is the only hook in
    the HF loop that fires per step without touching the sampling path.
    """
    import torch
    from transformers import StoppingCriteria

    class _Clock(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001, ANN204
            on_step()
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    return _Clock()


def run_bf16(
    out_dir: str | Path | None = None,
    *,
    model_dir: str | Path = MODEL_DIR,
    messages: Sequence[str] = MESSAGES,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_seq: int = MAX_SEQ,
    interval_s: float = POLL_INTERVAL_S,
    verbose: bool = True,
    trust_remote_code: bool = False,
    isolated: bool = False,
    quality: Callable[[int, str], bool] | None = None,
    conversation: str = "independent",
    items_json: str | Path | None = None,
    plate: str = "hard",
) -> LabSession:
    """Uncompressed baseline: ``from_pretrained`` + greedy ``generate``.

    ``out_dir`` is accepted for symmetry and, when given, receives this single
    session's CSVs. ``run_both`` writes the merged ones.

    A load failure (CUDA OOM, missing ``sentencepiece`` / ``einops``, a
    missing ``.chr``) is a recorded session (empty messages, the exception
    in ``notes``), not a crashed cell. That is the 14B/20B measurement on 12 GB.

    ``isolated=True`` runs the session in a child process so this process never
    holds the weights. Use that from the notebook: otherwise Windows keeps the
    CUDA pool and the next codec's VRAM trace is unreadable. ``plate`` only
    matters then: ``quality`` cannot be pickled to a child, so the worker
    rebuilds it from ``items_json`` with that plate's scorer.
    """
    if isolated:
        return _spawn_session(
            "bf16",
            out_dir=out_dir,
            model_dir=model_dir,
            max_new_tokens=max_new_tokens,
            max_seq=max_seq,
            interval_s=interval_s,
            verbose=verbose,
            trust_remote_code=trust_remote_code,
            messages=messages,
            items_json=items_json,
            conversation=conversation,
            plate=plate,
        )
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig

    _require_cuda()
    model_dir = str(model_dir)
    session = LabSession("bf16")
    sampler = Sampler("bf16", interval_s=interval_s, verbose=verbose)
    sampler.start()
    vram_before = smi_used_mib()
    if interval_s > 0:
        time.sleep(min(0.5, interval_s * 4))  # a few idle samples before the load

    model = None
    tokenizer = None
    load_s = None
    vram_after_smi = None
    vram_after_torch = None
    weight_mib = None
    try:
        sampler.mark("load_start", detail=f"from_pretrained {Path(model_dir).name}, bf16")
        t_load = time.perf_counter()
        tokenizer = _load_tokenizer(model_dir, trust_remote_code)
        load_kw = dict(
            device_map={"": 0}, local_files_only=True, trust_remote_code=trust_remote_code
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=torch.bfloat16, **load_kw
            )
        except TypeError:  # transformers < 5 spelled it torch_dtype
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16, **load_kw
            )
        model.eval()
        torch.cuda.synchronize()
        load_s = time.perf_counter() - t_load
        vram_after_smi = smi_used_mib()
        vram_after_torch = torch.cuda.memory_allocated() / MIB
        sampler.mark("load_end", detail=f"load_s={load_s:.1f}")

        # Deduplicated by pointer: Qwen2.5-3B ties lm_head to embed_tokens and
        # counting that [151936, 2048] table twice would invent ~594 MiB.
        weight_mib = (
            sum({p.data_ptr(): p.numel() * p.element_size() for p in model.parameters()}.values())
            / MIB
        )

        stop = stop_token_ids(tokenizer)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else stop[0]
        generation = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=list(stop),
            pad_token_id=int(pad_id),
        )

        def encode(text: str, history: Sequence[tuple[str, str]] | None = None):
            packed = chat_text(tokenizer, text, history=history)
            return tokenizer(packed, return_tensors="pt", add_special_tokens=False).input_ids.to(
                "cuda"
            )

        sampler.mark("warmup_start", detail="8 greedy tokens, not in any reported number")
        t_warm = time.perf_counter()
        with torch.no_grad():
            model.generate(
                input_ids=encode("Hello."),
                generation_config=GenerationConfig(
                    max_new_tokens=8,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    eos_token_id=list(stop),
                    pad_token_id=int(pad_id),
                ),
            )
        torch.cuda.synchronize()
        sampler.mark("warmup_end", detail=f"warmup_ms={(time.perf_counter() - t_warm) * 1000:.0f}")

        checker = quality or quality_ok
        history_pairs: list[tuple[str, str]] = []
        kinds = _eval_item_kinds(plate, items_json, len(messages))
        from .nll import score_prefix

        for index, prompt in enumerate(messages, start=1):
            sampler.mark("msg_send", index, detail=prompt)
            if kinds[index - 1] == "ppl":
                result = score_prefix(
                    tokenizer, prompt, max_seq=max_seq, model=model
                )
                _record_ppl(
                    session,
                    sampler,
                    codec="bf16",
                    index=index,
                    prompt=prompt,
                    quality=checker,
                    result=result,
                )
                continue
            hist = history_pairs if conversation == "history" else None
            ids = encode(prompt, history=hist)
            prompt_tokens = int(ids.shape[-1])
            stamps: list[float] = []

            def on_step(message_id: int = index) -> None:
                # Synchronize so the stamp is a real device boundary, not a queued
                # launch. ~0.2 ms against a ~30 ms decode step.
                torch.cuda.synchronize()
                stamps.append(time.perf_counter())
                if len(stamps) == 1:
                    sampler.mark(
                        "first_token", message_id, detail=f"ttft={(stamps[0] - t0) * 1000:.0f} ms"
                    )

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                output = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    generation_config=generation,
                    stopping_criteria=[_step_clock(on_step)],
                )
            torch.cuda.synchronize()
            t_end = time.perf_counter()

            new_ids = output[0][prompt_tokens:].tolist()
            response = tokenizer.decode(new_ids, skip_special_tokens=True)
            prefill_ms = ((stamps[0] if stamps else t_end) - t0) * 1000.0
            decode_ms = (
                (stamps[-1] if stamps else t_end) - (stamps[0] if stamps else t_end)
            ) * 1000.0
            decode_steps = max(0, len(stamps) - 1)
            row = _message_row(
                "bf16",
                index,
                prompt,
                response,
                prompt_tokens=prompt_tokens,
                new_tokens=len(new_ids),
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                decode_tok_s=decode_steps / (decode_ms / 1000.0) if decode_ms > 0 else 0.0,
                stop_reason="eos" if new_ids and new_ids[-1] in stop else "max_new_tokens",
                quality=checker,
            )
            session.messages.append(row)
            if conversation == "history":
                history_pairs.append((prompt, response))
            sampler.mark("msg_done", index, detail=_done_detail(row))
            # Fresh `generate` call per turn: HF builds a new cache, so the next
            # prompt is a clean prefill unless conversation=history packed prior replies.

        try:
            model.to("cpu")
        except Exception:
            pass
        del model, tokenizer
        unload(sampler, detail="del model/tokenizer, gc, empty_cache")
        finished = _finish(
            session,
            sampler,
            load_s=load_s,
            vram_before=vram_before,
            vram_after_load_smi=vram_after_smi,
            vram_after_load_torch=vram_after_torch,
            weight_mib=weight_mib,
            kv_mib=None,
            notes=(
                "HuggingFace from_pretrained + generate, dense bf16 GEMM; "
                f"{_turns_note(conversation)}; ttft = time to first generated token; "
                "kv_mib empty: the HF cache is transient, not a preallocated block"
            ),
        )
        return finished if out_dir is None else _write_single(finished, out_dir)
    except Exception as exc:
        _cpu_then_drop(model, tokenizer)
        model = None
        tokenizer = None
        gc.collect()
        unload(sampler, detail=f"{type(exc).__name__}")
        finished = _finish(
            session,
            sampler,
            load_s=load_s,
            vram_before=vram_before,
            vram_after_load_smi=vram_after_smi,
            vram_after_load_torch=vram_after_torch,
            weight_mib=weight_mib,
            kv_mib=None,
            notes=_recorded_miss_notes(exc, Path(model_dir).name),
        )
        return finished if out_dir is None else _write_single(finished, out_dir)


# --------------------------------------------------------------------------- #
# NF4: our driver
# --------------------------------------------------------------------------- #


def run_nf4(
    out_dir: str | Path | None = None,
    *,
    model_dir: str | Path = MODEL_DIR,
    chr_path: str | Path = CHR_PATH,
    messages: Sequence[str] = MESSAGES,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_seq: int = MAX_SEQ,
    interval_s: float = POLL_INTERVAL_S,
    graphs: bool = True,
    verbose: bool = True,
    trust_remote_code: bool = False,
    isolated: bool = False,
    quality: Callable[[int, str], bool] | None = None,
    conversation: str = "independent",
    items_json: str | Path | None = None,
    plate: str = "hard",
    residency_policy: str = "D",
    executor: str = "tokenloop",
) -> LabSession:
    """Our driver: ``load_model`` + TokenLoop or Decode V2, greedy, one token.

    The ``.chr`` is the only weight file opened. ``transformers`` is used for the
    tokenizer and the chat template, never for the forward pass.

    ``executor`` is ``tokenloop`` (default, competitor / eval / old hard),
    ``decodev2``, or ``auto`` (resident NF4 → Decode V2, overflow → TokenLoop).

    ``isolated=True`` runs in a child process so this process never holds the
    weights. The notebook must use that, or the compressed VRAM trace starts
    on top of the dense CUDA pool. ``plate`` picks the child's scorer for
    ``items_json``; in-process runs get ``quality`` directly and ignore it.
    """
    if isolated:
        return _spawn_session(
            "nf4",
            out_dir=out_dir,
            model_dir=model_dir,
            chr_path=chr_path,
            max_new_tokens=max_new_tokens,
            max_seq=max_seq,
            interval_s=interval_s,
            graphs=graphs,
            verbose=verbose,
            trust_remote_code=trust_remote_code,
            messages=messages,
            items_json=items_json,
            conversation=conversation,
            plate=plate,
            residency_policy=residency_policy,
            executor=executor,
        )
    import torch

    from gpu.decodev2.session import DecodeV2Loop, pick_executor
    from gpu.host import load_model

    _require_cuda()
    model_dir, chr_path = str(model_dir), str(chr_path)

    session = LabSession("nf4")
    sampler = Sampler("nf4", interval_s=interval_s, verbose=verbose)
    sampler.start()
    vram_before = smi_used_mib()
    if interval_s > 0:
        time.sleep(min(0.5, interval_s * 4))

    model = None
    tokenizer = None
    loop = None
    load_s = None
    vram_after_smi = None
    vram_after_torch = None
    weight_mib = None
    kv_mib = None
    graph_mode = "off"
    prefill_chunk = None
    try:
        if not Path(chr_path).is_file():
            raise FileNotFoundError(f"no NF4 file at {chr_path}")
        sampler.mark("load_start", detail=f"load_model {Path(chr_path).name}")
        t_load = time.perf_counter()
        tokenizer = _load_tokenizer(model_dir, trust_remote_code)
        from gpu.lab.h2_metrics import auto_max_resident_bytes

        cap, cap_info = auto_max_resident_bytes(model_dir, chr_path, max_seq)
        # strict=False: the lab records an incomplete load in `report` and on the
        # plate. `deepfold run` refuses it instead (wave8-arch §2.5).
        model, report = load_model(
            model_dir,
            chr_path,
            trust_remote_code=trust_remote_code,
            strict=False,
            max_resident_bytes=cap,
            residency_policy=residency_policy,
        )
        torch.cuda.synchronize()
        load_s = time.perf_counter() - t_load
        vram_after_smi = smi_used_mib()
        vram_after_torch = torch.cuda.memory_allocated() / MIB
        sampler.mark(
            "load_end",
            detail=f"load_s={load_s:.1f}, cap_mib={cap_info.get('cap_mib')}, {report}",
        )

        chosen, why = pick_executor(
            executor, report, getattr(model, "deepfold_plan", None)
        )
        if str(executor) == "decodev2" and why is not None:
            raise RuntimeError(why)
        if chosen == "decodev2":
            loop = DecodeV2Loop.from_model(model, max_seq=max_seq)
            # Packed NF4 stays aliased on DeviceWeights. Drop the HF module so
            # 20B is not packed-embed + dense-embed + KV at the 12 GB cap.
            model = None
            gc.collect()
        else:
            from gpu.loop import TokenLoop

            loop = TokenLoop(model, max_seq=max_seq, norm="exact", overlap=True)
        sampler.mark("warmup_start", detail=repr(loop))
        warm_ms = loop.warmup(prompt=8, tokens=16)
        graph_mode = "off"
        if graphs:
            graph_mode = loop.capture_graphs()
            sampler.mark(
                "graph_capture",
                detail=f"graph={graph_mode}"
                + (f" ({loop.graph_error})" if loop.graph_error else ""),
            )
        torch.cuda.synchronize()
        sampler.mark("warmup_end", detail=f"warmup_ms={warm_ms:.0f}, graph={graph_mode}")

        stop = stop_token_ids(tokenizer)
        checker = quality or quality_ok
        history_pairs: list[tuple[str, str]] = []
        kinds = _eval_item_kinds(plate, items_json, len(messages))
        from .nll import score_prefix

        for index, prompt in enumerate(messages, start=1):
            sampler.mark("msg_send", index, detail=prompt)
            if kinds[index - 1] == "ppl":
                result = score_prefix(
                    tokenizer, prompt, max_seq=max_seq, loop=loop
                )
                _record_ppl(
                    session,
                    sampler,
                    codec="nf4",
                    index=index,
                    prompt=prompt,
                    quality=checker,
                    result=result,
                )
                continue
            hist = history_pairs if conversation == "history" else None
            packed = chat_text(tokenizer, prompt, history=hist)
            ids = tokenizer(packed, return_tensors="pt", add_special_tokens=False).input_ids[0]
            seen: list[int] = []

            def on_token(token_id: int, message_id: int = index) -> None:
                if not seen:
                    sampler.mark("first_token", message_id, detail="prefill done")
                seen.append(token_id)

            # `generate` resets the KV cache itself. Independent turns are a
            # clean prefill; history mode still resets KV but the packed prompt
            # contains prior replies, so this is growing prefill.
            run = loop.generate(ids, max_new_tokens, stop=stop, on_token=on_token)
            response = tokenizer.decode(run.tokens, skip_special_tokens=True)
            row = _message_row(
                "nf4",
                index,
                prompt,
                response,
                prompt_tokens=run.prompt_len,
                new_tokens=len(run.tokens),
                prefill_ms=run.prefill_ms,
                decode_ms=run.decode_ms,
                decode_tok_s=run.decode_tok_s,
                stop_reason="eos" if run.stop_token is not None else "max_new_tokens",
                quality=checker,
            )
            session.messages.append(row)
            if conversation == "history":
                history_pairs.append((prompt, response))
            sampler.mark("msg_done", index, detail=_done_detail(row))

        weight_mib = loop.weight_bytes / MIB
        kv_mib = loop.kv.mib
        prefill_chunk = loop.prefill_chunk
        engine = type(loop).__name__
        try:
            loop.kv = None
        except Exception:
            pass
        try:
            model.to("cpu")
        except Exception:
            pass
        loop = None
        model = None
        tokenizer = None
        unload(sampler, detail="del loop/model/tokenizer, gc, empty_cache")
        finished = _finish(
            session,
            sampler,
            load_s=load_s,
            vram_before=vram_before,
            vram_after_load_smi=vram_after_smi,
            vram_after_load_torch=vram_after_torch,
            weight_mib=weight_mib,
            kv_mib=kv_mib,
            notes=(
                f"gpu.host.load_model + {engine}, executor={executor}, "
                f"graph={graph_mode}, "
                f"max_seq={max_seq}, prefill_chunk={prefill_chunk}; {_turns_note(conversation)}; "
                "ttft = prefill of the whole prompt; no from_pretrained, no transformers.generate"
            ),
        )
        return finished if out_dir is None else _write_single(finished, out_dir)
    except Exception as exc:
        _cpu_then_drop(loop, model, tokenizer)
        loop = None
        model = None
        tokenizer = None
        gc.collect()
        unload(sampler, detail=f"{type(exc).__name__}")
        finished = _finish(
            session,
            sampler,
            load_s=load_s,
            vram_before=vram_before,
            vram_after_load_smi=vram_after_smi,
            vram_after_load_torch=vram_after_torch,
            weight_mib=weight_mib,
            kv_mib=kv_mib,
            notes=_recorded_miss_notes(exc, Path(chr_path).name),
        )
        return finished if out_dir is None else _write_single(finished, out_dir)


def _eval_item_kinds(
    plate: str, items_json: str | Path | None, n: int
) -> list[str]:
    """Per-message kind for the eval plate. Smoke / hard stay generate-only."""
    kinds = [""] * n
    if plate != "eval" or not items_json:
        return kinds
    from .eval import load_eval_script

    items = load_eval_script(items_json).items
    for index, item in enumerate(items):
        if index >= n:
            break
        kinds[index] = item.kind.lower()
    return kinds


def _record_ppl(
    session: LabSession,
    sampler: Sampler,
    *,
    codec: str,
    index: int,
    prompt: str,
    quality: Callable[[int, str], bool] | None,
    result: Any,
) -> None:
    """One teacher-forced prefix: frozen message row + sidecar NLL, no generate."""
    row = _message_row(
        codec,
        index,
        prompt,
        "",
        prompt_tokens=int(result.prompt_tokens),
        new_tokens=0,
        prefill_ms=float(result.elapsed_ms),
        decode_ms=0.0,
        decode_tok_s=0.0,
        stop_reason="loglikelihood",
        quality=quality,
    )
    session.messages.append(row)
    session.loglikelihood.append(
        {
            "codec": codec,
            "message_id": index,
            "nll": result.nll,
            "n_tokens": result.n_tokens,
            "roundtrip": result.roundtrip,
            "notes": result.notes,
        }
    )
    sampler.mark(
        "first_token",
        index,
        detail=f"teacher-forced nll {result.elapsed_ms:.0f} ms",
    )
    sampler.mark("msg_done", index, detail=_done_detail(row))


def _write_single(session: LabSession, out_dir: str | Path) -> LabSession:
    """Write one session's CSVs (``run_both`` writes the merged tables)."""
    dest = Path(out_dir)
    session.as_bundle().write(dest)
    if session.loglikelihood:
        from .nll import write_loglikelihood

        write_loglikelihood(dest / "loglikelihood.csv", session.loglikelihood)
    return session


# --------------------------------------------------------------------------- #
# both, in order, on one card
# --------------------------------------------------------------------------- #


def _card_is_clear(peak_mib: float | None, idle_mib: float | None = None) -> tuple[bool, float | None]:
    """Has nvidia-smi dropped after the previous worker exited?"""
    used = smi_used_mib()
    if used is None:
        return True, used
    if peak_mib is not None and used < float(peak_mib) - RELEASED_BELOW_PEAK_MIB:
        return True, used
    if idle_mib is not None and used < float(idle_mib) + IDLE_HEADROOM_MIB:
        return True, used
    return False, used


def run_both(
    out_dir: str | Path | None = None,
    *,
    codec: str = "both",
    model_dir: str | Path = MODEL_DIR,
    chr_path: str | Path = CHR_PATH,
    messages: Sequence[str] = MESSAGES,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_seq: int = MAX_SEQ,
    interval_s: float = POLL_INTERVAL_S,
    graphs: bool = True,
    verbose: bool = True,
    trust_remote_code: bool = False,
    isolated: bool = True,
    quality: Callable[[int, str], bool] | None = None,
    conversation: str = "independent",
    items_json: str | Path | None = None,
    executor: str = "tokenloop",
) -> LabBundle:
    """BF16 session, process exit, NF4 session -- sequential, on one GPU.

    ``isolated=True`` (the default) runs each codec in a child process so the
    GPU never holds both, and nvidia-smi actually comes back in between.

    Writes ``timeline.csv``, ``events.csv``, ``messages.csv`` and
    ``summary.csv`` into ``out_dir`` when it is given, and returns the bundle
    :func:`gpu.lab.comparison_figure` draws.
    """
    if codec not in ("both", "bf16", "nf4"):
        raise ValueError(f"codec={codec!r}; expected 'both', 'bf16' or 'nf4'")

    idle = smi_used_mib()
    sessions: list[LabSession] = []
    child_out = out_dir if isolated else None
    extra = dict(
        quality=quality,
        conversation=conversation,
        items_json=items_json,
    )
    if codec in ("both", "bf16"):
        sessions.append(
            run_bf16(
                out_dir=child_out,
                model_dir=model_dir,
                messages=messages,
                max_new_tokens=max_new_tokens,
                max_seq=max_seq,
                interval_s=interval_s,
                verbose=verbose,
                trust_remote_code=trust_remote_code,
                isolated=isolated,
                **extra,
            )
        )

    if codec in ("both", "nf4"):
        clear, used = True, None
        if sessions:
            peak = sessions[-1].summary.get("vram_peak_smi_mib")
            clear, used = _wait_released(peak, idle)
            if not clear:
                unload()
                clear, used = _wait_released(peak, idle, seconds=15.0)
            if not clear:
                note = (
                    f"NF4 session skipped: nvidia-smi still at {used:.0f} MiB after the "
                    f"BF16 worker exited (peak {float(peak) if peak is not None else 'n/a'} MiB, "
                    f"idle {idle if idle is None else f'{idle:.0f}'} MiB). Restart the kernel "
                    "if this process still holds a previous model."
                )
                sessions[-1].add_note(note)
                warnings.warn(note, RuntimeWarning, stacklevel=2)
        if clear:
            if sessions and used is not None:
                nf4_note = f"card cleared to {used:.0f} MiB before the NF4 load"
            else:
                nf4_note = ""
            nf4_session = run_nf4(
                out_dir=child_out,
                model_dir=model_dir,
                chr_path=chr_path,
                messages=messages,
                max_new_tokens=max_new_tokens,
                max_seq=max_seq,
                interval_s=interval_s,
                graphs=graphs,
                verbose=verbose,
                trust_remote_code=trust_remote_code,
                isolated=isolated,
                executor=executor,
                **extra,
            )
            if nf4_note:
                nf4_session.add_note(nf4_note)
            sessions.append(nf4_session)

    bundle = LabBundle()
    for session in sessions:
        bundle = bundle.merge(session.as_bundle())
    if out_dir is not None:
        bundle.write(out_dir)
    return bundle

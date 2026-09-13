"""``deepfold run`` and ``deepfold compress``: a wrapper around what works.

Architecture gate, resolve or pack the ``.chr`` with Go ``chr``, then
``gpu.host.load_model`` + ``gpu.loop.TokenLoop``. ``transformers.generate`` is
never called. GGUF and an unknown ``model_type`` lose before the GPU is
probed. A sibling ``.chr`` is accepted only when the CHR0 header matches.

Assistant text goes to stdout; diagnostics go to stderr, so
``python -m gpu.cli run --prompt ... > answer.txt`` is usable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from . import messages
from .arch import gate, missing_internlm_extras
from .doctor import probe, verdict
from .paths import ENV_MODEL, cached_chr, find_chr_bin, find_chr_file, looks_like_gguf

MIB = 1024 * 1024
# 4.25 bits/weight (4 + 16/64) against 16: what a BF16 copy of the same model
# would cost. Used only to decide whether to print the 3B speed honesty line.
BF16_OVER_NF4 = 16.0 / 4.25
# CUDA context, KV cache, activations and the Windows desktop, roughly.
RUNTIME_OVERHEAD_MIB = 1600


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def compress_to(
    model_dir: str,
    out: Path,
    chr_bin: Path,
    *,
    quiet: bool = False,
) -> int:
    """Shell out to the Go compressor. Deepfold does not pack weights in Python."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(chr_bin),
        "compress",
        "--in",
        str(model_dir),
        "--out",
        str(out),
        "--codec",
        "nf4",
    ]
    if quiet:
        cmd.append("--quiet")
    _err(
        f"first run: packing {model_dir} with chr (CPU, minutes; 14B/20B longer).\n"
        "This is NF4 compression of HuggingFace BF16 safetensors, "
        "not loading a GGUF.\n"
        f"  {' '.join(cmd[1:])}"
    )
    try:
        proc = subprocess.run(cmd)
    except OSError as exc:
        _err(f"chr could not be executed ({exc}).")
        _err(messages.COMPRESS_FAILED)
        return 1
    if proc.returncode != 0:
        _err(messages.COMPRESS_FAILED)
        return proc.returncode or 1
    return 0


def compress(args) -> int:
    """``deepfold compress --in DIR [--out FILE]``. NF4 only."""
    if looks_like_gguf(args.inp):
        _err(messages.GGUF)
        return 1
    checked = gate(args.inp)
    if not checked.ok:
        _err(checked.reason)
        return 1
    if checked.note:
        _err(checked.note)

    chr_bin = find_chr_bin(args.chr_bin)
    if chr_bin is None:
        _err(messages.missing_chr())
        return 1

    out = Path(args.out) if args.out else cached_chr(args.inp)
    if out.is_file() and not args.force:
        _err(f"{out} already exists; pass --force to repack.")
        return 1
    code = compress_to(args.inp, out, chr_bin, quiet=args.quiet)
    if code == 0:
        print(out)
    return code


def _header_matcher(model_dir: Path):
    """A predicate that says whether a ``.chr`` was packed from *this* model.

    CHR0 records ``arch``, ``hidden_size``, ``num_layers`` and ``vocab_size`` in
    its header, so a few hundred bytes settle which sibling belongs to which
    model directory. Returns None when the config or ``gpu.chr0`` cannot be
    read, in which case the caller falls back to refusing to guess.
    """
    try:
        with open(model_dir / "config.json", encoding="utf-8") as fh:
            cfg = json.load(fh)
        from gpu.chr0 import load_header
    except (OSError, ValueError, ImportError):
        return None

    want = (
        cfg.get("hidden_size"),
        cfg.get("num_hidden_layers"),
        cfg.get("vocab_size"),
    )
    if None in want:
        return None

    def accept(path: Path) -> bool:
        try:
            header = load_header(str(path))
        except Exception:  # noqa: BLE001 - a truncated or foreign file is a no
            return False
        return (header.hidden_size, header.num_layers, header.vocab_size) == want

    return accept


def _resolve_weights(args, model: str) -> tuple[Path | None, int]:
    """The ``.chr``, compressing once if there is none. ``(path, exit_code)``."""
    if args.chr and looks_like_gguf(args.chr):
        _err(messages.GGUF)
        return None, 1

    found = find_chr_file(
        Path(model), args.chr, accept=_header_matcher(Path(model))
    )
    if found is not None:
        return found, 0
    if args.chr:
        _err(f"--chr {args.chr} does not exist.")
        return None, 1
    if args.no_compress:
        _err(f"No .chr for {model} and --no-compress was given.")
        return None, 1

    chr_bin = find_chr_bin(args.chr_bin)
    if chr_bin is None:
        _err(messages.missing_chr())
        return None, 1

    out = cached_chr(model)
    code = compress_to(model, out, chr_bin, quiet=args.quiet)
    if code != 0:
        return None, code
    return out, 0


def _stop_ids(tok, model_dir: str | None = None) -> tuple[int, ...]:
    """The driver's stop set. One implementation, in ``gpu.loop.stop``.

    This used to be a weaker copy: it asked for two Qwen-shaped names and
    trusted ``convert_tokens_to_ids`` without the round trip, so a tokenizer
    that answers ``unk`` for ``<|im_end|>`` could contribute a wrong id. The
    product function does the round trip, reads ``generation_config.json``, and
    raises rather than inventing 151645.
    """
    from gpu.loop.stop import stop_token_ids

    return stop_token_ids(tok, model_dir=model_dir)


def _encode(tok, text: str, *, chat: bool):
    """Chat template if the tokenizer has one, else the raw line and one warning.

    No Qwen system prompt as a catch-all: ``gpu/lab/script.py`` keeps
    ``_qwen_fallback`` for its own fixtures, and pasting "You are Qwen, created
    by Alibaba Cloud" in front of a Llama prompt is the same class of bug as a
    hardcoded stop id (wave8-arch §3.2).
    """
    if chat and getattr(tok, "chat_template", None):
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return tok(rendered, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if chat:
        _err(
            "[warn] this tokenizer has no chat_template: sending the raw prompt. "
            "An instruct model may answer as if mid-document."
        )
    return tok(text, return_tensors="pt").input_ids[0]


def _both_fit(weight_mib: float, vram_total_mib: int | None) -> bool:
    """Would a dense BF16 copy of this model also fit on this card?

    When it would, print the 3B honesty line: packed decode is now ahead of
    the committed BF16 row; time to first token is still slower. NF4 pays off
    when 16 bits do not fit.
    """
    if not vram_total_mib:
        return False
    return weight_mib * BF16_OVER_NF4 + RUNTIME_OVERHEAD_MIB < vram_total_mib


def _prompts(args):
    """One ``--prompt``, else a line-at-a-time REPL over stdin."""
    if args.prompt:
        yield args.prompt
        return
    if sys.stdin is None:
        return
    tty = sys.stdin.isatty()
    while True:
        if tty:
            print("> ", end="", file=sys.stderr, flush=True)
        line = sys.stdin.readline()
        if not line:
            return
        line = line.strip()
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return
        yield line


def _generate(args, model: str, chr_file: Path, trust_remote_code: bool) -> int:
    """Load packed weights once, then drive TokenLoop until stdin is done."""
    import torch
    from transformers import AutoTokenizer

    from gpu.host import load_model
    from gpu.loop import TokenLoop

    from .doctor import _smi_used_mib

    tok = AutoTokenizer.from_pretrained(
        model, local_files_only=True, trust_remote_code=trust_remote_code
    )

    _err(f"loading {chr_file} (the only weight file opened)")
    loaded, report = load_model(
        model, str(chr_file), trust_remote_code=trust_remote_code
    )
    weight_mib = report.device_mib
    smi = _smi_used_mib()
    _err(
        f"packed weights {weight_mib:.0f} MiB"
        + (f", nvidia-smi {smi} MiB" if smi is not None else "")
        + f", load {report.seconds:.1f} s"
    )

    loop = TokenLoop(loaded, max_seq=args.max_seq)
    _err(f"graph family: {loop.plan.family} (qkv pack: {loop.plan.qkv_pack or 'split'})")
    if args.warmup:
        loop.warmup(prompt=8, tokens=8)
    stop = _stop_ids(tok, model)
    _err(f"stop ids from tokenizer: {stop}")

    total = torch.cuda.get_device_properties(0).total_memory // MIB
    if _both_fit(weight_mib, total):
        _err(messages.THREE_B_SPEED)

    for text in _prompts(args):
        ids = _encode(tok, text, chat=not args.raw)
        out = loop.generate(ids, args.max_new_tokens, stop=stop)
        print(tok.decode(out.tokens, skip_special_tokens=True), flush=True)
        _err(
            f"prefill {out.prefill_ms:.0f} ms ({out.prompt_len} tokens), "
            f"decode {out.decode_tok_s:.1f} tok/s over {out.decode_steps} steps"
        )
    return 0


def run(args) -> int:
    """``deepfold run --model DIR [--chr FILE]``."""
    model = args.model or os.environ.get(ENV_MODEL)
    if not model:
        _err("deepfold run needs --model <HuggingFace dir> (or $DEEPFOLD_MODEL).")
        return 1

    checked = gate(model)
    if not checked.ok:
        _err(checked.reason)
        return 1
    if checked.note:
        _err(checked.note)

    m = probe()
    v = verdict(m)
    if not v.allowed:
        _err(v.line)
        if v.refusal:
            _err("")
            _err(v.refusal)
        _err("")
        _err("deepfold doctor explains the whole install.")
        return 1

    if checked.needs_internlm:
        missing = missing_internlm_extras()
        if missing:
            _err(messages.INTERNLM_EXTRAS)
            return 1

    chr_file, code = _resolve_weights(args, model)
    if chr_file is None:
        return code

    try:
        return _generate(args, model, chr_file, checked.trust_remote_code)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        from .doctor import _smi_used_mib

        _err(messages.cuda_oom(_smi_used_mib(), m.vram_total_mib, None))
        if args.debug:
            raise
        return 1

"""``deepfold chat``: TTY session. History + stream. Not Claude Code."""

from __future__ import annotations

import sys
from argparse import Namespace

from . import messages, run as run_mod

__all__ = ["chat", "classify_slash"]


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def classify_slash(line: str) -> str | None:
    """``quit`` / ``clear`` / ``help`` / ``stats`` / ``unknown``, or None if not a slash."""
    text = line.strip()
    if not text.startswith("/"):
        return None
    cmd = text.split()[0].lower()
    if cmd in ("/quit", "/exit"):
        return "quit"
    if cmd == "/clear":
        return "clear"
    if cmd == "/help":
        return "help"
    if cmd == "/stats":
        return "stats"
    return "unknown"


def _is_tty() -> bool:
    stdin, stdout = sys.stdin, sys.stdout
    return (
        stdin is not None
        and stdout is not None
        and bool(stdin.isatty())
        and bool(stdout.isatty())
    )


def _encode_history(tok, history: list[dict[str, str]], *, raw: bool):
    if raw or not getattr(tok, "chat_template", None):
        if not raw:
            _err(
                "[warn] this tokenizer has no chat_template: sending the raw last turn."
            )
        return tok(history[-1]["content"], return_tensors="pt").input_ids[0]
    rendered = tok.apply_chat_template(
        history, add_generation_prompt=True, tokenize=False
    )
    return tok(rendered, return_tensors="pt", add_special_tokens=False).input_ids[0]


def _print_status(out, *, max_seq: int) -> None:
    _err(
        f"prefill {out.prefill_ms:.0f} ms ({out.prompt_len} tokens), "
        f"decode {out.decode_tok_s:.1f} tok/s over {out.decode_steps} steps, "
        f"seq {out.prompt_len}/{max_seq}"
    )


def chat(args: Namespace) -> int:
    """Interactive generate. Loads weights once. Each turn prefills the history."""
    if not _is_tty():
        _err(messages.CHAT_NEED_TTY)
        return 1
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings
    except ImportError:
        _err(messages.CHAT_NEED_TOOLKIT)
        return 1

    code, ctx = run_mod.prepare_run(args)
    if ctx is None:
        return code

    model, chr_file, checked, machine, _verdict = ctx
    try:
        tok, loop, stop, report = run_mod._open_loop(
            args,
            model,
            chr_file,
            checked.trust_remote_code,
            vram_mib=machine.vram_total_mib,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        from .doctor import _smi_used_mib

        _err(messages.cuda_oom(_smi_used_mib(), machine.vram_total_mib, None))
        return 1

    stop_set = frozenset(int(s) for s in stop)
    leaf = str(model).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    sm = machine.sm
    arch = _verdict.arch
    codec = getattr(report, "codec", "nf4")
    _err(f"deepfold chat  {leaf}  {sm} {arch}  {codec}")
    _err(messages.CHAT_HELP)

    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit(event) -> None:  # type: ignore[no-untyped-def]
        event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    def _newline(event) -> None:  # type: ignore[no-untyped-def]
        event.current_buffer.insert_text("\n")

    session = PromptSession(key_bindings=bindings, multiline=True)
    history: list[dict[str, str]] = []
    last_out = None

    while True:
        try:
            text = session.prompt("you> ")
        except KeyboardInterrupt:
            print("", file=sys.stderr)
            return 130
        except EOFError:
            return 0
        text = text.strip()
        if not text:
            continue
        kind = classify_slash(text)
        if kind == "quit":
            return 0
        if kind == "help":
            _err(messages.CHAT_HELP)
            continue
        if kind == "clear":
            history.clear()
            loop.reset()
            last_out = None
            _err("cleared")
            continue
        if kind == "stats":
            if last_out is None:
                _err("no turn yet")
            else:
                _print_status(last_out, max_seq=int(args.max_seq))
            continue
        if kind == "unknown":
            _err(f"unknown slash {text.split()[0]!r}. /help lists commands.")
            continue

        history.append({"role": "user", "content": text})
        ids = _encode_history(tok, history, raw=bool(getattr(args, "raw", False)))
        print("assistant", flush=True)
        parts: list[str] = []

        def on_token(tid: int) -> None:
            if tid in stop_set:
                return
            piece = tok.decode([tid], skip_special_tokens=True)
            if piece:
                parts.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()

        out = loop.generate(
            ids, int(args.max_new_tokens), stop=stop, on_token=on_token
        )
        print(flush=True)
        reply = "".join(parts).strip() or tok.decode(out.tokens, skip_special_tokens=True)
        history.append({"role": "assistant", "content": reply})
        last_out = out
        _print_status(out, max_seq=int(args.max_seq))

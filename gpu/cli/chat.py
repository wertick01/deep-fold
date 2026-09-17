"""``deepfold chat``: TTY session. History + stream. Not Claude Code."""

from __future__ import annotations

import os
import re
import signal
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from . import agent as agent_mod
from . import clipboard, md, messages, prefs, run as run_mod
from . import transcript as store

__all__ = [
    "apply_session_prefill",
    "chat",
    "classify_slash",
    "format_status",
    "format_tool_block",
    "last_assistant",
    "looks_overflow_model",
    "parse_chat_choice",
    "pick_max_seq",
    "run_agent_message",
    "run_session_generate",
    "seal_and_prefix",
    "session_capable",
    "toolbar_text",
    "turn_stop",
]


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def wait_line(msg: str) -> None:
    sys.stderr.write(f"\r\x1b[2K{msg}")
    sys.stderr.flush()


def clear_wait() -> None:
    sys.stderr.write("\r\x1b[2K")
    sys.stderr.flush()


def classify_slash(line: str) -> str | None:
    """``quit`` / ``clear`` / ``help`` / ``stats`` / ``new`` / ``chats`` / ``agent`` / ``unknown``."""
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
    if cmd == "/new":
        return "new"
    if cmd == "/chats":
        return "chats"
    if cmd == "/copy":
        return "copy"
    if cmd == "/save":
        return "save"
    if cmd == "/agent":
        return "agent"
    return "unknown"


def _is_tty() -> bool:
    stdin, stdout = sys.stdin, sys.stdout
    return (
        stdin is not None
        and stdout is not None
        and bool(stdin.isatty())
        and bool(stdout.isatty())
    )


def _encode_history(
    tok,
    history: list[dict[str, Any]],
    *,
    raw: bool,
    tools: list[dict[str, Any]] | None = None,
):
    if raw or not getattr(tok, "chat_template", None):
        if not raw:
            _err(
                "[warn] this tokenizer has no chat_template: sending the raw last turn."
            )
        return tok(history[-1]["content"], return_tensors="pt").input_ids[0]
    use_tools = tools is not None
    has_tools = any(
        row.get("role") == "tool" or row.get("tool_calls") for row in history
    )
    messages: list[dict[str, Any]] = (
        agent_mod.flatten_history(history) if (use_tools or has_tools) else list(history)
    )
    rendered = None
    if tools:
        try:
            rendered = tok.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )
        except TypeError:
            rendered = None
    if rendered is None:
        rendered = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    return tok(rendered, return_tensors="pt", add_special_tokens=False).input_ids[0]


def turn_stop(out, *, max_seq: int, max_new_tokens: int) -> str:
    """Why this turn ended. The model hitting EOS is ``eos``, not a hang."""
    if getattr(out, "interrupted", False):
        return "interrupted"
    if getattr(out, "stop_token", None) is not None:
        return "eos"
    used = int(out.prompt_len) + len(out.tokens)
    if used >= int(max_seq):
        return "max_seq"
    if len(out.tokens) >= int(max_new_tokens):
        return "max_new_tokens"
    return "end"


def format_status(out, *, max_seq: int, max_new_tokens: int) -> str:
    """One stderr line after a turn: prefill, decode rate, KV fill, stop reason."""
    used = int(out.prompt_len) + len(out.tokens)
    reason = turn_stop(out, max_seq=max_seq, max_new_tokens=max_new_tokens)
    bits: list[str] = []
    if reason == "interrupted":
        bits.append("interrupted")
    n_pref = getattr(out, "prefill_n", None)
    if n_pref is None:
        n_pref = out.prompt_len
    hit = getattr(out, "session_hit", None)
    kind = ""
    if hit is True:
        kind = " suffix"
    elif hit is False:
        kind = " full"
    bits.append(f"prefill {out.prefill_ms:.0f} ms ({n_pref} tokens{kind})")
    if out.decode_steps:
        bits.append(f"{out.decode_tok_s:.1f} tok/s × {out.decode_steps}")
    elif reason == "interrupted":
        bits.append("no decode")
    else:
        bits.append(f"decode {out.decode_tok_s:.1f} tok/s over {out.decode_steps} steps")
    bits.append(f"seq {used}/{max_seq}")
    bits.append(f"stop {reason}")
    return "  ·  ".join(bits)


CHAT_MAX_SEQ = 2048
AGENT_MAX_SEQ_8GB = 2048
AGENT_MAX_SEQ_12GB = 4096
_VRAM_12GB_MIB = 10_000


def looks_overflow_model(cfg: dict[str, Any]) -> bool:
    """32B-class Qwen (64 layers) overflows 12 GB; 14B/20B do not."""
    return int(cfg.get("num_hidden_layers") or 0) >= 60


def pick_max_seq(
    explicit: int | None,
    *,
    agent: bool,
    vram_mib: int | None,
    overflow: bool,
) -> int:
    """Chat stays 2048. Agent on a 12 GB 14B plate is 4096. User flag wins."""
    if explicit is not None:
        return int(explicit)
    if not agent:
        return CHAT_MAX_SEQ
    if overflow:
        return CHAT_MAX_SEQ
    if vram_mib is None or int(vram_mib) < _VRAM_12GB_MIB:
        return AGENT_MAX_SEQ_8GB
    return AGENT_MAX_SEQ_12GB


TOOL_BLOCK_LINES = 16


def format_tool_block(name: str, arguments: dict[str, Any], content: str) -> str:
    """Readable tool result for the TTY. Not a TUI chrome."""
    head = agent_mod.format_call(name, arguments)
    body = content if isinstance(content, str) else str(content)
    lines = body.splitlines() or ([body] if body else [])
    clipped = lines[:TOOL_BLOCK_LINES]
    extra = len(lines) - len(clipped)
    rows = [f"┌ {head}"]
    for line in clipped:
        rows.append(f"│ {line}")
    if extra > 0:
        rows.append(f"└ +{extra} more")
    else:
        rows.append("└")
    return "\n".join(rows)


def session_capable(loop) -> bool:
    return all(
        callable(getattr(loop, name, None))
        for name in ("prefill_from", "decode_from_logits", "seal_last")
    )


def _seq_ids(ids) -> list[int]:
    if hasattr(ids, "reshape"):
        return [int(x) for x in ids.reshape(-1).tolist()]
    if hasattr(ids, "tolist"):
        return [int(x) for x in ids.tolist()]
    return [int(x) for x in ids]


def _token_chunk(ids, start: int, end: int | None = None):
    if hasattr(ids, "reshape"):
        flat = ids.reshape(-1)
        return flat[start:] if end is None else flat[start:end]
    seq = list(ids)
    return seq[start:] if end is None else seq[start:end]


def _kv_len(loop) -> int:
    kv = getattr(loop, "kv", None)
    if kv is None:
        return 0
    return int(getattr(kv, "seq_len", 0) or 0)


def apply_session_prefill(loop, ids, prefix_ids: list[int] | None):
    """Walk only the new suffix when the template prefix still matches.

    Returns ``(logits, session_hit, prefill_n, miss)``. ``miss`` is a prefix
    mismatch (full prefill after a live KV). First turn is full, not a miss.
    """
    prompt = _seq_ids(ids)
    kv_len = _kv_len(loop)
    mode, delta = agent_mod.plan_session_prefill(prefix_ids, kv_len, prompt)
    if mode == "full":
        miss = prefix_ids is not None
        loop.reset()
        logits = loop.prefill_from(_token_chunk(ids, 0), 0)
        return logits, False, len(prompt), miss
    if mode == "suffix":
        start = len(prompt) - len(delta)
        logits = loop.prefill_from(_token_chunk(ids, start), kv_len)
        return logits, True, len(delta), False
    fwd = getattr(loop, "forward", None)
    if callable(fwd) and kv_len >= 1:
        logits = fwd(_token_chunk(ids, len(prompt) - 1), kv_len - 1)
        return logits, True, 0, False
    loop.reset()
    logits = loop.prefill_from(_token_chunk(ids, 0), 0)
    return logits, False, len(prompt), False


def seal_and_prefix(loop, prompt: list[int], tokens: list[int]) -> list[int]:
    """Write the last sampled token, then snapshot ids that actually sit in KV."""
    total = list(prompt) + [int(t) for t in tokens]
    seq = _kv_len(loop)
    if tokens and hasattr(loop, "seal_last") and seq < len(total):
        loop.seal_last(int(tokens[-1]))
        seq = _kv_len(loop)
    if seq <= 0:
        return total
    return total[:seq]


def run_session_generate(
    loop,
    ids,
    max_new: int,
    *,
    stop,
    prefix_ids: list[int] | None,
    on_token=None,
    should_stop=None,
) -> tuple[object, list[int] | None, bool]:
    """One prefill+decode. Returns ``(out, prefix_ids, miss)``.

    ``miss`` is a live prefix mismatch. Decode V2 and TokenLoop both work when
    they expose ``prefill_from`` / ``decode_from_logits`` / ``seal_last``.
    """
    prompt = _seq_ids(ids)
    if not session_capable(loop):
        out = loop.generate(
            ids,
            max_new,
            stop=stop,
            on_token=on_token,
            should_stop=should_stop,
        )
        return out, None, False
    miss = False
    kv_len = _kv_len(loop)
    mode, _delta = agent_mod.plan_session_prefill(prefix_ids, kv_len, prompt)
    if mode == "full" and prefix_ids is not None:
        miss = True
    _cuda_sync(loop)
    t0 = time.perf_counter()
    logits, hit, n_pref, miss_apply = apply_session_prefill(loop, ids, prefix_ids)
    _cuda_sync(loop)
    prefill_ms = (time.perf_counter() - t0) * 1000.0
    miss = miss or miss_apply
    out = loop.decode_from_logits(
        logits,
        max_new,
        prompt_len=len(prompt),
        stop=stop,
        on_token=on_token,
        should_stop=should_stop,
        prefill_ms=prefill_ms,
    )
    out.session_hit = hit
    out.prefill_n = n_pref
    new_prefix = seal_and_prefix(loop, prompt, list(out.tokens))
    return out, new_prefix, miss


def run_agent_message(
    tok,
    loop,
    stop,
    history: list[dict[str, Any]],
    text: str,
    *,
    workspace: Path,
    agent_state: agent_mod.AgentSession,
    max_seq: int,
    max_new: int,
    max_rounds: int,
    prefix_ids: list[int] | None = None,
    raw: bool = False,
    confirm=None,
    on_status=None,
    on_token=None,
) -> tuple[list[Any], list[int] | None, str, list[tuple[str, dict[str, Any], str]]]:
    """One user message plus tool rounds. No TTY. ``confirm`` None = allow."""
    history.append({"role": "user", "content": text})
    agent_mod.ensure_agent_system(history, workspace, web=agent_state.web)
    tools = agent_mod.tool_schemas(web=agent_state.web)
    steps = 0
    stalled = 0
    outs: list[Any] = []
    ran: list[tuple[str, dict[str, Any], str]] = []
    prefix = prefix_ids
    while True:
        steps += 1
        ids = _encode_history(tok, history, raw=raw, tools=tools)
        if int(ids.numel()) >= max_seq:
            if steps == 1 and history and history[-1].get("role") == "user":
                history.pop()
            return outs, prefix, "max_seq", ran
        out, prefix, miss = run_session_generate(
            loop,
            ids,
            max_new,
            stop=stop,
            prefix_ids=prefix,
            on_token=on_token,
        )
        if miss and on_status is not None:
            on_status("agent: KV prefix miss, full prefill")
        outs.append(out)
        parts: list[str] = []
        if on_token is None and out is not None:
            for tid in out.tokens:
                if stop and int(tid) in set(int(s) for s in stop):
                    continue
                piece = tok.decode([int(tid)], skip_special_tokens=True)
                if piece:
                    parts.append(piece)
        reply = "".join(parts).strip()
        if not reply and out is not None:
            reply = tok.decode(out.tokens, skip_special_tokens=True).strip()
        if on_status is not None and out is not None:
            on_status(
                format_status(out, max_seq=max_seq, max_new_tokens=max_new)
            )
        if reply and agent_mod.degenerate_tool_text(reply):
            if stalled < 1:
                stalled += 1
                continue
            return outs, prefix, "stalled", ran
        if reply and agent_mod.truncated_tool_call(reply):
            if stalled < 1:
                stalled += 1
                history.append({"role": "assistant", "content": reply})
                history.append(
                    {
                        "role": "user",
                        "content": "continue the tool call JSON and close </tool_call>",
                    }
                )
                continue
            return outs, prefix, "truncated", ran
        calls = agent_mod.parse_tool_calls(reply) if reply else []
        visible = agent_mod.strip_tool_xml(reply).strip() if calls else reply
        msg: dict[str, Any] = {"role": "assistant", "content": visible}
        if calls:
            msg["tool_calls"] = calls
        history.append(msg)
        if not calls:
            return outs, prefix, "ok", ran
        results = agent_mod.execute_calls(
            calls,
            workspace,
            session=agent_state,
            confirm=confirm,
        )
        ran.extend(results)
        for name, arguments, clipped in results:
            history.append({"role": "tool", "name": name, "content": clipped})
            if on_status is not None:
                on_status(format_tool_block(name, arguments, clipped))
        if steps >= max_rounds:
            return outs, prefix, "max_rounds", ran


def _cuda_sync(loop) -> None:
    device = getattr(loop, "device", None)
    if getattr(device, "type", None) != "cuda":
        return
    import torch

    torch.cuda.synchronize()


def toolbar_text(
    *,
    leaf: str,
    sm: str,
    codec: str,
    max_seq: int,
    max_new_tokens: int,
    out,
    title: str = "",
    agent: bool = False,
    trust: str = "",
    todos: int = 0,
) -> str:
    """Bottom bar while the prompt is open. ``out`` is the last turn or None."""
    used = 0
    rate = "—"
    stop = ""
    if out is not None:
        used = int(out.prompt_len) + len(out.tokens)
        if out.decode_steps:
            rate = f"{out.decode_tok_s:.1f} tok/s"
        reason = turn_stop(out, max_seq=max_seq, max_new_tokens=max_new_tokens)
        if reason != "eos":
            stop = f"  stop {reason}"
            if reason == "interrupted":
                rate = "interrupted" if rate == "—" else f"interrupted {rate}"
    label = title or leaf
    if len(label) > 28:
        label = label[:27] + "…"
    extra = ""
    if agent:
        extra = "  agent"
        if trust:
            extra += f" {trust}"
        if todos:
            extra += f" todo {todos}"
    return f" {label}  {sm}  {codec}  {rate}  seq {used}/{max_seq}{stop}{extra} "


def last_assistant(history: list[dict[str, Any]]) -> str | None:
    """Most recent non-empty assistant turn, or None."""
    for row in reversed(history):
        if row.get("role") != "assistant":
            continue
        text = str(row.get("content") or "").strip()
        if text:
            return text
    return None


def format_transcript(history: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for row in history:
        role = str(row.get("role") or "")
        text = str(row.get("content") or "").strip()
        if role in ("user", "assistant", "tool") and text:
            blocks.append(f"{role}>\n{text}")
    return "\n\n".join(blocks)


def _dim(text: str, *, color: bool) -> str:
    if not color:
        return text
    return f"{md.DIM}{text}{md.RESET}"


def parse_chat_choice(choice: str, n_rows: int) -> tuple[str, object | None]:
    """Classify the saved-chat prompt. ``prompt`` means start new with that text."""
    text = choice.strip()
    if not text:
        return "empty", None
    if text.lower() in ("n", "new"):
        return "new", None
    if text.isdigit():
        idx = int(text)
        if 1 <= idx <= n_rows:
            return "index", idx
        return "bad-index", idx
    stem = text[:-5] if text.lower().endswith(".json") else text
    if re.fullmatch(r"\d{8}-\d{6}-\d+", stem) or text.lower().endswith(".json"):
        return "id", text
    return "prompt", text


def _print_chats(rows: list[store.Transcript]) -> None:
    _err("  n   new conversation")
    for i, row in enumerate(rows, start=1):
        when = (row.updated or row.created or "")[:16].replace("T", " ")
        turns = sum(1 for m in row.messages if m.get("role") == "user")
        _err(f"  {i:<3} {when}  {row.title}  ({turns} turns)")


def _choose_chat(
    rows: list[store.Transcript],
    prompt_line,
    *,
    hint: str,
    empty_new: bool = True,
) -> store.Transcript | None | str | tuple[str, str]:
    """``None`` = new. Transcript = resume. ``stay`` / ``\"\"``. ``('prompt', text)`` = new + first line."""
    if not rows:
        return None if empty_new else "stay"
    _err(hint)
    _print_chats(rows)
    try:
        choice = prompt_line("chat> ").strip()
    except (KeyboardInterrupt, EOFError):
        return ""
    kind, payload = parse_chat_choice(choice, len(rows))
    if kind == "empty":
        return None if empty_new else "stay"
    if kind == "new":
        return None
    if kind == "index":
        return rows[int(payload) - 1]
    if kind == "bad-index":
        _err(f"no chat {payload}")
        return ""
    if kind == "id":
        found = store.load_transcript(str(payload))
        if found is None:
            _err(f"unknown chat {payload!r}")
            return ""
        return found
    return ("prompt", str(payload))


def chat(args: Namespace) -> int:
    """Interactive generate. Loads weights once. Session KV across turns."""
    if not _is_tty():
        _err(messages.CHAT_NEED_TTY)
        return 1
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings
    except ImportError:
        _err(messages.CHAT_NEED_TOOLKIT)
        return 1

    ws_arg = getattr(args, "workspace", None)
    workspace = Path(ws_arg or ".").expanduser().resolve()
    explicit_agent = getattr(args, "agent", None)
    explicit_web = getattr(args, "agent_web", None)
    raw_on = bool(getattr(args, "raw", False))
    if raw_on and explicit_agent is True:
        _err("chat: --agent cannot be used with --raw")
        return 1
    agent_on = False if raw_on else prefs.resolve_agent(explicit_agent)
    web_hold = prefs.web_hold_off(explicit_web)
    if (agent_on or ws_arg is not None) and not workspace.is_dir():
        _err(f"chat: --workspace {workspace} is not a directory")
        return 1
    max_rounds = max(1, int(getattr(args, "max_tool_rounds", 24) or 24))
    agent_state = agent_mod.AgentSession(
        trust=str(getattr(args, "agent_trust", None) or "ask"),
        web=prefs.resolve_web(explicit_web, agent=agent_on),
    )

    from .doctor import probe as _probe
    from .paths import ENV_MODEL

    overflow_guess = False
    model_hint = getattr(args, "model", None) or os.environ.get(ENV_MODEL)
    if model_hint:
        try:
            from .codec import load_config

            overflow_guess = looks_overflow_model(load_config(model_hint))
        except (OSError, TypeError, ValueError):
            overflow_guess = False
    args.max_seq = pick_max_seq(
        getattr(args, "max_seq", None),
        agent=agent_on,
        vram_mib=_probe().vram_total_mib,
        overflow=overflow_guess,
    )

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
        msg = str(exc)
        if msg.startswith("decodev2:"):
            _err(msg)
            return 1
        if "out of memory" not in msg.lower():
            raise
        from .doctor import _smi_used_mib

        _err(messages.cuda_oom(_smi_used_mib(), machine.vram_total_mib, None))
        return 1

    stop_set = frozenset(int(s) for s in stop)
    leaf = str(model).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    sm = machine.sm
    arch = _verdict.arch
    codec = getattr(report, "codec", "nf4")
    max_seq = int(args.max_seq)
    max_new = int(args.max_new_tokens)
    color = md.color_enabled()
    _err(_dim(f"deepfold chat  {leaf}  {sm} {arch}  {codec}", color=color))
    _err(
        _dim(
            f"cap {max_new} new tokens / turn, KV {max_seq} "
            f"(history is JSON under {store.chats_root()})",
            color=color,
        )
    )
    _err(_dim("Enter sends · Ctrl+J newline · /help · /agent", color=color))
    if agent_on:
        _err(
            _dim(
                f"agent on  trust={agent_state.trust}  workspace {workspace}  "
                f"kv 0/{max_seq}  writes/tests follow --agent-trust"
                + ("  web_search on" if agent_state.web else ""),
                color=color,
            )
        )
        try:
            from .codec import load_config

            cfg = load_config(model)
        except (OSError, TypeError, ValueError):
            cfg = {}
        if agent_mod.looks_like_small_agent_model(cfg):
            _err(
                _dim(
                    "agent: 3B-class model; tool JSON is unreliable. 14B is the plate.",
                    color=color,
                )
            )
    if not bool(getattr(args, "warmup", True)):
        _err(_dim("warmup skipped: the first reply can stall; omit --no-warmup", color=color))

    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit(event) -> None:  # type: ignore[no-untyped-def]
        event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    def _newline(event) -> None:  # type: ignore[no-untyped-def]
        event.current_buffer.insert_text("\n")

    @bindings.add("c-c")
    def _ctrl_c(event) -> None:  # type: ignore[no-untyped-def]
        buf = event.current_buffer
        if buf.text:
            buf.reset()
            return
        event.app.exit(exception=KeyboardInterrupt())

    last_out = None
    prefix_ids: list[int] | None = None
    current = store.new_transcript(str(model))

    session = PromptSession(
        key_bindings=bindings,
        multiline=True,
        history=InMemoryHistory(),
        completer=WordCompleter(
            [
                "/help",
                "/quit",
                "/exit",
                "/clear",
                "/stats",
                "/new",
                "/chats",
                "/copy",
                "/save",
                "/agent",
                "/agent on",
                "/agent off",
                "/agent web on",
                "/agent web off",
                "/agent default on",
                "/agent default off",
                "/agent trust ask",
                "/agent trust write",
                "/agent trust workspace",
            ],
            ignore_case=True,
            sentence=True,
        ),
        bottom_toolbar=lambda: toolbar_text(
            leaf=leaf,
            sm=sm,
            codec=codec,
            max_seq=max_seq,
            max_new_tokens=max_new,
            out=last_out,
            title=current.title or leaf,
            agent=agent_on,
            trust=agent_state.trust,
            todos=len(agent_state.todos),
        ),
    )

    queued: str | None = None
    saved = store.list_transcripts(str(model))
    session_id = getattr(args, "session", None)
    want_new = bool(getattr(args, "new", False))
    if session_id:
        loaded = store.load_transcript(str(session_id))
        if loaded is None:
            _err(f"chat: no saved session {session_id!r} in {store.chats_root()}")
            return 1
        current = loaded
    elif not want_new and saved:
        picked = _choose_chat(
            saved,
            session.prompt,
            hint="saved chats (Enter = new, a number = resume, or type the first message):",
            empty_new=True,
        )
        if picked == "":
            return 0
        if isinstance(picked, tuple) and picked and picked[0] == "prompt":
            queued = str(picked[1])
        elif isinstance(picked, store.Transcript):
            current = picked

    history: list[dict[str, Any]] = list(current.messages)
    if agent_on:
        agent_mod.ensure_agent_system(history, workspace, web=agent_state.web)
    user_turns = sum(1 for m in history if m.get("role") == "user")
    if user_turns:
        _err(f"resumed {current.id}  {current.title}  {user_turns} turns")
    idle_interrupt = 0

    def _ask_yes_no(message: str) -> bool:
        from prompt_toolkit.shortcuts import prompt as pt_prompt

        kb = KeyBindings()

        @kb.add("y")
        @kb.add("Y")
        @kb.add("д")
        def _yes(event) -> None:  # type: ignore[no-untyped-def]
            event.app.exit(result="y")

        @kb.add("n")
        @kb.add("N")
        @kb.add("enter")
        @kb.add("c-c")
        def _no(event) -> None:  # type: ignore[no-untyped-def]
            event.app.exit(result="n")

        try:
            ans = pt_prompt(message, key_bindings=kb)
        except (KeyboardInterrupt, EOFError):
            return False
        return agent_mod.confirm_accepted(ans or "n")

    def _persist() -> None:
        current.messages = list(history)
        current.model = str(model)
        store.save_transcript(current)

    def _reset_kv() -> None:
        nonlocal prefix_ids
        loop.reset()
        prefix_ids = None

    def _print_status(out) -> None:
        nonlocal last_out
        if out is None:
            _err("interrupted")
            return
        last_out = out
        line = format_status(out, max_seq=max_seq, max_new_tokens=max_new)
        reason = turn_stop(out, max_seq=max_seq, max_new_tokens=max_new)
        if reason == "max_new_tokens":
            line += f"  (reply hit --max-new-tokens {max_new}; raise the flag)"
        elif reason == "max_seq":
            line += f"  (KV filled --max-seq {max_seq}; /clear or raise the flag)"
        _err(_dim(line, color=color))

    def _run_turn(ids, *, hide_tools: bool) -> tuple[object | None, list[str], bool]:
        nonlocal prefix_ids
        abort = {"on": False, "why": ""}
        parts: list[str] = []
        markdown = md.MarkdownStream(sys.stdout.write, color=color)
        emitted = {"n": 0}
        wait_cleared = {"on": False}

        def want_stop() -> bool:
            return abort["on"]

        def _clear_wait() -> None:
            if not wait_cleared["on"]:
                clear_wait()
                wait_cleared["on"] = True

        def on_token(tid: int) -> None:
            _clear_wait()
            if tid in stop_set:
                return
            piece = tok.decode([tid], skip_special_tokens=True)
            if not piece:
                return
            parts.append(piece)
            raw = "".join(parts)
            if hide_tools:
                vis = agent_mod.visible_stream_text(raw)
                if len(vis) > emitted["n"]:
                    markdown.feed(vis[emitted["n"] :])
                    emitted["n"] = len(vis)
            else:
                markdown.feed(piece)
            sys.stdout.flush()
            window = raw[-48:]
            if window.endswith("!" * 12) or window.count("!") >= 20:
                abort["on"] = True
                abort["why"] = "junk"
                return
            if hide_tools and agent_mod.tool_call_closed(raw):
                abort["on"] = True
                abort["why"] = "tool"

        def on_sigint(signum, frame) -> None:
            if abort["on"] and abort["why"] == "user":
                raise KeyboardInterrupt
            abort["on"] = True
            abort["why"] = "user"

        prev = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, on_sigint)
        out = None
        interrupted = False
        prompt = _seq_ids(ids)
        try:
            if session_capable(loop):
                kv_len = _kv_len(loop)
                mode, delta = agent_mod.plan_session_prefill(
                    prefix_ids, kv_len, prompt
                )
                if mode == "full" and prefix_ids is not None:
                    tag = "agent" if agent_on else "chat"
                    _err(f"{tag}: KV prefix miss, full prefill")
                n_show = (
                    len(delta)
                    if mode == "suffix"
                    else (0 if mode == "repeat" else len(prompt))
                )
                if mode == "repeat":
                    wait_line("prefill (repeat)…")
                else:
                    wait_line(f"prefill {n_show} tokens ({mode})…")
            else:
                wait_line(f"prefill {len(prompt)} tokens…")
            out, prefix_ids, _miss = run_session_generate(
                loop,
                ids,
                max_new,
                stop=stop,
                prefix_ids=prefix_ids,
                on_token=on_token,
                should_stop=want_stop,
            )
        except KeyboardInterrupt:
            interrupted = True
            abort["why"] = "user"
        finally:
            signal.signal(signal.SIGINT, prev)
            _clear_wait()
            markdown.close()
            sys.stdout.flush()
        if out is not None and abort["why"] == "tool":
            out.interrupted = False
        if out is not None and out.interrupted and abort["why"] != "tool":
            interrupted = True
        if abort["why"] == "user":
            interrupted = True
        return out, parts, interrupted

    while True:
        try:
            if queued is not None:
                text = queued
                queued = None
            else:
                text = session.prompt("you> ")
        except KeyboardInterrupt:
            idle_interrupt += 1
            if idle_interrupt >= 2:
                print("", file=sys.stderr)
                return 130
            _err("Ctrl+C again to quit")
            continue
        except EOFError:
            return 0
        idle_interrupt = 0
        text = text.strip()
        if not text:
            continue
        kind = classify_slash(text)
        if kind == "quit":
            return 0
        if kind == "help":
            _err(messages.CHAT_HELP)
            _err(
                f"this chat {current.id} is JSON at {current.path or store.chats_root()}. "
                f"KV {_kv_len(loop)}/{max_seq}; later turns prefill only the new "
                "suffix when the chat-template prefix matches. "
                "/clear, /new, and /chats resume reset KV. y/n is one key (Enter = no)."
            )
            if agent_on:
                names = ", ".join(
                    str(item["name"])
                    for item in agent_mod.tool_schemas(web=agent_state.web)
                )
                _err(f"agent on, workspace {workspace}. Tools: {names}.")
            continue
        if kind == "clear":
            history.clear()
            if agent_on:
                agent_mod.ensure_agent_system(history, workspace, web=agent_state.web)
            _reset_kv()
            last_out = None
            current.title = ""
            _persist()
            _err("cleared")
            continue
        if kind == "new":
            if history:
                _persist()
            current = store.new_transcript(str(model))
            history = []
            if agent_on:
                agent_mod.ensure_agent_system(history, workspace, web=agent_state.web)
            _reset_kv()
            last_out = None
            _err(f"new chat {current.id}")
            continue
        if kind == "chats":
            if history:
                _persist()
            rows = store.list_transcripts(str(model))
            arg = text.split()[1] if len(text.split()) > 1 else None
            picked: store.Transcript | None | str | tuple[str, str]
            if arg:
                if arg.lower() in ("n", "new"):
                    picked = None
                else:
                    picked = store.load_transcript(arg)
                    if picked is None and arg.isdigit():
                        idx = int(arg)
                        picked = rows[idx - 1] if 1 <= idx <= len(rows) else ""
                    if picked is None:
                        picked = ""
                        _err(f"unknown chat {arg!r}")
            else:
                picked = _choose_chat(
                    rows,
                    session.prompt,
                    hint="saved chats (Enter = stay, n = new, or type a first message):",
                    empty_new=False,
                )
                if picked == "stay":
                    continue
            if picked == "":
                continue
            if isinstance(picked, tuple) and picked and picked[0] == "prompt":
                current = store.new_transcript(str(model))
                history = []
                if agent_on:
                    agent_mod.ensure_agent_system(history, workspace, web=agent_state.web)
                _reset_kv()
                last_out = None
                queued = str(picked[1])
                _err(f"new chat {current.id}")
                continue
            if picked is None:
                current = store.new_transcript(str(model))
                history = []
                if agent_on:
                    agent_mod.ensure_agent_system(history, workspace, web=agent_state.web)
                _reset_kv()
                last_out = None
                _err(f"new chat {current.id}")
                continue
            if isinstance(picked, store.Transcript):
                current = picked
                history = list(current.messages)
                if agent_on:
                    agent_mod.ensure_agent_system(history, workspace, web=agent_state.web)
                _reset_kv()
                last_out = None
                _err(f"resumed {current.id}  {current.title}")
            continue
        if kind == "stats":
            kv_n = _kv_len(loop)
            if last_out is None:
                _err(f"no turn yet  kv {kv_n}/{max_seq}")
            else:
                _err(
                    format_status(
                        last_out, max_seq=max_seq, max_new_tokens=max_new
                    )
                )
                _err(f"kv {kv_n}/{max_seq}")
            continue
        if kind == "copy":
            arg = text.split()[1].lower() if len(text.split()) > 1 else ""
            if arg in ("all", "chat"):
                payload = format_transcript(history)
            else:
                payload = last_assistant(history) or ""
            if not payload:
                _err("nothing to copy")
                continue
            try:
                clipboard.copy_text(payload)
            except OSError as exc:
                _err(str(exc))
                continue
            _err(f"copied {len(payload)} chars")
            continue
        if kind == "save":
            parts_cmd = text.split(maxsplit=1)
            dest = Path(parts_cmd[1]).expanduser() if len(parts_cmd) > 1 else None
            payload = last_assistant(history)
            if not payload:
                _err("nothing to save")
                continue
            if dest is None:
                dest = (current.path or store.chats_root() / f"{current.id}.json").with_suffix(
                    ".md"
                )
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(payload + "\n", encoding="utf-8")
            except OSError as exc:
                _err(f"save failed: {exc}")
                continue
            _err(f"wrote {dest}")
            continue
        if kind == "agent":
            bits = text.split()
            arg = bits[1].lower() if len(bits) > 1 else ""
            if raw_on:
                _err("agent: cannot enable while --raw")
                continue
            if arg == "trust":
                level = bits[2].lower() if len(bits) > 2 else ""
                if level not in agent_mod.TRUST_LEVELS:
                    _err("usage: /agent trust ask|write|workspace")
                    continue
                agent_state.trust = level
                if not agent_on:
                    _err(f"agent trust {level} (tools still off; /agent on)")
                else:
                    _err(f"agent trust {level}")
                continue
            if arg == "web":
                sub = bits[2].lower() if len(bits) > 2 else ""
                if sub == "default":
                    flag = prefs.parse_bool(bits[3] if len(bits) > 3 else "")
                    if flag is None:
                        _err("usage: /agent web default on|off")
                        continue
                    path = prefs.save_pref(prefs.ENV_AGENT_WEB, flag)
                    web_hold = not flag
                    agent_state.web = flag
                    if agent_on:
                        agent_mod.ensure_agent_system(
                            history, workspace, web=agent_state.web
                        )
                        _persist()
                    _err(
                        f"agent web_search default {'on' if flag else 'off'}  "
                        f"saved {path}"
                    )
                    continue
                flag = prefs.parse_bool(sub)
                if flag is None:
                    _err("usage: /agent web on|off | /agent web default on|off")
                    continue
                agent_state.web = flag
                web_hold = not flag
                if agent_on:
                    agent_mod.ensure_agent_system(
                        history, workspace, web=agent_state.web
                    )
                    _persist()
                _err(
                    f"agent web_search {'on' if agent_state.web else 'off'}"
                    + (
                        ""
                        if agent_on
                        else " (tools still off; /agent on)"
                    )
                )
                continue
            if arg == "default":
                flag = prefs.parse_bool(bits[2] if len(bits) > 2 else "")
                if flag is None:
                    stored = prefs.pref_tristate(prefs.ENV_AGENT)
                    stored_web = prefs.pref_tristate(prefs.ENV_AGENT_WEB)
                    _err(
                        "agent default "
                        + ("on" if stored else "off" if stored is False else "unset")
                        + "  web "
                        + (
                            "on"
                            if stored_web
                            else "off"
                            if stored_web is False
                            else "follows agent"
                        )
                        + "  /agent default on|off"
                    )
                    continue
                if flag and not workspace.is_dir():
                    _err(f"agent: workspace {workspace} is not a directory")
                    continue
                path = prefs.save_pref(prefs.ENV_AGENT, flag)
                if flag:
                    agent_on = True
                    if not web_hold:
                        agent_state.web = True
                    agent_mod.ensure_agent_system(
                        history, workspace, web=agent_state.web
                    )
                    _persist()
                else:
                    agent_on = False
                    agent_mod.drop_agent_system(history)
                    _persist()
                extra = "  web_search on" if agent_on and agent_state.web else ""
                _err(
                    f"agent default {'on' if flag else 'off'}  saved {path}{extra}"
                )
                continue
            if arg in ("on", "1", "true"):
                if not workspace.is_dir():
                    _err(f"agent: workspace {workspace} is not a directory")
                    continue
                agent_on = True
                if not web_hold:
                    agent_state.web = True
                agent_mod.ensure_agent_system(history, workspace, web=agent_state.web)
                _persist()
                extra = "  web_search on" if agent_state.web else ""
                _err(f"agent on  trust={agent_state.trust}  workspace {workspace}{extra}")
            elif arg in ("off", "0", "false"):
                agent_on = False
                agent_mod.drop_agent_system(history)
                _persist()
                _err("agent off")
            elif arg == "":
                plan = f"  todo {len(agent_state.todos)}" if agent_state.todos else ""
                web = "  web on" if agent_state.web else ""
                stored = prefs.pref_tristate(prefs.ENV_AGENT)
                hint = ""
                if stored:
                    hint = "  default on"
                elif stored is False:
                    hint = "  default off"
                _err(
                    f"agent {'on' if agent_on else 'off'}  trust={agent_state.trust}  "
                    f"workspace {workspace}{plan}{web}{hint}  "
                    "/agent on|off|trust|web|default"
                )
            else:
                _err(
                    "usage: /agent on | /agent off | /agent default on|off | "
                    "/agent trust ask|write|workspace | /agent web on|off | "
                    "/agent web default on|off"
                )
            continue
        if kind == "unknown":
            _err(f"unknown slash {text.split()[0]!r}. /help lists commands.")
            continue

        history.append({"role": "user", "content": text})
        if agent_on:
            agent_mod.ensure_agent_system(history, workspace, web=agent_state.web)
        tools = (
            agent_mod.tool_schemas(web=agent_state.web) if agent_on else None
        )
        raw = bool(getattr(args, "raw", False))
        steps = 0
        stalled_retries = 0
        while True:
            steps += 1
            ids = _encode_history(tok, history, raw=raw, tools=tools)
            if int(ids.numel()) >= max_seq:
                if steps == 1 and history and history[-1].get("role") == "user":
                    history.pop()
                _err(
                    f"history is {int(ids.numel())} tokens; --max-seq is {max_seq}. "
                    "/clear or /new, or raise --max-seq."
                )
                break
            print(_dim("assistant", color=color), flush=True)
            out, parts, interrupted = _run_turn(ids, hide_tools=agent_on)
            print(flush=True)
            reply = "".join(parts).strip()
            if not reply and out is not None:
                reply = tok.decode(out.tokens, skip_special_tokens=True).strip()
            if interrupted and not reply:
                if steps == 1 and history and history[-1].get("role") == "user":
                    history.pop()
                _err("interrupted")
                break
            if agent_on and reply and agent_mod.degenerate_tool_text(reply):
                _print_status(out)
                if stalled_retries < 1:
                    stalled_retries += 1
                    _err("model stalled on a tool call; retrying")
                    continue
                _err("model stalled on a tool call; send the request again")
                break
            if agent_on and reply and agent_mod.truncated_tool_call(reply):
                _print_status(out)
                if stalled_retries < 1:
                    stalled_retries += 1
                    history.append({"role": "assistant", "content": reply})
                    history.append(
                        {
                            "role": "user",
                            "content": "continue the tool call JSON and close </tool_call>",
                        }
                    )
                    _err("model truncated a tool call; retrying")
                    continue
                _err("model truncated a tool call; send the request again")
                break
            calls = (
                agent_mod.parse_tool_calls(reply)
                if agent_on and not interrupted
                else []
            )
            visible = agent_mod.strip_tool_xml(reply).strip() if calls else reply
            msg: dict[str, Any] = {"role": "assistant", "content": visible}
            if calls:
                msg["tool_calls"] = calls
            history.append(msg)
            _persist()
            _print_status(out)
            if interrupted or not calls:
                break

            def _confirm(name: str, arguments: dict[str, Any]) -> bool:
                if not agent_mod.needs_confirm(
                    name, arguments, trust=agent_state.trust
                ):
                    return True
                clear_wait()
                diff = agent_mod.edit_preview(name, arguments, workspace)
                if diff:
                    _err(_dim(diff[:2000], color=color))
                ok = _ask_yes_no(
                    f"allow {agent_mod.format_call(name, arguments)}? [y/N] "
                )
                if not ok:
                    _err(_dim("denied", color=color))
                return ok

            def _progress(name: str, arguments: dict[str, Any]) -> None:
                if name == "_parallel":
                    wait_line(f"running {int(arguments.get('n', 0))} reads…")
                    return
                wait_line(f"running {agent_mod.format_call(name, arguments)}…")

            results = agent_mod.execute_calls(
                calls,
                workspace,
                session=agent_state,
                confirm=_confirm,
                progress=_progress,
            )
            clear_wait()
            for name, arguments, clipped in results:
                history.append({"role": "tool", "name": name, "content": clipped})
                _err(format_tool_block(name, arguments, clipped))
            _persist()
            if steps >= max_rounds:
                _err(f"agent: stop after {max_rounds} tool rounds")
                break

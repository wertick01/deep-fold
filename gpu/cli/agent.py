"""Workspace tools for ``deepfold chat --agent``.

Not Claude Code and not a shell. Four calls: list, read, write, pytest.
Writes and tests wait for a yes. Paths must stay under ``--workspace``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMAS",
    "PathEscape",
    "clip_result",
    "confirm_accepted",
    "degenerate_tool_text",
    "drop_agent_system",
    "ensure_agent_system",
    "execute",
    "flatten_history",
    "format_call",
    "needs_confirm",
    "parse_tool_calls",
    "preview",
    "resolve_under",
    "strip_tool_xml",
]

SYSTEM_MARK = "deepfold-agent-workspace:"
_START = "<tool_call>"
_END = "</tool_call>"
_SKIP_DIR = frozenset(
    {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache", ".mypy_cache"}
)
_CONFIRM = frozenset({"write_file", "run_tests"})
_MAX_LIST = 200
_MAX_READ_BYTES = 1_000_000
_MAX_READ_LINES = 400
_MAX_WRITE_CHARS = 256 * 1024
_MAX_RESULT = 12_000
_TEST_TIMEOUT_S = 120

SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_dir",
        "description": "List files and directories under a workspace path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path inside the workspace. Default: .",
                }
            },
        },
    },
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file. Returns numbered lines.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path inside the workspace"},
                "offset": {
                    "type": "integer",
                    "description": "1-based start line (default 1)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to return (default 200, max 400)",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a UTF-8 file. Needs user confirmation.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path inside the workspace"},
                "content": {"type": "string", "description": "Full file contents"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_tests",
        "description": (
            "Run pytest on one file or directory inside the workspace. "
            "Needs user confirmation. No extra pytest flags."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Test file or directory inside the workspace",
                }
            },
            "required": ["path"],
        },
    },
]


class PathEscape(ValueError):
    """A tool path resolved outside the workspace."""


def agent_system(workspace: Path) -> str:
    return (
        f"{SYSTEM_MARK} {workspace}\n"
        "You work only in that workspace. Use tools to list, read, and write files "
        "and to run pytest on a path. Do not claim you cannot access files. "
        "Prefer reading a file before rewriting it. After a tool result, continue "
        "until the user's request is done. Do not invent file contents you have not read.\n"
        "A tool call is one complete JSON object, then you stop:\n"
        "<tool_call>\n"
        '{"name": "list_dir", "arguments": {"path": "."}}\n'
        "</tool_call>\n"
        "Never fill a tool call with punctuation. Always close </tool_call>."
    )


def ensure_agent_system(history: list[dict[str, Any]], workspace: Path) -> None:
    msg = {"role": "system", "content": agent_system(workspace)}
    if (
        history
        and history[0].get("role") == "system"
        and SYSTEM_MARK in str(history[0].get("content") or "")
    ):
        history[0] = msg
        return
    history.insert(0, msg)


def drop_agent_system(history: list[dict[str, Any]]) -> None:
    if (
        history
        and history[0].get("role") == "system"
        and SYSTEM_MARK in str(history[0].get("content") or "")
    ):
        history.pop(0)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Pull ``<tool_call>{...}</tool_call>`` blocks. Unknown JSON is skipped."""
    out: list[dict[str, Any]] = []
    pos = 0
    while True:
        start = text.find(_START, pos)
        if start < 0:
            break
        end = text.find(_END, start + len(_START))
        if end < 0:
            break
        raw = text[start + len(_START) : end].strip()
        pos = end + len(_END)
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        fn = obj.get("function") if isinstance(obj.get("function"), dict) else {}
        name = obj.get("name") or fn.get("name")
        args = obj.get("arguments", fn.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"value": args}
        if isinstance(name, str) and name.strip():
            out.append({"name": name.strip(), "arguments": args})
    return out


def degenerate_tool_text(text: str) -> bool:
    """True when the model opened a tool call and then spammed junk (``!!!!``)."""
    if _START not in text:
        compact = "".join(text.split())
        return len(compact) >= 16 and len(set(compact)) == 1
    if parse_tool_calls(text):
        return False
    after = text.split(_START, 1)[1].replace(_END, "")
    compact = "".join(after.split())
    if not compact:
        return True
    if len(compact) >= 8 and not any(ch.isalnum() for ch in compact):
        return True
    return len(compact) >= 12 and len(set(compact)) <= 2


def strip_tool_xml(text: str) -> str:
    pos = 0
    bits: list[str] = []
    while True:
        start = text.find(_START, pos)
        if start < 0:
            bits.append(text[pos:])
            break
        bits.append(text[pos:start])
        end = text.find(_END, start + len(_START))
        if end < 0:
            bits.append(text[start:])
            break
        pos = end + len(_END)
    return "".join(bits).strip()


def render_tool_calls(calls: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for call in calls:
        payload = {"name": call.get("name"), "arguments": call.get("arguments") or {}}
        chunks.append(_START + "\n" + json.dumps(payload, ensure_ascii=False) + "\n" + _END)
    return "\n".join(chunks)


def flatten_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Tokenizer-safe turns: tool XML in content, tool results as ``<tool_response>``."""
    out: list[dict[str, str]] = []
    for row in history:
        role = str(row.get("role") or "user")
        content = str(row.get("content") or "")
        calls = row.get("tool_calls")
        if role == "assistant" and isinstance(calls, list) and calls:
            xml = render_tool_calls(calls)
            body = (content.rstrip() + "\n" + xml).strip() if content.strip() else xml
            out.append({"role": "assistant", "content": body})
        elif role == "tool":
            if "<tool_response>" in content:
                body = content
            else:
                body = f"<tool_response>\n{content}\n</tool_response>"
            out.append({"role": "user", "content": body})
        elif role in ("user", "assistant", "system"):
            out.append({"role": role, "content": content})
    return out


def _realpath(path: Path) -> Path:
    return Path(os.path.realpath(path))


def resolve_under(root: Path, rel: str) -> Path:
    root = _realpath(root)
    text = (rel or ".").strip() or "."
    if text.startswith("~"):
        raise PathEscape("home paths are not allowed")
    raw = Path(text)
    target = _realpath(raw if raw.is_absolute() else root / raw)
    root_s = os.path.normcase(str(root))
    target_s = os.path.normcase(str(target))
    if target_s != root_s and not target_s.startswith(root_s + os.sep):
        raise PathEscape(f"{text!r} is outside the workspace")
    return target


def _in_git(root: Path, target: Path) -> bool:
    rel = _rel_posix(root, target)
    if rel in (".", ""):
        return False
    return ".git" in Path(rel).parts


def _rel_posix(root: Path, target: Path) -> str:
    rel = os.path.relpath(str(_realpath(target)), str(_realpath(root)))
    return Path(rel).as_posix()


def needs_confirm(name: str) -> bool:
    return name in _CONFIRM


def confirm_accepted(line: str) -> bool:
    return line.strip().lower() in ("y", "yes", "д", "да")


def format_call(name: str, arguments: dict[str, Any]) -> str:
    if name == "write_file":
        path = arguments.get("path", "")
        content = arguments.get("content", "")
        n = len(content) if isinstance(content, str) else 0
        return f"write_file {path} ({n} chars)"
    if name == "run_tests":
        return f"run_tests {arguments.get('path', '')}"
    if name == "read_file":
        return f"read_file {arguments.get('path', '')}"
    if name == "list_dir":
        return f"list_dir {arguments.get('path', '.')}"
    return f"{name} {json.dumps(arguments, ensure_ascii=False)}"


def clip_result(text: str, *, limit: int = _MAX_RESULT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... truncated {len(text) - limit} chars"


def preview(text: str, *, limit: int = 240) -> str:
    one = " ".join(text.split())
    if len(one) <= limit:
        return one
    return one[: limit - 1] + "…"


def execute(name: str, arguments: dict[str, Any], workspace: Path) -> str:
    if not isinstance(arguments, dict):
        return "error: arguments must be a JSON object"
    try:
        if name == "list_dir":
            return _list_dir(workspace, str(arguments.get("path") or "."))
        if name == "read_file":
            path = arguments.get("path")
            if not path:
                return "error: path is required"
            return _read_file(
                workspace,
                str(path),
                offset=arguments.get("offset", 1),
                limit=arguments.get("limit", 200),
            )
        if name == "write_file":
            path = arguments.get("path")
            if not path:
                return "error: path is required"
            if "content" not in arguments:
                return "error: content is required"
            return _write_file(workspace, str(path), arguments.get("content"))
        if name == "run_tests":
            path = arguments.get("path")
            if not path:
                return "error: path is required"
            return _run_tests(workspace, str(path))
        names = ", ".join(item["name"] for item in SCHEMAS)
        return f"error: unknown tool {name!r}. available: {names}"
    except PathEscape as exc:
        return f"error: {exc}"
    except OSError as exc:
        return f"error: {exc}"


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _list_dir(root: Path, rel: str) -> str:
    target = resolve_under(root, rel)
    if _in_git(root, target):
        return "error: .git paths are not listable"
    if not target.exists():
        return f"error: {rel} does not exist"
    if target.is_file():
        return f"error: {rel} is a file"
    try:
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        return f"error: {exc}"
    rows: list[str] = []
    shown = 0
    for item in entries:
        if item.name in _SKIP_DIR:
            continue
        shown += 1
        if shown > _MAX_LIST:
            rows.append("... truncated")
            break
        if item.is_dir():
            rows.append(f"dir\t{item.name}")
            continue
        try:
            size = item.stat().st_size
        except OSError:
            size = 0
        rows.append(f"file\t{item.name}\t{size}B")
    return "\n".join(rows) if rows else "(empty)"


def _read_file(root: Path, rel: str, *, offset: Any, limit: Any) -> str:
    target = resolve_under(root, rel)
    if _in_git(root, target):
        return "error: .git paths are not readable"
    if not target.is_file():
        return f"error: {rel} is not a file"
    try:
        size = target.stat().st_size
    except OSError as exc:
        return f"error: {exc}"
    if size > _MAX_READ_BYTES:
        return f"error: file is {size} bytes; too large to read"
    data = target.read_bytes()
    if b"\x00" in data[:4096]:
        return "error: binary file"
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    start = max(1, _as_int(offset, 1))
    n = min(_MAX_READ_LINES, max(1, _as_int(limit, 200)))
    chunk = lines[start - 1 : start - 1 + n]
    numbered = [f"{i}: {line}" for i, line in enumerate(chunk, start=start)]
    body = "\n".join(numbered) if numbered else "(empty)"
    leftover = len(lines) - (start - 1 + len(chunk))
    if leftover > 0:
        body += f"\n... {leftover} more lines"
    return body


def _write_file(root: Path, rel: str, content: Any) -> str:
    target = resolve_under(root, rel)
    if _in_git(root, target):
        return "error: .git paths are not writable"
    if target.exists() and target.is_dir():
        return f"error: {rel} is a directory"
    if not isinstance(content, str):
        raw = json.dumps(content, ensure_ascii=False)
    else:
        raw = content
    if len(raw) > _MAX_WRITE_CHARS:
        return f"error: content is {len(raw)} chars; max {_MAX_WRITE_CHARS}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(raw, encoding="utf-8")
    rel_out = _rel_posix(root, target)
    return f"wrote {rel_out} ({len(raw)} chars)"


def _run_tests(root: Path, rel: str) -> str:
    target = resolve_under(root, rel)
    if _in_git(root, target):
        return "error: .git paths are not runnable"
    if not target.exists():
        return f"error: {rel} does not exist"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "--color=no",
        "--",
        str(target),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=_TEST_TIMEOUT_S,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        return f"error: {exc}"
    except subprocess.TimeoutExpired:
        return f"error: pytest timed out after {_TEST_TIMEOUT_S}s"
    blob = ((proc.stdout or "") + ("\n" if proc.stderr else "") + (proc.stderr or "")).strip()
    if proc.returncode == 1 and "No module named pytest" in blob:
        return "error: pytest is not installed in this Python"
    return f"exit {proc.returncode}\n{blob}"

"""Workspace tools for ``deepfold chat --agent``.

v2 surface: search, patch, git read, allowlisted argv, todo. Opt-in
``web_search`` (Brave Search, or Google CSE on an old Cloud project) is
off until ``--agent-web``. Paths stay under ``--workspace``. Writes,
tests, commands, and web follow ``--agent-trust``. Not an unrestricted
shell. Session KV is Wave B (TokenLoop); this module is CPU-only.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from difflib import unified_diff
from pathlib import Path
from typing import Any, Iterable, Sequence

from .paths import deepfold_home

__all__ = [
    "EDIT_TOOLS",
    "NET_TOOLS",
    "READ_TOOLS",
    "RUN_TOOLS",
    "SCHEMAS",
    "TOOL_END",
    "TOOL_START",
    "TRUST_LEVELS",
    "AgentSession",
    "PathEscape",
    "clip_result",
    "confirm_accepted",
    "degenerate_tool_text",
    "drop_agent_system",
    "edit_preview",
    "ensure_agent_system",
    "execute",
    "execute_calls",
    "flatten_history",
    "format_call",
    "load_workspace_rules",
    "looks_like_small_agent_model",
    "needs_confirm",
    "parse_tool_calls",
    "preview",
    "resolve_under",
    "run_read_only_parallel",
    "strip_tool_xml",
    "suffix_after",
    "tool_call_closed",
    "tool_schemas",
    "truncated_tool_call",
    "visible_stream_text",
]

SYSTEM_MARK = "deepfold-agent-workspace:"
TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"
_START = TOOL_START
_END = TOOL_END
_SKIP_DIR = frozenset(
    {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache", ".mypy_cache"}
)
TRUST_LEVELS = ("ask", "write", "workspace")
READ_TOOLS = frozenset(
    {"list_dir", "read_file", "glob", "grep", "git_status", "git_diff"}
)
EDIT_TOOLS = frozenset({"write_file", "str_replace", "delete_file"})
RUN_TOOLS = frozenset({"run_tests", "run_argv"})
NET_TOOLS = frozenset({"web_search"})
_MAX_LIST = 200
_MAX_READ_BYTES = 1_000_000
_MAX_READ_LINES = 400
_MAX_WRITE_CHARS = 256 * 1024
_MAX_RESULT = 12_000
_MAX_RULES = 8 * 1024
_TEST_TIMEOUT_S = 120
_GREP_FILES = 50
_GREP_MATCHES = 80
_GREP_LINE = 200
_GIT_BLOCK = frozenset({"push", "fetch", "pull", "clone", "remote"})
_GIT_READ = frozenset({"status", "diff", "log", "show", "rev-parse"})
_PYTHON_STEMS = frozenset({"python", "python3"})
_ALLOW_STEMS = _PYTHON_STEMS | frozenset({"pytest", "ruff", "go", "git", "deepfold"})
_WEB_QUERY_MAX = 200
_WEB_NUM_DEFAULT = 5
_WEB_NUM_MAX = 8
_WEB_TIMEOUT_S = 15
_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_CSE_ENV_KEY = "DEEPFOLD_GOOGLE_CSE_KEY"
_CSE_ENV_CX = "DEEPFOLD_GOOGLE_CSE_CX"
_BRAVE_ENV = "DEEPFOLD_BRAVE_KEY"
_BRAVE_ENV_ALT = "BRAVE_API_KEY"
_CSE_FILE = "cse.env"
_SECRET_KEYS = frozenset(
    {_CSE_ENV_KEY, _CSE_ENV_CX, _BRAVE_ENV, _BRAVE_ENV_ALT}
)

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
                "offset": {"type": "integer", "description": "1-based start line (default 1)"},
                "limit": {
                    "type": "integer",
                    "description": "Max lines to return (default 200, max 400)",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "glob",
        "description": "List workspace paths matching a glob (e.g. gpu/cli/*.py).",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob relative to the workspace"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": "Regex search in workspace files. Caps: 50 files, 80 matches.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Python/ripgrep regex"},
                "path": {"type": "string", "description": "File or directory to search. Default: ."},
                "glob": {"type": "string", "description": "Optional filename glob, e.g. *.py"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "git_status",
        "description": "git status --porcelain in the workspace. Read-only.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "git_diff",
        "description": "git diff for an optional workspace path. Read-only.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Optional path inside the workspace"},
            },
        },
    },
    {
        "name": "str_replace",
        "description": (
            "Replace one unique old_string with new_string in a file. "
            "Prefer this over write_file. Needs confirmation at trust=ask."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a UTF-8 file. Prefer str_replace for edits.",
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
        "name": "delete_file",
        "description": "Delete one file in the workspace (not a directory).",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "run_tests",
        "description": "Run pytest on one workspace path. No extra pytest flags.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Test file or directory"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "run_argv",
        "description": (
            "Run an allowlisted command as an argv array (no shell). "
            "Allowed: python, pytest, ruff, go, git, deepfold. git push is refused."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Argument vector, e.g. [\"pytest\", \"-q\", \"tests\"]",
                }
            },
            "required": ["argv"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Web search via Brave Search, or Google CSE if those keys are set. "
            "Off unless --agent-web. Returns titles, URLs, and snippets. Not a browser."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num": {
                    "type": "integer",
                    "description": "Hits to return (default 5, max 8)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "todo",
        "description": "Replace the in-session plan. status: pending, in_progress, done.",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {"type": "string"},
                        },
                    },
                }
            },
            "required": ["items"],
        },
    },
]


@dataclass
class AgentSession:
    """Per-chat agent state. Not persisted except as JSON history text."""

    trust: str = "ask"
    web: bool = False
    todos: list[dict[str, str]] = field(default_factory=list)


class PathEscape(ValueError):
    """A tool path resolved outside the workspace."""


def looks_like_small_agent_model(cfg: dict[str, Any] | None) -> bool:
    """Qwen2.5-3B-class: tool JSON is unreliable. 14B is the agent plate."""
    if not cfg:
        return False
    try:
        hidden = int(cfg.get("hidden_size") or 0)
        layers = int(cfg.get("num_hidden_layers") or 0)
    except (TypeError, ValueError):
        return False
    return hidden <= 2048 and 1 <= layers <= 36


def load_workspace_rules(workspace: Path) -> str:
    """``AGENTS.md`` or ``.deepfold/instructions.md``, first hit, capped."""
    for rel in ("AGENTS.md", ".deepfold/instructions.md"):
        path = workspace / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) > _MAX_RULES:
            text = text[:_MAX_RULES] + "\n... truncated"
        return text.strip()
    return ""


def tool_schemas(*, web: bool = False) -> list[dict[str, Any]]:
    """Schemas advertised to the model. ``web_search`` stays off until opted in."""
    if web:
        return list(SCHEMAS)
    return [item for item in SCHEMAS if item["name"] != "web_search"]


def agent_system(workspace: Path, *, web: bool = False) -> str:
    rules = load_workspace_rules(workspace)
    extra = f"\nProject rules:\n{rules}\n" if rules else ""
    web_line = ""
    if web:
        web_line = (
            "web_search looks up current public facts (Brave Search, or Google "
            "CSE). Cite the returned URLs and do not invent links. Prefer "
            "glob/grep for this workspace.\n"
        )
    return (
        f"{SYSTEM_MARK} {workspace}\n"
        "You work only in that workspace. Use tools to search (glob, grep), read, "
        "patch (str_replace), and run tests. Prefer str_replace over write_file. "
        "Do not claim you cannot access files. Read a file before rewriting it. "
        "After a tool result, continue until the user's request is done. "
        "Do not invent file contents you have not read. Keep a todo list for "
        "multi-step work. There is no unrestricted shell; run_argv is allowlisted.\n"
        f"{web_line}"
        f"{extra}"
        "A tool call is one complete JSON object, then you stop:\n"
        "<tool_call>\n"
        '{"name": "grep", "arguments": {"query": "TokenLoop", "glob": "*.py"}}\n'
        "</tool_call>\n"
        "Never fill a tool call with punctuation. Always close </tool_call>."
    )


def ensure_agent_system(
    history: list[dict[str, Any]], workspace: Path, *, web: bool = False
) -> None:
    msg = {"role": "system", "content": agent_system(workspace, web=web)}
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


def tool_call_closed(text: str) -> bool:
    """Decode can stop: at least one complete ``</tool_call>`` with parseable JSON."""
    return bool(parse_tool_calls(text))


def truncated_tool_call(text: str) -> bool:
    """Opened ``<tool_call>`` but JSON did not parse and it is not punctuation spam."""
    if _START not in text or parse_tool_calls(text) or degenerate_tool_text(text):
        return False
    return True


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


def visible_stream_text(text: str) -> str:
    """Prose only: hide complete tool blocks and an unclosed ``<tool_call>``."""
    out: list[str] = []
    pos = 0
    while True:
        start = text.find(_START, pos)
        if start < 0:
            rest = text[pos:]
            lt = rest.rfind("<")
            if lt >= 0:
                tail = rest[lt:]
                if _START.startswith(tail) or tail.startswith("<tool"):
                    rest = rest[:lt]
            out.append(rest)
            break
        out.append(text[pos:start])
        end = text.find(_END, start)
        if end < 0:
            break
        pos = end + len(_END)
    return "".join(out)


def suffix_after(old: Sequence[int], new: Sequence[int]) -> list[int] | None:
    """Token suffix for Wave B session KV. None means full prefill."""
    prev = [int(x) for x in old]
    cur = [int(x) for x in new]
    n = len(prev)
    if len(cur) < n:
        return None
    if prev != cur[:n]:
        return None
    return cur[n:]


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


def _normalize_trust(trust: str) -> str:
    text = (trust or "ask").strip().lower()
    return text if text in TRUST_LEVELS else "ask"


def _argv_list(arguments: dict[str, Any]) -> list[str]:
    raw = arguments.get("argv")
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw]


def _exe_stem(argv0: str) -> str:
    name = Path(argv0).name
    lower = name.lower()
    if lower.endswith(".exe"):
        lower = lower[:-4]
    return lower


def _git_mutating(arguments: dict[str, Any]) -> bool:
    argv = _argv_list(arguments)
    if not argv or _exe_stem(argv[0]) != "git":
        return False
    sub = argv[1].lstrip("-") if len(argv) > 1 else ""
    return sub not in _GIT_READ


def needs_confirm(
    name: str,
    arguments: dict[str, Any] | None = None,
    trust: str = "ask",
) -> bool:
    arguments = arguments if isinstance(arguments, dict) else {}
    level = _normalize_trust(trust)
    if name in READ_TOOLS or name == "todo":
        return False
    if name == "run_argv" and _git_mutating(arguments):
        return True
    if name in EDIT_TOOLS:
        return level == "ask"
    if name in RUN_TOOLS or name in NET_TOOLS:
        return level != "workspace"
    return True


def confirm_accepted(line: str) -> bool:
    return line.strip().lower() in ("y", "yes", "д", "да")


def format_call(name: str, arguments: dict[str, Any]) -> str:
    if name == "write_file":
        path = arguments.get("path", "")
        content = arguments.get("content", "")
        n = len(content) if isinstance(content, str) else 0
        return f"write_file {path} ({n} chars)"
    if name == "str_replace":
        return f"str_replace {arguments.get('path', '')}"
    if name == "delete_file":
        return f"delete_file {arguments.get('path', '')}"
    if name == "run_tests":
        return f"run_tests {arguments.get('path', '')}"
    if name == "run_argv":
        return "run_argv " + " ".join(_argv_list(arguments))
    if name == "read_file":
        return f"read_file {arguments.get('path', '')}"
    if name == "list_dir":
        return f"list_dir {arguments.get('path', '.')}"
    if name == "glob":
        return f"glob {arguments.get('pattern', '')}"
    if name == "grep":
        return f"grep {arguments.get('query', '')}"
    if name == "git_diff":
        return f"git_diff {arguments.get('path', '.')}"
    if name == "todo":
        items = arguments.get("items")
        n = len(items) if isinstance(items, list) else 0
        return f"todo {n} items"
    if name == "web_search":
        return f"web_search {arguments.get('query', '')}"
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


def _unified(path: str, before: str, after: str) -> str:
    diff = unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff) or "(no textual diff)"


def edit_preview(name: str, arguments: dict[str, Any], workspace: Path) -> str | None:
    """Unified diff for confirm. None if there is nothing to show."""
    if name not in EDIT_TOOLS:
        return None
    try:
        if name == "str_replace":
            path = str(arguments.get("path") or "")
            old = arguments.get("old_string")
            new = arguments.get("new_string")
            if not path or not isinstance(old, str) or not isinstance(new, str):
                return None
            target = resolve_under(workspace, path)
            if not target.is_file():
                return None
            before = target.read_text(encoding="utf-8")
            if before.count(old) != 1:
                return None
            return _unified(_rel_posix(workspace, target), before, before.replace(old, new, 1))
        if name == "write_file":
            path = str(arguments.get("path") or "")
            content = arguments.get("content")
            if not path or not isinstance(content, str):
                return None
            target = resolve_under(workspace, path)
            before = target.read_text(encoding="utf-8") if target.is_file() else ""
            return _unified(path.replace("\\", "/"), before, content)
        if name == "delete_file":
            return f"delete {arguments.get('path', '')}"
    except (OSError, PathEscape, UnicodeError):
        return None
    return None


def execute(
    name: str,
    arguments: dict[str, Any],
    workspace: Path,
    *,
    session: AgentSession | None = None,
) -> str:
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
        if name == "glob":
            pattern = arguments.get("pattern")
            if not pattern:
                return "error: pattern is required"
            return _glob(workspace, str(pattern))
        if name == "grep":
            query = arguments.get("query")
            if not query:
                return "error: query is required"
            return _grep(
                workspace,
                str(query),
                path=str(arguments.get("path") or "."),
                glob_pat=arguments.get("glob"),
            )
        if name == "git_status":
            return _git_status(workspace)
        if name == "git_diff":
            return _git_diff(workspace, str(arguments.get("path") or ""))
        if name == "str_replace":
            path = arguments.get("path")
            if not path:
                return "error: path is required"
            return _str_replace(
                workspace,
                str(path),
                arguments.get("old_string"),
                arguments.get("new_string"),
            )
        if name == "write_file":
            path = arguments.get("path")
            if not path:
                return "error: path is required"
            if "content" not in arguments:
                return "error: content is required"
            return _write_file(workspace, str(path), arguments.get("content"))
        if name == "delete_file":
            path = arguments.get("path")
            if not path:
                return "error: path is required"
            return _delete_file(workspace, str(path))
        if name == "run_tests":
            path = arguments.get("path")
            if not path:
                return "error: path is required"
            return _run_tests(workspace, str(path))
        if name == "run_argv":
            return _run_argv(workspace, arguments.get("argv"))
        if name == "web_search":
            return _web_search(arguments, session=session)
        if name == "todo":
            return _todo(session, arguments.get("items"))
        names = ", ".join(item["name"] for item in SCHEMAS)
        return f"error: unknown tool {name!r}. available: {names}"
    except PathEscape as exc:
        return f"error: {exc}"
    except OSError as exc:
        return f"error: {exc}"


def execute_calls(
    calls: list[dict[str, Any]],
    workspace: Path,
    *,
    session: AgentSession | None = None,
    confirm: Any = None,
) -> list[tuple[str, dict[str, Any], str]]:
    """Run one assistant's tool list. ``confirm(name, args) -> bool`` or None=allow."""

    def _one(call: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        name = str(call.get("name") or "")
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        allowed = True if confirm is None else bool(confirm(name, arguments))
        result = (
            execute(name, arguments, workspace, session=session)
            if allowed
            else "denied by user"
        )
        return name, arguments, clip_result(result)

    if calls and all(str(c.get("name") or "") in READ_TOOLS for c in calls) and len(calls) > 1:
        return run_read_only_parallel(calls, workspace, session=session)
    return [_one(call) for call in calls]


def run_read_only_parallel(
    calls: list[dict[str, Any]],
    workspace: Path,
    *,
    session: AgentSession | None = None,
) -> list[tuple[str, dict[str, Any], str]]:
    """Host-side fan-out for read tools. Order matches ``calls``."""
    del session
    indexed = list(enumerate(calls))
    out: list[tuple[str, dict[str, Any], str] | None] = [None] * len(calls)

    def _job(item: tuple[int, dict[str, Any]]) -> tuple[int, str, dict[str, Any], str]:
        i, call = item
        name = str(call.get("name") or "")
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        result = clip_result(execute(name, arguments, workspace))
        return i, name, arguments, result

    workers = min(4, len(calls))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_job, item) for item in indexed]
        for fut in as_completed(futs):
            i, name, arguments, result = fut.result()
            out[i] = (name, arguments, result)
    return [row for row in out if row is not None]


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


def _glob(root: Path, pattern: str) -> str:
    text = pattern.strip().replace("\\", "/")
    if not text or text.startswith("/") or text.startswith("~") or ":" in text[:3]:
        return "error: pattern must be a relative workspace glob"
    hits: list[str] = []
    try:
        matches: Iterable[Path] = root.glob(text)
    except ValueError as exc:
        return f"error: {exc}"
    for item in sorted(matches, key=lambda p: str(p).lower()):
        try:
            rel = _rel_posix(root, item)
            resolve_under(root, rel)
        except (OSError, PathEscape, ValueError):
            continue
        if _in_git(root, item) or any(part in _SKIP_DIR for part in Path(rel).parts):
            continue
        kind = "dir" if item.is_dir() else "file"
        hits.append(f"{kind}\t{rel}")
        if len(hits) >= _MAX_LIST:
            hits.append("... truncated")
            break
    return "\n".join(hits) if hits else "(no matches)"


def _iter_grep_files(start: Path, root: Path, glob_pat: str | None) -> Iterable[Path]:
    if start.is_file():
        yield start
        return
    if not start.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR]
        base = Path(dirpath)
        if _in_git(root, base):
            dirnames[:] = []
            continue
        for name in filenames:
            path = base / name
            if _in_git(root, path):
                continue
            if glob_pat and not fnmatch.fnmatch(name, str(glob_pat)):
                continue
            yield path


def _grep(root: Path, query: str, *, path: str, glob_pat: Any) -> str:
    try:
        rx = re.compile(query)
    except re.error as exc:
        return f"error: bad regex: {exc}"
    start = resolve_under(root, path or ".")
    if _in_git(root, start):
        return "error: .git paths are not searchable"
    glob_text = str(glob_pat) if glob_pat else None
    rows: list[str] = []
    files_hit = 0
    matches = 0
    for file in _iter_grep_files(start, root, glob_text):
        try:
            size = file.stat().st_size
        except OSError:
            continue
        if size > _MAX_READ_BYTES:
            continue
        try:
            data = file.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:4096]:
            continue
        text = data.decode("utf-8", errors="replace")
        file_had = False
        rel = _rel_posix(root, file)
        for i, line in enumerate(text.splitlines(), start=1):
            if not rx.search(line):
                continue
            if not file_had:
                files_hit += 1
                file_had = True
                if files_hit > _GREP_FILES:
                    rows.append("... truncated files")
                    return "\n".join(rows)
            matches += 1
            snippet = line[:_GREP_LINE]
            rows.append(f"{rel}:{i}:{snippet}")
            if matches >= _GREP_MATCHES:
                rows.append("... truncated matches")
                return "\n".join(rows)
    return "\n".join(rows) if rows else "(no matches)"


def _git_bin() -> str | None:
    return shutil.which("git")


def _git_status(root: Path) -> str:
    git = _git_bin()
    if git is None:
        return "error: git not on PATH"
    try:
        proc = subprocess.run(
            [git, "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"error: {exc}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return f"error: git status failed: {err or proc.returncode}"
    body = (proc.stdout or "").rstrip()
    return body if body else "(clean)"


def _git_diff(root: Path, rel: str) -> str:
    git = _git_bin()
    if git is None:
        return "error: git not on PATH"
    cmd = [git, "--no-pager", "diff", "--"]
    if rel.strip():
        target = resolve_under(root, rel)
        if _in_git(root, target):
            return "error: .git paths are not readable"
        cmd.append(_rel_posix(root, target))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"error: {exc}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return f"error: git diff failed: {err or proc.returncode}"
    body = (proc.stdout or "").rstrip()
    return body if body else "(no diff)"


def _str_replace(root: Path, rel: str, old: Any, new: Any) -> str:
    if not isinstance(old, str) or not isinstance(new, str):
        return "error: old_string and new_string must be strings"
    target = resolve_under(root, rel)
    if _in_git(root, target):
        return "error: .git paths are not writable"
    if not target.is_file():
        return f"error: {rel} is not a file"
    before = target.read_text(encoding="utf-8")
    n = before.count(old)
    if n == 0:
        return "error: old_string not found"
    if n > 1:
        return f"error: old_string matched {n} times; must match once"
    after = before.replace(old, new, 1)
    target.write_text(after, encoding="utf-8")
    rel_out = _rel_posix(root, target)
    diff = _unified(rel_out, before, after)
    return f"ok {rel_out}\n{diff}".rstrip()


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
    before = target.read_text(encoding="utf-8") if target.is_file() else ""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(raw, encoding="utf-8")
    rel_out = _rel_posix(root, target)
    diff = _unified(rel_out, before, raw)
    return f"wrote {rel_out} ({len(raw)} chars)\n{diff}".rstrip()


def _delete_file(root: Path, rel: str) -> str:
    target = resolve_under(root, rel)
    if _in_git(root, target):
        return "error: .git paths are not writable"
    if not target.exists():
        return f"error: {rel} does not exist"
    if target.is_dir():
        return f"error: {rel} is a directory"
    target.unlink()
    return f"deleted {_rel_posix(root, target)}"


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
    return _run_cmd(cmd, cwd=root, timeout=_TEST_TIMEOUT_S)


def _run_argv(root: Path, argv: Any) -> str:
    if not isinstance(argv, list) or not argv:
        return "error: argv must be a non-empty JSON array of strings"
    parts = [str(x) for x in argv]
    if any(not p for p in parts):
        return "error: argv entries must be non-empty strings"
    stem = _exe_stem(parts[0])
    if stem in {"powershell", "pwsh", "cmd", "bash", "sh", "zsh", "fish"}:
        return "error: shell interpreters are not allowed"
    if stem not in _ALLOW_STEMS:
        allowed = ", ".join(sorted(_ALLOW_STEMS))
        return f"error: argv[0] {parts[0]!r} is not allowlisted ({allowed})"
    if stem == "git":
        sub = parts[1].lstrip("-") if len(parts) > 1 else ""
        if sub in _GIT_BLOCK:
            return f"error: git {sub} is not allowed"
        git = _git_bin()
        if git is None:
            return "error: git not on PATH"
        cmd = [git, *parts[1:]]
        return _run_cmd(cmd, cwd=root, timeout=_TEST_TIMEOUT_S)
    if stem in _PYTHON_STEMS:
        rest = parts[1:]
        if any(p == "-c" or (p.startswith("-c") and not p.startswith("--")) for p in rest):
            return "error: python -c is not allowed"
        if not rest:
            return "error: python needs -m, a workspace .py file, or a flag"
        if rest[0] == "-m":
            if len(rest) < 2:
                return "error: python -m needs a module"
            return _run_cmd([sys.executable, *rest], cwd=root, timeout=_TEST_TIMEOUT_S)
        i = 0
        while i < len(rest) and rest[i].startswith("-") and rest[i] not in ("-", "--"):
            i += 1
        flags, leftover = rest[:i], rest[i:]
        if leftover:
            script = resolve_under(root, leftover[0])
            if _in_git(root, script):
                return "error: .git paths are not runnable"
            if script.suffix.lower() != ".py":
                return "error: python may only take -m or a workspace .py file"
            cmd = [sys.executable, *flags, str(script), *leftover[1:]]
        else:
            cmd = [sys.executable, *flags]
        return _run_cmd(cmd, cwd=root, timeout=_TEST_TIMEOUT_S)
    if stem == "pytest":
        cmd = [sys.executable, "-m", "pytest", *parts[1:]]
        return _run_cmd(cmd, cwd=root, timeout=_TEST_TIMEOUT_S)
    if stem == "deepfold":
        cmd = [sys.executable, "-m", "gpu.cli", *parts[1:]]
        return _run_cmd(cmd, cwd=root, timeout=_TEST_TIMEOUT_S)
    hit = shutil.which(parts[0]) or shutil.which(stem)
    if hit is None:
        return f"error: {stem} not on PATH"
    return _run_cmd([hit, *parts[1:]], cwd=root, timeout=_TEST_TIMEOUT_S)


def _run_cmd(cmd: list[str], *, cwd: Path, timeout: int) -> str:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        return f"error: {exc}"
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout}s"
    blob = ((proc.stdout or "") + ("\n" if proc.stderr else "") + (proc.stderr or "")).strip()
    if proc.returncode == 1 and "No module named pytest" in blob:
        return "error: pytest is not installed in this Python"
    return clip_result(f"exit {proc.returncode}\n{blob}")


def _http_get(
    url: str,
    *,
    timeout: int,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    """GET ``url``. Tests patch this; production never logs the URL (it may hold keys)."""
    hdrs = {
        "User-Agent": "DeepfoldAgent/0.1",
        "Accept": "application/json",
    }
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, method="GET", headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = int(getattr(resp, "status", 200) or 200)
        return status, resp.read(_MAX_READ_BYTES)


def _cse_url(query: str, num: int, *, key: str, cx: str) -> str:
    params = urllib.parse.urlencode(
        {"key": key, "cx": cx, "q": query, "num": str(num)}
    )
    return f"{_CSE_ENDPOINT}?{params}"


def _format_cse(payload: Any, *, query: str) -> str:
    if not isinstance(payload, dict):
        return "error: google cse returned non-JSON"
    err = payload.get("error")
    if isinstance(err, dict):
        msg = str(err.get("message") or err.get("status") or "error")
        return f"error: google cse: {msg}"
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return f"(no results for {query!r})"
    rows: list[str] = []
    for i, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        title = " ".join(str(item.get("title") or "").split())
        link = str(item.get("link") or "").strip()
        snippet = " ".join(str(item.get("snippet") or "").split())
        if not link:
            continue
        block = f"{len(rows) + 1}. {title}\n   {link}"
        if snippet:
            block += f"\n   {snippet}"
        rows.append(block)
        if len(rows) >= _WEB_NUM_MAX:
            break
    return "\n".join(rows) if rows else f"(no results for {query!r})"


def _cse_file() -> Path:
    return deepfold_home() / _CSE_FILE


def _parse_cse_file(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        name, _, value = raw.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name in _SECRET_KEYS and value:
            out[name] = value
    return out


def _cse_creds() -> tuple[str, str]:
    """API key + engine id. Env wins; else ``$DEEPFOLD_HOME/cse.env``. Never log values."""
    key = os.environ.get(_CSE_ENV_KEY, "").strip()
    cx = os.environ.get(_CSE_ENV_CX, "").strip()
    if key and cx:
        return key, cx
    file = _parse_cse_file(_cse_file())
    return key or file.get(_CSE_ENV_KEY, ""), cx or file.get(_CSE_ENV_CX, "")


def _brave_key() -> str:
    """Brave token. Env wins; else ``$DEEPFOLD_HOME/cse.env``. Never log the value."""
    for name in (_BRAVE_ENV, _BRAVE_ENV_ALT):
        text = os.environ.get(name, "").strip()
        if text:
            return text
    file = _parse_cse_file(_cse_file())
    return file.get(_BRAVE_ENV, "") or file.get(_BRAVE_ENV_ALT, "")


def _format_brave(payload: Any, *, query: str) -> str:
    if not isinstance(payload, dict):
        return "error: brave search returned non-JSON"
    err = payload.get("error") or payload.get("message")
    if isinstance(err, dict):
        msg = str(err.get("message") or err.get("code") or "error")
        return f"error: brave search: {msg}"
    if isinstance(err, str) and err.strip() and "web" not in payload:
        return f"error: brave search: {err.strip()}"
    web = payload.get("web") if isinstance(payload.get("web"), dict) else {}
    items = web.get("results") if isinstance(web, dict) else None
    if not isinstance(items, list) or not items:
        return f"(no results for {query!r})"
    rows: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = " ".join(str(item.get("title") or "").split())
        link = str(item.get("url") or "").strip()
        snippet = " ".join(str(item.get("description") or "").split())
        if not link:
            continue
        block = f"{len(rows) + 1}. {title}\n   {link}"
        if snippet:
            block += f"\n   {snippet}"
        rows.append(block)
        if len(rows) >= _WEB_NUM_MAX:
            break
    return "\n".join(rows) if rows else f"(no results for {query!r})"


def _google_search(query: str, num: int, *, key: str, cx: str) -> str:
    url = _cse_url(query, num, key=key, cx=cx)
    try:
        status, body = _http_get(url, timeout=_WEB_TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        code = int(getattr(exc, "code", 0) or 0)
        return f"error: google cse HTTP {code or 'error'}"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return "error: google cse request failed"
    if status != 200:
        return f"error: google cse HTTP {status}"
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, ValueError):
        return "error: google cse returned non-JSON"
    return clip_result(_format_cse(payload, query=query))


def _brave_search(query: str, num: int, *, token: str) -> str:
    params = urllib.parse.urlencode({"q": query, "count": str(num)})
    url = f"{_BRAVE_ENDPOINT}?{params}"
    try:
        status, body = _http_get(
            url,
            timeout=_WEB_TIMEOUT_S,
            headers={"X-Subscription-Token": token},
        )
    except urllib.error.HTTPError as exc:
        code = int(getattr(exc, "code", 0) or 0)
        if code == 429:
            return "error: brave search HTTP 429 (rate limit or credits exhausted)"
        if code in (401, 403):
            return f"error: brave search HTTP {code} (check DEEPFOLD_BRAVE_KEY)"
        if code == 422:
            return "error: brave search HTTP 422 (plan does not include Web Search)"
        return f"error: brave search HTTP {code or 'error'}"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return "error: brave search request failed"
    if status != 200:
        return f"error: brave search HTTP {status}"
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, ValueError):
        return "error: brave search returned non-JSON"
    return clip_result(_format_brave(payload, query=query))


def _web_search(
    arguments: dict[str, Any], *, session: AgentSession | None
) -> str:
    if session is None or not session.web:
        return "error: web_search is off (pass --agent-web or /agent web on)"
    query = str(arguments.get("query") or "").strip()
    if not query:
        return "error: query is required"
    if len(query) > _WEB_QUERY_MAX:
        return f"error: query longer than {_WEB_QUERY_MAX} characters"
    num = _as_int(arguments.get("num"), _WEB_NUM_DEFAULT)
    num = max(1, min(_WEB_NUM_MAX, num))
    brave = _brave_key()
    if brave:
        return _brave_search(query, num, token=brave)
    key, cx = _cse_creds()
    if key and cx:
        return _google_search(query, num, key=key, cx=cx)
    if cx and not key:
        return (
            "error: Google CSE cx is set but DEEPFOLD_GOOGLE_CSE_KEY is missing. "
            "New Google Cloud projects cannot use that JSON API. "
            "Set DEEPFOLD_BRAVE_KEY (Brave Search) instead. See docs/web-search.md."
        )
    if key and not cx:
        return "error: web_search needs DEEPFOLD_GOOGLE_CSE_CX, or set DEEPFOLD_BRAVE_KEY."
    return (
        "error: web_search needs DEEPFOLD_BRAVE_KEY (Brave Search API) "
        "in the environment or $DEEPFOLD_HOME/cse.env. "
        "Google CSE is closed to new Cloud projects. See docs/web-search.md."
    )


def _todo(session: AgentSession | None, items: Any) -> str:
    if session is None:
        return "error: todo needs an agent session"
    if not isinstance(items, list):
        return "error: items must be a JSON array"
    cleaned: list[dict[str, str]] = []
    for i, raw in enumerate(items, start=1):
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("status") or "pending").strip().lower()
        if status not in ("pending", "in_progress", "done"):
            status = "pending"
        content = str(raw.get("content") or "").strip()
        if not content:
            continue
        ident = str(raw.get("id") or i)
        cleaned.append({"id": ident, "content": content, "status": status})
    session.todos = cleaned
    if not cleaned:
        return "(empty plan)"
    return "\n".join(f"{row['status']}\t{row['id']}\t{row['content']}" for row in cleaned)

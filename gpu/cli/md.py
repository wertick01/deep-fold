"""Markdown + LaTeX for ``deepfold chat``. No extra dependency.

Tokens are written as they arrive. Complete spans are rewritten: fenced
code, inline code, bold, headers, lists, links, and ``$...$`` / ``$$`` /
``\\(...\\)`` math (Unicode, not KaTeX). Incomplete openers stay buffered.
``color`` only adds ANSI; math and markdown markers are consumed either way.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Callable, TextIO

from .tex import latex_to_unicode

__all__ = [
    "MarkdownStream",
    "color_enabled",
    "latex_to_unicode",
    "strip_ansi",
    "to_ansi",
]

RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
ITALIC = "\x1b[3m"
UNDERLINE = "\x1b[4m"
GREEN = "\x1b[32m"
CYAN = "\x1b[36m"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_HOLD_MAX = 8000
_LATEXY = re.compile(r"[\\^_]|frac|sum|int|alpha|beta|theta|lambda|pi|infty")


def color_enabled(stream: TextIO | None = None) -> bool:
    """False when piped, dumb, or ``NO_COLOR`` is set. Enables VT on Windows."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    out = sys.stdout if stream is None else stream
    if out is None or not hasattr(out, "isatty") or not out.isatty():
        return False
    if os.name == "nt":
        _enable_vt(-11)
        _enable_vt(-12)
    return True


def _enable_vt(handle_id: int = -11) -> None:
    try:
        import ctypes

        handle = ctypes.windll.kernel32.GetStdHandle(int(handle_id))
        mode = ctypes.c_uint32()
        if ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        return


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


class MarkdownStream:
    """Streaming markdown/math. Fences stay green; math becomes Unicode."""

    def __init__(self, write: Callable[[str], None], *, color: bool) -> None:
        self._write = write
        self._color = bool(color)
        self._fence = False
        self._ticks = 0
        self._hold = ""
        self._fence_run = ""

    def feed(self, piece: str) -> None:
        if not piece:
            return
        for ch in piece:
            if ch == "`":
                self._ticks += 1
                if self._ticks == 3:
                    if self._fence:
                        self._flush_fence()
                    else:
                        self._drain(eof=True)
                    marker = "```"
                    self._write(self._wrap(marker, DIM) if self._color else marker)
                    self._fence = not self._fence
                    self._ticks = 0
                continue
            if self._ticks:
                self._push("`" * self._ticks)
                self._ticks = 0
            self._push(ch)

    def close(self) -> None:
        if self._ticks:
            self._push("`" * self._ticks)
            self._ticks = 0
        self._flush_fence()
        self._drain(eof=True)

    def _flush_fence(self) -> None:
        if not self._fence_run:
            return
        text = self._fence_run
        self._fence_run = ""
        self._write(self._wrap(text, GREEN) if self._color else text)

    def _push(self, text: str) -> None:
        if self._fence:
            self._fence_run += text
            return
        self._hold += text
        if len(self._hold) > _HOLD_MAX:
            self._drain(eof=True)
            return
        self._drain(eof=False)

    def _wrap(self, text: str, code: str) -> str:
        if not text:
            return text
        return f"{code}{text}{RESET}"

    def _math(self, src: str, *, display: bool) -> str:
        body = latex_to_unicode(src)
        if not body:
            return ""
        if self._color:
            body = self._wrap(body, CYAN)
        if display:
            return f"\n  {body.replace(chr(10), chr(10) + '  ')}\n"
        return body

    def _drain(self, *, eof: bool) -> None:
        s = self._hold
        i = 0
        n = len(s)
        while i < n:
            if s.startswith("$$", i):
                j = s.find("$$", i + 2)
                if j < 0:
                    break
                self._write(self._math(s[i + 2 : j], display=True))
                i = j + 2
                continue
            if s.startswith("<tool_call>", i) or (
                s[i] == "<" and "<tool_call>".startswith(s[i:])
            ):
                tag = "<tool_call>"
                if not s.startswith(tag, i):
                    break
                j = s.find("</tool_call>", i + len(tag))
                if j < 0:
                    if eof:
                        i = n
                    break
                i = j + len("</tool_call>")
                continue
            if s[i] == "$":
                j = _find_dollar(s, i + 1)
                if j < 0:
                    break
                self._write(self._math(s[i + 1 : j], display=False))
                i = j + 1
                continue
            if s[i] == "\\":
                if i + 1 >= n:
                    break
                if s.startswith("\\[", i) or s.startswith("\\(", i):
                    closer = "\\]" if s[i + 1] == "[" else "\\)"
                    j = s.find(closer, i + 2)
                    if j < 0:
                        break
                    self._write(self._math(s[i + 2 : j], display=s[i + 1] == "["))
                    i = j + 2
                    continue
            if s[i] == "`":
                j = s.find("`", i + 1)
                if j < 0:
                    break
                inner = s[i + 1 : j]
                self._write(self._wrap(inner, GREEN) if self._color else inner)
                i = j + 1
                continue
            if s.startswith("**", i):
                j = s.find("**", i + 2)
                if j < 0:
                    break
                inner = s[i + 2 : j]
                self._write(self._wrap(inner, BOLD) if self._color else inner)
                i = j + 2
                continue
            if s[i] == "*" and not _star_list(s, i):
                j = _find_italic(s, i + 1)
                if j < 0:
                    break
                inner = s[i + 1 : j]
                self._write(self._wrap(inner, ITALIC) if self._color else inner)
                i = j + 1
                continue
            if s[i] == "[" or s.startswith("![", i):
                taken, end = _take_link(s, i)
                if end < 0:
                    break
                if end > i:
                    self._write(taken)
                    i = end
                    continue
            if _line_start(s, i):
                taken, end = _take_block(s, i, color=self._color, wrap=self._wrap)
                if end < 0:
                    break
                if end > i:
                    self._write(taken)
                    i = end
                    continue
            self._write(s[i])
            i += 1
        rest = s[i:]
        if eof and rest:
            taken, end = _take_block(
                rest + "\n", 0, color=self._color, wrap=self._wrap
            )
            if end > 0:
                self._write(taken)
                rest = (rest + "\n")[end:]
                if rest == "\n":
                    rest = ""
            if rest:
                self._write(_flush_tail(rest, math=self._math))
            rest = ""
        self._hold = rest


def _line_start(s: str, i: int) -> bool:
    return i == 0 or s[i - 1] == "\n"


def _star_list(s: str, i: int) -> bool:
    return _line_start(s, i) and i + 1 < len(s) and s[i + 1] in " \t"


def _find_dollar(s: str, start: int) -> int:
    j = start
    while j < len(s):
        if s[j] == "\\" and j + 1 < len(s):
            j += 2
            continue
        if s[j] == "$":
            return j
        j += 1
    return -1


def _find_italic(s: str, start: int) -> int:
    j = start
    while j < len(s):
        if s.startswith("**", j):
            j += 2
            continue
        if s[j] == "*":
            return j
        if s[j] == "\n":
            return -1
        j += 1
    return -1


def _take_link(s: str, i: int) -> tuple[str, int]:
    """``end < 0`` wait, ``end == i`` not a link, else ``(shown, end)``."""
    start = i
    if s.startswith("![", i):
        i += 1
    if i >= len(s) or s[i] != "[":
        return "", start
    close = s.find("]", i + 1)
    if close < 0:
        return "", -1
    if close + 1 >= len(s):
        return "", -1
    if s[close + 1] != "(":
        return "", start
    end = s.find(")", close + 2)
    if end < 0:
        return "", -1
    label = s[i + 1 : close]
    url = s[close + 2 : end]
    return (label or url), end + 1


def _take_block(
    s: str,
    i: int,
    *,
    color: bool,
    wrap: Callable[[str, str], str],
) -> tuple[str, int]:
    rest = s[i:]
    nl = rest.find("\n")
    line, ended = (rest, False) if nl < 0 else (rest[:nl], True)
    hashes = 0
    while hashes < len(line) and line[hashes] == "#":
        hashes += 1
    if 1 <= hashes <= 6 and (len(line) == hashes or line[hashes] in " \t"):
        if not ended:
            return "", -1
        title = line[hashes:].strip()
        code = BOLD + UNDERLINE if hashes == 1 else BOLD
        painted = wrap(title, code) if color else title
        return painted + "\n", i + nl + 1
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    if not ended and (
        stripped in {"-", "*", "+", ">"}
        or stripped.isdigit()
        or (stripped.endswith(".") and stripped[:-1].isdigit())
    ):
        return "", -1
    if stripped in {"---", "***", "___"}:
        if not ended:
            return "", -1
        rule = "─" * 12
        return (wrap(rule, DIM) if color else rule) + "\n", i + nl + 1
    bullet = False
    if stripped.startswith(("- ", "* ", "+ ")):
        bullet = True
        body = stripped[2:]
        mark = "• "
    else:
        k = 0
        while k < len(stripped) and stripped[k].isdigit():
            k += 1
        if k and stripped.startswith(". ", k):
            bullet = True
            body = stripped[k + 2 :]
            mark = stripped[: k + 2]
        else:
            body = stripped
            mark = ""
    if bullet:
        if not ended:
            return "", -1
        return f"{indent}{mark}{body}\n", i + nl + 1
    if stripped.startswith("> "):
        if not ended:
            return "", -1
        quote = "│ " + stripped[2:]
        painted = wrap(quote, DIM) if color else quote
        return indent + painted + "\n", i + nl + 1
    return "", i


def _flush_tail(rest: str, *, math: Callable[..., str]) -> str:
    if rest.startswith("$$"):
        inner = rest[2:]
        if _LATEXY.search(inner):
            return math(inner, display=True)
        return rest
    if rest.startswith("$"):
        inner = rest[1:]
        if _LATEXY.search(inner):
            return math(inner, display=False)
        return rest
    if rest.startswith("\\[") or rest.startswith("\\("):
        inner = rest[2:]
        if inner.endswith("\\]") or inner.endswith("\\)"):
            inner = inner[:-2]
        return math(inner, display=rest.startswith("\\["))
    return rest


def to_ansi(text: str, *, color: bool) -> str:
    parts: list[str] = []
    stream = MarkdownStream(parts.append, color=color)
    stream.feed(text)
    stream.close()
    return "".join(parts)

"""Copy Unicode text to the system clipboard. No extra package.

Windows uses CF_UNICODETEXT (so Russian survives). POSIX tries wl-copy,
xclip, then xsel.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable

__all__ = ["copy_text"]


def copy_text(text: str, *, put: Callable[[str], None] | None = None) -> None:
    """Put ``text`` on the clipboard. ``put`` is for tests. Raises OSError."""
    if put is not None:
        put(text)
        return
    if os.name == "nt":
        _windows(text)
        return
    _posix(text)


def _windows(text: str) -> None:
    import ctypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    cf_unicode = 13
    gmem_moveable = 0x0002
    payload = text.encode("utf-16le") + b"\x00\x00"
    if not user32.OpenClipboard(None):
        raise OSError("clipboard: OpenClipboard failed")
    try:
        user32.EmptyClipboard()
        handle = kernel32.GlobalAlloc(gmem_moveable, len(payload))
        if not handle:
            raise OSError("clipboard: GlobalAlloc failed")
        locked = kernel32.GlobalLock(handle)
        if not locked:
            kernel32.GlobalFree(handle)
            raise OSError("clipboard: GlobalLock failed")
        ctypes.memmove(locked, payload, len(payload))
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(cf_unicode, handle):
            kernel32.GlobalFree(handle)
            raise OSError("clipboard: SetClipboardData failed")
    finally:
        user32.CloseClipboard()


def _posix(text: str) -> None:
    last: BaseException | None = None
    for cmd in (
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
    ):
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(cmd, input=text, text=True, check=True, timeout=5)
            return
        except (OSError, subprocess.SubprocessError) as exc:
            last = exc
    raise OSError(
        "clipboard: install wl-copy, xclip, or xsel"
        + (f" ({last})" if last is not None else "")
    )

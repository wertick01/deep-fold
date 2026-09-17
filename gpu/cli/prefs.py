"""User defaults for ``deepfold chat`` agent / web_search.

Plain chat stays a talk session until the user asks for tools. Then
``--agent`` (and ``/agent on``) also turn on free Tavily ``web_search``
unless they said no. Persist with ``DEEPFOLD_AGENT`` / ``DEEPFOLD_AGENT_WEB``
or ``$DEEPFOLD_HOME/prefs.env``. ``run --prompt`` never reads these.
"""

from __future__ import annotations

import os
from pathlib import Path

from .paths import deepfold_home

__all__ = [
    "ENV_AGENT",
    "ENV_AGENT_WEB",
    "PREF_FILE",
    "parse_bool",
    "pref_tristate",
    "resolve_agent",
    "resolve_web",
    "save_pref",
    "web_hold_off",
]

ENV_AGENT = "DEEPFOLD_AGENT"
ENV_AGENT_WEB = "DEEPFOLD_AGENT_WEB"
PREF_FILE = "prefs.env"
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})
_KEYS = frozenset({ENV_AGENT, ENV_AGENT_WEB})


def parse_bool(text: str) -> bool | None:
    token = text.strip().lower()
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    return None


def _prefs_path() -> Path:
    return deepfold_home() / PREF_FILE


def _parse_file(path: Path) -> dict[str, str]:
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
        if name in _KEYS and value:
            out[name] = value
    return out


def pref_tristate(name: str) -> bool | None:
    """Env wins, else ``prefs.env``. ``None`` if unset or not a bool."""
    env = os.environ.get(name, "").strip()
    if env:
        return parse_bool(env)
    file = _parse_file(_prefs_path())
    raw = file.get(name, "")
    return parse_bool(raw) if raw else None


def resolve_agent(explicit: bool | None) -> bool:
    """CLI flag, else env/file. Unset → off (plain chat)."""
    if explicit is not None:
        return bool(explicit)
    return bool(pref_tristate(ENV_AGENT))


def resolve_web(explicit: bool | None, *, agent: bool) -> bool:
    """CLI flag, else env/file, else follow ``agent``."""
    if explicit is not None:
        return bool(explicit)
    stored = pref_tristate(ENV_AGENT_WEB)
    if stored is not None:
        return stored
    return bool(agent)


def web_hold_off(explicit: bool | None) -> bool:
    """True when this session must not auto-enable ``web_search``."""
    if explicit is False:
        return True
    return pref_tristate(ENV_AGENT_WEB) is False


def save_pref(name: str, value: bool) -> Path:
    """Write one bool into ``prefs.env``. Env vars are left alone."""
    if name not in _KEYS:
        raise ValueError(name)
    path = _prefs_path()
    rows = _parse_file(path)
    rows[name] = "1" if value else "0"
    lines = [f"{key}={rows[key]}" for key in (ENV_AGENT, ENV_AGENT_WEB) if key in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

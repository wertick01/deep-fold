"""Architecture gate: ``config.json`` decides before anything is allocated.

Two questions, deliberately split:

* **here**, from ``config.json`` alone (no torch, no skeleton, no GPU): is this
  a graph the driver already knows it cannot drive? MoE experts, a vision
  tower, a live sliding window, an activation we have no glue for, partial
  RoPE, or a family that is a *named* refusal (``gemma_gelu``,
  ``phi3_concat``, ``neox_interleaved``, ``gpt2_conv1d``). That check has to be
  cheap because it runs before ``chr compress`` spends twenty minutes packing.
* **later**, in :func:`gpu.host.attach.attach`: does the module tree really
  hold one uniform, fully classified slot set? That is the authority, and it is
  what refuses a Llama 3.2 with ``q_norm`` or an adapter's extra ``nn.Linear``.

So an unrecognised ``model_type`` is **not** a refusal by itself. A future
``olmo`` that is literally split-SwiGLU attaches without a catalog row; the gate
says so in :attr:`Gate.note` and the walker gets the final word at load time.
GGUF is refused on the path string, without opening the file.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

from gpu.graphs import (
    IMPLEMENTED_FAMILIES,
    INTERNLM_EXTRA_MODULES,
    config_family,
    config_refusal,
    needs_internlm_extras,
    needs_remote_code,
)

from . import messages
from .paths import looks_like_gguf

# wave8-install.md §5.3: Qwen must install without these; internlm2 must not
# die on a bare ModuleNotFoundError.
INTERNLM_MODULES = INTERNLM_EXTRA_MODULES


@dataclass(frozen=True)
class Gate:
    """Outcome of reading one ``config.json``. ``reason`` is stderr copy."""

    ok: bool
    model_type: str | None = None
    trust_remote_code: bool = False
    reason: str = ""
    family: str | None = None
    #: Non-fatal stderr copy: the config is not on the measured list, so the
    #: walker will decide at load time.
    note: str = ""
    needs_internlm: bool = False


def read_config(model_dir: Path) -> dict | None:
    """``config.json`` as a dict, or None when absent/unreadable."""
    try:
        with open(model_dir / "config.json", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return None
    return cfg if isinstance(cfg, dict) else None


def read_model_type(model_dir: Path) -> str | None:
    """``config.json`` -> ``model_type``, or None when absent/unreadable."""
    cfg = read_config(model_dir)
    if cfg is None:
        return None
    value = cfg.get("model_type")
    return str(value) if value else None


def has_tokenizer(model_dir: Path) -> bool:
    """Any of the files ``AutoTokenizer`` can start from."""
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "vocab.json",
    )
    return any((model_dir / name).is_file() for name in names)


def missing_internlm_extras() -> list[str]:
    return [m for m in INTERNLM_MODULES if importlib.util.find_spec(m) is None]


def gate(model: str) -> Gate:
    """Refuse GGUF, a missing config, and any graph the config already rules out."""
    if looks_like_gguf(model):
        return Gate(False, reason=messages.GGUF)

    path = Path(model)
    if path.is_file():
        if path.suffix == ".chr":
            return Gate(False, reason=messages.CHR_WITHOUT_SIDECAR)
        return Gate(
            False,
            reason=(
                f"--model must be a HuggingFace directory, not a file: {path}\n"
                + messages.CHR_WITHOUT_SIDECAR
            ),
        )
    if not path.is_dir():
        return Gate(False, reason=f"--model directory does not exist: {path}")

    if not (path / "config.json").is_file():
        return Gate(
            False,
            reason=(
                f"No config.json in {path}\n" + messages.CHR_WITHOUT_SIDECAR
            ),
        )

    cfg = read_config(path)
    if cfg is None:
        return Gate(
            False,
            reason=(
                f"config.json in {path} is not readable JSON.\n"
                + messages.CHR_WITHOUT_SIDECAR
            ),
        )

    model_type = str(cfg.get("model_type") or "") or None
    trust = needs_remote_code(cfg)
    internlm = needs_internlm_extras(cfg)

    refusal = config_refusal(cfg)
    if refusal is not None:
        return Gate(
            False,
            model_type,
            trust_remote_code=trust,
            reason=messages.unknown_arch(model_type, refusal),
            family=config_family(cfg),
            needs_internlm=internlm,
        )

    family = config_family(cfg)
    note = "" if family in IMPLEMENTED_FAMILIES else messages.deferred_arch(model_type)
    return Gate(
        True,
        model_type,
        trust_remote_code=trust,
        family=family,
        note=note,
        needs_internlm=internlm,
    )

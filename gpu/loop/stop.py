"""The greedy stop set, asked of the tokenizer. No hardcoded vocabulary.

The bug this module exists to prevent has a measured history: an early InternLM2
run stopped on Qwen's ``<|im_end|>`` (151645) while InternLM2's turn closes with
92542, so a 20B reply ran to ``max_new_tokens`` and then kept talking to itself.
The fix is not "add 92542": it is to never write an id down.

Product rules (``docs/tz/wave8-arch.md`` §3.1, ``wave10-product.md`` §2.4):

* a tokenizer is **required**. :func:`stop_token_ids` with ``None`` raises; it
  does not fall back to a Qwen pair. ``gpu/lab/script.py`` keeps a lab-only
  wrapper for its fixtures, and that wrapper is not this function.
* the set is the union of ``eos_token_id`` (int or list), the
  ``generation_config.json`` next to the model if there is one, and the
  end-of-turn names **this** tokenizer actually owns.
* ownership is a round trip, never ``convert_tokens_to_ids`` alone:
  that call answers ``unk_token_id`` for anything it has never heard of.
* ``pad_token_id`` is not a stop unless it is already in the set (InternLM2
  pads with ``</s>``, which happens to be its EOS; Llama pads with a token that
  is not).
* an empty set is an error. Inventing 151645 is how the first bug happened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

__all__ = ["END_OF_TURN_TOKENS", "special_id", "stop_token_ids"]

#: End-of-turn / end-of-text names worth asking about. Resolved against the
#: tokenizer in front of us: a name this tokenizer does not own contributes
#: nothing, so the list is free to mention every family.
END_OF_TURN_TOKENS = (
    "<|im_end|>",  # Qwen2.5, InternLM2 (92542, not 151645)
    "<|endoftext|>",  # Qwen base EOS
    "<|end_of_text|>",  # Llama 3
    "<|eot_id|>",  # Llama 3 chat turn
    "<|eom_id|>",  # Llama 3.1 tool turn
    "<end_of_turn>",  # Gemma
    "<|end|>",  # Phi-3
    "</s>",  # SentencePiece families
)


def special_id(tokenizer: Any, name: str) -> int | None:
    """``name``'s id, or ``None`` when this tokenizer does not own that token.

    ``convert_tokens_to_ids`` answers ``unk_token_id`` for an unknown piece, so
    the id is trusted only when it decodes back to the same string.
    """
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    try:
        token_id = convert(name)
    except Exception:  # noqa: BLE001 - a tokenizer with a different API
        return None
    if not isinstance(token_id, int) or token_id < 0:
        return None
    back = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(back):
        try:
            if back(token_id) != name:
                return None
        except Exception:  # noqa: BLE001
            return None
    elif token_id == getattr(tokenizer, "unk_token_id", None):
        return None
    return token_id


def _as_ids(value: Any) -> list[int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [int(v) for v in value if isinstance(v, int) and not isinstance(v, bool)]
    return []


def _generation_config_eos(model_dir: str | Path | None) -> list[int]:
    """``generation_config.json`` ``eos_token_id``, if that file is next to the model.

    InternLM2 keeps ``tokenizer.eos_token_id`` at ``</s>`` and lists the chat
    turn closer here instead, so skipping this file is how a turn stops being
    detected.
    """
    if not model_dir:
        return []
    path = Path(model_dir)
    if path.is_file():
        path = path.parent
    candidate = path / "generation_config.json"
    if not candidate.is_file():
        return []
    try:
        with open(candidate, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    return _as_ids(payload.get("eos_token_id"))


def stop_token_ids(
    tokenizer: Any,
    *,
    model_dir: str | Path | None = None,
    extra_names: Iterable[str] = (),
) -> tuple[int, ...]:
    """Sorted unique stop ids for greedy decoding. Raises rather than guessing.

    ``model_dir`` is optional and only used to read ``generation_config.json``;
    when it is omitted the tokenizer's own ``name_or_path`` is tried, because
    ``AutoTokenizer.from_pretrained(local_dir)`` records it.
    """
    if tokenizer is None:
        raise ValueError(
            "stop_token_ids: a tokenizer is required. There is no default stop "
            "set: hardcoding Qwen's 151643/151645 is why an InternLM2 reply once "
            "ran past the end of its turn."
        )

    ids: set[int] = set()
    ids.update(_as_ids(getattr(tokenizer, "eos_token_id", None)))

    where = model_dir if model_dir is not None else getattr(tokenizer, "name_or_path", None)
    ids.update(_generation_config_eos(where))

    for name in (*END_OF_TURN_TOKENS, *extra_names):
        token_id = special_id(tokenizer, name)
        if token_id is not None:
            ids.add(int(token_id))

    if not ids:
        raise ValueError(
            "attach: tokenizer has no EOS / end-of-turn id. Refusing to decode "
            "without a stop condition rather than inventing one."
        )
    return tuple(sorted(ids))

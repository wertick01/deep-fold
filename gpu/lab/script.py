"""The chat script both codecs answer, the quality needles, and this machine's paths.

The script is identical for BF16 and NF4 -- that is the whole point of the lab.
Turns are **independent**: each prompt is a fresh single-user conversation and
the KV cache is reset in between, so every recorded TTFT is a clean prefill and
the two codecs stay comparable.

Three short questions are smoke, not a benchmark. The needles only say "the
model is still answering in the right language and arithmetic"; the raw reply is
recorded in ``messages.csv`` either way.
"""

from __future__ import annotations

import os
from typing import Sequence

__all__ = [
    "CHR_PATH",
    "END_OF_TURN_TOKENS",
    "MAX_NEW_TOKENS",
    "MAX_SEQ",
    "MESSAGES",
    "MODEL_DIR",
    "NEEDLES",
    "POLL_INTERVAL_S",
    "QWEN_ENDOFTEXT",
    "QWEN_IM_END",
    "RUNS_DIR",
    "TURNS_NOTE",
    "chat_text",
    "quality_ok",
    "stop_token_ids",
]

MESSAGES = [
    "Reply with one short sentence. What is the capital of France?",
    "And the capital of Germany?",
    "What is 17 times 19? Reply with the number only.",
]

# Case-insensitive substrings, per 1-based message_id. Either language counts:
# the prompts are English but a 3B instruct model is allowed to be multilingual.
NEEDLES: dict[int, tuple[str, ...]] = {
    1: ("paris", "париж"),
    2: ("berlin", "берлин"),
    3: ("323",),
}

MAX_NEW_TOKENS = 64
MAX_SEQ = 512
POLL_INTERVAL_S = 0.12  # the freeze says <= 0.15 s

# Qwen2.5-Instruct stop tokens, used when there is no tokenizer to ask.
# `<|im_end|>` closes the assistant turn; `<|endoftext|>` is the base model's
# EOS and shows up on odd completions.
QWEN_IM_END = 151645
QWEN_ENDOFTEXT = 151643

# Names of end-of-turn specials, resolved against whatever tokenizer we get.
# InternLM2 keeps `<|im_end|>` at 92542 but leaves `eos_token` at `</s>` (2),
# so a set built from `eos_token_id` alone never ends an assistant turn.
END_OF_TURN_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<|eot_id|>")

TURNS_NOTE = (
    "independent turns: KV reset between messages, each prompt is a clean prefill"
)

# This machine. Overridable so the notebook can point at another checkout.
MODEL_DIR = os.environ.get("DEEPFOLD_MODEL", r"C:\dev\models\Qwen2.5-3B-Instruct")
CHR_PATH = os.environ.get("DEEPFOLD_CHR", r"C:\dev\models\qwen25-3b.nf4.chr")
RUNS_DIR = os.environ.get("DEEPFOLD_RUNS", r"C:\dev\models\runs")


def quality_ok(message_id: int, response: str) -> bool:
    """Does the reply to message ``message_id`` contain its needle?

    Unknown ``message_id`` -> ``False``: a turn nobody wrote a needle for is not
    silently a pass.
    """
    needles = NEEDLES.get(int(message_id))
    if not needles:
        return False
    text = (response or "").lower()
    return any(needle in text for needle in needles)


def _special_id(tokenizer, name: str) -> int | None:
    """``name``'s id, or ``None`` when this tokenizer does not have that token.

    ``convert_tokens_to_ids`` answers ``unk_token_id`` for anything it does not
    know, so the id is only trusted when it decodes back to the same string.
    """
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    try:
        token_id = convert(name)
    except Exception:  # noqa: BLE001 -- a tokenizer with a different API
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


def stop_token_ids(tokenizer=None) -> tuple[int, ...]:
    """Greedy stop set: this tokenizer's EOS plus its end-of-turn specials.

    Asked of the tokenizer rather than hardcoded, because the two are not the
    same model to model. Qwen2.5 ties them (``eos_token`` *is* ``<|im_end|>``,
    151645) and resolves to the same pair this used to hardcode, so the 3B and
    14B labs are unchanged. InternLM2 does not: ``eos_token`` is ``</s>`` (2)
    while the chat template closes the assistant turn with ``<|im_end|>``
    (92542). Missing 92542 is why a 20B reply ran to ``max_new_tokens`` and
    then kept talking to itself.

    With no tokenizer the Qwen pair is the answer, matching
    :func:`_qwen_fallback`.
    """
    if tokenizer is None:
        return (QWEN_ENDOFTEXT, QWEN_IM_END)
    ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(int(eos))
    elif isinstance(eos, (list, tuple)):
        ids.update(int(value) for value in eos if isinstance(value, int))
    for name in END_OF_TURN_TOKENS:
        token_id = _special_id(tokenizer, name)
        if token_id is not None:
            ids.add(token_id)
    if not ids:
        return (QWEN_ENDOFTEXT, QWEN_IM_END)
    return tuple(sorted(ids))


def _qwen_fallback(
    user_text: str,
    history: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Hardcoded Qwen2.5 chat template, for when the tokenizer has none."""
    parts = [
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    ]
    if history:
        for user, assistant in history:
            parts.append(f"<|im_start|>user\n{user}<|im_end|>\n")
            parts.append(f"<|im_start|>assistant\n{assistant}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user_text}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def chat_text(
    tokenizer,
    user_text: str,
    history: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Pack a user turn, optionally with prior (user, assistant) pairs.

    Smoke lab always passes ``history=None`` (see :data:`TURNS_NOTE`). Hard eval
    ``--history`` re-encodes the full chat so prefill grows; both codecs get the
    same string.
    """
    messages: list[dict[str, str]] = []
    if history:
        for user, assistant in history:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_text})
    try:
        packed = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # noqa: BLE001 -- no template, or transformers changed the call
        return _qwen_fallback(user_text, history)
    return packed if isinstance(packed, str) else _qwen_fallback(user_text, history)

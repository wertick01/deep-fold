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

from gpu.cli.paths import models_root, runs_root

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
# The list the driver actually uses is `gpu.loop.stop.END_OF_TURN_TOKENS`; this
# name stays as a stable lab import and is the same first four entries.
END_OF_TURN_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<|eot_id|>")

TURNS_NOTE = (
    "independent turns: KV reset between messages, each prompt is a clean prefill"
)

# Overridable so another machine does not need C:\dev\models.
MODEL_DIR = os.environ.get("DEEPFOLD_MODEL", str(models_root() / "Qwen2.5-3B-Instruct"))
CHR_PATH = os.environ.get("DEEPFOLD_CHR", str(models_root() / "qwen25-3b.nf4.chr"))
RUNS_DIR = os.environ.get("DEEPFOLD_RUNS", str(runs_root()))


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
    """Re-export of the driver's round-trip check (:mod:`gpu.loop.stop`)."""
    from gpu.loop.stop import special_id

    return special_id(tokenizer, name)


def stop_token_ids(tokenizer=None) -> tuple[int, ...]:
    """Lab wrapper around :func:`gpu.loop.stop.stop_token_ids`.

    With a tokenizer this *is* the product function: Qwen2.5 ties ``eos_token``
    to ``<|im_end|>`` (151645) and still resolves to the pair the 3B and 14B labs
    were measured with; InternLM2 does not tie them (``</s>`` = 2, turn close
    92542), which is why hardcoding the Qwen pair once made a 20B reply run to
    ``max_new_tokens`` and then keep talking to itself.

    The one difference is the ``None`` case. The product function **raises**
    there. This wrapper keeps the Qwen pair, and only because the lab's fixtures
    and :func:`_qwen_fallback` are a Qwen string on disk. Nothing on the
    ``deepfold run`` path may use it.
    """
    if tokenizer is None:
        return (QWEN_ENDOFTEXT, QWEN_IM_END)
    # Imported here, not at module scope: `gpu.loop` pulls torch in, and
    # `gpu.lab.plot` / `fixture` read MESSAGES and NEEDLES out of this module on
    # a box that only has plotly.
    from gpu.loop.stop import stop_token_ids as product_stop_token_ids

    try:
        return product_stop_token_ids(tokenizer)
    except ValueError:
        # A fixture tokenizer that owns no end-of-turn id at all: the lab
        # records the miss rather than dying inside a worker.
        return (QWEN_ENDOFTEXT, QWEN_IM_END)


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

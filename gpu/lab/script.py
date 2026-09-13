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

__all__ = [
    "CHR_PATH",
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

# Qwen2.5-Instruct stop tokens. `<|im_end|>` closes the assistant turn;
# `<|endoftext|>` is the base model's EOS and shows up on odd completions.
QWEN_IM_END = 151645
QWEN_ENDOFTEXT = 151643

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


def stop_token_ids(tokenizer=None) -> tuple[int, ...]:
    """Greedy stop set: the tokenizer's EOS plus the two Qwen specials."""
    ids = {QWEN_IM_END, QWEN_ENDOFTEXT}
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(int(eos))
    elif isinstance(eos, (list, tuple)):
        ids.update(int(value) for value in eos if isinstance(value, int))
    return tuple(sorted(ids))


def _qwen_fallback(user_text: str) -> str:
    """Hardcoded Qwen2.5 chat template, for when the tokenizer has none."""
    return (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def chat_text(tokenizer, user_text: str) -> str:
    """One independent turn as a prompt string: ``[{"role": "user", ...}]``.

    No history is carried between messages (see :data:`TURNS_NOTE`), so the same
    text is produced for both codecs and both tokenize to the same ids.
    """
    try:
        packed = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # noqa: BLE001 -- no template, or transformers changed the call
        return _qwen_fallback(user_text)
    return packed if isinstance(packed, str) else _qwen_fallback(user_text)

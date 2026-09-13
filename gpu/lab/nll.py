"""Teacher-forced prefix NLL for ``kind=ppl`` items.

The frozen ``messages.csv`` schema cannot grow a column
(:mod:`gpu.lab.bundle`). This module is the side channel: compute
``-sum log p(token_t | prefix_<t)`` on both codecs, persist it as
``loglikelihood.csv``, and let :func:`gpu.lab.eval.score_messages` fill
``eval_scores.csv``. An empty cell stays empty; a missing adapter is never a
0.0 and never a published WikiText PPL.

Neither path uses the chat template. WikiText / fixture prefixes are raw text.
A prefix longer than ``max_seq`` is an empty cell, not a silent truncate:
rolling windows are a later plate.

    python -m gpu.lab.test_nll
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .eval import nll_from_logprobs, roundtrip_ok

__all__ = [
    "LOG_LIKELIHOOD_COLUMNS",
    "PrefixNLL",
    "perplexity",
    "prefix_input_ids",
    "read_loglikelihood",
    "score_prefix",
    "teacher_forced_logprobs",
    "write_loglikelihood",
]

LOG_LIKELIHOOD_COLUMNS = (
    "codec",
    "message_id",
    "nll",
    "n_tokens",
    "roundtrip",
    "notes",
)


@dataclass(frozen=True)
class PrefixNLL:
    """One prefix. ``nll is None`` means the CSV cell stays empty."""

    nll: float | None
    n_tokens: int | None
    prompt_tokens: int
    elapsed_ms: float
    roundtrip: bool
    notes: str = ""


def perplexity(nll: float | None, n_tokens: int | None) -> float | None:
    """``exp(nll / n_tokens)``. None when the adapter did not produce a number."""
    if nll is None or n_tokens is None or int(n_tokens) < 1:
        return None
    try:
        value = math.exp(float(nll) / int(n_tokens))
    except (OverflowError, ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def prefix_input_ids(tokenizer: Any, text: str) -> list[int] | None:
    """Same encode path as :func:`roundtrip_ok`, so a False round-trip is the same tokens."""
    try:
        encoded = tokenizer(text)
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return [int(x) for x in list(ids)]
    except Exception:  # noqa: BLE001 -- a tokenizer with another API is a miss, not a crash
        return None


def _as_token_logits(logits: torch.Tensor) -> torch.Tensor:
    """``[T, vocab]``. HuggingFace adds a batch axis; the loop may return ``[vocab]``."""
    if logits.dim() == 3:
        if logits.shape[0] != 1:
            raise ValueError(f"expected batch 1, got logits {tuple(logits.shape)}")
        logits = logits[0]
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if logits.dim() != 2:
        raise ValueError(f"logits must be [T, vocab], got {tuple(logits.shape)}")
    return logits


def teacher_forced_logprobs(
    logits: torch.Tensor, input_ids: torch.Tensor
) -> list[float]:
    """``log p`` of ``input_ids[1:]`` under ``logits[:-1]``. Empty when ``T < 2``.

    ``logits[t]`` is P(next | tokens ``0..t``), the usual causal-LM shift. The
    first token is context, not a scored target. Log-softmax is fp32.
    """
    logits = _as_token_logits(logits)
    ids = input_ids.reshape(-1).to(device=logits.device, dtype=torch.long)
    t_ids = int(ids.numel())
    t_logits = int(logits.shape[0])
    if t_ids < 2:
        return []
    if t_logits != t_ids:
        raise ValueError(
            f"logits T={t_logits} vs ids T={t_ids}; teacher-forced NLL needs one row per token"
        )
    logp = F.log_softmax(logits[:-1].float(), dim=-1)
    gathered = logp.gather(-1, ids[1:].unsqueeze(-1)).squeeze(-1)
    return [float(x) for x in gathered.detach().cpu().tolist()]


def _prepare(tokenizer: Any, text: str, max_seq: int) -> tuple[PrefixNLL | None, list[int]]:
    """``(blocked PrefixNLL, [])`` or ``(None, ids)`` ready for a forward."""
    roundtrip = roundtrip_ok(tokenizer, text)
    ids = prefix_input_ids(tokenizer, text) or []
    n = len(ids)
    if not roundtrip:
        return (
            PrefixNLL(
                None,
                None,
                n,
                0.0,
                False,
                "tokenizer does not round-trip this prefix; empty cell, not a number",
            ),
            [],
        )
    if n < 2:
        return (
            PrefixNLL(None, None, n, 0.0, True, "prefix shorter than 2 tokens"),
            ids,
        )
    if n > int(max_seq):
        return (
            PrefixNLL(
                None,
                None,
                n,
                0.0,
                True,
                f"prefix {n} tokens exceeds max_seq={max_seq}; no silent truncate "
                "(rolling windows are not this adapter)",
            ),
            ids,
        )
    return None, ids


@torch.no_grad()
def _bf16_logits(model: Any, ids: Sequence[int]) -> tuple[torch.Tensor, float]:
    device = next(model.parameters()).device
    x = torch.tensor([list(ids)], dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
    logits = out.logits if hasattr(out, "logits") else out[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    return _as_token_logits(logits), (time.perf_counter() - t0) * 1000.0


def score_prefix(
    tokenizer: Any,
    text: str,
    *,
    max_seq: int,
    model: Any | None = None,
    loop: Any | None = None,
) -> PrefixNLL:
    """Teacher-forced NLL of ``text``. Pass either a HF causal model or a ``TokenLoop``.

    No chat template. Failures (round-trip, short prefix, over-long, a raised
    forward) become ``nll=None`` plus a note, never a fabricated PPL.
    """
    if (model is None) == (loop is None):
        raise ValueError("score_prefix needs exactly one of model= (BF16) or loop= (NF4)")
    blocked, ids = _prepare(tokenizer, text, max_seq)
    if blocked is not None:
        return blocked
    try:
        if loop is not None:
            tensor = torch.tensor(ids, dtype=torch.long)
            logits, scored_ids, elapsed_ms = loop.loglikelihood(tensor)
            logprobs = teacher_forced_logprobs(logits, scored_ids)
            elapsed = float(elapsed_ms)
        else:
            logits, elapsed = _bf16_logits(model, ids)
            logprobs = teacher_forced_logprobs(
                logits, torch.tensor(ids, dtype=torch.long, device=logits.device)
            )
    except Exception as exc:  # noqa: BLE001 -- one prefix must not kill the plate
        return PrefixNLL(
            None,
            None,
            len(ids),
            0.0,
            True,
            f"{type(exc).__name__} during teacher-forced NLL: {exc}",
        )
    nll = nll_from_logprobs(logprobs)
    n_tokens = len(logprobs) if nll is not None else None
    return PrefixNLL(
        nll=nll,
        n_tokens=n_tokens,
        prompt_tokens=len(ids),
        elapsed_ms=float(elapsed),
        roundtrip=True,
        notes="" if nll is not None else "no finite loglikelihood from teacher-forced logprobs",
    )


def _cell_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _cell_int(value: Any) -> int | None:
    number = _cell_float(value)
    return None if number is None else int(round(number))


def _cell_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def write_loglikelihood(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path | None:
    """Sidecar next to ``messages.csv``. No rows -> no file (hard plate stays clean)."""
    material = [dict(row) for row in rows]
    if not material:
        return None
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(LOG_LIKELIHOOD_COLUMNS)
        for row in material:
            nll = row.get("nll")
            n_tokens = row.get("n_tokens")
            roundtrip = row.get("roundtrip")
            writer.writerow(
                [
                    str(row.get("codec") or ""),
                    str(row.get("message_id") or ""),
                    "" if nll is None else f"{float(nll):.6g}",
                    "" if n_tokens is None else str(int(n_tokens)),
                    "" if roundtrip is None else ("true" if roundtrip else "false"),
                    str(row.get("notes") or ""),
                ]
            )
    return dest


def read_loglikelihood(path: str | Path) -> list[dict[str, Any]]:
    """Missing file is an empty list, not an error: generate-only runs have no sidecar."""
    dest = Path(path)
    if not dest.is_file():
        return []
    with dest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for raw in reader:
            rows.append(
                {
                    "codec": str(raw.get("codec") or ""),
                    "message_id": _cell_int(raw.get("message_id")),
                    "nll": _cell_float(raw.get("nll")),
                    "n_tokens": _cell_int(raw.get("n_tokens")),
                    "roundtrip": _cell_bool(raw.get("roundtrip")),
                    "notes": str(raw.get("notes") or ""),
                }
            )
        return rows

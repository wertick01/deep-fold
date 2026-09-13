"""GPU-free tests for the prefix NLL adapter. No 3B, no WikiText number.

    python -m gpu.lab.test_nll
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab.eval import load_eval_fixture, score_messages, FIXTURE_PATH  # noqa: E402
from gpu.lab.nll import (  # noqa: E402
    perplexity,
    prefix_input_ids,
    read_loglikelihood,
    score_prefix,
    teacher_forced_logprobs,
    write_loglikelihood,
)
from gpu.lab.sessions import LabSession, _message_row  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


class _TinyTok:
    """Eight characters, one id each. Round-trips ``abcdefgh``."""

    alphabet = "abcdefgh"

    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": [self.alphabet.index(ch) for ch in text]}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(self.alphabet[int(i)] for i in ids)


class _LossyTok(_TinyTok):
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return super().decode(ids, skip_special_tokens)[:-1]


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 8, hidden: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, use_cache=False):  # noqa: ANN001
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.out(self.embed(input_ids)))


def gate_teacher_forced_shift() -> None:
    ids = torch.tensor([0, 1, 2, 3])
    logits = torch.zeros(4, 8)
    logits[0, 1] = 20.0
    logits[1, 2] = 20.0
    logits[2, 3] = 20.0
    logp = teacher_forced_logprobs(logits, ids)
    check("three scored tokens, first is context", len(logp) == 3, str(len(logp)))
    nll = -sum(logp)
    check("peaked logits at the true next tokens are ~0 nll", nll < 1e-4, str(nll))

    uniform = teacher_forced_logprobs(torch.zeros(4, 8), ids)
    expected = 3 * math.log(8)
    got = -sum(uniform)
    check(
        "uniform vocab-8 is (T-1) log V",
        abs(got - expected) < 1e-5,
        f"{got} vs {expected}",
    )
    check("T<2 is an empty list, not a zero", teacher_forced_logprobs(logits[:1], ids[:1]) == [], "[]")
    try:
        teacher_forced_logprobs(torch.zeros(3, 8), ids)
        check("T mismatch raises", False, "no raise")
    except ValueError as exc:
        check("T mismatch is a ValueError", "logits T=3" in str(exc), str(exc)[:60])


def gate_perplexity_none() -> None:
    check("perplexity of an empty cell is None", perplexity(None, 10) is None, "None")
    check("perplexity of n_tokens=0 is None", perplexity(1.0, 0) is None, "None")
    ppl = perplexity(math.log(4) * 2, 2)
    check("exp(nll/n) for a 4-way uniform bigram", abs(ppl - 4.0) < 1e-9, str(ppl))


def gate_prepare_blocks() -> None:
    tok = _TinyTok()
    model = _TinyLM()
    model.eval()
    ok = score_prefix(tokenizer=tok, text="abcd", max_seq=32, model=model)
    check("a round-tripping prefix on CPU is a finite nll", ok.nll is not None, str(ok.notes))
    check("n_tokens is T-1", ok.n_tokens == 3, str(ok.n_tokens))
    check("prompt_tokens is T", ok.prompt_tokens == 4, str(ok.prompt_tokens))
    check("PPL is defined only after a finite nll", perplexity(ok.nll, ok.n_tokens) is not None, "ppl")

    lossy = score_prefix(tokenizer=_LossyTok(), text="abcd", max_seq=32, model=model)
    check("a lossy tokenizer leaves nll empty", lossy.nll is None, lossy.notes[:50])
    check("the note is not a zero", "round-trip" in lossy.notes, lossy.notes[:60])

    short = score_prefix(tokenizer=tok, text="a", max_seq=32, model=model)
    check("one token is empty, not a PPL of 1", short.nll is None, short.notes)

    long = score_prefix(tokenizer=tok, text="abcdefgh", max_seq=4, model=model)
    check(
        "over-long prefix is empty, not truncated",
        long.nll is None and "max_seq" in long.notes and "truncate" in long.notes,
        long.notes[:80],
    )


def gate_sidecar_roundtrip() -> None:
    rows = [
        {
            "codec": "bf16",
            "message_id": 7,
            "nll": 41.25,
            "n_tokens": 38,
            "roundtrip": True,
            "notes": "",
        },
        {
            "codec": "bf16",
            "message_id": 8,
            "nll": None,
            "n_tokens": None,
            "roundtrip": False,
            "notes": "tokenizer does not round-trip this prefix",
        },
    ]
    with tempfile.TemporaryDirectory(prefix="nll-csv-") as tmp:
        path = write_loglikelihood(Path(tmp) / "loglikelihood.csv", rows)
        check("a sidecar is written when there are ppl rows", path is not None, str(path))
        loaded = read_loglikelihood(path)
        check("two rows come back", len(loaded) == 2, str(len(loaded)))
        check("finite nll survives the CSV", loaded[0]["nll"] == 41.25, str(loaded[0]["nll"]))
        check("an empty nll stays empty, not 0", loaded[1]["nll"] is None, str(loaded[1]["nll"]))
        check("missing file is []", read_loglikelihood(Path(tmp) / "nope.csv") == [], "[]")
        check("no rows means no file", write_loglikelihood(Path(tmp) / "empty.csv", []) is None, "None")


def gate_bundle_stays_frozen() -> None:
    session = LabSession("bf16")
    session.messages.append(
        _message_row(
            "bf16",
            7,
            "prefix",
            "",
            prompt_tokens=8,
            new_tokens=0,
            prefill_ms=1.0,
            decode_ms=0.0,
            decode_tok_s=0.0,
            stop_reason="loglikelihood",
        )
    )
    session.loglikelihood.append(
        {
            "codec": "bf16",
            "message_id": 7,
            "nll": 1.5,
            "n_tokens": 7,
            "roundtrip": True,
            "notes": "",
        }
    )
    bundle = session.as_bundle()
    check("as_bundle still accepts the frozen message columns", len(bundle.messages) == 1, "ok")
    check("nll is not a messages.csv column", "nll" not in bundle.messages[0], str(bundle.messages[0].keys()))


def gate_score_messages_sidecar() -> None:
    items = load_eval_fixture(FIXTURE_PATH)
    ppl_index = next(i for i, item in enumerate(items, start=1) if item.kind == "ppl")
    messages = [
        {"codec": "nf4", "message_id": index, "response": ""}
        for index, _item in enumerate(items, start=1)
    ]
    sidecar = [
        {"message_id": ppl_index, "nll": 12.5, "n_tokens": 40, "notes": ""},
    ]
    rows = score_messages(messages, items, loglikelihood=sidecar)
    scored = next(row for row in rows if row["message_id"] == ppl_index)
    check(
        "sidecar nll lands in eval_scores shape",
        scored["nll"] == 12.5 and scored["n_tokens"] == 40 and scored["correct"] is None,
        f"nll={scored['nll']} n={scored['n_tokens']}",
    )
    other = [row for row in rows if row["kind"] == "ppl" and row["message_id"] != ppl_index]
    check("the other ppl row stays empty without a sidecar line", other[0]["nll"] is None, str(other[0]["nll"]))


def gate_ids_match_roundtrip() -> None:
    tok = _TinyTok()
    text = "cade"
    ids = prefix_input_ids(tok, text)
    check("encode is the alphabet index", ids == [2, 0, 3, 4], str(ids))


def _gate(name: str, body: Callable[[], Any]) -> None:
    try:
        body()
    except Exception as error:  # noqa: BLE001
        check(name, False, f"{type(error).__name__}: {error}")


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print("gpu/lab prefix NLL adapter (CPU tensors, no 3B, no WikiText number)\n")
    for name, body in (
        ("gate_teacher_forced_shift", gate_teacher_forced_shift),
        ("gate_perplexity_none", gate_perplexity_none),
        ("gate_prepare_blocks", gate_prepare_blocks),
        ("gate_sidecar_roundtrip", gate_sidecar_roundtrip),
        ("gate_bundle_stays_frozen", gate_bundle_stays_frozen),
        ("gate_score_messages_sidecar", gate_score_messages_sidecar),
        ("gate_ids_match_roundtrip", gate_ids_match_roundtrip),
    ):
        _gate(name, body)
    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"\n{sum(1 for _, ok, _ in CHECKS if ok)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

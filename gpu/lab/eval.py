"""Eval-harness stub: eight committed items, no corpus, no Hub, no download flag.

This is a **second** plate. It does not touch the 12-item hard plate
(``gpu/lab/data/hard_items.json``, ``python -m gpu.lab.hard``), which stays the
reasoning fixture. What this module owns is the shape a real ``lm-eval``-style
run would have to fit into: task names, an extractor per task, a loglikelihood
cell, and one CSV per codec.

The whole point is that CI can run it. Therefore:

* The items are in-repo (:data:`FIXTURE_PATH`), eight of them, golds included.
* ``DEEPFOLD_EVAL`` may point at an **already-downloaded** local JSON/JSONL or a
  directory of them. Unset, missing, or empty -> the committed fixture and a note
  saying so. Nothing here calls ``datasets.load_dataset``.
* There is no ``--download``. There is no "fetch GSM8K" flag. Asking for one is
  an error with a pointer to ``DEEPFOLD_EVAL``.
* :func:`assert_offline` puts the HuggingFace offline switches into the
  environment before any child process starts, so a cache miss fails loudly
  instead of quietly reaching the network.

Scoring shares the hard plate's number path (:func:`gpu.lab.hard.extract_number`)
and adds two kinds the hard set does not have: ``mcq`` (letter extract, fail
closed) and ``ppl`` (no accuracy at all -- a loglikelihood cell, empty until a
codec adapter fills it). ``truthful`` items fail closed and are allowed to come
back ``pending_human`` when the reply commits to neither the accepted phrasing
nor the myth.

    python -m gpu.lab.eval --list
    python -m gpu.lab.eval --lab qwen25-3b --codecs bf16,nf4
    python -m gpu.lab.test_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .catalog import SUPPORTED_NF4, LabModel, lab_by_slug
from .hard import extract_number, extract_yesno, hard_max_seq, numbers_equal
from .script import RUNS_DIR

__all__ = [
    "EVAL_CODECS",
    "EVAL_COLUMNS",
    "EVAL_ENV",
    "EVAL_KINDS",
    "EVAL_MAX_NEW_TOKENS",
    "FIXTURE_PATH",
    "NO_DOWNLOAD",
    "OFFLINE_ENV",
    "EvalItem",
    "EvalRun",
    "EvalScore",
    "EvalScript",
    "EvalSource",
    "accuracy",
    "accuracy_by_task",
    "assert_offline",
    "codec_slot",
    "eval_source",
    "extract_choice",
    "load_eval_fixture",
    "load_eval_script",
    "nll_from_logprobs",
    "quality_fn_for",
    "roundtrip_ok",
    "run_eval",
    "run_eval_one",
    "score_eval_item",
    "score_messages",
    "write_eval_script",
    "write_eval_scores",
]

FIXTURE_PATH = Path(__file__).resolve().parent / "data" / "eval_items.json"

#: Directory or file of **already-downloaded** local sets. Never a Hub id.
EVAL_ENV = "DEEPFOLD_EVAL"

#: Long enough for a shown-steps GSM8K item; the mcq items answer in one token.
EVAL_MAX_NEW_TOKENS = 256

#: Kinds this plate knows how to score. Anything else fails closed.
EVAL_KINDS = ("gsm8k", "exact", "mcq", "yesno", "truthful", "ppl")

#: Codec slots. ``bnb`` is bitsandbytes NF4 and only runs when that stack is
#: importable *and* the BF16 tree is on disk; otherwise it is a skip row.
EVAL_CODECS = ("bf16", "nf4", "bnb")

#: What a codec's rows look like. ``correct`` is empty for ``ppl`` items -- a
#: loglikelihood prefix has no right answer, and writing ``false`` there would
#: invent a miss.
EVAL_COLUMNS = (
    "codec",
    "task",
    "message_id",
    "item_id",
    "kind",
    "gold",
    "extracted",
    "correct",
    "pending_human",
    "nll",
    "n_tokens",
    "notes",
)

#: Set before any child starts. A stale cache is then a loud miss, not a silent
#: download.
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_HASH_CHOICE = re.compile(r"####\s*\(?([A-Za-z])\)?")
_SAID_CHOICE = re.compile(
    r"(?:final\s+answer|answer|option|choice)\s*(?:is)?\s*[:\-]?\s*\(?([A-Za-z])\)?\b",
    re.IGNORECASE,
)
_BARE_CHOICE = re.compile(r"^\s*\(?([A-Za-z])\)?\s*[.:)]?\s*$")


@dataclass(frozen=True)
class EvalItem:
    """One fixture row. ``task`` is the lm-eval-style name a real harness emits."""

    id: str
    kind: str
    prompt: str
    task: str = ""
    gold: str = ""
    choices: tuple[str, ...] = ()
    needles: tuple[str, ...] = ()
    rejects: tuple[str, ...] = ()
    note: str = ""

    @property
    def scoreable(self) -> bool:
        """``ppl`` prefixes carry no gold and are never counted as accuracy."""
        return self.kind.lower() != "ppl"


@dataclass(frozen=True)
class EvalScore:
    """One scored reply. ``correct is None`` means "this kind has no accuracy"."""

    item_id: str
    task: str
    kind: str
    gold: str
    extracted: str
    correct: bool | None
    pending_human: bool = False
    nll: float | None = None
    n_tokens: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class EvalSource:
    """Where the items came from, and the sentence that says so in a run dir."""

    path: Path
    mode: str  # "fixture" | "local"
    note: str

    @property
    def is_fixture(self) -> bool:
        return self.mode == "fixture"


@dataclass(frozen=True)
class EvalScript:
    """The prompts a worker is asked, frozen next to the CSVs at launch."""

    items: tuple[EvalItem, ...]
    source: EvalSource
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS

    @property
    def prompts(self) -> tuple[str, ...]:
        return tuple(item.prompt for item in self.items)

    @property
    def quality_fn(self) -> Callable[[int, str], bool]:
        return quality_fn_for(self.items)


# --------------------------------------------------------------------------- #
# where the items come from -- never the Hub
# --------------------------------------------------------------------------- #


def assert_offline() -> dict[str, str]:
    """Put the HuggingFace offline switches in ``os.environ`` and return them.

    Child workers inherit this. A tokenizer that is not already on disk then
    raises instead of downloading, which is the behaviour CI needs: this plate is
    allowed to be skipped, it is not allowed to fetch a corpus.
    """
    for name, value in OFFLINE_ENV.items():
        os.environ.setdefault(name, value)
    return dict(OFFLINE_ENV)


def _local_candidates(directory: Path) -> list[Path]:
    """Local set files inside a ``DEEPFOLD_EVAL`` directory, preferred name first."""
    preferred = directory / FIXTURE_PATH.name
    found = [preferred] if preferred.is_file() else []
    for pattern in ("*.json", "*.jsonl"):
        found.extend(path for path in sorted(directory.glob(pattern)) if path not in found)
    return found


def eval_source() -> EvalSource:
    """Resolve ``DEEPFOLD_EVAL``, falling back to the committed eight items.

    A missing or empty override is **not** an error and **not** a download: it is
    the fixture plus a note. That is what keeps ``--list`` and the tests
    identical on a machine with no corpus.
    """
    override = os.environ.get(EVAL_ENV, "").strip()
    if not override:
        return EvalSource(
            FIXTURE_PATH,
            "fixture",
            f"{EVAL_ENV} unset: committed {FIXTURE_PATH.name} only, no corpus, no download",
        )
    path = Path(override)
    if path.is_file():
        return EvalSource(path, "local", f"{EVAL_ENV}={path} (local file, already on disk)")
    if path.is_dir():
        candidates = _local_candidates(path)
        if candidates:
            return EvalSource(
                candidates[0],
                "local",
                f"{EVAL_ENV}={path} (local dir; using {candidates[0].name})",
            )
        return EvalSource(
            FIXTURE_PATH,
            "fixture",
            f"{EVAL_ENV}={path} holds no .json/.jsonl set: fixture only, nothing downloaded",
        )
    return EvalSource(
        FIXTURE_PATH,
        "fixture",
        f"{EVAL_ENV}={path} does not exist: fixture only, nothing downloaded",
    )


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(entry) for entry in value)


def _item_from_dict(row: Mapping[str, Any]) -> EvalItem:
    gold = row.get("gold")
    return EvalItem(
        id=str(row["id"]),
        kind=str(row["kind"]),
        prompt=str(row["prompt"]),
        task=str(row.get("task") or ""),
        gold="" if gold is None else str(gold),
        choices=_tuple(row.get("choices")),
        needles=_tuple(row.get("needles")),
        rejects=_tuple(row.get("rejects")),
        note=str(row.get("note") or ""),
    )


def _rows_from_payload(path: Path) -> tuple[list[Mapping[str, Any]], int]:
    """Item dicts out of a JSON object, a JSON list, or a JSONL file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return rows, EVAL_MAX_NEW_TOKENS
    data = json.loads(text)
    if isinstance(data, list):
        return list(data), EVAL_MAX_NEW_TOKENS
    rows = data.get("items")
    if rows is None:
        # Accept the hard plate's shape too, so a local set written for that
        # harness can be pointed at this one without a converter.
        rows = data.get("independent") or []
    return list(rows), int(data.get("max_new_tokens") or EVAL_MAX_NEW_TOKENS)


def load_eval_fixture(path: str | Path | None = None) -> tuple[EvalItem, ...]:
    """The items, from ``path``, or from :func:`eval_source`."""
    source = Path(path) if path is not None else eval_source().path
    rows, _max_new = _rows_from_payload(source)
    return tuple(_item_from_dict(row) for row in rows)


def load_eval_script(path: str | Path | None = None) -> EvalScript:
    """Items plus the provenance note. ``path`` bypasses ``DEEPFOLD_EVAL``."""
    if path is None:
        source = eval_source()
    else:
        source = EvalSource(Path(path), "local", f"script {Path(path)}")
    rows, max_new = _rows_from_payload(source.path)
    return EvalScript(
        items=tuple(_item_from_dict(row) for row in rows),
        source=source,
        max_new_tokens=max_new,
    )


def write_eval_script(path: str | Path, *, script: EvalScript | None = None) -> Path:
    """Freeze the prompts into the run root so a later read is not a guess."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    script = script or load_eval_script()
    payload = {
        "plate": "eval",
        "max_new_tokens": script.max_new_tokens,
        "source": str(script.source.path),
        "source_mode": script.source.mode,
        "note": script.source.note,
        "items": [
            {
                "id": item.id,
                "kind": item.kind,
                "task": item.task,
                "prompt": item.prompt,
                "gold": item.gold,
                "choices": list(item.choices),
                "needles": list(item.needles),
                "rejects": list(item.rejects),
                "note": item.note,
            }
            for item in script.items
        ],
    }
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dest


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def extract_choice(text: str, choices: Sequence[str] = ()) -> str | None:
    """The letter an mcq reply committed to, or ``None``.

    ``#### B`` wins, then "the answer is B", then a reply that is nothing but a
    letter, then exactly one choice's text appearing verbatim. A letter outside
    the offered range is ``None``: a model that answered ``F`` to four options
    has not answered.
    """
    if not text:
        return None
    limit = len(choices) if choices else len(_LETTERS)
    allowed = set(_LETTERS[:limit])

    def accept(letter: str | None) -> str | None:
        if letter is None:
            return None
        upper = letter.upper()
        return upper if upper in allowed else None

    for pattern in (_HASH_CHOICE, _SAID_CHOICE):
        found = pattern.search(text)
        letter = accept(found.group(1) if found else None)
        if letter is not None:
            return letter
    bare = _BARE_CHOICE.match(text.strip())
    letter = accept(bare.group(1) if bare else None)
    if letter is not None:
        return letter
    if choices:
        lowered = text.lower()
        hits = [
            _LETTERS[index]
            for index, choice in enumerate(choices)
            if choice.strip() and choice.strip().lower() in lowered
        ]
        if len(hits) == 1:
            return hits[0]
    return None


def nll_from_logprobs(logprobs: Sequence[float]) -> float | None:
    """Sum of ``-log p`` over the scored tokens, or ``None`` when that is not a number.

    The loglikelihood smoke the ``ppl`` items exist for: an empty list, a ``-inf``
    logprob or a ``nan`` is not a perplexity and must not be written into a CSV
    as one. Q3's ppl script divides this by the token count; the stub only has to
    prove the cell is finite or empty.
    """
    values = list(logprobs)
    if not values:
        return None
    total = 0.0
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            return None
        total -= number
    return total if math.isfinite(total) else None


def roundtrip_ok(tokenizer: Any, text: str) -> bool:
    """Does this tokenizer encode+decode ``text`` back to the same words?

    The other half of the ``ppl`` smoke: a prefix whose NLL is finite but whose
    tokens do not decode back to the prompt is scoring something else. Whitespace
    is normalized, specials are dropped; anything the tokenizer raises is a
    ``False``, not a traceback.
    """
    try:
        encoded = tokenizer(text)
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        decoded = tokenizer.decode(list(ids), skip_special_tokens=True)
    except Exception:  # noqa: BLE001 -- a tokenizer with another API is a False
        return False
    return " ".join(str(decoded).split()) == " ".join(text.split())


def score_eval_item(
    item: EvalItem,
    response: str,
    *,
    nll: float | None = None,
    n_tokens: int | None = None,
) -> EvalScore:
    """Fail closed. Unknown kind, empty reply or missing gold is never a pass."""
    text = response or ""
    kind = item.kind.lower()
    base = dict(item_id=item.id, task=item.task, kind=item.kind, gold=item.gold)

    if kind == "ppl":
        finite = nll is not None and math.isfinite(float(nll))
        return EvalScore(
            **base,
            extracted="" if not finite else f"{float(nll):.4f}",
            correct=None,
            nll=float(nll) if finite else None,
            n_tokens=n_tokens,
            notes=(
                ""
                if finite
                else "no finite loglikelihood: the codec adapter did not report one "
                "(Q3 owns the ppl script); empty cell, not a zero"
            ),
        )

    if kind == "gsm8k":
        extracted = extract_number(text) or ""
        ok = bool(extracted) and bool(item.gold) and numbers_equal(extracted, item.gold)
        return EvalScore(
            **base, extracted=extracted, correct=ok, notes="" if ok else "number miss"
        )

    if kind == "exact":
        extracted = " ".join(text.split()).strip()
        ok = bool(item.gold) and extracted.casefold() == item.gold.strip().casefold()
        return EvalScore(
            **base, extracted=extracted, correct=ok, notes="" if ok else "exact miss"
        )

    if kind == "mcq":
        extracted = extract_choice(text, item.choices) or ""
        ok = bool(extracted) and extracted == item.gold.strip().upper()
        return EvalScore(
            **base,
            extracted=extracted,
            correct=ok,
            notes="" if ok else ("letter miss" if extracted else "no letter in the reply"),
        )

    if kind == "yesno":
        extracted = extract_yesno(text) or ""
        ok = bool(extracted) and extracted == item.gold.strip().lower()
        return EvalScore(
            **base, extracted=extracted, correct=ok, notes="" if ok else "yes/no miss"
        )

    if kind == "truthful":
        lowered = text.lower()
        myth = next((bad for bad in item.rejects if bad.lower() in lowered), "")
        hit = next((good for good in item.needles if good.lower() in lowered), "")
        if myth:
            return EvalScore(**base, extracted=myth, correct=False, notes="restated the myth")
        if hit:
            return EvalScore(**base, extracted=hit, correct=True)
        return EvalScore(
            **base,
            extracted="",
            correct=False,
            pending_human=True,
            notes="committed to neither the accepted phrasing nor the myth: unscoreable, "
            "raw text in messages.csv, human rating later",
        )

    return EvalScore(
        **base,
        extracted="",
        correct=False,
        notes=f"unknown kind {item.kind!r}; known: {', '.join(EVAL_KINDS)}",
    )


def quality_fn_for(items: Sequence[EvalItem]) -> Callable[[int, str], bool]:
    """``quality_ok(message_id, response)`` for the worker's messages.csv column.

    1-based ``message_id`` indexes the script. A ``ppl`` prefix has no right
    answer, so its cell is ``False`` and the note in ``eval_scores.csv`` is what a
    reader should quote. Unknown ids fail closed.
    """

    def fn(message_id: int, response: str) -> bool:
        index = int(message_id) - 1
        if index < 0 or index >= len(items):
            return False
        return score_eval_item(items[index], response).correct is True

    return fn


def score_messages(
    messages: Iterable[Mapping[str, Any]],
    items: Sequence[EvalItem],
) -> list[dict[str, Any]]:
    """One row per reply, keyed to the fixture by 1-based ``message_id``."""
    rows: list[dict[str, Any]] = []
    items_list = list(items)
    for row in messages:
        message_id = int(row["message_id"])
        index = message_id - 1
        if index < 0 or index >= len(items_list):
            scored = EvalScore(
                item_id=f"msg-{message_id}",
                task="",
                kind="unknown",
                gold="",
                extracted="",
                correct=False,
                notes="message_id has no fixture item",
            )
        else:
            scored = score_eval_item(items_list[index], str(row.get("response") or ""))
        rows.append(
            {
                "codec": str(row.get("codec") or ""),
                "task": scored.task,
                "message_id": message_id,
                "item_id": scored.item_id,
                "kind": scored.kind,
                "gold": scored.gold,
                "extracted": scored.extracted,
                "correct": scored.correct,
                "pending_human": scored.pending_human,
                "nll": scored.nll,
                "n_tokens": scored.n_tokens,
                "notes": scored.notes,
            }
        )
    return rows


def _counted(rows: Iterable[Mapping[str, Any]], codec: str) -> list[Mapping[str, Any]]:
    """Rows that carry an accuracy: not ``ppl``, not ``pending_human``."""
    return [
        row
        for row in rows
        if row.get("codec") == codec
        and row.get("correct") is not None
        and not row.get("pending_human")
    ]


def accuracy(rows: Sequence[Mapping[str, Any]], codec: str) -> float | None:
    """Fraction correct over the auto-scored rows. ``None`` when there are none."""
    counted = _counted(rows, codec)
    if not counted:
        return None
    return sum(1 for row in counted if row.get("correct")) / len(counted)


def accuracy_by_task(rows: Sequence[Mapping[str, Any]], codec: str) -> dict[str, float]:
    """Per-task accuracy, which is the column a real harness would report."""
    buckets: dict[str, list[bool]] = {}
    for row in _counted(rows, codec):
        buckets.setdefault(str(row.get("task") or "untasked"), []).append(
            bool(row.get("correct"))
        )
    return {task: sum(hits) / len(hits) for task, hits in sorted(buckets.items())}


def write_eval_scores(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """``eval_scores.csv``. Empty cells stay empty; a missing NLL is never a 0."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(EVAL_COLUMNS)
        for row in rows:
            correct = row.get("correct")
            nll = row.get("nll")
            n_tokens = row.get("n_tokens")
            writer.writerow(
                [
                    str(row.get("codec") or ""),
                    str(row.get("task") or ""),
                    str(row.get("message_id") or ""),
                    str(row.get("item_id") or ""),
                    str(row.get("kind") or ""),
                    str(row.get("gold") or ""),
                    str(row.get("extracted") or ""),
                    "" if correct is None else ("true" if correct else "false"),
                    "true" if row.get("pending_human") else "false",
                    "" if nll is None else f"{float(nll):.6g}",
                    "" if n_tokens is None else str(int(n_tokens)),
                    str(row.get("notes") or ""),
                ]
            )
    return dest


# --------------------------------------------------------------------------- #
# codec slots: bf16, nf4, and bnb only when that stack is really here
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalRun:
    """One (lab, codec) eval session reduced to what a reader would quote."""

    slug: str
    codec: str
    out_dir: Path
    source_note: str = ""
    accuracy: float | None = None
    by_task: Mapping[str, float] = field(default_factory=dict)
    n_messages: int = 0
    mean_ttft_ms: float | None = None
    mean_decode_tok_s: float | None = None
    notes: str = ""
    scores: tuple[Mapping[str, Any], ...] = ()

    @property
    def ran(self) -> bool:
        return self.n_messages > 0


def codec_slot(lab: LabModel, codec: str) -> tuple[bool, str]:
    """Can this codec be attempted at all? ``(ready, why not)``.

    ``bnb`` is the interesting one: importable **and** the BF16 tree on disk, per
    the TZ. Even then the session body belongs to the competitor harness
    (:mod:`gpu.lab.competitor`), so this returns the reason that harness gives --
    a skip row, never a fabricated score.
    """
    if codec not in EVAL_CODECS:
        raise ValueError(f"codec={codec!r}; expected one of {', '.join(EVAL_CODECS)}")
    if codec == "bf16":
        if not Path(lab.model_dir).is_dir():
            return False, f"bf16 skipped: model dir missing: {lab.model_dir}"
        return True, ""
    if codec == "nf4":
        if lab.nf4_driver not in SUPPORTED_NF4:
            return False, f"nf4 skipped: no TokenLoop driver for {lab.nf4_driver!r}"
        if not Path(lab.model_dir).is_dir():
            return False, f"nf4 skipped: model dir missing (tokenizer): {lab.model_dir}"
        if not Path(lab.chr_path).is_file():
            return False, (
                f"nf4 skipped: NF4 file missing: {lab.chr_path}. Compress on CPU first: "
                f"{lab.compress_cmd or '(no compress command in the catalog)'}"
            )
        return True, ""

    from .competitor import detect, stack_by_name

    detection = detect(stack_by_name("bitsandbytes-nf4"), model_dir=lab.model_dir)
    return False, f"bnb skipped: {detection.skip_reason}"


def _recorded_miss(
    lab: LabModel,
    codec: str,
    dest: Path,
    note: str,
    source_note: str,
    *,
    verbose: bool = True,
) -> EvalRun:
    """A skip, a missing stack or a dead worker is a row with notes, not a traceback."""
    if verbose:
        print(f"[{lab.slug} {codec}] {note}", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    write_eval_scores(dest / "eval_scores.csv", [])
    return EvalRun(
        slug=lab.slug,
        codec=codec,
        out_dir=dest,
        source_note=source_note,
        notes=note,
    )


def run_eval_one(
    lab: LabModel | str,
    out_root: str | Path,
    *,
    codec: str,
    script: EvalScript | None = None,
    script_path: str | Path | None = None,
    isolated: bool = True,
    graphs: bool = True,
    verbose: bool = True,
) -> EvalRun:
    """One codec of one model in its own process, into ``<slug>-<codec>/``.

    Never loads a model in this process and never raises: a missing model, a
    missing ``.chr``, an absent bitsandbytes, a CUDA OOM or a worker that died
    before writing ``summary.csv`` all come back as an :class:`EvalRun` whose
    notes say what happened. Same rule as :func:`gpu.lab.hard.run_hard_one`.
    """
    if isinstance(lab, str):
        lab = lab_by_slug(lab)
    assert_offline()

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    script = script or load_eval_script()
    if script_path is None:
        script_path = write_eval_script(root / "eval_script.json", script=script)
    dest = root / f"{lab.slug}-{codec}"

    ready, why = codec_slot(lab, codec)
    if not ready:
        return _recorded_miss(
            lab,
            codec,
            dest,
            f"{why}. Recorded miss, not a crashed cell.",
            script.source.note,
            verbose=verbose,
        )

    from .sessions import run_bf16, run_nf4

    common = dict(
        out_dir=dest,
        model_dir=lab.model_dir,
        messages=script.prompts,
        max_new_tokens=script.max_new_tokens,
        trust_remote_code=lab.trust_remote_code,
        isolated=isolated,
        items_json=script_path,
        plate="eval",
        quality=script.quality_fn,
        verbose=verbose,
    )
    try:
        if codec == "bf16":
            session = run_bf16(**common)
        else:
            session = run_nf4(
                chr_path=lab.chr_path,
                max_seq=hard_max_seq(lab),
                graphs=graphs,
                **common,
            )
    except Exception as exc:  # noqa: BLE001 -- a dead worker is data, not a traceback
        return _recorded_miss(
            lab,
            codec,
            dest,
            f"{type(exc).__name__} from the isolated {codec} worker: {exc}",
            script.source.note,
            verbose=verbose,
        )

    bundle = session.as_bundle()
    bundle.write(dest)
    scores = score_messages(bundle.messages, script.items)
    write_eval_scores(dest / "eval_scores.csv", scores)
    summary = bundle.summary_for(codec) or {}
    return EvalRun(
        slug=lab.slug,
        codec=codec,
        out_dir=dest,
        source_note=script.source.note,
        accuracy=accuracy(scores, codec),
        by_task=accuracy_by_task(scores, codec),
        n_messages=int(summary.get("n_messages") or 0),
        mean_ttft_ms=summary.get("mean_ttft_ms"),
        mean_decode_tok_s=summary.get("mean_decode_tok_s"),
        notes=str(summary.get("notes") or ""),
        scores=tuple(scores),
    )


def run_eval(
    lab: LabModel | str,
    out_root: str | Path,
    *,
    codecs: Sequence[str] = ("bf16", "nf4"),
    isolated: bool = True,
    graphs: bool = True,
    verbose: bool = True,
) -> list[EvalRun]:
    """Each codec in turn, one isolated worker each, merged into ``eval_scores.csv``."""
    if isinstance(lab, str):
        lab = lab_by_slug(lab)
    unknown = [codec for codec in codecs if codec not in EVAL_CODECS]
    if unknown:
        raise ValueError(f"unknown codec(s) {unknown}; expected {', '.join(EVAL_CODECS)}")

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    script = load_eval_script()
    script_path = write_eval_script(root / "eval_script.json", script=script)

    runs: list[EvalRun] = []
    merged: list[Mapping[str, Any]] = []
    for codec in codecs:
        if verbose:
            print(
                f"\n=== eval {codec.upper()}  {lab.title}  ({len(script.items)} items) ===",
                flush=True,
            )
        run = run_eval_one(
            lab,
            root,
            codec=codec,
            script=script,
            script_path=script_path,
            isolated=isolated,
            graphs=graphs,
            verbose=verbose,
        )
        runs.append(run)
        merged.extend(run.scores)
        write_eval_scores(root / "eval_scores.csv", merged)
    return runs


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

#: Printed when someone asks for a download flag.
NO_DOWNLOAD = (
    "there is no --download and no fetch flag in this harness. WikiText, C4, MMLU, "
    "GSM8K, ARC, HellaSwag, Winogrande and TruthfulQA are never pulled from the Hub "
    f"here. Download them yourself, then point {EVAL_ENV} at the local directory or "
    "file; unset, it runs the committed 8-item fixture and stops."
)

_DOWNLOAD_FLAGS = ("--download", "--fetch", "--hub", "--datasets", "--lm-eval")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gpu.lab.eval",
        description=(
            "Eval-harness stub: 8 committed items, offline. Not the 12-item hard "
            "plate, not lm-eval."
        ),
        epilog=NO_DOWNLOAD,
    )
    parser.add_argument(
        "--lab", default="", help="catalog slug: qwen25-3b, qwen25-14b, internlm20b"
    )
    parser.add_argument(
        "--out", default="", help="output dir; default $DEEPFOLD_RUNS/eval-<slug>-<time>"
    )
    parser.add_argument(
        "--codecs",
        default="bf16,nf4",
        help=(
            f"comma-separated slots from {', '.join(EVAL_CODECS)} "
            "(bnb is a skip unless that stack is here)"
        ),
    )
    parser.add_argument(
        "--list", action="store_true", help="print the items and the source, then exit (no GPU)"
    )
    parser.add_argument(
        "--tasks", action="store_true", help="print the task names this plate emits, then exit"
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _report(runs: Sequence[EvalRun]) -> None:
    print("\ncodec  items  accuracy  ttft ms  tok/s  by task")
    for run in runs:
        acc = "n/a" if run.accuracy is None else f"{100 * run.accuracy:.0f}%"
        by_task = ", ".join(f"{task} {100 * value:.0f}%" for task, value in run.by_task.items())
        ttft = "n/a" if run.mean_ttft_ms is None else f"{run.mean_ttft_ms:.0f}"
        tok_s = "n/a" if run.mean_decode_tok_s is None else f"{run.mean_decode_tok_s:.1f}"
        print(
            f"{run.codec:<6} {run.n_messages:>5}  {acc:>8}  {ttft:>7}  {tok_s:>5}  "
            f"{by_task or '-'}"
        )
        if run.notes:
            print(f"  notes: {run.notes[:240]}")
    print(
        "\nEight items is a harness smoke, not a quality claim. The 12-item hard plate "
        "(python -m gpu.lab.hard) is unchanged, and neither is lm-eval."
    )


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    asked = [flag for flag in raw if flag.split("=")[0].lower() in _DOWNLOAD_FLAGS]
    if asked:
        print(f"{asked[0]}: {NO_DOWNLOAD}", file=sys.stderr)
        return 2

    parser = _parser()
    args = parser.parse_args(raw)
    assert_offline()
    source = eval_source()

    if args.tasks:
        tasks = sorted({item.task or "untasked" for item in load_eval_fixture(source.path)})
        for task in tasks:
            print(task)
        print(f"\n{len(tasks)} task names, fixture-shaped. {NO_DOWNLOAD}")
        return 0

    if args.list:
        items = load_eval_fixture(source.path)
        print(f"{len(items)} items  source={source.mode}  {source.path}")
        for index, item in enumerate(items, start=1):
            gold = item.gold or (
                "(no gold: loglikelihood prefix)" if item.kind == "ppl" else "(human)"
            )
            print(f"  {index:>2}. {item.id:26} {item.kind:9} {item.task:14} gold={gold!r}")
        print(f"\n{source.note}")
        print(NO_DOWNLOAD)
        return 0

    if not args.lab:
        parser.error("--lab is required for a live run (or pass --list / --tasks)")

    import time

    lab = lab_by_slug(args.lab)
    codecs = [part.strip() for part in args.codecs.split(",") if part.strip()]
    out_dir = Path(args.out) if args.out else Path(RUNS_DIR) / time.strftime(
        f"eval-{lab.slug}-%Y%m%d-%H%M%S"
    )
    print(f"lab {lab.slug}  codecs={codecs}  out={out_dir}", flush=True)
    print(source.note, flush=True)
    runs = run_eval(lab, out_dir, codecs=codecs, verbose=not args.quiet)
    _report(runs)
    print(f"\nwrote {out_dir / 'eval_scores.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

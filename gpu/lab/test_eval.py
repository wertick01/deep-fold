"""GPU-free tests for the eval-harness stub. No model, no Hub, no corpus.

The point of this file is the thing it does *not* do: nothing here imports
``datasets``, calls ``load_dataset``, or reaches the network. It scores synthetic
replies against the committed eight items.

    python -m gpu.lab.test_eval
    python gpu/lab/test_eval.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab.catalog import LabModel, lab_by_slug  # noqa: E402
from gpu.lab.eval import (  # noqa: E402
    EVAL_CODECS,
    EVAL_COLUMNS,
    EVAL_ENV,
    EVAL_KINDS,
    EVAL_MAX_NEW_TOKENS,
    FIXTURE_PATH,
    NO_DOWNLOAD,
    OFFLINE_ENV,
    EvalItem,
    accuracy,
    accuracy_by_task,
    assert_offline,
    codec_slot,
    eval_source,
    extract_choice,
    load_eval_fixture,
    load_eval_script,
    nll_from_logprobs,
    quality_fn_for,
    roundtrip_ok,
    run_eval_one,
    score_eval_item,
    score_messages,
    write_eval_script,
    write_eval_scores,
)
from gpu.lab.hard import FIXTURE_PATH as HARD_FIXTURE_PATH  # noqa: E402
from gpu.lab.hard import load_fixture as load_hard_fixture  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


@contextmanager
def _env(name: str, value: str | None) -> Iterator[None]:
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


class _CharTokenizer:
    """A tokenizer-shaped object with no model behind it: one id per character."""

    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(value) for value in ids)


class _LossyTokenizer(_CharTokenizer):
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return super().decode(ids, skip_special_tokens)[:-1]


def gate_fixture() -> None:
    items = load_eval_fixture(FIXTURE_PATH)
    check("the committed fixture holds exactly 8 items", len(items) == 8, f"n={len(items)}")
    check(
        "the fixture is a committed file in the repo",
        FIXTURE_PATH.is_file() and FIXTURE_PATH.parent.name == "data",
        str(FIXTURE_PATH.relative_to(_REPO)),
    )
    ids = [item.id for item in items]
    check("item ids are unique", len(ids) == len(set(ids)), str(ids))
    kinds = [item.kind for item in items]
    check(
        "every kind is one this plate can score",
        all(kind in EVAL_KINDS for kind in kinds),
        str(sorted(set(kinds))),
    )
    counts = {kind: kinds.count(kind) for kind in set(kinds)}
    check("2 number items", counts.get("gsm8k") == 2, str(counts))
    check("2 multiple-choice items", counts.get("mcq") == 2, str(counts))
    check("1 binary item", counts.get("yesno") == 1, str(counts))
    check("1 truthfulqa-shaped item", counts.get("truthful") == 1, str(counts))
    check("2 loglikelihood prefixes", counts.get("ppl") == 2, str(counts))
    check(
        "every scoreable item carries a gold in the JSON",
        all(item.gold for item in items if item.scoreable),
        "no gold is off in a dataset somewhere",
    )
    check(
        "ppl prefixes carry no gold and no choices",
        all(not item.gold and not item.choices for item in items if item.kind == "ppl"),
        "loglikelihood only",
    )
    check(
        "mcq items ship their choices",
        all(len(item.choices) >= 2 for item in items if item.kind == "mcq"),
        str([len(i.choices) for i in items if i.kind == "mcq"]),
    )
    check(
        "the truthful item ships both an accept list and a myth list",
        all(item.needles and item.rejects for item in items if item.kind == "truthful"),
        "accept + reject",
    )
    tasks = {item.task for item in items}
    check(
        "tasks use lm-eval spelling so a real harness can emit the same names",
        {"gsm8k", "mmlu", "arc_challenge", "winogrande", "truthfulqa"} <= tasks,
        str(sorted(tasks)),
    )
    check(
        "every item declares a task",
        all(item.task for item in items),
        str(sorted(tasks)),
    )
    check(
        "prefixes are short enough to be a loglikelihood smoke",
        all(len(item.prompt.split()) <= 64 for item in items if item.kind == "ppl"),
        str([len(i.prompt.split()) for i in items if i.kind == "ppl"]),
    )
    text = FIXTURE_PATH.read_text(encoding="utf-8").lower()
    check(
        "the fixture holds no URL",
        "http://" not in text and "https://" not in text,
        "nothing to fetch",
    )
    check(
        "no item id looks like a corpus row that could be re-fetched",
        all(item.id.startswith("eval-") for item in items),
        str(ids),
    )
    check(
        "no item points at a dataset split",
        not any(mark in text for mark in ("/train", "/validation", "/test[", "split=")),
        "golds are in this file",
    )


def gate_hard_plate_untouched() -> None:
    """Q2 adds a plate. It does not shrink, rename or replace the 12."""
    hard = load_hard_fixture(HARD_FIXTURE_PATH)
    check(
        "the hard plate still has 12 independent items",
        len(hard["independent"]) == 12,
        f"n={len(hard['independent'])}",
    )
    check("the hard 4-turn history is still there", len(hard["history"]) == 4, "4 turns")
    check(
        "the two fixtures are different files",
        FIXTURE_PATH != HARD_FIXTURE_PATH and HARD_FIXTURE_PATH.is_file(),
        HARD_FIXTURE_PATH.name,
    )
    eval_ids = {item.id for item in load_eval_fixture(FIXTURE_PATH)}
    hard_ids = {item.id for item in hard["independent"]}
    check(
        "no eval item reuses a hard item id",
        not (eval_ids & hard_ids),
        str(sorted(eval_ids & hard_ids)) or "disjoint",
    )


def gate_offline() -> None:
    check(
        "the offline switches cover hub, datasets and transformers",
        set(OFFLINE_ENV) == {"HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"},
        str(sorted(OFFLINE_ENV)),
    )
    applied = assert_offline()
    check(
        "assert_offline puts them in the environment for the children",
        all(os.environ.get(name) for name in applied),
        str({name: os.environ.get(name) for name in applied}),
    )
    # Calls, not words: the docstring names `datasets.load_dataset` in order to
    # say it is never called, and that sentence must not fail its own gate. This
    # scans the module under test only -- a file cannot grep itself for strings
    # it has to contain in order to do the grepping.
    source = Path(__file__).with_name("eval.py").read_text(encoding="utf-8")
    for call in ("load_dataset(", "snapshot_download(", "hf_hub_download(", "requests."):
        check(
            f"gpu/lab/eval.py never calls {call.rstrip('.(')}",
            call not in source,
            "offline by construction",
        )
    for statement in ("import datasets", "from datasets", "import huggingface_hub"):
        check(
            f"gpu/lab/eval.py has no `{statement}`",
            statement not in source,
            "no corpus imported",
        )


def gate_eval_source() -> None:
    with _env(EVAL_ENV, None):
        source = eval_source()
        check(
            "unset DEEPFOLD_EVAL is the committed fixture",
            source.is_fixture and source.path == FIXTURE_PATH,
            source.note,
        )
        check("the note says nothing was downloaded", "no download" in source.note, source.note)
    with _env(EVAL_ENV, str(_REPO / "does-not-exist-anywhere")):
        source = eval_source()
        check(
            "a missing DEEPFOLD_EVAL path falls back, it does not fetch",
            source.is_fixture and "nothing downloaded" in source.note,
            source.note,
        )
    with tempfile.TemporaryDirectory(prefix="eval-src-") as tmp:
        empty = Path(tmp) / "empty"
        empty.mkdir()
        with _env(EVAL_ENV, str(empty)):
            source = eval_source()
            check(
                "an empty DEEPFOLD_EVAL dir is the fixture plus a note",
                source.is_fixture and "no .json/.jsonl" in source.note,
                source.note,
            )
        local = Path(tmp) / "local.jsonl"
        local.write_text(
            json.dumps({"id": "local-1", "kind": "gsm8k", "task": "gsm8k", "gold": "7",
                        "prompt": "Say 7. ####"})
            + "\n",
            encoding="utf-8",
        )
        with _env(EVAL_ENV, str(local)):
            source = eval_source()
            items = load_eval_fixture()
            check(
                "DEEPFOLD_EVAL may be a local JSONL file",
                source.mode == "local" and len(items) == 1 and items[0].id == "local-1",
                source.note,
            )
        with _env(EVAL_ENV, str(Path(tmp))):
            source = eval_source()
            check(
                "DEEPFOLD_EVAL may be a local directory",
                source.mode == "local" and source.path == local,
                source.note,
            )
    with _env(EVAL_ENV, None):
        check(
            "the default source is restored after the overrides",
            eval_source().path == FIXTURE_PATH,
            str(eval_source().path.name),
        )


def gate_no_download_flag() -> None:
    from gpu.lab.eval import main as eval_main

    for flag in ("--download", "--fetch", "--hub", "--lm-eval"):
        code = eval_main([flag])
        check(f"{flag} is refused with a non-zero exit", code == 2, f"exit={code}")
    check(
        "the refusal points at DEEPFOLD_EVAL instead",
        EVAL_ENV in NO_DOWNLOAD and "no --download" in NO_DOWNLOAD,
        NO_DOWNLOAD[:60],
    )
    check("--list runs offline and exits 0", eval_main(["--list"]) == 0, "list")
    check("--tasks runs offline and exits 0", eval_main(["--tasks"]) == 0, "tasks")
    parser_source = Path(__file__).with_name("eval.py").read_text(encoding="utf-8")
    check(
        "there is no add_argument for a download flag",
        'add_argument("--download"' not in parser_source,
        "no fetch flag exists",
    )


def gate_extract_choice() -> None:
    four = ("Data link layer", "Network layer", "Transport layer", "Session layer")
    check("#### B wins", extract_choice("blah\n#### B", four) == "B", "B")
    check("the answer is C", extract_choice("I think the answer is C.", four) == "C", "C")
    check("a bare letter", extract_choice(" b ", four) == "B", "B")
    check("a parenthesised letter", extract_choice("(D)", four) == "D", "D")
    check(
        "a letter outside the offered range is not an answer",
        extract_choice("#### F", four) is None,
        "F of four options",
    )
    check(
        "exactly one choice text quoted is accepted",
        extract_choice("It has to be the Transport layer.", four) == "C",
        "C",
    )
    check(
        "two choice texts quoted is ambiguous, so no answer",
        extract_choice("either the Network layer or the Transport layer", four) is None,
        "fail closed",
    )
    check("prose with no letter is None", extract_choice("I do not know.", four) is None, "None")
    check("empty is None", extract_choice("", four) is None, "None")


def gate_nll() -> None:
    check(
        "a finite sum of -log p",
        nll_from_logprobs([-1.0, -2.0, -0.5]) == 3.5,
        str(nll_from_logprobs([-1.0, -2.0, -0.5])),
    )
    check("no logprobs is empty, not zero", nll_from_logprobs([]) is None, "None")
    check("-inf is not a perplexity", nll_from_logprobs([-1.0, float("-inf")]) is None, "None")
    check("nan is not a perplexity", nll_from_logprobs([float("nan")]) is None, "None")


def gate_roundtrip() -> None:
    text = "The packed weights never leave their four-bit form."
    check("a lossless tokenizer round-trips", roundtrip_ok(_CharTokenizer(), text), "ok")
    check("a lossy decode is a False", not roundtrip_ok(_LossyTokenizer(), text), "clipped")
    check("a tokenizer that raises is a False", not roundtrip_ok(object(), text), "no crash")


def gate_score_items() -> None:
    items = {item.id: item for item in load_eval_fixture(FIXTURE_PATH)}

    jars = items["eval-gsm8k-jars"]
    check("gsm8k hit", score_eval_item(jars, "8*12=96, +24\n#### 120").correct is True, "120")
    check("gsm8k miss", score_eval_item(jars, "#### 119").correct is False, "119")
    check(
        "gsm8k with no number is a miss, not a pass",
        score_eval_item(jars, "I cannot say.").correct is False,
        "no number",
    )

    osi = items["eval-mmlu-osi"]
    check("mcq hit", score_eval_item(osi, "B").correct is True, "B")
    check("mcq miss", score_eval_item(osi, "#### C").correct is False, "C")
    unanswered = score_eval_item(osi, "The OSI model has seven layers.")
    check(
        "mcq with no letter fails closed and says so",
        unanswered.correct is False and "no letter" in unanswered.notes,
        unanswered.notes,
    )

    trophy = items["eval-winogrande-trophy"]
    check("yesno hit", score_eval_item(trophy, "No").correct is True, "No")
    check("yesno miss", score_eval_item(trophy, "Yes").correct is False, "Yes")

    senses = items["eval-truthfulqa-senses"]
    check(
        "truthful accept phrase",
        score_eval_item(senses, "Humans have more than five senses.").correct is True,
        "accepted",
    )
    myth = score_eval_item(senses, "Humans have exactly five senses.")
    check(
        "truthful myth is a miss, and the myth is quoted",
        myth.correct is False and "exactly five" in myth.extracted,
        myth.notes,
    )
    vague = score_eval_item(senses, "Physiology is a broad field.")
    check(
        "truthful with neither phrase is pending_human, never a silent pass",
        vague.pending_human and vague.correct is False,
        vague.notes,
    )

    prefix = items["eval-ppl-registers"]
    empty = score_eval_item(prefix, "")
    check(
        "a ppl item has no accuracy at all",
        empty.correct is None and empty.nll is None,
        empty.notes[:60],
    )
    check(
        "an unreported loglikelihood is an empty cell with a reason",
        "not a zero" in empty.notes,
        empty.notes[-30:],
    )
    filled = score_eval_item(prefix, "", nll=41.25, n_tokens=38)
    check(
        "a finite loglikelihood is recorded with its token count",
        filled.correct is None and filled.nll == 41.25 and filled.n_tokens == 38,
        f"nll={filled.nll} n={filled.n_tokens}",
    )
    infinite = score_eval_item(prefix, "", nll=float("inf"))
    check("an infinite nll is dropped, not written", infinite.nll is None, "empty cell")

    unknown = score_eval_item(EvalItem("x", "weird", "p", gold="1"), "1")
    check("an unknown kind fails closed", unknown.correct is False, unknown.notes)


def gate_gold_round_trip() -> None:
    for item in load_eval_fixture(FIXTURE_PATH):
        kind = item.kind
        if kind == "ppl":
            scored = score_eval_item(item, "", nll=1.0)
            ok = scored.correct is None
        elif kind == "mcq":
            scored = score_eval_item(item, f"#### {item.gold}")
            ok = scored.correct is True
        elif kind == "yesno":
            scored = score_eval_item(item, item.gold)
            ok = scored.correct is True
        elif kind == "truthful":
            scored = score_eval_item(item, f"Humans have {item.gold} senses.")
            ok = scored.correct is True
        else:
            scored = score_eval_item(item, f"steps...\n#### {item.gold}")
            ok = scored.correct is True
        check(f"gold round-trip {item.id}", ok, f"gold={item.gold!r} -> {scored.extracted!r}")


def gate_quality_fn() -> None:
    items = load_eval_fixture(FIXTURE_PATH)
    fn = quality_fn_for(items)
    check("message_id 1 is the first item", fn(1, "#### 120"), items[0].id)
    check("a wrong reply is False", not fn(1, "#### 0"), "miss")
    check("an unknown message_id fails closed", not fn(99, "#### 120"), "id=99")
    ppl_index = next(i for i, item in enumerate(items, start=1) if item.kind == "ppl")
    check(
        "a ppl prefix is never a quality_ok pass",
        not fn(ppl_index, "anything at all"),
        f"message_id={ppl_index}",
    )


def gate_score_messages() -> None:
    items = load_eval_fixture(FIXTURE_PATH)
    replies = {
        "eval-gsm8k-jars": "#### 120",
        "eval-gsm8k-posts": "#### 20",
        "eval-mmlu-osi": "B",
        "eval-arc-ice": "C",
        "eval-winogrande-trophy": "No",
        "eval-truthfulqa-senses": "Humans have more than five senses.",
        "eval-ppl-registers": "",
        "eval-ppl-rumour": "",
    }
    messages = [
        {"codec": "nf4", "message_id": index, "response": replies[item.id]}
        for index, item in enumerate(items, start=1)
    ]
    rows = score_messages(messages, items)
    check("one row per reply", len(rows) == 8, str(len(rows)))
    acc = accuracy(rows, "nf4")
    check(
        "accuracy counts the 6 scoreable items, 5 of them right",
        acc is not None and abs(acc - 5 / 6) < 1e-9,
        str(acc),
    )
    check(
        "the two ppl rows are excluded from accuracy, not counted as misses",
        sum(1 for row in rows if row["correct"] is None) == 2,
        "correct is None",
    )
    by_task = accuracy_by_task(rows, "nf4")
    check(
        "per-task accuracy splits gsm8k from the choice tasks",
        abs(by_task["gsm8k"] - 0.5) < 1e-9 and by_task["mmlu"] == 1.0,
        str(by_task),
    )
    check("ppl_fixture is not a task with an accuracy", "ppl_fixture" not in by_task, str(by_task))
    check(
        "another codec with no rows has no accuracy",
        accuracy(rows, "bf16") is None,
        "None, not 0",
    )
    ppl_index = next(i for i, item in enumerate(items, start=1) if item.kind == "ppl")
    filled = score_messages(
        [{"codec": "nf4", "message_id": ppl_index, "response": "", "nll": 9.25, "n_tokens": 17}],
        items,
    )
    check(
        "nll on the message dict fills the ppl cell",
        filled[0]["nll"] == 9.25 and filled[0]["n_tokens"] == 17,
        str(filled[0]["nll"]),
    )
    sidecar = score_messages(
        [{"codec": "nf4", "message_id": ppl_index, "response": ""}],
        items,
        loglikelihood=[{"message_id": ppl_index, "nll": 3.5, "n_tokens": 11, "notes": ""}],
    )
    check(
        "nll from the sidecar fills the ppl cell when messages.csv has none",
        sidecar[0]["nll"] == 3.5 and sidecar[0]["n_tokens"] == 11,
        str(sidecar[0]["nll"]),
    )
    orphan = score_messages([{"codec": "nf4", "message_id": 99, "response": "#### 1"}], items)
    check(
        "an orphan message_id fails closed",
        orphan[0]["correct"] is False and "no fixture item" in orphan[0]["notes"],
        orphan[0]["notes"],
    )


def gate_write_scores() -> None:
    items = load_eval_fixture(FIXTURE_PATH)
    rows = score_messages(
        [
            {"codec": "nf4", "message_id": 1, "response": "#### 120"},
            {"codec": "nf4", "message_id": 7, "response": ""},
        ],
        items,
    )
    with tempfile.TemporaryDirectory(prefix="eval-csv-") as tmp:
        path = write_eval_scores(Path(tmp) / "eval_scores.csv", rows)
        lines = path.read_text(encoding="utf-8").splitlines()
        check("header is the frozen schema", lines[0] == ",".join(EVAL_COLUMNS), lines[0])
        check(
            "the TZ's minimum columns are all present",
            {"task", "item_id", "gold", "extracted", "correct", "nll"} <= set(EVAL_COLUMNS),
            str(EVAL_COLUMNS),
        )
        check("a scored row writes true", lines[1].split(",")[7] == "true", lines[1])
        ppl_cells = lines[2].split(",")
        check(
            "a ppl row leaves correct empty rather than writing false",
            ppl_cells[7] == "",
            f"correct={ppl_cells[7]!r}",
        )
        check("a ppl row leaves nll empty rather than writing 0", ppl_cells[9] == "", lines[2])


def gate_script_freeze() -> None:
    with tempfile.TemporaryDirectory(prefix="eval-script-") as tmp:
        root = Path(tmp)
        script = load_eval_script(FIXTURE_PATH)
        path = write_eval_script(root / "eval_script.json", script=script)
        payload = json.loads(path.read_text(encoding="utf-8"))
        check("the frozen script says which plate it is", payload["plate"] == "eval", "eval")
        check("it freezes all 8 prompts", len(payload["items"]) == 8, str(len(payload["items"])))
        check(
            "it records where the items came from",
            payload["source"].endswith(FIXTURE_PATH.name),
            payload["source_mode"],
        )
        reloaded = load_eval_script(path)
        check(
            "a frozen script reloads to the same prompts",
            reloaded.prompts == script.prompts,
            f"{len(reloaded.prompts)} prompts",
        )
        check(
            "golds and choices survive the freeze",
            [item.gold for item in reloaded.items] == [item.gold for item in script.items]
            and [item.choices for item in reloaded.items]
            == [item.choices for item in script.items],
            "round trip",
        )
        check(
            "max_new_tokens is long enough for a shown-steps item",
            reloaded.max_new_tokens >= 128 and reloaded.max_new_tokens == EVAL_MAX_NEW_TOKENS,
            str(reloaded.max_new_tokens),
        )


def gate_codec_slots() -> None:
    check("the slots are bf16, nf4 and bnb", EVAL_CODECS == ("bf16", "nf4", "bnb"), str(EVAL_CODECS))
    nowhere = LabModel(
        slug="nowhere",
        notebook="none.ipynb",
        title="Qwen2.5-3B-Instruct",
        model_dir=r"C:\dev\models\does-not-exist",
        chr_path=r"C:\dev\models\does-not-exist.chr",
        bf16_fits=True,
        nf4_fits=True,
        nf4_driver="qwen2",
        trust_remote_code=False,
        note="test only",
        compress_cmd="chr.exe compress --in ... --out ...",
    )
    ready, why = codec_slot(nowhere, "bf16")
    check("a missing model dir is not ready", not ready and "model dir missing" in why, why[:70])
    ready, why = codec_slot(nowhere, "nf4")
    check("a missing .chr is not ready", not ready, why[:70])
    unsupported = LabModel(**{**nowhere.__dict__, "nf4_driver": "unsupported"})
    ready, why = codec_slot(unsupported, "nf4")
    check("an unsupported driver is not ready", not ready and "driver" in why, why[:70])
    ready, why = codec_slot(nowhere, "bnb")
    check(
        "bnb is a skip, and the reason names the competitor harness rule",
        not ready and why.startswith("bnb skipped:") and "SKIP:" in why,
        why[:110],
    )
    check(
        "the bnb skip never carries a number",
        "tok/s" not in why.split("SKIP:")[-1].split(".")[0],
        "no fabricated speed",
    )
    try:
        codec_slot(nowhere, "int8")
        check("an unknown codec raises", False, "no raise")
    except ValueError as error:
        check("an unknown codec raises ValueError", "int8" in str(error), str(error)[:60])
    three = lab_by_slug("qwen25-3b")
    ready, why = codec_slot(three, "bf16")
    if Path(three.model_dir).is_dir():
        check("the real 3B bf16 slot is ready on this box", ready, "model dir present")
    else:
        check(
            "no 3B tree on this box: bf16 is a skip, not a failure",
            not ready and "model dir missing" in why,
            "GPU-free box",
        )


def gate_recorded_miss() -> None:
    """A missing model is an EvalRun with notes and an empty CSV, not a traceback."""
    nowhere = LabModel(
        slug="nowhere",
        notebook="none.ipynb",
        title="Qwen2.5-3B-Instruct",
        model_dir=r"C:\dev\models\does-not-exist",
        chr_path=r"C:\dev\models\does-not-exist.chr",
        bf16_fits=True,
        nf4_fits=True,
        nf4_driver="qwen2",
        trust_remote_code=False,
        note="test only",
        compress_cmd="chr.exe compress --in ... --out ...",
    )
    with tempfile.TemporaryDirectory(prefix="eval-miss-") as tmp:
        root = Path(tmp)
        run = run_eval_one(nowhere, root, codec="bf16", verbose=False)
        check("a missing model does not raise", not run.ran, run.notes[:70])
        check("the miss says what was missing", "model dir missing" in run.notes, run.notes[:90])
        check("the miss is recorded, not crashed", "Recorded miss" in run.notes, run.notes[-40:])
        check(
            "a recorded miss still writes its CSV",
            (run.out_dir / "eval_scores.csv").is_file(),
            run.out_dir.name,
        )
        check(
            "a recorded miss invents no accuracy and no speed",
            run.accuracy is None
            and run.mean_ttft_ms is None
            and run.mean_decode_tok_s is None
            and run.by_task == {},
            "all empty",
        )
        check(
            "the run root freezes the script even when every codec skipped",
            (root / "eval_script.json").is_file(),
            "eval_script.json",
        )
        header = (run.out_dir / "eval_scores.csv").read_text(encoding="utf-8").strip()
        check("the skipped CSV is a header and nothing else", header == ",".join(EVAL_COLUMNS), header[:40])


def gate_worker_cli() -> None:
    from gpu.lab.worker import _parser

    args = _parser().parse_args(
        ["--codec", "nf4", "--out", "x", "--items-json", "s.json", "--plate", "eval"]
    )
    check("the worker accepts --plate eval", args.plate == "eval", args.plate)
    default = _parser().parse_args(["--codec", "bf16", "--out", "x"])
    check("the worker still defaults to the hard plate", default.plate == "hard", default.plate)
    import inspect

    from gpu.lab.eval import FIXTURE_PATH, load_eval_fixture
    from gpu.lab.sessions import _eval_item_kinds, _spawn_session, run_bf16, run_nf4
    from gpu.lab.worker import main as worker_main

    src = inspect.getsource(worker_main)
    check(
        "the isolated worker forwards items_json so kind=ppl can skip generate",
        'extra["items_json"] = args.items_json' in src,
        "items_json must reach run_bf16/run_nf4",
    )
    check(
        "the isolated worker forwards plate=eval with the script",
        'extra["plate"] = args.plate' in src,
        "plate must reach _eval_item_kinds",
    )
    check(
        "the isolated worker forwards max_seq into the session",
        "max_seq=args.max_seq" in src,
        "max_seq=",
    )
    items = load_eval_fixture(FIXTURE_PATH)
    kinds = _eval_item_kinds("eval", FIXTURE_PATH, len(items))
    check(
        "eval kinds include the fixture ppl prefixes",
        kinds.count("ppl") == 2 and "ppl" in kinds,
        str(kinds),
    )
    skipped = _eval_item_kinds("hard", FIXTURE_PATH, len(items))
    check(
        "a hard-plate child does not treat eval kinds as ppl",
        all(kind == "" for kind in skipped),
        str(skipped),
    )

    for name, fn in (("run_bf16", run_bf16), ("run_nf4", run_nf4)):
        check(
            f"{name} forwards plate to the isolated child",
            "plate" in inspect.signature(fn).parameters,
            "plate=",
        )
    spawn_src = inspect.getsource(_spawn_session)
    check(
        "isolated children inherit UTF-8 IO so WikiText CJK is not a crash",
        "PYTHONIOENCODING" in spawn_src,
        "PYTHONIOENCODING=utf-8",
    )


def gate_module_runs_offline() -> None:
    """`python -m gpu.lab.eval --list` in a fresh process, with no GPU and no network."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop(EVAL_ENV, None)
    completed = subprocess.run(
        [sys.executable, "-m", "gpu.lab.eval", "--list"],
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    check(
        "python -m gpu.lab.eval --list exits 0 in a fresh process",
        completed.returncode == 0,
        (completed.stderr or "")[-200:] or "exit 0",
    )
    check(
        "it prints the eight items and says the fixture is the source",
        "8 items" in completed.stdout and "fixture" in completed.stdout,
        completed.stdout.splitlines()[0] if completed.stdout else "no output",
    )
    check(
        "it repeats that there is no download flag",
        "no --download" in completed.stdout,
        "honest epilog",
    )


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
    print("gpu/lab eval-harness stub (8 committed items, no GPU, no Hub)\n")
    for name, body in (
        ("gate_fixture", gate_fixture),
        ("gate_hard_plate_untouched", gate_hard_plate_untouched),
        ("gate_offline", gate_offline),
        ("gate_eval_source", gate_eval_source),
        ("gate_no_download_flag", gate_no_download_flag),
        ("gate_extract_choice", gate_extract_choice),
        ("gate_nll", gate_nll),
        ("gate_roundtrip", gate_roundtrip),
        ("gate_score_items", gate_score_items),
        ("gate_gold_round_trip", gate_gold_round_trip),
        ("gate_quality_fn", gate_quality_fn),
        ("gate_score_messages", gate_score_messages),
        ("gate_write_scores", gate_write_scores),
        ("gate_script_freeze", gate_script_freeze),
        ("gate_codec_slots", gate_codec_slots),
        ("gate_recorded_miss", gate_recorded_miss),
        ("gate_worker_cli", gate_worker_cli),
        ("gate_module_runs_offline", gate_module_runs_offline),
    ):
        _gate(name, body)
    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

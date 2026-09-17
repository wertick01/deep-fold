"""GPU-free scoring tests for the hard eval. Never loads a model.

    python -m gpu.lab.test_hard
    python gpu/lab/test_hard.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab.catalog import LABS, LabModel, lab_by_slug  # noqa: E402
from gpu.lab.hard import (  # noqa: E402
    FIXTURE_PATH,
    HARD_MAX_NEW_TOKENS,
    RUN_ORDER,
    HardItem,
    HardRun,
    accuracy,
    answer_cell,
    answer_rows,
    expected_weight_mib,
    extract_number,
    extract_yesno,
    hard_max_seq,
    items_path,
    load_fixture,
    load_script,
    pad_runs,
    quality_fn_for,
    read_matrix,
    run_hard_one,
    runs_from_matrix,
    score_item,
    score_messages,
    size_label,
    write_matrix_csv,
    write_run_script,
    write_scores,
)
from gpu.lab.script import chat_text  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def _gate(name: str, body: Callable[[], tuple[bool, str]]) -> None:
    try:
        ok, detail = body()
    except Exception as error:  # noqa: BLE001
        check(name, False, f"{type(error).__name__}: {error}")
        return
    check(name, ok, detail)


class _FakeTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        packed = "||".join(f"{m['role']}:{m['content']}" for m in messages)
        return packed + (">>" if add_generation_prompt else "")


def gate_fixture() -> None:
    fixture = load_fixture()
    independent = fixture["independent"]
    history = fixture["history"]
    check(
        "independent set is 8–16 items",
        8 <= len(independent) <= 16,
        f"n={len(independent)}",
    )
    check("history protocol is 4 turns", len(history) == 4, f"n={len(history)}")
    ids = [item.id for item in independent]
    check("item ids are unique", len(ids) == len(set(ids)), str(ids))
    check(
        "every independent item has a prompt and a gold",
        all(item.prompt.strip() and item.gold for item in independent),
        "prompt+gold",
    )
    check(
        "fixture is a committed JSON file",
        FIXTURE_PATH.is_file() and items_path() == FIXTURE_PATH,
        str(FIXTURE_PATH.relative_to(_REPO)),
    )
    kinds = {item.kind for item in independent}
    check("set includes gsm8k and yesno", {"gsm8k", "yesno"} <= kinds, str(sorted(kinds)))
    check(
        "hard max_new_tokens is longer than smoke 64",
        HARD_MAX_NEW_TOKENS >= 128,
        str(HARD_MAX_NEW_TOKENS),
    )


def gate_extract_number() -> None:
    check(
        "#### wins over later numbers",
        extract_number("Monday 35, total 999\n#### 164\n") == "164",
        "#### 164",
    )
    check(
        "the answer is beats a trailing distractor",
        extract_number("I added 35 and 47. The answer is 164, not 35.") == "164",
        "the answer is 164",
    )
    check("last number fallback", extract_number("junk 12 then 44") == "44", "44")
    check("commas", extract_number("#### 1,234") == "1234", "1234")
    check("0.05 stays 0.05", extract_number("#### 0.05") == "0.05", extract_number("#### 0.05"))
    check("integer 164.0", extract_number("#### 164.0") == "164", extract_number("#### 164.0"))
    check("empty is None", extract_number("") is None, "None")
    check("no number is None", extract_number("I do not know") is None, "None")


def gate_extract_yesno() -> None:
    check("last yes/no wins", extract_yesno("Yes or No? No.") == "no", "no")
    check("Yes only", extract_yesno("Yes") == "yes", "yes")
    check("none", extract_yesno("maybe") is None, "None")


def gate_score_items() -> None:
    lamps = HardItem("gsm8k-lamps", "gsm8k", "unused", gold="164")
    check(
        "gsm8k #### hit",
        score_item(lamps, "35+47+82=164\n#### 164").correct,
        "164",
    )
    check(
        "gsm8k wrong number",
        not score_item(lamps, "#### 163").correct,
        "163",
    )
    check(
        "gsm8k 0.05 vs 0.10",
        not score_item(
            HardItem("trap-batball", "gsm8k", "unused", gold="0.05"),
            "The ball is 0.10 dollars.\n#### 0.10",
        ).correct,
        "0.10 is a miss",
    )
    check(
        "gsm8k 0.05 hit",
        score_item(
            HardItem("trap-batball", "gsm8k", "unused", gold="0.05"),
            "ball + (ball+1) = 1.10 so ball=0.05\n#### 0.05",
        ).correct,
        "0.05",
    )
    yesno = HardItem("logic-yesno", "yesno", "unused", gold="no")
    check("yesno No", score_item(yesno, "No").correct, "No")
    check("yesno Yes is a miss", not score_item(yesno, "Yes").correct, "Yes")
    open_item = HardItem("open-1", "open", "explain IEEE", gold="")
    scored = score_item(open_item, "because mantissa")
    check(
        "open item is pending, not a silent pass",
        scored.pending_human and not scored.correct,
        scored.notes,
    )
    unknown = score_item(HardItem("x", "weird", "p", gold="1"), "1")
    check("unknown kind fails closed", not unknown.correct, unknown.notes)
    needle = HardItem("n", "needle", "p", needles=("7429",))
    check("needle hit", score_item(needle, "the code is 7429.").correct, "7429")
    exact = HardItem("e", "exact", "p", gold="33")
    check("exact hit", score_item(exact, "33").correct, "33")
    check("exact miss", not score_item(exact, "32").correct, "32")


def gate_quality_fn() -> None:
    items = load_fixture()["independent"]
    fn = quality_fn_for(items)
    lamps = next(i for i in items if i.id == "gsm8k-lamps")
    check("message_id 1 is lamps", items[0].id == "gsm8k-lamps", items[0].id)
    check("quality_fn hit", fn(1, "#### 164"), "id=1")
    check("quality_fn miss", not fn(1, "#### 0"), "id=1 miss")
    check("unknown message_id fails closed", not fn(99, "#### 164"), "id=99")
    # same prompts for both codecs: the fixture is the prompt list
    check(
        "prompts are a fixed tuple",
        lamps.prompt == items[0].prompt and "####" in lamps.prompt,
        "shared script",
    )


def gate_score_messages() -> None:
    items = load_fixture()["independent"][:3]
    messages = [
        {"codec": "bf16", "message_id": 1, "response": "#### 164"},
        {"codec": "bf16", "message_id": 2, "response": "#### 44"},
        {"codec": "bf16", "message_id": 3, "response": "#### 0"},
        {"codec": "nf4", "message_id": 1, "response": "#### 164"},
        {"codec": "nf4", "message_id": 2, "response": "I give up"},
        {"codec": "nf4", "message_id": 3, "response": "#### 90"},
    ]
    rows = score_messages(messages, items)
    check("scored both codecs", len(rows) == 6, str(len(rows)))
    acc_bf16 = accuracy(rows, "bf16")
    acc_nf4 = accuracy(rows, "nf4")
    check("bf16 2/3", acc_bf16 is not None and abs(acc_bf16 - 2 / 3) < 1e-9, str(acc_bf16))
    check("nf4 2/3", acc_nf4 is not None and abs(acc_nf4 - 2 / 3) < 1e-9, str(acc_nf4))
    extra = score_messages(
        [{"codec": "bf16", "message_id": 9, "response": "#### 1"}],
        items,
    )
    check("orphan message_id fails closed", not extra[0]["correct"], extra[0]["notes"])


def gate_write_scores() -> None:
    with tempfile.TemporaryDirectory(prefix="hard-eval-") as tmp:
        dest = Path(tmp) / "hard_scores.csv"
        rows = [
            {
                "codec": "bf16",
                "message_id": 1,
                "item_id": "gsm8k-lamps",
                "kind": "gsm8k",
                "gold": "164",
                "extracted": "164",
                "correct": True,
                "pending_human": False,
                "notes": "",
            }
        ]
        write_scores(dest, rows)
        text = dest.read_text(encoding="utf-8")
        header = text.splitlines()[0]
        check(
            "hard_scores.csv header",
            header == "codec,message_id,item_id,kind,gold,extracted,correct,pending_human,notes",
            header,
        )
        check("true is written as true", "true" in text.splitlines()[1], text.splitlines()[1])


def gate_run_script() -> None:
    with tempfile.TemporaryDirectory(prefix="hard-script-") as tmp:
        independent = write_run_script(Path(tmp) / "ind.json", history=False)
        history = write_run_script(Path(tmp) / "hist.json", history=True)
        ind = load_script(independent)
        hist = load_script(history)
        check("independent conversation", ind.conversation == "independent", ind.conversation)
        check("history conversation", hist.conversation == "history", hist.conversation)
        check("independent has 12 prompts", len(ind.prompts) == 12, str(len(ind.prompts)))
        check("history has 4 prompts", len(hist.prompts) == 4, str(len(hist.prompts)))
    check(
        "history golds are the running crate",
        [item.gold for item in hist.items] == ["33", "28", "48", "16"],
        str([item.gold for item in hist.items]),
    )


def gate_gold_round_trip() -> None:
    for item in load_fixture()["independent"]:
        if item.kind == "yesno":
            reply = item.gold
        else:
            reply = f"working...\n#### {item.gold}"
        scored = score_item(item, reply)
        check(f"gold round-trip {item.id}", scored.correct, f"gold={item.gold!r} extracted={scored.extracted!r}")
    for item in load_fixture()["history"]:
        scored = score_item(item, f"#### {item.gold}")
        check(f"history gold {item.id}", scored.correct, item.gold)


def gate_deepfold_hard_override() -> None:
    with tempfile.TemporaryDirectory(prefix="hard-override-") as tmp:
        payload = {
            "independent": [
                {
                    "id": "only-one",
                    "kind": "gsm8k",
                    "gold": "1",
                    "prompt": "Say 1. ####",
                }
            ],
            "history": [],
        }
        path = Path(tmp) / "extra.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        previous = os.environ.get("DEEPFOLD_HARD")
        os.environ["DEEPFOLD_HARD"] = str(path)
        try:
            check("DEEPFOLD_HARD points at the override", items_path() == path, str(items_path()))
            loaded = load_fixture()
            check(
                "override fixture loads",
                len(loaded["independent"]) == 1 and loaded["independent"][0].id == "only-one",
                loaded["independent"][0].id,
            )
        finally:
            if previous is None:
                os.environ.pop("DEEPFOLD_HARD", None)
            else:
                os.environ["DEEPFOLD_HARD"] = previous
        check("default fixture restored", items_path() == FIXTURE_PATH, str(items_path()))


def gate_catalog() -> None:
    check(
        "four catalog labs",
        [lab.slug for lab in LABS]
        == ["qwen25-3b", "qwen25-14b", "internlm20b", "qwen25-32b"],
        "",
    )
    three = lab_by_slug("qwen25-3b")
    check("3B slug", three.slug == "qwen25-3b" and three.nf4_driver == "qwen2", three.notebook)
    intern = lab_by_slug("internlm20b")
    check("20B max_seq is tighter", hard_max_seq(intern) < hard_max_seq(three), str(hard_max_seq(intern)))
    thirtytwo = lab_by_slug("qwen25-32b")
    check("32B is qwen2 overflow", thirtytwo.nf4_driver == "qwen2" and thirtytwo.nf4_fits is False, thirtytwo.note[:40])
    check("32B size label", size_label(thirtytwo) == "32B", size_label(thirtytwo))
    try:
        lab_by_slug("nope")
        check("unknown slug raises", False, "no raise")
    except KeyError as error:
        check("unknown slug raises KeyError", "qwen25-3b" in str(error), str(error)[:80])


def gate_chat_history() -> None:
    tok = _FakeTok()
    one = chat_text(tok, "hello")
    check("independent pack is one user turn", one == "user:hello>>", one)
    packed = chat_text(tok, "next", history=[("hello", "hi")])
    check(
        "history packs prior assistant text",
        packed == "user:hello||assistant:hi||user:next>>",
        packed,
    )


def _synthetic_bundle(codec: str, *, weight: float, torch_mib: float):
    """A summary + timeline shaped like a finished session. Not a measurement."""
    from gpu.lab.bundle import LabBundle

    return LabBundle.of(
        timeline=[
            {"t_s": 0.0, "codec": codec, "used_mib": 1200.0, "total_mib": 12288.0},
            {
                "t_s": 1.0,
                "codec": codec,
                "used_mib": 8000.0,
                "total_mib": 12288.0,
                "torch_alloc_mib": torch_mib,
                "torch_reserved_mib": torch_mib + 512.0,
            },
        ],
        messages=[
            {"codec": codec, "message_id": 1, "response": "#### 164", "prompt": "p"},
            {"codec": codec, "message_id": 2, "response": "no idea", "prompt": "p"},
        ],
        summary=[
            {
                "codec": codec,
                "weight_mib": weight,
                "vram_after_load_torch_mib": torch_mib,
                "vram_after_load_smi_mib": 8000.0,
                "vram_peak_smi_mib": 8200.0,
                "mean_ttft_ms": 53.0,
                "mean_decode_tok_s": 22.3,
                "n_messages": 2,
                "notes": "FIXTURE",
            }
        ],
        source="fixture",
    )


def gate_run_order() -> None:
    check(
        "run order is BF16 3B, BF16 14B, NF4 3B, NF4 14B",
        RUN_ORDER
        == (
            ("qwen25-3b", "bf16"),
            ("qwen25-14b", "bf16"),
            ("qwen25-3b", "nf4"),
            ("qwen25-14b", "nf4"),
        ),
        str(RUN_ORDER),
    )
    check(
        "uncompressed pair comes before the compressed pair",
        [codec for _slug, codec in RUN_ORDER] == ["bf16", "bf16", "nf4", "nf4"],
        str([codec for _slug, codec in RUN_ORDER]),
    )
    check(
        "3B is in both codecs and was not dropped for 14B",
        [slug for slug, _codec in RUN_ORDER].count("qwen25-3b") == 2,
        str([slug for slug, _codec in RUN_ORDER]),
    )
    known = {lab.slug for lab in LABS}
    check(
        "every run-order slug is in the catalog",
        all(slug in known for slug, _codec in RUN_ORDER),
        str(sorted(known)),
    )


def gate_size_and_weights() -> None:
    check("3B size label", size_label(lab_by_slug("qwen25-3b")) == "3B", size_label(lab_by_slug("qwen25-3b")))
    check("14B size label", size_label(lab_by_slug("qwen25-14b")) == "14B", size_label(lab_by_slug("qwen25-14b")))
    check("32B size label", size_label(lab_by_slug("qwen25-32b")) == "32B", size_label(lab_by_slug("qwen25-32b")))
    three = expected_weight_mib("qwen25-3b")
    fourteen = expected_weight_mib("qwen25-14b")
    check("expected weights have both codecs", set(three) == {"bf16", "nf4"}, str(sorted(three)))
    missing = [
        name
        for name, value in (("3B", three["bf16"]), ("14B", fourteen["bf16"]))
        if value is None
    ]
    if missing:
        check(
            "weights on disk: skipped, model dirs not present",
            True,
            f"missing {missing} -- GPU-free box, nothing to measure",
        )
        return
    check(
        "14B BF16 weighs more than 3B BF16 on disk",
        fourteen["bf16"] > three["bf16"] * 3,
        f"3B={three['bf16']:.0f} MiB, 14B={fourteen['bf16']:.0f} MiB",
    )
    check(
        "14B BF16 does not fit the 12288 MiB card",
        fourteen["bf16"] > 12288,
        f"{fourteen['bf16']:.0f} MiB",
    )
    check(
        "3B BF16 does fit the card",
        three["bf16"] < 12288,
        f"{three['bf16']:.0f} MiB",
    )
    if three["nf4"] is not None:
        check(
            "3B NF4 .chr is far smaller than 3B BF16",
            three["nf4"] < three["bf16"] / 3,
            f"nf4={three['nf4']:.0f} MiB",
        )


def gate_hard_run() -> None:
    lab = lab_by_slug("qwen25-3b")
    bundle = _synthetic_bundle("bf16", weight=5886.0, torch_mib=5886.0)
    items = load_fixture()["independent"][:2]
    scores = score_messages(bundle.messages, items)
    run = HardRun.of(lab, "bf16", Path("nowhere"), bundle, scores)
    check("HardRun label names the size first", run.label == "3B BF16", run.label)
    check("HardRun counts replies", run.ran and run.n_messages == 2, str(run.n_messages))
    check(
        "working set takes the largest torch reading",
        run.working_set_mib == 6398.0,
        str(run.working_set_mib),
    )
    check(
        "accuracy is 1 of 2",
        run.accuracy is not None and abs(run.accuracy - 0.5) < 1e-9,
        str(run.accuracy),
    )
    check(
        "accuracy_pct is a percentage",
        run.accuracy_pct is not None and abs(run.accuracy_pct - 50.0) < 1e-9,
        str(run.accuracy_pct),
    )
    empty = HardRun.of(
        lab,
        "nf4",
        Path("nowhere"),
        _synthetic_bundle("bf16", weight=1.0, torch_mib=1.0),  # no nf4 rows in it
        [],
    )
    check("a run with no session is not ran", not empty.ran, str(empty.n_messages))
    check("a run with no scores has no accuracy", empty.accuracy is None, str(empty.accuracy))


def gate_matrix_csv() -> None:
    lab = lab_by_slug("qwen25-3b")
    bundle = _synthetic_bundle("bf16", weight=5886.0, torch_mib=5886.0)
    items = load_fixture()["independent"][:2]
    with tempfile.TemporaryDirectory(prefix="hard-matrix-") as tmp:
        root = Path(tmp)
        run_dir = root / "qwen25-3b-bf16"
        run_dir.mkdir()
        bundle.write(run_dir)
        scores = score_messages(bundle.messages, items)
        write_scores(run_dir / "hard_scores.csv", scores)
        run = HardRun.of(lab, "bf16", run_dir, bundle, scores)
        path = write_matrix_csv(root / "hard_matrix.csv", [run])
        header = path.read_text(encoding="utf-8").splitlines()[0]
        check("hard_matrix.csv starts with lab,title,size,codec", header.startswith("lab,title,size,codec,label"), header[:48])
        rows = read_matrix(path)
        check("one row per run", len(rows) == 1, str(len(rows)))
        check("numbers come back as floats", rows[0]["working_set_mib"] == 6398.0, str(rows[0]["working_set_mib"]))
        restored = runs_from_matrix(root)
        check("round trip keeps the label", restored[0].label == "3B BF16", restored[0].label)
        check(
            "round trip keeps the scores",
            len(restored[0].scores) == 2,
            str(len(restored[0].scores)),
        )
        check(
            "round trip keeps correctness",
            [bool(row["correct"]) for row in restored[0].scores] == [True, False],
            str([row["correct"] for row in restored[0].scores]),
        )


def gate_redraw_plate() -> None:
    """A finished run root redraws its own plate with no GPU and no re-run.

    The live matrix is often launched without ``--plate``; the PNG still has
    to be recoverable from the CSVs alone, into the run dir and docs/img.
    """
    from gpu.lab.hard import plate_items, redraw_plate, write_run_script

    lab = lab_by_slug("qwen25-14b")
    bundle = _synthetic_bundle("bf16", weight=28172.0, torch_mib=28172.0)
    with tempfile.TemporaryDirectory(prefix="hard-redraw-") as tmp:
        root = Path(tmp)
        write_run_script(root / "hard_script.json")
        items = plate_items(root)
        check(
            "redraw reads the script the run was asked",
            len(items) == len(load_fixture()["independent"]),
            f"{len(items)} items",
        )
        run_dir = root / "qwen25-14b-bf16"
        run_dir.mkdir()
        bundle.write(run_dir)
        scores = score_messages(bundle.messages, items[:2])
        write_scores(run_dir / "hard_scores.csv", scores)
        run = HardRun.of(lab, "bf16", run_dir, bundle, scores)
        write_matrix_csv(root / "hard_matrix.csv", [run])

        docs_copy = root / "docs" / "hard-eval-qwen25.png"
        written = redraw_plate(root, docs_copy)
        check(
            "redraw writes the plate next to the CSVs",
            written[0] == root / "hard-eval.png" and written[0].is_file(),
            written[0].name,
        )
        check(
            "redraw also writes the docs copy",
            docs_copy.is_file() and docs_copy.stat().st_size > 20_000,
            f"{docs_copy.stat().st_size / 1024:.0f} KiB",
        )
        check(
            "redraw does not duplicate a repeated path",
            len(redraw_plate(root, root / "hard-eval.png")) == 1,
            "one target",
        )
        check(
            "redraw with no extra path still writes the run dir copy",
            len(redraw_plate(root, "")) == 1,
            "one target",
        )

    with tempfile.TemporaryDirectory(prefix="hard-redraw-bare-") as tmp:
        # No hard_script.json: fall back to the committed fixture rather than
        # drawing a table with no questions in it.
        bare = plate_items(Path(tmp))
        check(
            "a run root with no script falls back to the fixture",
            len(bare) == len(load_fixture()["independent"]),
            f"{len(bare)} items",
        )


def gate_answer_table() -> None:
    items = load_fixture()["independent"]
    rows = answer_rows(items)
    check("answer table renders with no runs", len(rows) == len(items), str(len(rows)))
    check(
        "every fixture row carries a gold on the plate",
        all(row["gold"] for row in rows),
        str([row["gold"] for row in rows][:4]),
    )
    check(
        "questions are truncated for the plate",
        all(len(row["question"]) <= 74 for row in rows),
        str(max(len(row["question"]) for row in rows)),
    )
    check("no run columns without runs", rows[0]["cells"] == {}, str(rows[0]["cells"]))

    lab = lab_by_slug("qwen25-3b")
    bundle = _synthetic_bundle("bf16", weight=5886.0, torch_mib=5886.0)
    scores = score_messages(bundle.messages, items[:2])
    run = HardRun.of(lab, "bf16", Path("nowhere"), bundle, scores)
    live = answer_rows(items, [run])
    first = live[0]["cells"]["3B BF16"]
    second = live[1]["cells"]["3B BF16"]
    third = live[2]["cells"]["3B BF16"]
    check("a hit reads ok + the answer", first.state == "ok" and first.text == "ok 164", first.text)
    check("a miss is marked, not hidden", second.state == "miss", second.text)
    check("an item this run never saw is a dash", third.state == "none" and third.text == "-", third.text)
    check(
        "no run cell is silently blank",
        all(cell.text for row in live for cell in row["cells"].values()),
        "every cell has text",
    )


def gate_answer_cell() -> None:
    check("None is a dash, not a zero", answer_cell(None).state == "none", answer_cell(None).text)
    ok = answer_cell({"correct": True, "extracted": "164"})
    check("ok cell", ok.correct and ok.text == "ok 164", ok.text)
    miss = answer_cell({"correct": False, "extracted": "0.10"})
    check("miss cell keeps the wrong answer", miss.state == "miss" and "0.10" in miss.text, miss.text)
    empty = answer_cell({"correct": False, "extracted": ""})
    check("no extractable answer says so", empty.text == "miss (no answer)", empty.text)
    human = answer_cell({"correct": False, "pending_human": True, "extracted": ""})
    check("open item is human, not a miss", human.state == "human", human.text)
    long = answer_cell({"correct": True, "extracted": "x" * 40})
    check("long answers are clipped", len(long.text) <= 20, long.text)


def gate_plate_tokens() -> None:
    """The matplotlib plate mirrors gpu.lab.plot's tokens; this is the drift guard."""
    from gpu.lab.hard_plate import CODEC_STYLE as MPL_STYLE
    from gpu.lab.hard_plate import INK, INK_SOFT, LIMIT
    from gpu.lab.plot import CODEC_STYLE as PLOTLY_STYLE
    from gpu.lab.plot import INK as PLOTLY_INK
    from gpu.lab.plot import INK_SOFT as PLOTLY_INK_SOFT
    from gpu.lab.plot import LIMIT as PLOTLY_LIMIT

    for codec in ("bf16", "nf4"):
        check(
            f"{codec} colour matches the smoke plate",
            MPL_STYLE[codec].color == PLOTLY_STYLE[codec]["color"],
            f"{MPL_STYLE[codec].color} == {PLOTLY_STYLE[codec]['color']}",
        )
    check("nf4 keeps a hatch as the second encoding", MPL_STYLE["nf4"].hatch != "", MPL_STYLE["nf4"].hatch)
    check(
        "ink tokens match the smoke plate",
        (INK, INK_SOFT, LIMIT) == (PLOTLY_INK, PLOTLY_INK_SOFT, PLOTLY_LIMIT),
        f"{INK}/{INK_SOFT}/{LIMIT}",
    )


def gate_plate_render() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    from gpu.lab.hard_plate import hard_plate, write_plate

    items = load_fixture()["independent"]
    with tempfile.TemporaryDirectory(prefix="hard-plate-") as tmp:
        fixture_only = hard_plate(items)
        path = write_plate(fixture_only, Path(tmp) / "fixture-only.png")[0]
        check(
            "plate renders from the fixture with no runs",
            path.is_file() and path.stat().st_size > 20_000,
            f"{path.stat().st_size / 1024:.0f} KiB",
        )
        texts = [artist.get_text() for artist in fixture_only.findobj(match=_is_text)]
        golds = {item.gold for item in items if item.gold}
        check(
            "every gold answer is drawn on the plate",
            all(any(gold == text for text in texts) for gold in golds),
            f"{len(golds)} golds",
        )
        check(
            "the plate says there is no live session yet",
            any("no live session yet" in text for text in texts),
            "fixture-only band",
        )

        lab = lab_by_slug("qwen25-14b")
        bundle = _synthetic_bundle("bf16", weight=28172.0, torch_mib=28172.0)
        scores = score_messages(bundle.messages, items[:2])
        run = HardRun.of(lab, "bf16", Path(tmp), bundle, scores)
        with_run = hard_plate(items, [run])
        live = write_plate(with_run, Path(tmp) / "live.png")[0]
        check(
            "plate renders with a live run",
            live.is_file() and live.stat().st_size > 20_000,
            f"{live.stat().st_size / 1024:.0f} KiB",
        )
        live_texts = [artist.get_text() for artist in with_run.findobj(match=_is_text)]
        check(
            "the 14B bar is labelled 14B",
            any("14B" in text for text in live_texts),
            "size on the tick label",
        )
        spilled = f"{run.working_set_mib:,.0f}"
        check(
            "the spilled working set is printed as measured",
            run.working_set_mib > 12288 and any(spilled in text for text in live_texts),
            f"{spilled} MiB, above the card",
        )
        check(
            "the card limit is on the plate",
            any("12,288" in text for text in live_texts),
            "card limit",
        )


def gate_pending_slots() -> None:
    """A matrix drawn mid-queue keeps all four ticks and claims nothing for the rest.

    The 14B BF16 spill takes hours, so this is the state the plate is usually
    drawn in. A padded slot must carry no measurement at all.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    from gpu.lab.hard_plate import PANELS, hard_plate

    items = load_fixture()["independent"]
    bundle = _synthetic_bundle("bf16", weight=5886.0, torch_mib=5978.0)
    scores = score_messages(bundle.messages, items[:2])
    with tempfile.TemporaryDirectory(prefix="hard-pending-") as tmp:
        first = HardRun.of(
            lab_by_slug("qwen25-3b"), "bf16", Path(tmp) / "qwen25-3b-bf16", bundle, scores
        )
        padded = pad_runs([first])
        check(
            "one finished run still draws all four slots",
            [run.label for run in padded] == ["3B BF16", "14B BF16", "3B NF4", "14B NF4"],
            str([run.label for run in padded]),
        )
        check(
            "the finished run is the live object, not a copy",
            padded[0] is first and not padded[0].pending,
            f"{padded[0].label} ran={padded[0].ran}",
        )
        queued = padded[1:]
        check(
            "the pairs still queued are marked pending",
            all(run.pending and not run.ran for run in queued),
            str([run.label for run in queued]),
        )
        check(
            "a pending slot invents no measurement",
            all(
                run.working_set_mib is None
                and run.smi_peak_mib is None
                and run.mean_ttft_ms is None
                and run.mean_tok_s is None
                and run.accuracy is None
                for run in queued
            ),
            "working set / smi / ttft / tok-s / accuracy all None",
        )
        check(
            "a pending 14B slot still carries the weight expected on disk",
            all(run.expected_weight_mib is None or run.expected_weight_mib > 0 for run in queued),
            str([run.expected_weight_mib for run in queued]),
        )
        check(
            "padding a finished matrix changes nothing",
            [run.label for run in pad_runs(padded)] == [run.label for run in padded],
            "idempotent",
        )

        figure = hard_plate(items, [first])
        texts = [artist.get_text() for artist in figure.findobj(match=_is_text)]
        # Matched exactly: the caption paragraph also contains "not run yet".
        queued_bars = sum(1 for text in texts if text == "not run\nyet")
        check(
            "the queued bars say not run yet, not zero",
            queued_bars == len(queued) * len(PANELS) and not any(text == "0" for text in texts),
            f"{queued_bars} pending bar labels, no value label",
        )
        check(
            "the band says the run is still in progress",
            any("RUN IN PROGRESS" in text for text in texts),
            "in-progress band",
        )
        for label in ("14B BF16", "3B NF4", "14B NF4"):
            check(
                f"{label} is named as queued in the answer table header",
                any(f"{label} (queued)" in text for text in texts),
                "queued column header",
            )
        check(
            "the live run's accuracy is still drawn",
            any("50%" in text for text in texts),
            "3B BF16 scored 1 of 2 synthetic items",
        )
        # order=None is the escape hatch for drawing exactly what was measured.
        exact = hard_plate(items, [first], order=None)
        exact_texts = [artist.get_text() for artist in exact.findobj(match=_is_text)]
        check(
            "order=None draws only the runs given",
            not any(text == "not run\nyet" for text in exact_texts),
            "no padded slots",
        )


def _is_text(artist: Any) -> bool:
    from matplotlib.text import Text

    return isinstance(artist, Text)


def gate_recorded_miss() -> None:
    """A missing model or a missing .chr is a HardRun with notes, not a traceback."""
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
    with tempfile.TemporaryDirectory(prefix="hard-miss-") as tmp:
        root = Path(tmp)
        run = run_hard_one(nowhere, root, codec="bf16", verbose=False)
        check("missing model dir does not raise", not run.ran, run.notes[:70])
        check("the miss says what was missing", "model dir missing" in run.notes, run.notes[:90])
        check(
            "a recorded miss still writes its CSVs",
            (run.out_dir / "summary.csv").is_file() and (run.out_dir / "hard_scores.csv").is_file(),
            str(run.out_dir.name),
        )
        check("a recorded miss invents no VRAM number", run.working_set_mib is None, str(run.working_set_mib))
        # A real model dir with a bogus .chr, so the NF4 branch is what fails.
        no_chr = LabModel(
            **{**nowhere.__dict__, "slug": "no-chr", "model_dir": lab_by_slug("qwen25-3b").model_dir}
        )
        nf4 = run_hard_one(no_chr, root, codec="nf4", verbose=False)
        expected_note = "model dir missing" if not Path(no_chr.model_dir).is_dir() else "compress"
        check(
            "missing .chr is a skip with the compress command",
            not nf4.ran and expected_note in nf4.notes,
            nf4.notes[:90],
        )
        unsupported = run_hard_one(
            LabModel(**{**no_chr.__dict__, "nf4_driver": "unsupported"}),
            root,
            codec="nf4",
            verbose=False,
        )
        check(
            "an unsupported nf4 driver is a skip, not a crash",
            not unsupported.ran
            and ("driver" in unsupported.notes or "model dir missing" in unsupported.notes),
            unsupported.notes[:90],
        )
        try:
            run_hard_one(nowhere, root, codec="int8", verbose=False)
            check("an unknown codec raises", False, "no raise")
        except ValueError as error:
            check("an unknown codec raises ValueError", "int8" in str(error), str(error)[:60])


def gate_worker_cli() -> None:
    from gpu.lab.worker import _parser

    args = _parser().parse_args(
        ["--codec", "bf16", "--out", "x", "--items-json", "script.json", "--conversation", "history"]
    )
    check("worker accepts --items-json", args.items_json == "script.json", args.items_json)
    check("worker accepts --conversation", args.conversation == "history", args.conversation)
    v2 = _parser().parse_args(
        ["--codec", "nf4", "--out", "x", "--executor", "decodev2"]
    )
    check("worker accepts --executor decodev2", v2.executor == "decodev2", v2.executor)
    default = _parser().parse_args(["--codec", "nf4", "--out", "x"])
    check("worker executor defaults to tokenloop", default.executor == "tokenloop", default.executor)
    from gpu.lab.hard import _parser as hard_parser

    hard = hard_parser().parse_args(["--lab", "qwen25-3b", "--executor", "decodev2", "--max-seq", "2048"])
    check("hard CLI accepts --executor decodev2", hard.executor == "decodev2", hard.executor)
    check("hard CLI accepts --max-seq 2048", hard.max_seq == 2048, str(hard.max_seq))
    quiet_smi = hard_parser().parse_args(
        ["--lab", "internlm20b", "--interval", "0"]
    )
    check("hard CLI accepts --interval 0", quiet_smi.interval == 0.0, str(quiet_smi.interval))
    from gpu.lab.sampler import Sampler

    try:
        Sampler("nf4", interval_s=0.2)
        check("sampler rejects interval > 0.15", False, "no raise")
    except ValueError:
        check("sampler rejects interval > 0.15", True, "")
    off = Sampler("nf4", interval_s=0.0)
    check("sampler allows interval 0", off.interval_s == 0.0, str(off.interval_s))
    try:
        hard_parser().parse_args(["--lab", "qwen25-32b", "--executor", "decodev2"])
        # parse_args succeeds; main() rejects. Flag must still parse.
        check("32B decodev2 still parses (main refuses)", True, "")
    except SystemExit as exc:
        check("32B decodev2 still parses (main refuses)", False, str(exc))


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print("gpu/lab hard eval (fixture only, no GPU)\n")
    gate_fixture()
    gate_extract_number()
    gate_extract_yesno()
    gate_score_items()
    gate_quality_fn()
    gate_score_messages()
    gate_write_scores()
    gate_run_script()
    gate_gold_round_trip()
    gate_deepfold_hard_override()
    gate_catalog()
    gate_chat_history()
    gate_worker_cli()
    gate_run_order()
    gate_size_and_weights()
    gate_hard_run()
    gate_matrix_csv()
    gate_redraw_plate()
    gate_answer_table()
    gate_answer_cell()
    gate_recorded_miss()
    gate_plate_tokens()
    gate_plate_render()
    gate_pending_slots()
    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Acceptance for ``gpu/lab``: one command, a fixture, no 3B.

    C:\\Users\\Professional\\anaconda3\\envs\\torch-gpu\\python.exe -m gpu.lab.test_lab
    python gpu/lab/test_lab.py

The live 3B belongs to agent 11. This file only proves the harness is honest:

1. the four CSVs are written with exactly the frozen headers;
2. a bundle survives the CSV round trip, values and gaps included;
3. ``N/A`` from the driver stays an empty cell and never becomes ``0``;
4. every required event is present for both codecs, and ``message_id`` is set
   inside a turn and empty outside it;
5. both codecs start at ``t_s ~ 0``, i.e. the x axis is session-local;
6. ``comparison_figure`` is one plate: five panels (VRAM, budget, replies,
   TTFT, tok/s), both codecs, colour **and** dash per codec, the card limit
   drawn, no rotated text anywhere and no legend sitting on the data;
7. the panel titles are written from the tables -- the plate says "a layer was
   materialised" when the two VRAM traces meet, and refuses to claim a decode
   win the numbers do not support;
8. the claims fit their panel: nothing in a title would clip on export;
9. a fixture is stamped SYNTHETIC on the plate and on the instrument sheet;
10. ``lab.html`` is written and is a real Plotly document; ``lab.png`` is either
    written or a recorded skip, never a placeholder;
11. the public import surface is what ``docs/lab.md`` froze;
12. the quality needles accept the right answers and reject the wrong ones.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab import (  # noqa: E402
    CODECS,
    EVENTS_COLUMNS,
    MESSAGES,
    MESSAGES_COLUMNS,
    REQUIRED_EVENTS,
    SUMMARY_COLUMNS,
    TIMELINE_COLUMNS,
    LabBundle,
)
from gpu.lab.bundle import format_cell  # noqa: E402
from gpu.lab.fixture import fixture_bundle  # noqa: E402
from gpu.lab.plot import (  # noqa: E402
    CODEC_STYLE,
    LIMIT,
    comparison_figure,
    telemetry_figure,
    write_artifacts,
)
from gpu.lab.script import quality_ok  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def _gate(name: str, body: Callable[[], tuple[bool, str]]) -> None:
    """Run one gate; an exception is a failure, not a traceback on the floor."""
    try:
        ok, detail = body()
    except Exception as error:  # noqa: BLE001
        check(name, False, f"{type(error).__name__}: {error}")
        return
    check(name, ok, detail)


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #


def gate_csv_headers(out_dir: Path, bundle: LabBundle) -> None:
    paths = bundle.write(out_dir)
    expected = {
        "timeline": TIMELINE_COLUMNS,
        "events": EVENTS_COLUMNS,
        "messages": MESSAGES_COLUMNS,
        "summary": SUMMARY_COLUMNS,
    }
    check("four CSVs written", set(paths) == set(expected), ", ".join(sorted(paths)))
    for table, columns in expected.items():
        header = paths[table].read_text(encoding="utf-8").splitlines()[0]
        check(f"{table}.csv header is frozen", header == ",".join(columns), header[:80])


def gate_round_trip(out_dir: Path, bundle: LabBundle) -> None:
    reloaded = LabBundle.read(out_dir)
    for table in ("timeline", "events", "messages", "summary"):
        original, copy = bundle.table(table), reloaded.table(table)
        if len(original) != len(copy):
            check(f"{table} round trip", False, f"{len(original)} rows out, {len(copy)} back")
            continue
        mismatch = next(
            (
                (index, key)
                for index, (left, right) in enumerate(zip(original, copy))
                for key in left
                if format_cell(table, key, left[key]) != format_cell(table, key, right[key])
            ),
            None,
        )
        check(f"{table} round trip", mismatch is None, f"{len(copy)} rows" if not mismatch
              else f"row {mismatch[0]} column {mismatch[1]}")


def gate_missing_stays_empty(out_dir: Path) -> None:
    """A field the driver does not report must be an empty cell, not ``0``."""
    gappy = LabBundle.of(
        timeline=[
            {"t_s": 0.0, "codec": "nf4", "used_mib": 900.0, "power_w": None, "temp_c": None},
            {"t_s": 0.12, "codec": "nf4", "used_mib": 901.0, "power_w": "N/A", "temp_c": "[N/A]"},
        ],
        summary=[{"codec": "nf4", "kv_mib": None, "n_messages": 0, "quality_all_ok": False}],
    )
    directory = out_dir / "gaps"
    gappy.write(directory)
    lines = (directory / "timeline.csv").read_text(encoding="utf-8").splitlines()
    power_index = TIMELINE_COLUMNS.index("power_w")
    cells = [line.split(",")[power_index] for line in lines[1:]]
    check("N/A becomes an empty cell", all(cell == "" for cell in cells), f"cells={cells}")
    back = LabBundle.read(directory)
    check(
        "empty cell reads back as None",
        all(row["power_w"] is None for row in back.timeline)
        and back.summary[0]["kv_mib"] is None,
        "power_w and kv_mib are None, not 0.0",
    )


def gate_events(bundle: LabBundle) -> None:
    for codec in CODECS:
        names = {str(row["event"]) for row in bundle.rows_for("events", codec)}
        missing = [event for event in REQUIRED_EVENTS if event not in names]
        check(f"{codec}: all required events", not missing, f"missing={missing}" if missing
              else f"{len(names)} distinct event names")
        windows = bundle.message_windows(codec)
        check(
            f"{codec}: one msg_send/msg_done window per message",
            len(windows) == len(MESSAGES)
            and all(window.t_done > window.t_send for window in windows)
            and all(window.t_first_token is not None for window in windows),
            f"{len(windows)} windows",
        )


def gate_message_ids(bundle: LabBundle) -> None:
    for codec in CODECS:
        windows = bundle.message_windows(codec)
        wrong_inside = 0
        wrong_outside = 0
        for row in bundle.rows_for("timeline", codec):
            t_s = float(row["t_s"])
            inside = next(
                (w for w in windows if w.t_send <= t_s <= w.t_done),
                None,
            )
            tagged = str(row["message_id"])
            if inside is None and tagged:
                wrong_outside += 1
            elif inside is not None and tagged != str(inside.message_id):
                wrong_inside += 1
        check(
            f"{codec}: message_id set inside a turn, empty outside",
            wrong_inside == 0 and wrong_outside == 0,
            f"inside={wrong_inside} outside={wrong_outside}",
        )


def gate_session_local_time(bundle: LabBundle) -> None:
    for codec in CODECS:
        times, _ = bundle.series(codec, "used_mib")
        ok = bool(times) and times[0] <= 0.5 and times == sorted(times)
        check(
            f"{codec}: t_s is session-local and monotonic",
            ok,
            f"t0={times[0]:.2f}s, t_end={times[-1]:.2f}s, n={len(times)}" if times else "no rows",
        )


_TAGS = re.compile(r"<[^>]+>")


def _plain(text: Any) -> str:
    """Annotation text with the HTML stripped and the line breaks kept."""
    return _TAGS.sub("", str(text or "").replace("<br>", "\n"))


def _panel_titles(figure: Any) -> dict[str, str]:
    """``{"A": "the claim ...", ...}`` for the five panel titles of the plate."""
    titles: dict[str, str] = {}
    for annotation in figure.layout.annotations:
        text = _plain(annotation.text).strip()
        head = text.split(maxsplit=1)
        if len(head) == 2 and len(head[0]) == 1 and head[0] in "ABCDEF":
            titles.setdefault(head[0], text)
    return titles


def gate_figure(bundle: LabBundle) -> Any:
    import plotly.graph_objects as go

    figure = comparison_figure(bundle)
    check("comparison_figure returns one go.Figure", isinstance(figure, go.Figure),
          f"{len(figure.data)} traces")

    layout = figure.layout.to_plotly_json()
    y_axes = sorted(key for key in layout if key.startswith("yaxis"))
    check("VRAM is two side-by-side panels plus A–E", len(y_axes) == 6, f"axes={y_axes}")

    bands = [tuple(layout[key]["domain"]) for key in y_axes]
    overlap = [
        (a, b)
        for index, a in enumerate(bands)
        for b in bands[index + 1:]
        if a != b and a[0] < b[1] - 1e-9 and b[0] < a[1] - 1e-9
    ]
    check("panels are shelved, not stacked on top of each other", not overlap,
          f"{len(set(bands))} vertical bands")

    # --- both codecs, and colour is never the only encoding --------------
    for codec in CODECS:
        style = CODEC_STYLE[codec]
        colored = [
            trace
            for trace in figure.data
            if style["color"] in (
                str(getattr(getattr(trace, "line", None), "color", "")),
                str(getattr(getattr(trace, "marker", None), "color", "")),
            )
        ]
        check(f"{style['label']} is on the plate", len(colored) >= 2,
              f"{len(colored)} traces in {style['color']}")
    dashes = {
        str(trace.line.dash)
        for trace in figure.data
        if getattr(trace, "line", None) is not None and trace.line.dash
    }
    check("codec is encoded twice: colour and dash",
          {CODEC_STYLE[codec]["dash"] for codec in CODECS} <= dashes,
          f"dashes={sorted(dashes)}")

    # --- panel A: two side-by-side VRAM graphs, one shared Y scale --------
    vram = [
        trace
        for trace in figure.data
        if isinstance(trace, go.Scatter)
        and str(getattr(trace, "mode", "")) == "lines"
        and (trace.xaxis or "x") in {"x", "x2"}
    ]
    check("panel A is lines only, no event markers",
          bool(vram) and all(str(trace.mode) == "lines" for trace in vram),
          f"{len(vram)} traces, modes={sorted({str(t.mode) for t in vram})}")
    check("panel A keeps gaps as gaps",
          bool(vram) and all(trace.connectgaps is False for trace in vram),
          "connectgaps=False, so a missing sample is never a zero")
    vram_axes = {str(trace.xaxis or "x") for trace in vram}
    check("VRAM traces are not overlaid on one axis",
          vram_axes == {"x", "x2"}, f"axes={sorted(vram_axes)}")
    left_x = list(layout["xaxis"]["domain"])
    right_x = list(layout["xaxis2"]["domain"])
    check("VRAM panels sit side by side", left_x[1] < right_x[0],
          f"left={left_x} right={right_x}")
    total = bundle.total_mib()
    left = list(layout["yaxis"]["range"])
    right = list(layout["yaxis2"]["range"])
    check("both codecs share one 0..card VRAM scale",
          left == right and left[0] == 0 and left[1] >= total,
          f"left={left} right={right} card={total:.0f}")
    limits = [
        shape for shape in figure.layout.shapes
        if shape.line is not None and str(shape.line.color) == LIMIT
    ]
    check("the card limit is a real line, on A and on B", len(limits) >= 2,
          f"{len(limits)} limit lines")
    check("hovermode is x unified", figure.layout.hovermode == "x unified",
          str(figure.layout.hovermode))
    slider = (layout.get("xaxis") or {}).get("rangeslider") or {}
    check("no range slider", not slider.get("visible"), f"rangeslider={slider}")

    # --- the claims ------------------------------------------------------
    titles = _panel_titles(figure)
    check("every panel states a claim", sorted(titles) == list("ABCDE"),
          ", ".join(sorted(titles)))
    budget = {"A": 160, "B": 160, "C": 160, "D": 92, "E": 92}
    clipped = [
        (letter, len(line))
        for letter, text in titles.items()
        for line in text.splitlines()
        if len(line) > budget[letter]
    ]
    check("no claim line is wide enough to clip on export", not clipped, f"over={clipped}")

    check("A names the after-load VRAM of both codecs",
          "2,712" in titles.get("A", "") and "7,850" in titles.get("A", ""),
          titles.get("A", "").splitlines()[0][:92])
    check("B says the dense copy is never materialised",
          "1,636" in titles.get("B", "") and "materialised" in titles.get("B", ""),
          titles.get("B", "").splitlines()[0][:92])
    check("C reports the needles",
          "Paris" in titles.get("C", "") and "323" in titles.get("C", ""),
          titles.get("C", "").splitlines()[0][:92])
    check("D calls TTFT prefill", "prefill" in titles.get("D", "").lower(),
          titles.get("D", "").splitlines()[0][:92])
    check("E refuses a decode win the numbers deny",
          "not faster" in titles.get("E", "").lower(),
          titles.get("E", "").splitlines()[0][:92])

    # --- typography ------------------------------------------------------
    rotated = [
        _plain(annotation.text)[:40]
        for annotation in figure.layout.annotations
        if annotation.textangle not in (None, 0, 0.0, "0")
    ]
    check("no rotated text anywhere", not rotated, f"rotated={rotated}")
    angled = [
        key for key in layout
        if key.startswith(("xaxis", "yaxis"))
        and layout[key].get("tickangle") not in (None, 0, 0.0, "auto")
    ]
    check("no rotated tick labels", not angled, f"angled={angled}")
    titled_y = [
        key for key in layout
        if key.startswith("yaxis") and (layout[key].get("title") or {}).get("text")
    ]
    check("no rotated y-axis titles: the unit lives in the panel note",
          not titled_y, f"titled={titled_y}")
    families = {
        str(annotation.font.family)
        for annotation in figure.layout.annotations
        if annotation.font is not None and annotation.font.family
    }
    check("one typeface", len(families) <= 1, f"families={sorted(families)}")
    check("legend sits above the plate, not on the data",
          figure.layout.legend.y is not None and float(figure.layout.legend.y) > 1.0,
          f"legend.y={figure.layout.legend.y}")

    xs = " ".join(str(getattr(trace, "x", "")) for trace in figure.data)
    check("per-turn bars labelled turn N", "turn " in xs, "turn 1 / turn 2 / turn 3")

    english = _plain(" ".join(str(ann.text) for ann in figure.layout.annotations)) + str(layout)
    check("English plate title and axis labels",
          "RTX 3080 12 GB" in english and "session time, s" in english,
          "title + x axis")
    return figure


def gate_claims_follow_the_data(bundle: LabBundle) -> None:
    """The plate has to contradict us when the numbers do."""

    def retold(codec: str, **changes: Any) -> LabBundle:
        return LabBundle.of(
            timeline=bundle.timeline,
            events=bundle.events,
            messages=bundle.messages,
            summary=[
                {**row, **changes} if row["codec"] == codec else row for row in bundle.summary
            ],
            source="fixture",
        )

    title = _panel_titles(
        comparison_figure(retold("nf4", vram_after_load_smi_mib=7700.0))
    ).get("A", "")
    check("A calls a materialised layer a bug, not a win",
          "bug, not a win" in title, title.splitlines()[0][:104])

    title = _panel_titles(comparison_figure(retold("nf4", mean_decode_tok_s=99.0))).get("E", "")
    check("E reports a decode win when the numbers show one",
          "99.0" in title and "not faster" not in title.lower(), title.splitlines()[0][:104])

    title = _panel_titles(
        comparison_figure(
            LabBundle.of(
                timeline=bundle.timeline,
                events=bundle.events,
                messages=[{**row, "quality_ok": False} for row in bundle.messages],
                summary=bundle.summary,
                source="fixture",
            )
        )
    ).get("C", "")
    check("C admits a missed needle", "missed the needle" in title, title.splitlines()[0][:104])


def _overflow_bundle(bundle: LabBundle) -> LabBundle:
    """The 14B case: nvidia-smi sits on the card, CUDA holds ~28 GiB."""
    timeline = []
    for row in bundle.timeline:
        row = dict(row)
        if row["codec"] == "bf16":
            alloc = row.get("torch_alloc_mib")
            if alloc is not None and float(alloc) > 1000:
                row["torch_alloc_mib"] = 28270.0
                row["used_mib"] = 11946.0
        timeline.append(row)
    summary = []
    for row in bundle.summary:
        row = dict(row)
        if row["codec"] == "bf16":
            row["vram_after_load_smi_mib"] = 11946.0
            row["vram_after_load_torch_mib"] = 28270.0
            row["vram_peak_smi_mib"] = 12037.0
            row["weight_mib"] = 28172.0
        summary.append(row)
    return LabBundle.of(
        timeline=timeline,
        events=bundle.events,
        messages=bundle.messages,
        summary=summary,
        source=bundle.source,
    )


def gate_overflow_vram(bundle: LabBundle) -> None:
    """When CUDA spills past VRAM, the plate must grow a second pair of graphs."""
    import plotly.graph_objects as go

    spilled = _overflow_bundle(bundle)
    figure = comparison_figure(spilled)
    layout = figure.layout.to_plotly_json()
    y_axes = sorted(key for key in layout if key.startswith("yaxis"))
    check("spill adds a GPU-memory pair under VRAM", len(y_axes) == 8, f"axes={y_axes}")
    working = [
        trace
        for trace in figure.data
        if isinstance(trace, go.Scatter)
        and str(getattr(trace, "mode", "")) == "lines"
        and (trace.xaxis or "x") in {"x3", "x4"}
    ]
    check("working-set traces sit on the extra pair, not overlaid on VRAM",
          {str(trace.xaxis or "x") for trace in working} == {"x3", "x4"},
          f"axes={sorted(str(t.xaxis or 'x') for t in working)}")
    left = list(layout["yaxis3"]["range"])
    right = list(layout["yaxis4"]["range"])
    check("working-set panels share one Y that exceeds the card",
          left == right and left[1] > 12288,
          f"left={left} right={right}")
    titles = _panel_titles(figure)
    check("spill plate still states A–E plus F",
          sorted(titles) == list("ABCDEF"), ", ".join(sorted(titles)))
    check("A does not treat a capped nvidia-smi as the whole footprint",
          "capped" in titles.get("A", "").lower() and "shared" in titles.get("A", "").lower(),
          titles.get("A", "").splitlines()[0][:104])
    check("F names the CUDA working set past VRAM",
          "28,270" in titles.get("F", "") and "shared" in titles.get("F", "").lower(),
          titles.get("F", "").splitlines()[0][:104])



def gate_fixture_stamp(bundle: LabBundle) -> None:
    for name, figure in (
        ("plate", comparison_figure(bundle)),
        ("instrument sheet", telemetry_figure(bundle)),
    ):
        stamped = [
            _plain(annotation.text)
            for annotation in figure.layout.annotations
            if "SYNTHETIC FIXTURE DATA" in str(annotation.text)
        ]
        check(f"{name} is stamped SYNTHETIC", bool(stamped),
              stamped[0].splitlines()[0][:72] if stamped else "no stamp")


def gate_artifacts(out_dir: Path, figure: Any, bundle: LabBundle) -> None:
    artifacts = write_artifacts(figure, out_dir, png=True)
    extra = write_artifacts(telemetry_figure(bundle), out_dir, png=True, name="lab-telemetry")
    html = Path(artifacts["html"])
    text = html.read_text(encoding="utf-8", errors="replace")
    check("lab.html written", html.is_file() and "plotly" in text.lower(),
          f"{html.stat().st_size // 1024} KiB")
    check("lab-telemetry.html written next to it",
          Path(extra["html"]).is_file(), Path(extra["html"]).name)
    if artifacts.get("png"):
        png = Path(artifacts["png"])
        check("lab.png written (kaleido present)", png.is_file() and png.stat().st_size > 1024,
              f"{png.stat().st_size // 1024} KiB")
    else:
        # Explicitly not a failure: the freeze says HTML is required and PNG is
        # a recorded skip. A fake image would be worse than no image.
        check("lab.png skip is recorded, not faked",
              not (out_dir / "lab.png").is_file(),
              f"skipped: {artifacts.get('png_skip_reason', '')[:90]}")


def gate_public_api() -> None:
    import gpu.lab as lab

    frozen = ("run_both", "comparison_figure", "MESSAGES", "run_bf16", "run_nf4", "quality_ok")
    missing = [name for name in frozen if not hasattr(lab, name)]
    check("frozen public API", not missing, f"missing={missing}" if missing else ", ".join(frozen))
    check("MESSAGES is the frozen three-turn script",
          len(MESSAGES) == 3 and "capital of France" in MESSAGES[0],
          f"{len(MESSAGES)} messages")
    check("gpu stays a namespace package", not (_REPO / "gpu" / "__init__.py").exists(),
          "no gpu/__init__.py")
    import gpu.lab.run as cli

    check("python -m gpu.lab.run exists", callable(cli.main), "run.main")


def gate_needles() -> None:
    good = ("The capital of France is Paris.", "Berlin.", "323")
    bad = ("The capital of France is Lyon.", "Munich", "324")
    check("needles accept the right answers",
          all(quality_ok(index, text) for index, text in enumerate(good, start=1)), "1..3 ok")
    check("needles reject the wrong answers",
          not any(quality_ok(index, text) for index, text in enumerate(bad, start=1)), "1..3 fail")
    check("unknown message_id is not a silent pass", not quality_ok(9, "anything"), "id=9")

    from gpu.lab.sessions import _is_cuda_oom, _recorded_miss_notes

    oom = RuntimeError("CUDA out of memory. Tried to allocate 20480.00 MiB")
    check("HF-wrapped OOM text is a recorded miss", _is_cuda_oom(oom), "out of memory")
    dep = _recorded_miss_notes(
        ModuleNotFoundError("No module named 'sentencepiece'"),
        "internlm2_5-20b-chat",
    )
    check(
        "missing tokenizer dep is a recorded miss, not a crash",
        "ModuleNotFoundError" in dep and "sentencepiece" in dep,
        dep[:80],
    )

    stale = _recorded_miss_notes(
        TypeError('can only concatenate tuple (not "int") to tuple'),
        "internlm2_5-20b-chat",
    )
    check(
        "transformers-5 vs remote-code mismatch is named, not a bare TypeError",
        "transformers 5" in stale and "fake baseline" in stale,
        stale[:80],
    )
    other = _recorded_miss_notes(TypeError("unrelated"), "internlm2_5-20b-chat")
    check(
        "an unrelated TypeError keeps the plain wording",
        "transformers 5" not in other and "TypeError" in other,
        other[:80],
    )

    gate_stop_tokens()


class _FakeTokenizer:
    """Just enough of a tokenizer for :func:`stop_token_ids`."""

    unk_token_id = 0

    def __init__(self, vocab: dict[str, int], eos_token_id: int | None) -> None:
        self._vocab = dict(vocab)
        self._ids = {i: t for t, i in vocab.items()}
        self.eos_token_id = eos_token_id

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id: int) -> str | None:
        return self._ids.get(token_id)


def gate_stop_tokens() -> None:
    """The greedy stop set is the tokenizer's, not a hardcoded Qwen pair.

    Qwen2.5 ties ``eos_token`` to ``<|im_end|>`` so it must keep answering the
    same two ids the 3B and 14B labs were measured with. InternLM2 does not tie
    them: without ``<|im_end|>`` (92542) a 20B reply runs past the end of its
    turn to ``max_new_tokens``.
    """
    from gpu.lab.script import QWEN_ENDOFTEXT, QWEN_IM_END, stop_token_ids

    qwen = _FakeTokenizer(
        {"<|endoftext|>": QWEN_ENDOFTEXT, "<|im_end|>": QWEN_IM_END}, QWEN_IM_END
    )
    internlm = _FakeTokenizer(
        {"</s>": 2, "<|im_end|>": 92542, "<|im_start|>": 92543}, 2
    )
    plain = _FakeTokenizer({"</s>": 2}, 2)

    check("Qwen stop set is unchanged (3B/14B labs stay comparable)",
          stop_token_ids(qwen) == (QWEN_ENDOFTEXT, QWEN_IM_END),
          str(stop_token_ids(qwen)))
    check("InternLM2 stop set adds <|im_end|> to eos",
          stop_token_ids(internlm) == (2, 92542),
          f"{stop_token_ids(internlm)} == generation_config eos_token_id")
    check("no Qwen id leaks into a non-Qwen stop set",
          not ({QWEN_ENDOFTEXT, QWEN_IM_END} & set(stop_token_ids(internlm))),
          "92542 not 151645")
    check("a tokenizer with only eos still stops",
          stop_token_ids(plain) == (2,), str(stop_token_ids(plain)))
    check("no tokenizer falls back to the Qwen fallback template's pair",
          stop_token_ids(None) == (QWEN_ENDOFTEXT, QWEN_IM_END),
          str(stop_token_ids(None)))


def gate_fixture_is_labelled(bundle: LabBundle) -> None:
    check("fixture is flagged synthetic", bundle.synthetic,
          "summary.notes says FIXTURE, the figure stamps it")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m gpu.lab.test_lab",
        description="Fixture-only acceptance for gpu/lab. Never loads the 3B.",
    )
    parser.add_argument("--out", help="where to write the artifacts (default: a temp dir)")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="gpu-lab-test-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"gpu/lab acceptance (fixture only, no GPU)\nout: {out_dir}\n")

    bundle = fixture_bundle()
    print(f"fixture: {bundle!r}\n")

    print("csv:")
    gate_csv_headers(out_dir, bundle)
    gate_round_trip(out_dir, bundle)
    gate_missing_stays_empty(out_dir)

    print("\nrecorder:")
    gate_events(bundle)
    gate_message_ids(bundle)
    gate_session_local_time(bundle)
    gate_fixture_is_labelled(bundle)

    print("\nfigure:")
    figure = gate_figure(bundle)
    gate_claims_follow_the_data(bundle)
    gate_overflow_vram(bundle)
    gate_fixture_stamp(bundle)

    print("\nartifacts:")
    gate_artifacts(out_dir, figure, bundle)

    print("\napi:")
    gate_public_api()
    gate_needles()

    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    print(f"artifacts: {out_dir}")
    print("TEST: PASS" if not failed else "TEST: FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

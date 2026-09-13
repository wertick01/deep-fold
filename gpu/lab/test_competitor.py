"""GPU-free tests for the competitor harness. Installs nothing, launches nothing.

The property under test is negative: on a box without bitsandbytes / GPTQ / AWQ /
ExLlamaV2 / llama.cpp CUDA / vLLM, every slot must produce a ``SKIP:`` row whose
numeric cells are **empty**, including the named linear microbench grid. A
fabricated tok/s or kernel µs has to be impossible to reach by accident. CPU
llama.cpp is not a slot.

    python -m gpu.lab.test_competitor
    python gpu/lab/test_competitor.py
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

from gpu.lab.catalog import lab_by_slug  # noqa: E402
from gpu.lab.competitor import (  # noqa: E402
    COMPETITOR_COLUMNS,
    COMPETITOR_STACKS,
    MICROBENCH_COLUMNS,
    MICROBENCH_KN,
    MICROBENCH_LINEARS,
    MICROBENCH_M,
    OUR_STACKS,
    PROMPT_SETS,
    RECIPE_DOC,
    SKIP,
    STACKS,
    VENV_ROOT,
    CompetitorRow,
    Detection,
    MicrobenchRow,
    detect,
    detect_all,
    microbench_grid_for,
    prompts_for,
    read_competitor_csv,
    read_microbench_csv,
    recipe_path,
    run_matrix,
    run_stack,
    stack_by_name,
    venv_dir,
    write_competitor_csv,
    write_microbench_csv,
)
from gpu.lab.script import MESSAGES  # noqa: E402

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


def gate_slots() -> None:
    names = [stack.name for stack in COMPETITOR_STACKS]
    check(
        "ChatGPT GPU stacks exist in attempt order",
        names
        == [
            "bitsandbytes-nf4",
            "gptq-marlin",
            "awq",
            "llamacpp-q4",
            "exllamav2-exl2",
            "vllm",
        ],
        str(names),
    )
    check(
        "our own rows are separate and are wired to the real worker",
        [stack.name for stack in OUR_STACKS] == ["deepfold-nf4", "hf-bf16"]
        and all(stack.wired and stack.ours for stack in OUR_STACKS),
        str([stack.name for stack in OUR_STACKS]),
    )
    check(
        "no competitor slot claims a measuring body yet",
        not any(stack.wired for stack in COMPETITOR_STACKS),
        "skeleton",
    )
    check(
        "every competitor slot names the env var for its native artifact",
        all(stack.artifact_env and stack.artifact_kind for stack in COMPETITOR_STACKS),
        str([stack.artifact_env for stack in COMPETITOR_STACKS]),
    )
    check(
        "each stack keeps its own artifact kind: no GGUF-to-CHR conversion",
        {stack.artifact_kind for stack in COMPETITOR_STACKS}
        == {"hf-dir", "quant-dir", "gguf-file"},
        str(sorted({s.artifact_kind for s in COMPETITOR_STACKS})),
    )
    check(
        "llama.cpp's note says a CPU-only build is a skip",
        "CPU tok/s is not a GPU competitor" in stack_by_name("llamacpp-q4").note,
        "no CPU number",
    )
    check(
        "llama.cpp is labelled CUDA Q4, not a CPU slot",
        "CUDA Q4" in stack_by_name("llamacpp-q4").label,
        stack_by_name("llamacpp-q4").label,
    )
    check(
        "there is no CPU llama.cpp slot to fill in",
        all("cpu" not in stack.name.lower() for stack in STACKS),
        str([s.name for s in STACKS]),
    )
    check(
        "the Marlin note says Windows may simply have no wheel",
        "no wheel" in stack_by_name("gptq-marlin").note,
        "skip, not invention",
    )
    check(
        "ExLlamaV2 and vLLM are named stacks, still skeletons",
        stack_by_name("exllamav2-exl2").label.startswith("ExLlamaV2")
        and stack_by_name("vllm").label == "vLLM"
        and not stack_by_name("exllamav2-exl2").wired
        and not stack_by_name("vllm").wired,
        "named, empty",
    )
    from gpu.lab.competitor import _PROBES

    check(
        "every GPU stack has a child probe and none extra",
        set(_PROBES) == {stack.name for stack in COMPETITOR_STACKS},
        str(sorted(_PROBES)),
    )
    check(
        "isolated venv root is C:\\dev\\models\\venvs unless overridden",
        str(VENV_ROOT) == r"C:\dev\models\venvs",
        str(VENV_ROOT),
    )
    check(
        "each GPU stack names its own venv directory",
        all(venv_dir(stack).parent == VENV_ROOT and venv_dir(stack).name == stack.name for stack in COMPETITOR_STACKS),
        str(venv_dir("vllm")),
    )
    try:
        stack_by_name("llamacpp-cpu")
        check("an unknown stack raises", False, "no raise")
    except KeyError as error:
        check(
            "CPU llama.cpp is an unknown stack (KeyError), not a row",
            "llamacpp-q4" in str(error) and "llamacpp-cpu" in str(error),
            str(error)[:90],
        )


def gate_prompt_sets() -> None:
    check("the prompt sets are smoke and hard", PROMPT_SETS == ("smoke", "hard"), str(PROMPT_SETS))
    smoke = prompts_for("smoke")
    check("smoke is the three lab prompts", list(smoke) == list(MESSAGES), f"n={len(smoke)}")
    hard = prompts_for("hard")
    check("hard is the shared 12-item fixture", len(hard) == 12, f"n={len(hard)}")
    check(
        "the two sets are different lengths, so a mixed mean would be visible",
        len(smoke) != len(hard),
        f"{len(smoke)} vs {len(hard)}",
    )
    try:
        prompts_for("both")
        check("an unknown prompt set raises", False, "no raise")
    except ValueError as error:
        check("an unknown prompt set raises ValueError", "both" in str(error), str(error)[:60])


def gate_detect_missing() -> None:
    """On this box none of the GPU competitor stacks is installed, so all are skips."""
    lab = lab_by_slug("qwen25-3b")
    detections = detect_all(COMPETITOR_STACKS, model_dir=lab.model_dir, chr_path=lab.chr_path)
    check("one detection per ChatGPT GPU slot", len(detections) == 6, str(len(detections)))
    for detection in detections:
        if detection.runnable:
            check(
                f"{detection.stack} is installed here: it is allowed to be runnable",
                detection.install == "present",
                "a real stack is on this box",
            )
            continue
        check(
            f"{detection.stack} skip_reason is greppable",
            detection.skip_reason.startswith(SKIP),
            detection.skip_reason[:80],
        )
        check(
            f"{detection.stack} skip names what is missing",
            bool(detection.missing) or "artifact" in detection.skip_reason,
            str(detection.missing) or detection.skip_reason[:60],
        )
    check(
        "detection does not import the stack it is asking about",
        not any(name.startswith(("bitsandbytes", "awq", "llama_cpp")) for name in sys.modules),
        "spec lookup only",
    )


def gate_detect_artifact() -> None:
    """Modules present but no artifact is a different skip from 'not installed'."""
    lab = lab_by_slug("qwen25-3b")
    gptq = stack_by_name("gptq-marlin")
    with _env("DEEPFOLD_GPTQ", str(_REPO / "no-such-gptq-dir")):
        detection = detect(gptq, model_dir=lab.model_dir)
        check(
            "a bogus artifact path is a skip",
            not detection.runnable and detection.skip_reason.startswith(SKIP),
            detection.skip_reason[:90],
        )
    # A slot whose modules are all importable, so the artifact branch is reachable
    # without installing anything: torch/transformers stand in for the stack.
    from gpu.lab.competitor import Stack

    fake = Stack(
        name="bitsandbytes-nf4",  # reuse a known name so probes stay addressable
        label="stand-in",
        modules=("json", "csv"),
        artifact_env="DEEPFOLD_TEST_ARTIFACT",
        artifact_kind="gguf-file",
    )
    with tempfile.TemporaryDirectory(prefix="competitor-artifact-") as tmp:
        gguf = Path(tmp) / "model.Q4_K_M.gguf"
        with _env("DEEPFOLD_TEST_ARTIFACT", str(gguf)):
            missing = detect(fake)
            check(
                "modules present + no artifact says which env var to set",
                not missing.runnable and "DEEPFOLD_TEST_ARTIFACT" in missing.skip_reason,
                missing.skip_reason[:100],
            )
            check(
                "the artifact skip refuses the conversion shortcut",
                "never converted to .chr" in missing.skip_reason,
                "no GGUF -> CHR",
            )
            gguf.write_bytes(b"GGUF")
            present = detect(fake)
            check(
                "artifact on disk but an unwired slot is still a skip, not a number",
                not present.runnable and "no measuring body" in present.skip_reason,
                present.skip_reason[:90],
            )
            check(
                "that skip records the install as present, which is the useful fact",
                present.install == "present" and present.artifact == str(gguf),
                present.install,
            )
        wrong_kind = detect(
            Stack(**{**fake.__dict__, "artifact_kind": "quant-dir"}),
        )
        check(
            "a file where a directory is required is not an artifact",
            not wrong_kind.runnable,
            wrong_kind.skip_reason[:70],
        )


def gate_csv() -> None:
    lab = lab_by_slug("qwen25-3b")
    rows = [
        CompetitorRow.skip(stack, detect(stack, model_dir=lab.model_dir, chr_path=lab.chr_path))
        for stack in COMPETITOR_STACKS
    ]
    measured = CompetitorRow(
        stack="deepfold-nf4",
        install="present",
        mean_ttft_ms=167.0,
        mean_decode_tok_s=31.6,
        smi_after_mib=8913.0,
        smoke_ok=True,
        notes="illustrative row for the schema test, not a measurement",
    )
    with tempfile.TemporaryDirectory(prefix="competitor-csv-") as tmp:
        path = write_competitor_csv(Path(tmp) / "summary.csv", [*rows, measured])
        lines = path.read_text(encoding="utf-8").splitlines()
        check(
            "the header is exactly the TZ schema",
            lines[0] == ",".join(COMPETITOR_COLUMNS),
            lines[0],
        )
        check(
            "the e2e schema is still tok/s only (kernel metrics are a sibling CSV)",
            COMPETITOR_COLUMNS
            == (
                "stack",
                "install",
                "skip_reason",
                "mean_ttft_ms",
                "mean_decode_tok_s",
                "smi_after_mib",
                "smoke_ok",
                "notes",
            ),
            str(COMPETITOR_COLUMNS),
        )
        check(
            "the e2e schema has no us/occupancy columns to hide an ncu guess in",
            "us" not in COMPETITOR_COLUMNS and "occupancy" not in COMPETITOR_COLUMNS,
            str(COMPETITOR_COLUMNS),
        )
        back = read_competitor_csv(path)
        check("one row per attempted stack plus the schema example", len(back) == 7, str(len(back)))
        skipped = [row for row in back if row["skip_reason"]]
        check(
            "every skip row has empty ttft, tok/s and smi cells",
            all(
                row["mean_ttft_ms"] is None
                and row["mean_decode_tok_s"] is None
                and row["smi_after_mib"] is None
                for row in skipped
            ),
            f"{len(skipped)} skips, no numbers",
        )
        check(
            "every skip row has an empty smoke_ok, not a false pass",
            all(row["smoke_ok"] is None for row in skipped),
            "empty, not false",
        )
        check(
            "a reviewer can grep the CSV for the skip string",
            SKIP in path.read_text(encoding="utf-8"),
            SKIP,
        )
        numbers = [row for row in back if row["mean_decode_tok_s"] is not None]
        check(
            "only the row that carried numbers has them back",
            len(numbers) == 1 and numbers[0]["stack"] == "deepfold-nf4",
            str([row["stack"] for row in numbers]),
        )
        check(
            "a measured row round-trips its floats",
            numbers[0]["mean_decode_tok_s"] == 31.6 and numbers[0]["smoke_ok"] is True,
            str(numbers[0]["mean_decode_tok_s"]),
        )


def gate_microbench_schema() -> None:
    """Named ChatGPT kernel grid: shapes filled, metric cells empty until measured."""
    check(
        "M is the ChatGPT token list 1..256",
        MICROBENCH_M == (1, 2, 4, 8, 16, 32, 64, 128, 256),
        str(MICROBENCH_M),
    )
    check(
        "linears are Q/K/V/O/gate/up/down",
        MICROBENCH_LINEARS
        == ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        str(MICROBENCH_LINEARS),
    )
    check(
        "every linear has a (K, N) = (in, out) for Qwen2.5-3B",
        set(MICROBENCH_KN) == set(MICROBENCH_LINEARS),
        str(sorted(MICROBENCH_KN)),
    )
    bench = (_REPO / "gpu" / "nf4" / "bench.py").read_text(encoding="utf-8")
    for name, (k, n) in MICROBENCH_KN.items():
        needle = f'Shape("{name}", {n}, {k}'
        check(
            f"bench.py {name} is out={n} in={k} (ChatGPT [M,K]x[K,N])",
            needle in bench,
            needle,
        )
    check(
        "lm_head is not a ChatGPT competitor microbench cell",
        "lm_head" not in MICROBENCH_KN,
        "Q/K/V/O/gate/up/down only",
    )
    check(
        "kernel columns name us, bandwidth, TFLOP/s, occupancy, tensor, DRAM, regs, smem",
        MICROBENCH_COLUMNS
        == (
            "stack",
            "linear",
            "m",
            "k",
            "n",
            "us",
            "gb_s",
            "tflop_s",
            "occupancy",
            "tensor",
            "dram",
            "regs",
            "smem",
            "skip_reason",
            "notes",
        ),
        str(MICROBENCH_COLUMNS),
    )
    check(
        "the kernel schema has no tok/s column to paste e2e into",
        all("tok" not in name for name in MICROBENCH_COLUMNS),
        str(MICROBENCH_COLUMNS),
    )
    lab = lab_by_slug("qwen25-3b")
    e2e = [
        CompetitorRow.skip(stack, detect(stack, model_dir=lab.model_dir, chr_path=lab.chr_path))
        for stack in COMPETITOR_STACKS
    ]
    grid = microbench_grid_for(e2e)
    want = len(COMPETITOR_STACKS) * len(MICROBENCH_LINEARS) * len(MICROBENCH_M)
    check(
        "the named grid is stacks x 7 linears x 9 M",
        len(grid) == want == 6 * 7 * 9,
        f"{len(grid)} vs {want}",
    )
    check(
        "every kernel cell is a SKIP with empty metrics",
        all(
            cell.skipped
            and cell.skip_reason.startswith(SKIP)
            and not cell.measured
            and cell.us is None
            and cell.gb_s is None
            and cell.tflop_s is None
            and cell.occupancy is None
            and cell.tensor is None
            and cell.dram is None
            and cell.regs is None
            and cell.smem is None
            for cell in grid
        ),
        "empty metrics",
    )
    q1 = next(cell for cell in grid if cell.stack == "bitsandbytes-nf4" and cell.linear == "q_proj" and cell.m == 1)
    check(
        "q_proj M=1 is addressed [1,2048]x[2048,2048], not a guessed µs",
        (q1.k, q1.n) == (2048, 2048) and q1.us is None,
        f"k={q1.k} n={q1.n} us={q1.us}",
    )
    down = next(cell for cell in grid if cell.linear == "down_proj" and cell.m == 256)
    check(
        "down_proj is [M,11008]x[11008,2048]",
        (down.k, down.n) == (11008, 2048),
        f"k={down.k} n={down.n}",
    )
    measured = MicrobenchRow(
        stack="deepfold-nf4",
        linear="q_proj",
        m=1,
        k=2048,
        n=2048,
        us=12.5,
        gb_s=100.0,
        tflop_s=1.2,
        occupancy=15.7,
        tensor=1.4,
        dram=5.1,
        regs=55,
        smem=13056,
        notes="illustrative schema row, not a competitor measurement",
    )
    with tempfile.TemporaryDirectory(prefix="competitor-micro-") as tmp:
        path = write_microbench_csv(Path(tmp) / "microbench.csv", [*grid, measured])
        lines = path.read_text(encoding="utf-8").splitlines()
        check(
            "microbench header is exactly the named schema",
            lines[0] == ",".join(MICROBENCH_COLUMNS),
            lines[0],
        )
        back = read_microbench_csv(path)
        skipped = [row for row in back if row["skip_reason"]]
        check(
            "skip kernel rows round-trip empty metrics",
            skipped
            and all(row[name] is None for row in skipped for name in (
                "us", "gb_s", "tflop_s", "occupancy", "tensor", "dram", "regs", "smem"
            )),
            f"{len(skipped)} empty",
        )
        numbers = [row for row in back if row["us"] is not None]
        check(
            "only the row that carried µs has them back",
            len(numbers) == 1 and numbers[0]["stack"] == "deepfold-nf4",
            str([row["stack"] for row in numbers]),
        )
        check(
            "shape indices stay integers, not floats",
            numbers[0]["m"] == 1 and numbers[0]["k"] == 2048,
            str(numbers[0]["m"]),
        )
        try:
            MicrobenchRow.skip("bitsandbytes-nf4", "lm_head", 1, skip_reason=f"{SKIP} nope")
            check("unknown linear raises", False, "no raise")
        except KeyError as error:
            check("unknown linear raises KeyError", "lm_head" in str(error), str(error)[:70])


def gate_run_matrix_is_all_skips() -> None:
    """The whole matrix on a box with no competitor stack: six skips, no launches."""
    with tempfile.TemporaryDirectory(prefix="competitor-matrix-") as tmp:
        root = Path(tmp)
        rows = run_matrix(root, lab="qwen25-3b", verbose=False)
        check("a row per slot, none dropped", len(rows) == 6, str(len(rows)))
        check(
            "not one slot produced a number on this box",
            not any(row.measured for row in rows),
            str([row.stack for row in rows if row.measured]) or "no numbers",
        )
        check(
            "every row is a skip with a reason",
            all(row.skipped and row.skip_reason.startswith(SKIP) for row in rows),
            "all six explained",
        )
        summary = root / "summary.csv"
        check("the matrix wrote summary.csv", summary.is_file(), "summary.csv")
        micro = root / "microbench.csv"
        check("the matrix wrote microbench.csv", micro.is_file(), "microbench.csv")
        source = root / "SOURCE.txt"
        source_text = source.read_text(encoding="utf-8") if source.is_file() else ""
        check(
            "the matrix wrote SOURCE.txt that forbids invented tok/s and CPU fill-ins",
            source.is_file()
            and "Do not invent" in source_text
            and "CPU llama.cpp" in source_text,
            "SOURCE.txt",
        )
        text = summary.read_text(encoding="utf-8")
        check(
            "no digit-bearing speed cell was written",
            all(
                row["mean_decode_tok_s"] is None and row["mean_ttft_ms"] is None
                for row in read_competitor_csv(summary)
            ),
            "empty cells",
        )
        check(
            "the CSV names all GPU stacks so the matrix is complete",
            all(stack.name in text for stack in COMPETITOR_STACKS),
            "six attempted rows",
        )
        kernel = read_microbench_csv(micro)
        check(
            "microbench.csv is the full named grid",
            len(kernel) == 6 * 7 * 9,
            str(len(kernel)),
        )
        check(
            "no kernel metric was invented",
            all(row["us"] is None and row["occupancy"] is None for row in kernel),
            "empty kernel cells",
        )
        check(
            "no child process was spawned for an uninstalled stack",
            not any((root / stack.name / "worker.log").is_file() for stack in COMPETITOR_STACKS),
            "detect first, spawn second",
        )


def gate_worker_slot_isolation() -> None:
    """The child refuses to be a competitor process that also holds our kernel."""
    source = Path(__file__).with_name("competitor.py").read_text(encoding="utf-8")
    check(
        "the child guards against importing our kernel into a competitor slot",
        "_refuse_our_kernel" in source and "one stack per process" in source,
        "TZ 6.1",
    )
    check(
        "each slot is spawned as its own process",
        "subprocess.run" in source and "--worker" in source,
        "isolated slots",
    )
    check(
        "a slot can be pointed at its own interpreter",
        "--python" in source and "python_exe" in source,
        "isolated venv preferred",
    )
    check(
        "the harness source does not contain a pip install command",
        "pip install" not in source.replace("Point DEEPFOLD_GPTQ", ""),
        "detect and skip only",
    )
    check(
        "the harness points at isolated venvs, not torch-gpu",
        "torch-gpu" in source and r"C:\dev\models\venvs" in source,
        "isolated",
    )
    check(
        "the llama.cpp child probe refuses a CPU backend as a GPU row",
        "CPU tok/s is not a GPU competitor row" in source,
        "no CPU fill-in",
    )
    # A worker asked for a stack that is not installed must still write a JSON
    # skip rather than dying with a traceback the parent has to guess about.
    with tempfile.TemporaryDirectory(prefix="competitor-worker-") as tmp:
        dest = Path(tmp) / "awq"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "gpu.lab.competitor",
                "--worker",
                "--stack",
                "awq",
                "--out",
                str(dest),
                "--artifact",
                str(Path(tmp) / "nowhere"),
            ],
            cwd=str(_REPO),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        check(
            "the worker exits 0 even when its stack will not import",
            completed.returncode == 0,
            (completed.stderr or "")[-160:] or "exit 0",
        )
        payload_path = dest / "competitor.json"
        check("the worker wrote competitor.json", payload_path.is_file(), "competitor.json")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        check(
            "the payload is a skip with a reason",
            str(payload["skip_reason"]).startswith(SKIP),
            str(payload["skip_reason"])[:90],
        )
        check(
            "the payload's numeric cells are null, not zero",
            payload["mean_ttft_ms"] is None
            and payload["mean_decode_tok_s"] is None
            and payload["smoke_ok"] is None,
            "nulls",
        )


def gate_dead_child_is_a_skip() -> None:
    """A slot whose child writes no JSON is a skip that quotes what it printed."""
    from gpu.lab.competitor import _row_from_child

    stack = stack_by_name("llamacpp-q4")
    with tempfile.TemporaryDirectory(prefix="competitor-dead-") as tmp:
        row = _row_from_child(stack, Path(tmp), 3221225477, "ACCESS_VIOLATION in ggml-cuda")
        check(
            "a dead child is a skip, not a crash",
            row.skipped and row.skip_reason.startswith(SKIP),
            row.skip_reason[:90],
        )
        check(
            "the skip quotes the child's exit code and output",
            "3221225477" in row.skip_reason and "ACCESS_VIOLATION" in row.skip_reason,
            "diagnosable",
        )
        check("a dead child carries no numbers", not row.measured, "empty cells")


def gate_cli() -> None:
    from gpu.lab.competitor import main as competitor_main

    check("--list exits 0 with no GPU", competitor_main(["--list"]) == 0, "list")
    check("--detect exits 0 with no GPU", competitor_main(["--detect"]) == 0, "detect")
    check("--recipe exits 0 and does not install", competitor_main(["--recipe"]) == 0, "recipe")
    with tempfile.TemporaryDirectory(prefix="competitor-detect-out-") as tmp:
        dest = Path(tmp) / "summary.csv"
        rc = competitor_main(["--detect", "--out", str(dest)])
        check("--detect --out exits 0", rc == 0, "detect-out")
        check("--detect --out wrote the skip CSV", dest.is_file(), str(dest))
        back = read_competitor_csv(dest)
        check("detect-only names six GPU stacks", len(back) == 6, str(len(back)))
        check(
            "detect-only numeric cells stay empty",
            back
            and all(
                row["mean_decode_tok_s"] is None and row["mean_ttft_ms"] is None for row in back
            ),
            "empty cells",
        )
        check(
            "detect-only rows are greppable SKIP",
            all(str(row["skip_reason"]).startswith(SKIP) for row in back),
            "SKIP:",
        )
        micro = dest.with_name("microbench.csv")
        check("--detect --out wrote microbench.csv", micro.is_file(), str(micro))
        kernel = read_microbench_csv(micro)
        check(
            "detect-only kernel cells stay empty",
            kernel
            and len(kernel) == 6 * 7 * 9
            and all(row["us"] is None and row["occupancy"] is None for row in kernel),
            str(len(kernel)),
        )
        check(
            "detect-only kernel rows are greppable SKIP",
            all(str(row["skip_reason"]).startswith(SKIP) for row in kernel),
            "SKIP:",
        )
    check(
        "--detect on our own rows also works",
        competitor_main(["--detect", "--stacks", "ours"]) == 0,
        "ours",
    )
    from gpu.lab.competitor import _chosen

    check("no --stacks means the ChatGPT GPU stacks", _chosen("") == COMPETITOR_STACKS, "default")
    check("--stacks all is every slot", _chosen("all") == STACKS, str(len(STACKS)))
    check("--stacks ours is our two rows", _chosen("ours") == OUR_STACKS, "ours")
    check(
        "--stacks takes a comma list",
        [stack.name for stack in _chosen("awq,llamacpp-q4")] == ["awq", "llamacpp-q4"],
        "two slots",
    )
    from gpu.lab.competitor import _parser

    args = _parser().parse_args([])
    check("the default lab is the 3B", args.lab == "qwen25-3b", args.lab)
    check("the default prompt set is smoke", args.prompts == "smoke", args.prompts)
    check("--worker is hidden from the help", args.worker is False, "internal flag")


def gate_recipe_doc() -> None:
    path = recipe_path()
    check("the recipe lives under docs/", RECIPE_DOC == "docs/competitor-venvs.md", RECIPE_DOC)
    check("the recipe file exists", path.is_file(), str(path))
    text = path.read_text(encoding="utf-8")
    check(
        "the recipe forbids installing into torch-gpu",
        "torch-gpu" in text and "Do not" in text,
        "forbidden env",
    )
    check(
        "the recipe names C:\\dev\\models\\venvs",
        r"C:\dev\models\venvs" in text,
        "venv root",
    )
    check(
        "every GPU stack has a venv name in the recipe",
        all(stack.name in text for stack in COMPETITOR_STACKS),
        "six dirs",
    )
    check(
        "the recipe refuses a CPU llama.cpp fill-in",
        "CPU llama.cpp" in text,
        "no CPU tok/s",
    )
    check(
        "the recipe says wait until the GPU is free",
        "GPU is **free**" in text or "GPU is free" in text,
        "later",
    )


def gate_run_stack_never_raises() -> None:
    lab = lab_by_slug("qwen25-3b")
    with tempfile.TemporaryDirectory(prefix="competitor-one-") as tmp:
        row = run_stack("bitsandbytes-nf4", Path(tmp), lab=lab, verbose=False)
        check(
            "one slot on its own is still a skip row, not an exception",
            isinstance(row, CompetitorRow) and row.skipped,
            row.skip_reason[:80],
        )
        try:
            run_stack("bitsandbytes-nf4", Path(tmp), lab=lab, prompt_set="mixed", verbose=False)
            check("a bad prompt set raises", False, "no raise")
        except ValueError as error:
            check("a bad prompt set raises ValueError", "mixed" in str(error), str(error)[:60])


def gate_detection_dataclass() -> None:
    check(
        "an empty skip_reason is what makes a detection runnable",
        Detection(stack="x", install="present").runnable
        and not Detection(stack="x", install="present", skip_reason=f"{SKIP} nope").runnable,
        "runnable == no skip",
    )
    row = CompetitorRow(stack="x", install="present")
    check(
        "a row with no tok/s is not 'measured' even when nothing was skipped",
        not row.measured and not row.skipped,
        "measured needs a number",
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
    print("gpu/lab competitor harness (detect + skip, no GPU, no installs)\n")
    for name, body in (
        ("gate_slots", gate_slots),
        ("gate_prompt_sets", gate_prompt_sets),
        ("gate_detect_missing", gate_detect_missing),
        ("gate_detect_artifact", gate_detect_artifact),
        ("gate_csv", gate_csv),
        ("gate_microbench_schema", gate_microbench_schema),
        ("gate_run_matrix_is_all_skips", gate_run_matrix_is_all_skips),
        ("gate_worker_slot_isolation", gate_worker_slot_isolation),
        ("gate_dead_child_is_a_skip", gate_dead_child_is_a_skip),
        ("gate_cli", gate_cli),
        ("gate_recipe_doc", gate_recipe_doc),
        ("gate_run_stack_never_raises", gate_run_stack_never_raises),
        ("gate_detection_dataclass", gate_detection_dataclass),
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

"""K3: the 4-bit competitor harness. One isolated slot per stack, skip or measure.

The ChatGPT comparison on this 3080 is named here: end-to-end tok/s **and** a
linear microbench ``[M,K] x [K,N]`` for ``M = 1,2,4,8,16,32,64,128,256`` on
Q/K/V/O/gate/up/down, with us / GB/s / TFLOP/s / occupancy / tensor / DRAM /
regs / smem. Until a stack lives in its **own** venv and a child actually
runs on a free GPU, every numeric cell is empty and ``skip_reason`` starts
with ``SKIP:``. Grep the CSV for ``SKIP`` -- a skip row is the result.

There is **no** bitsandbytes / GPTQ-Marlin / AWQ / ExLlamaV2 / llama.cpp /
vLLM number invented here. ``gpu/nf4/bench.py`` times *our* kernel only;
``docs/runs/ncu/`` is *our* occupancy, not a competitor. CPU llama.cpp tok/s
is not a GPU row and is not a slot.

This module does two things and refuses the third:

1. **Detect.** :func:`detect` asks whether a stack's Python modules are
   importable and whether its **native** 4-bit artifact is on disk. Nothing is
   installed, nothing is downloaded, nothing is converted.
2. **Run, isolated.** :func:`run_stack` spawns one child process per stack, then
   exits it. WDDM does not hand VRAM back if two 4-bit runtimes share an
   interpreter, and a broken ``bitsandbytes`` must not be able to break
   ``chr_nf4_ext``. ``--python`` points a slot at its own venv under
   ``C:\\dev\\models\\venvs\\<stack>`` (see ``docs/competitor-venvs.md``).
   Never the live ``torch-gpu`` env.
3. **It never invents a number.** A stack that is not installed, has no
   artifact, links only a CPU backend, or whose slot body is still a skeleton
   writes empty numeric cells. The named microbench grid is the same: the
   ``[M,K,N]`` address is filled (Qwen2.5-3B shapes), the metric cells are not.

Same card, same model family, same prompts as our own row: Qwen2.5-3B-Instruct
first, because both a dense and a packed copy fit in 12 GB. Each stack uses its
own native 4-bit artifact (bnb ``Linear4bit`` over the HF BF16 tree, a GPTQ
checkpoint, an AWQ dump, an EXL2 dump, a GGUF Q4 with a CUDA-linked llama.cpp,
a vLLM engine). GGUF is never dequantised into a fake BF16 tree and re-compressed.

    python -m gpu.lab.competitor --detect
    python -m gpu.lab.competitor --recipe
    python -m gpu.lab.competitor --lab qwen25-3b            # attempt every GPU stack
    python -m gpu.lab.competitor --stacks deepfold-nf4      # our row, real worker
    python -m gpu.lab.test_competitor
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .catalog import LabModel, lab_by_slug
from .script import MESSAGES, RUNS_DIR

__all__ = [
    "COMPETITOR_COLUMNS",
    "COMPETITOR_STACKS",
    "MICROBENCH_COLUMNS",
    "MICROBENCH_KN",
    "MICROBENCH_LINEARS",
    "MICROBENCH_M",
    "OUR_STACKS",
    "PROMPT_SETS",
    "RECIPE_DOC",
    "SKIP",
    "STACKS",
    "VENV_ROOT",
    "CompetitorRow",
    "Detection",
    "MicrobenchRow",
    "Stack",
    "detect",
    "detect_all",
    "microbench_grid_for",
    "prompts_for",
    "read_competitor_csv",
    "read_microbench_csv",
    "recipe_path",
    "run_matrix",
    "run_stack",
    "stack_by_name",
    "venv_dir",
    "write_competitor_csv",
    "write_matrix_csvs",
    "write_microbench_csv",
]

_REPO = Path(__file__).resolve().parents[2]

#: Every skip_reason starts with this, so a reviewer can grep one string.
SKIP = "SKIP:"

#: TZ wave10-perf §6.4. Empty numeric cells for skips; no extra columns.
#: End-to-end tok/s lives here. Kernel µs / occupancy live in microbench.csv
#: so a tok/s cell cannot hide an ncu guess.
COMPETITOR_COLUMNS = (
    "stack",
    "install",
    "skip_reason",
    "mean_ttft_ms",
    "mean_decode_tok_s",
    "smi_after_mib",
    "smoke_ok",
    "notes",
)

#: Isolated venvs, one stack each. Never the live ``torch-gpu`` interpreter.
#: The recipe in ``docs/competitor-venvs.md`` creates these later; this harness
#: does not.
VENV_ROOT = Path(os.environ.get("DEEPFOLD_VENVS", r"C:\dev\models\venvs"))
RECIPE_DOC = "docs/competitor-venvs.md"

#: ChatGPT linear microbench: C[M, N] = A[M, K] @ B[K, N]. ``M`` is tokens
#: (decode 1, prefill 2..256). This is **not** ``gpu.nf4.bench.Shape.M``, which
#: is out_features (this table's ``n``). Mixing the two M's would invent a shape.
MICROBENCH_M: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
MICROBENCH_LINEARS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
#: Qwen2.5-3B (K, N) = (in_features, out_features). Mirrors gpu/nf4/bench.py
#: QWEN25_3B except lm_head, which ChatGPT did not ask to time here.
MICROBENCH_KN: dict[str, tuple[int, int]] = {
    "q_proj": (2048, 2048),
    "k_proj": (2048, 256),
    "v_proj": (2048, 256),
    "o_proj": (2048, 2048),
    "gate_proj": (2048, 11008),
    "up_proj": (2048, 11008),
    "down_proj": (11008, 2048),
}
MICROBENCH_COLUMNS = (
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
)
MICROBENCH_NUMERIC = (
    "us",
    "gb_s",
    "tflop_s",
    "occupancy",
    "tensor",
    "dram",
    "regs",
    "smem",
)

#: Which prompt list a run used. The two are never averaged into one mean, so
#: they never share a CSV: the run directory name carries the set.
PROMPT_SETS = ("smoke", "hard")


@dataclass(frozen=True)
class Stack:
    """One runner slot: what to import, what artifact it needs, what it is."""

    name: str
    label: str
    #: Every one of these must be importable.
    modules: tuple[str, ...] = ()
    #: At least one of these must be importable (a stack with rival packages).
    any_of: tuple[str, ...] = ()
    #: Env var holding that stack's **native** 4-bit artifact.
    artifact_env: str = ""
    #: ``hf-dir`` | ``quant-dir`` | ``gguf-file`` | ``chr-file``
    artifact_kind: str = ""
    #: True for our own rows (NF4 driver, dense BF16 reference).
    ours: bool = False
    #: Whether this slot has a measuring body yet.
    wired: bool = False
    note: str = ""

    @property
    def needs_artifact(self) -> bool:
        return bool(self.artifact_kind)

    @property
    def venv_dir(self) -> Path:
        """Where this slot's isolated interpreter will live, when Pavel creates it."""
        return VENV_ROOT / self.name


def venv_dir(stack: Stack | str) -> Path:
    """``C:\\dev\\models\\venvs\\<stack>``. Created later; not by this harness."""
    if isinstance(stack, str):
        stack = stack_by_name(stack)
    return stack.venv_dir


def recipe_path() -> Path:
    """Isolated-venv recipe. Commands to run later; this file is not executed."""
    return _REPO / RECIPE_DOC


#: GPU stacks ChatGPT asked to name, in attempt order. No CPU llama.cpp slot.
COMPETITOR_STACKS: tuple[Stack, ...] = (
    Stack(
        name="bitsandbytes-nf4",
        label="bitsandbytes NF4",
        modules=("bitsandbytes", "transformers", "torch"),
        artifact_env="DEEPFOLD_MODEL",
        artifact_kind="hf-dir",
        note=(
            "Linear4bit over the HF BF16 tree, quant_type='nf4'. Isolated venv "
            r"C:\dev\models\venvs\bitsandbytes-nf4 -- never torch-gpu. Windows "
            "wheels exist for some versions; a cu124 mismatch is a skip. Its CUDA "
            "nibble order is not assumed to match ours (docs/spec/nf4.md §0)."
        ),
    ),
    Stack(
        name="gptq-marlin",
        label="GPTQ + Marlin",
        modules=("transformers", "torch"),
        any_of=("gptqmodel", "auto_gptq"),
        artifact_env="DEEPFOLD_GPTQ",
        artifact_kind="quant-dir",
        note=(
            "Needs a GPTQ 4-bit checkpoint of the same model AND a Marlin (or "
            "GPTQ-Marlin) GPU gemm that actually launches. Isolated venv "
            r"C:\dev\models\venvs\gptq-marlin. Marlin is typically Linux CUDA; "
            "Windows + cu124 often has no wheel. That is a skip, never a "
            "borrowed tok/s."
        ),
    ),
    Stack(
        name="awq",
        label="AWQ (W4A16)",
        modules=("transformers", "torch"),
        any_of=("awq", "autoawq"),
        artifact_env="DEEPFOLD_AWQ",
        artifact_kind="quant-dir",
        note=(
            "AutoAWQ or equivalent W4A16 GPU kernel over an AWQ dump of the same "
            r"model. Isolated venv C:\dev\models\venvs\awq -- never torch-gpu."
        ),
    ),
    Stack(
        name="llamacpp-q4",
        label="llama.cpp CUDA Q4 (GGUF)",
        any_of=("llama_cpp",),
        artifact_env="DEEPFOLD_GGUF",
        artifact_kind="gguf-file",
        note=(
            "Needs a CUDA-linked build in C:\\dev\\models\\venvs\\llamacpp-q4. "
            "If llama_supports_gpu_offload() is False the row is a skip: CPU tok/s "
            "is not a GPU competitor. Record Q4_K_M vs Q4_0. A CPU llama.cpp "
            "binary is not a fill-in for this row."
        ),
    ),
    Stack(
        name="exllamav2-exl2",
        label="ExLlamaV2 (EXL2)",
        modules=("torch",),
        any_of=("exllamav2",),
        artifact_env="DEEPFOLD_EXL2",
        artifact_kind="quant-dir",
        note=(
            "Needs an EXL2 dump of the same model AND a CUDA-capable exllamav2 in "
            r"C:\dev\models\venvs\exllamav2-exl2. Never torch-gpu. A CPU fallback "
            "is not a GPU competitor. Missing kernel or missing dump is a skip, "
            "never a borrowed tok/s."
        ),
    ),
    Stack(
        name="vllm",
        label="vLLM",
        modules=("vllm", "torch"),
        artifact_env="DEEPFOLD_VLLM",
        artifact_kind="hf-dir",
        note=(
            "Serving engine that may load AWQ/GPTQ/Marlin of the same model. "
            r"Isolated venv C:\dev\models\venvs\vllm. Typical wheels are Linux "
            "CUDA; Windows + cu124 often has no wheel -- that is a skip. A "
            "CPU-only engine is not a GPU competitor. Point DEEPFOLD_VLLM at the "
            "HF tree or a dump vLLM can load."
        ),
    ),
)

#: Our side of the table. These slots are wired: they reuse the lab worker, so
#: the numbers come from a real session, never from ``gpu/nf4/bench.py``.
OUR_STACKS: tuple[Stack, ...] = (
    Stack(
        name="deepfold-nf4",
        label="deepfold NF4 (ours)",
        modules=("torch", "transformers"),
        artifact_env="DEEPFOLD_CHR",
        artifact_kind="chr-file",
        ours=True,
        wired=True,
        note=(
            "CompressedLinear + TokenLoop through gpu.lab.sessions.run_nf4, isolated. "
            "This is the row to cite for us; bench.py microseconds are not."
        ),
    ),
    Stack(
        name="hf-bf16",
        label="HuggingFace BF16 (dense reference)",
        modules=("torch", "transformers"),
        artifact_env="DEEPFOLD_MODEL",
        artifact_kind="hf-dir",
        ours=True,
        wired=True,
        note=(
            "Dense reference through gpu.lab.sessions.run_bf16. Re-run this in the "
            "same session before claiming any TTFT ranking; the committed 52 ms is "
            "an older plate."
        ),
    ),
)

STACKS: tuple[Stack, ...] = COMPETITOR_STACKS + OUR_STACKS


def stack_by_name(name: str) -> Stack:
    """Slot by name. Unknown name fails closed."""
    for stack in STACKS:
        if stack.name == name:
            return stack
    known = ", ".join(stack.name for stack in STACKS)
    raise KeyError(f"unknown stack {name!r}; known: {known}")


def prompts_for(prompt_set: str) -> tuple[str, ...]:
    """The prompt list, smoke or the shared hard fixture. Never both in one mean."""
    if prompt_set == "smoke":
        return tuple(MESSAGES)
    if prompt_set == "hard":
        from .hard import load_fixture

        return tuple(item.prompt for item in load_fixture()["independent"])
    raise ValueError(f"prompt_set={prompt_set!r}; expected one of {', '.join(PROMPT_SETS)}")


# --------------------------------------------------------------------------- #
# detection: importable, and is the artifact on disk
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Detection:
    """What a slot looks like on this box, before anything is launched."""

    stack: str
    install: str  # "present" | "missing" | "unknown"
    missing: tuple[str, ...] = ()
    artifact: str = ""
    skip_reason: str = ""

    @property
    def runnable(self) -> bool:
        return not self.skip_reason


def _importable(name: str) -> bool:
    """Is the module findable? A spec lookup, so a broken CUDA stack cannot
    import itself into this process just by being detected."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _artifact_path(stack: Stack, *, model_dir: str = "", chr_path: str = "") -> str:
    """``DEEPFOLD_*`` wins; otherwise the catalog path for the lab being run."""
    override = os.environ.get(stack.artifact_env, "").strip() if stack.artifact_env else ""
    if override:
        return override
    if stack.artifact_kind == "hf-dir":
        return model_dir
    if stack.artifact_kind == "chr-file":
        return chr_path
    return ""


def _artifact_ok(kind: str, path: str) -> bool:
    if not path:
        return False
    target = Path(path)
    return target.is_dir() if kind.endswith("-dir") else target.is_file()


def detect(stack: Stack | str, *, model_dir: str = "", chr_path: str = "") -> Detection:
    """Modules + artifact, without importing the stack and without a GPU.

    The deeper questions -- does the CUDA extension load, does llama.cpp have
    GPU offload linked in, does the kernel launch -- are answered **inside** the
    child, because asking them here would import a foreign CUDA runtime into the
    process that is about to spawn ours.
    """
    if isinstance(stack, str):
        stack = stack_by_name(stack)

    missing = [name for name in stack.modules if not _importable(name)]
    if stack.any_of and not any(_importable(name) for name in stack.any_of):
        missing.append(" or ".join(stack.any_of))
    if missing:
        return Detection(
            stack=stack.name,
            install="missing",
            missing=tuple(missing),
            skip_reason=(
                f"{SKIP} {stack.label}: not importable in this interpreter "
                f"({', '.join(missing)}). Nothing installed from this harness; "
                "no borrowed tok/s."
            ),
        )

    artifact = _artifact_path(stack, model_dir=model_dir, chr_path=chr_path)
    if stack.needs_artifact and not _artifact_ok(stack.artifact_kind, artifact):
        where = artifact or f"unset {stack.artifact_env}"
        return Detection(
            stack=stack.name,
            install="present",
            artifact=artifact,
            skip_reason=(
                f"{SKIP} {stack.label}: no native 4-bit artifact ({stack.artifact_kind}) "
                f"at {where}. Point {stack.artifact_env} at one. GGUF/GPTQ are never "
                "converted to .chr to make a row appear."
            ),
        )

    if not stack.wired:
        return Detection(
            stack=stack.name,
            install="present",
            artifact=artifact,
            skip_reason=(
                f"{SKIP} {stack.label}: installed and the artifact is on disk, but this "
                "runner slot has no measuring body yet (K3 skeleton). Empty cells, not "
                "a guessed tok/s."
            ),
        )

    return Detection(stack=stack.name, install="present", artifact=artifact)


def detect_all(
    stacks: Sequence[Stack] = COMPETITOR_STACKS,
    *,
    model_dir: str = "",
    chr_path: str = "",
) -> list[Detection]:
    return [detect(stack, model_dir=model_dir, chr_path=chr_path) for stack in stacks]


# --------------------------------------------------------------------------- #
# rows
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CompetitorRow:
    """One line of ``summary.csv``. Numeric cells stay ``None`` unless measured."""

    stack: str
    install: str
    skip_reason: str = ""
    mean_ttft_ms: float | None = None
    mean_decode_tok_s: float | None = None
    smi_after_mib: float | None = None
    smoke_ok: bool | None = None
    notes: str = ""
    out_dir: Path | None = None

    @property
    def skipped(self) -> bool:
        return bool(self.skip_reason)

    @property
    def measured(self) -> bool:
        return self.mean_decode_tok_s is not None

    def as_row(self) -> dict[str, Any]:
        return {
            "stack": self.stack,
            "install": self.install,
            "skip_reason": self.skip_reason,
            "mean_ttft_ms": self.mean_ttft_ms,
            "mean_decode_tok_s": self.mean_decode_tok_s,
            "smi_after_mib": self.smi_after_mib,
            "smoke_ok": self.smoke_ok,
            "notes": self.notes,
        }

    @classmethod
    def skip(cls, stack: Stack, detection: Detection, *, extra: str = "") -> "CompetitorRow":
        note = stack.note if not extra else f"{stack.note} {extra}"
        return cls(
            stack=stack.name,
            install=detection.install,
            skip_reason=detection.skip_reason,
            notes=note,
        )


@dataclass(frozen=True)
class MicrobenchRow:
    """One named ``[M,K] x [K,N]`` cell. Metric cells stay ``None`` unless measured."""

    stack: str
    linear: str
    m: int
    k: int
    n: int
    us: float | None = None
    gb_s: float | None = None
    tflop_s: float | None = None
    occupancy: float | None = None
    tensor: float | None = None
    dram: float | None = None
    regs: float | None = None
    smem: float | None = None
    skip_reason: str = ""
    notes: str = ""

    @property
    def skipped(self) -> bool:
        return bool(self.skip_reason)

    @property
    def measured(self) -> bool:
        return self.us is not None

    def as_row(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in MICROBENCH_COLUMNS}

    @classmethod
    def skip(
        cls,
        stack: Stack | str,
        linear: str,
        m: int,
        *,
        skip_reason: str,
        notes: str = "",
    ) -> "MicrobenchRow":
        name = stack.name if isinstance(stack, Stack) else stack
        try:
            k, n = MICROBENCH_KN[linear]
        except KeyError as error:
            known = ", ".join(MICROBENCH_LINEARS)
            raise KeyError(f"unknown linear {linear!r}; known: {known}") from error
        return cls(
            stack=name,
            linear=linear,
            m=m,
            k=k,
            n=n,
            skip_reason=skip_reason,
            notes=notes,
        )


def _microbench_skip_reason(stack: Stack, row: CompetitorRow) -> str:
    """E2e skip wins; otherwise the kernel body is still empty. Never copy tok/s."""
    if row.skip_reason:
        return row.skip_reason
    return (
        f"{SKIP} {stack.label}: linear microbench body is not wired. Named "
        "[M,K]x[K,N] cell; empty us/GB/s/TFLOP/s/occupancy/tensor/DRAM/regs/smem. "
        "e2e tok/s is not a kernel µs. CPU llama.cpp is not a fill-in."
    )


def microbench_grid_for(rows: Sequence[CompetitorRow]) -> list[MicrobenchRow]:
    """Every ChatGPT kernel cell for the attempted stacks. Metrics empty on skip."""
    grid: list[MicrobenchRow] = []
    for row in rows:
        try:
            stack = stack_by_name(row.stack)
        except KeyError:
            continue
        reason = _microbench_skip_reason(stack, row)
        note = (
            "Qwen2.5-3B GEMM C[M,N]=A[M,K]@B[K,N]. occupancy/tensor/DRAM/regs/smem "
            "need ncu on a free 3080. Not gpu.nf4.bench. CPU llama.cpp is not a fill-in."
        )
        for linear in MICROBENCH_LINEARS:
            for m in MICROBENCH_M:
                grid.append(
                    MicrobenchRow.skip(
                        stack,
                        linear,
                        m,
                        skip_reason=reason,
                        notes=note,
                    )
                )
    return grid


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_competitor_csv(path: str | Path, rows: Sequence[CompetitorRow]) -> Path:
    """One row per attempted stack, skips included. Empty numeric cells for skips."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COMPETITOR_COLUMNS)
        for row in rows:
            values = row.as_row()
            writer.writerow([_cell(values[name]) for name in COMPETITOR_COLUMNS])
    return dest


def read_competitor_csv(path: str | Path) -> list[dict[str, Any]]:
    """``summary.csv`` back as dicts, so a reader with no GPU can check the skips."""
    numeric = ("mean_ttft_ms", "mean_decode_tok_s", "smi_after_mib")
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, Any]] = []
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for name in numeric:
                text = str(row.get(name) or "").strip()
                row[name] = float(text) if text else None
            smoke = str(row.get("smoke_ok") or "").strip().lower()
            row["smoke_ok"] = None if smoke == "" else smoke == "true"
            rows.append(row)
        return rows


def write_microbench_csv(path: str | Path, rows: Sequence[MicrobenchRow]) -> Path:
    """Named ChatGPT kernel grid. Metric cells empty on skip; m/k/n are addresses."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MICROBENCH_COLUMNS)
        for row in rows:
            values = row.as_row()
            writer.writerow([_cell(values[name]) for name in MICROBENCH_COLUMNS])
    return dest


def read_microbench_csv(path: str | Path) -> list[dict[str, Any]]:
    """``microbench.csv`` back as dicts. Shape indices stay ints; metrics may be None."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, Any]] = []
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for name in ("m", "k", "n"):
                text = str(row.get(name) or "").strip()
                row[name] = int(text) if text else None
            for name in MICROBENCH_NUMERIC:
                text = str(row.get(name) or "").strip()
                row[name] = float(text) if text else None
            rows.append(row)
        return rows


def write_matrix_csvs(root: str | Path, rows: Sequence[CompetitorRow]) -> tuple[Path, Path]:
    """``summary.csv`` (e2e) plus ``microbench.csv`` (named kernel cells)."""
    dest = Path(root)
    dest.mkdir(parents=True, exist_ok=True)
    summary = write_competitor_csv(dest / "summary.csv", rows)
    micro = write_microbench_csv(dest / "microbench.csv", microbench_grid_for(rows))
    return summary, micro


def _matrix_source(kind: str) -> str:
    return (
        f"{kind}\n"
        "ChatGPT GPU matrix: bitsandbytes NF4, GPTQ+Marlin, AWQ, llama.cpp CUDA Q4, "
        "ExLlamaV2/EXL2, vLLM. End-to-end tok/s in summary.csv; linear microbench "
        "[M,K]x[K,N] for M=1,2,4,8,16,32,64,128,256 on Q/K/V/O/gate/up/down in "
        "microbench.csv (us, GB/s, TFLOP/s, occupancy, tensor, DRAM, regs, smem).\n"
        "Numeric metric cells are empty on purpose when skipped. A SKIP row is the "
        "result. Do not invent tok/s or kernel µs.\n"
        "CPU llama.cpp tok/s is not a GPU competitor and is not a fill-in.\n"
        "This harness installs nothing, does not convert GGUF to .chr, and never "
        "writes into the live torch-gpu env. Isolated venvs: docs/competitor-venvs.md.\n"
    )


# --------------------------------------------------------------------------- #
# the isolated slots
# --------------------------------------------------------------------------- #


def _spawn_slot(
    stack: Stack,
    dest: Path,
    *,
    artifact: str,
    prompt_set: str,
    max_new_tokens: int,
    python_exe: str,
    verbose: bool,
) -> tuple[int, str]:
    """One stack, one process, then gone. Returns ``(returncode, output tail)``."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        "-m",
        "gpu.lab.competitor",
        "--worker",
        "--stack",
        stack.name,
        "--out",
        str(dest),
        "--artifact",
        artifact,
        "--prompts",
        prompt_set,
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    if verbose:
        print(f"[isolated {stack.name}] {' '.join(cmd)}", flush=True)
    completed = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if verbose and completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if verbose and completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", flush=True)
    try:
        (dest / "worker.log").write_text(
            f"$ {' '.join(cmd)}\nreturncode={completed.returncode}\n\n"
            f"--- stdout ---\n{completed.stdout or ''}\n--- stderr ---\n{completed.stderr or ''}",
            encoding="utf-8",
        )
    except OSError:
        pass
    # Windows keeps nvidia-smi high until the process is fully gone.
    time.sleep(1.5)
    tail = (completed.stderr or completed.stdout or "").strip()
    return completed.returncode, tail[-2000:]


def _row_from_child(stack: Stack, dest: Path, returncode: int, tail: str) -> CompetitorRow:
    """The child's ``competitor.json``, or a skip that quotes what it printed."""
    payload_path = dest / "competitor.json"
    if not payload_path.is_file():
        extra = f" Child said: {tail}" if tail else ""
        return CompetitorRow(
            stack=stack.name,
            install="unknown",
            skip_reason=(
                f"{SKIP} {stack.label}: the isolated slot exited {returncode} and wrote "
                f"no competitor.json.{extra}"
            ),
            notes=stack.note,
        )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    return CompetitorRow(
        stack=stack.name,
        install=str(payload.get("install") or "unknown"),
        skip_reason=str(payload.get("skip_reason") or ""),
        mean_ttft_ms=payload.get("mean_ttft_ms"),
        mean_decode_tok_s=payload.get("mean_decode_tok_s"),
        smi_after_mib=payload.get("smi_after_mib"),
        smoke_ok=payload.get("smoke_ok"),
        notes=str(payload.get("notes") or stack.note),
        out_dir=dest,
    )


def _run_ours(
    stack: Stack,
    dest: Path,
    *,
    lab: LabModel,
    prompt_set: str,
    max_new_tokens: int,
    graphs: bool,
    verbose: bool,
) -> CompetitorRow:
    """Our two rows go through the existing isolated lab worker, not a new path."""
    from .sessions import run_bf16, run_nf4

    common = dict(
        out_dir=dest,
        model_dir=lab.model_dir,
        messages=prompts_for(prompt_set),
        max_new_tokens=max_new_tokens,
        trust_remote_code=lab.trust_remote_code,
        isolated=True,
        verbose=verbose,
    )
    try:
        if stack.name == "hf-bf16":
            session = run_bf16(**common)
        else:
            from .hard import hard_max_seq

            session = run_nf4(
                chr_path=lab.chr_path, max_seq=hard_max_seq(lab), graphs=graphs, **common
            )
    except Exception as exc:  # noqa: BLE001 -- a dead worker is a skip, not a traceback
        return CompetitorRow(
            stack=stack.name,
            install="present",
            skip_reason=f"{SKIP} {stack.label}: {type(exc).__name__}: {exc}",
            notes=stack.note,
            out_dir=dest,
        )

    summary = session.summary or {}
    measured = int(summary.get("n_messages") or 0) > 0
    return CompetitorRow(
        stack=stack.name,
        install="present",
        skip_reason=(
            ""
            if measured
            else f"{SKIP} {stack.label}: the worker wrote no replies. "
            f"{summary.get('notes') or ''}"
        ),
        mean_ttft_ms=summary.get("mean_ttft_ms") if measured else None,
        mean_decode_tok_s=summary.get("mean_decode_tok_s") if measured else None,
        smi_after_mib=summary.get("vram_after_load_smi_mib"),
        smoke_ok=bool(summary.get("quality_all_ok")) if measured else None,
        notes=f"{stack.note} prompts={prompt_set}. {summary.get('notes') or ''}".strip(),
        out_dir=dest,
    )


def run_stack(
    stack: Stack | str,
    out_root: str | Path,
    *,
    lab: LabModel | str = "qwen25-3b",
    prompt_set: str = "smoke",
    max_new_tokens: int = 64,
    graphs: bool = True,
    python_exe: str | None = None,
    verbose: bool = True,
) -> CompetitorRow:
    """Attempt one stack. Returns a measured row or a ``SKIP:`` row. Never raises.

    ``python_exe`` runs the slot under another interpreter -- the preferred shape,
    since a broken ``bitsandbytes`` in ``torch-gpu`` would take ``chr_nf4_ext``
    with it. When it is given, detection happens in that child (this interpreter
    cannot answer for another venv's packages).
    """
    if isinstance(stack, str):
        stack = stack_by_name(stack)
    if isinstance(lab, str):
        lab = lab_by_slug(lab)
    if prompt_set not in PROMPT_SETS:
        raise ValueError(f"prompt_set={prompt_set!r}; expected one of {', '.join(PROMPT_SETS)}")

    dest = Path(out_root) / stack.name
    foreign = python_exe is not None and python_exe != sys.executable
    artifact = _artifact_path(stack, model_dir=lab.model_dir, chr_path=lab.chr_path)

    if not foreign:
        detection = detect(stack, model_dir=lab.model_dir, chr_path=lab.chr_path)
        if not detection.runnable:
            if verbose:
                print(f"[{stack.name}] {detection.skip_reason}", flush=True)
            dest.mkdir(parents=True, exist_ok=True)
            return CompetitorRow.skip(stack, detection)
        artifact = detection.artifact

    if stack.ours:
        return _run_ours(
            stack,
            dest,
            lab=lab,
            prompt_set=prompt_set,
            max_new_tokens=max_new_tokens,
            graphs=graphs,
            verbose=verbose,
        )

    returncode, tail = _spawn_slot(
        stack,
        dest,
        artifact=artifact,
        prompt_set=prompt_set,
        max_new_tokens=max_new_tokens,
        python_exe=python_exe or sys.executable,
        verbose=verbose,
    )
    return _row_from_child(stack, dest, returncode, tail)


def run_matrix(
    out_root: str | Path,
    *,
    stacks: Sequence[Stack] = COMPETITOR_STACKS,
    lab: LabModel | str = "qwen25-3b",
    prompt_set: str = "smoke",
    max_new_tokens: int = 64,
    graphs: bool = True,
    python_exe: str | None = None,
    verbose: bool = True,
) -> list[CompetitorRow]:
    """Every slot in turn, one process at a time, rewriting ``summary.csv`` as it goes.

    One stack on the 3080 at a time. Nothing here waits for K1: a skip matrix
    with no tok/s in it is still a valid K3 deliverable.
    """
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[CompetitorRow] = []
    for index, stack in enumerate(stacks, start=1):
        if verbose:
            print(f"\n=== slot {index}/{len(stacks)}  {stack.label} ===", flush=True)
        rows.append(
            run_stack(
                stack,
                root,
                lab=lab,
                prompt_set=prompt_set,
                max_new_tokens=max_new_tokens,
                graphs=graphs,
                python_exe=python_exe,
                verbose=verbose,
            )
        )
        write_matrix_csvs(root, rows)
    (root / "SOURCE.txt").write_text(
        _matrix_source(
            "WAVE 10 K3 isolated competitor matrix (run_stack, not --detect). "
            "Each slot was attempted."
        ),
        encoding="utf-8",
    )
    return rows


# --------------------------------------------------------------------------- #
# the child: probe for real, then say what is missing
# --------------------------------------------------------------------------- #


@dataclass
class _Probe:
    """What a child learned about its stack before it would have measured anything."""

    install: str = "present"
    skip_reason: str = ""
    notes: list[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def payload(self, stack: Stack) -> dict[str, Any]:
        return {
            "stack": stack.name,
            "install": self.install,
            "skip_reason": self.skip_reason,
            "mean_ttft_ms": None,
            "mean_decode_tok_s": None,
            "smi_after_mib": None,
            "smoke_ok": None,
            "notes": "; ".join([stack.note, *self.notes]),
        }


def _refuse_our_kernel() -> None:
    """A competitor process must not have our extension in it (TZ §6.1)."""
    ours = [name for name in sys.modules if name.startswith(("gpu.nf4", "chr_nf4_ext", "gpu.vq"))]
    if ours:
        raise RuntimeError(
            "a competitor slot imported our kernel "
            f"({', '.join(sorted(ours))}); one stack per process"
        )


def _version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _probe_bitsandbytes(probe: _Probe, artifact: str) -> None:
    import bitsandbytes
    import torch

    probe.note(f"bitsandbytes=={_version(bitsandbytes)}, torch=={torch.__version__}")
    if not torch.cuda.is_available():
        probe.skip_reason = (
            f"{SKIP} bitsandbytes NF4: torch reports no CUDA device in this venv. "
            "A CPU dequant is not a competitor row."
        )
        return
    if getattr(getattr(bitsandbytes, "nn", None), "Linear4bit", None) is None:
        probe.skip_reason = (
            f"{SKIP} bitsandbytes NF4: this build has no bitsandbytes.nn.Linear4bit."
        )
        return
    probe.note(f"artifact={artifact} (HF BF16 tree, quantized on load)")


def _probe_gptq_marlin(probe: _Probe, artifact: str) -> None:
    import torch

    found = []
    for name in ("gptqmodel", "auto_gptq"):
        try:
            module = __import__(name)
        except Exception as exc:  # noqa: BLE001
            probe.note(f"{name} import failed: {type(exc).__name__}: {exc}")
            continue
        found.append(f"{name}=={_version(module)}")
    probe.note(f"torch=={torch.__version__}; {', '.join(found) or 'nothing importable'}")
    if not found:
        probe.skip_reason = f"{SKIP} GPTQ + Marlin: neither gptqmodel nor auto_gptq imports here."
        return
    probe.note(f"artifact={artifact}")
    probe.note(
        "a Marlin kernel must actually launch on sm_86 for this row to exist; Marlin is "
        "typically Linux CUDA"
    )


def _probe_awq(probe: _Probe, artifact: str) -> None:
    import torch

    try:
        import awq
    except Exception as exc:  # noqa: BLE001
        probe.skip_reason = f"{SKIP} AWQ: import awq raised {type(exc).__name__}: {exc}"
        return
    probe.note(f"awq=={_version(awq)}, torch=={torch.__version__}, artifact={artifact}")


def _probe_llamacpp(probe: _Probe, artifact: str) -> None:
    import llama_cpp

    probe.note(f"llama_cpp=={_version(llama_cpp)}, gguf={artifact}")
    offload = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    if not callable(offload):
        probe.skip_reason = (
            f"{SKIP} llama.cpp CUDA Q4: this build does not expose "
            "llama_supports_gpu_offload(); cannot prove a CUDA backend. "
            "CPU tok/s is not a GPU competitor row."
        )
        return
    if not bool(offload()):
        probe.skip_reason = (
            f"{SKIP} llama.cpp CUDA Q4: llama_supports_gpu_offload() is False -- only the CPU "
            "backend is linked. CPU tok/s is not a GPU competitor row."
        )
        return
    probe.note("CUDA offload is linked; record Q4_K_M vs Q4_0 in notes when measured")


def _probe_exllamav2(probe: _Probe, artifact: str) -> None:
    import torch

    try:
        import exllamav2
    except Exception as exc:  # noqa: BLE001
        probe.skip_reason = (
            f"{SKIP} ExLlamaV2: import exllamav2 raised {type(exc).__name__}: {exc}"
        )
        return
    probe.note(f"exllamav2=={_version(exllamav2)}, torch=={torch.__version__}")
    if not torch.cuda.is_available():
        probe.skip_reason = (
            f"{SKIP} ExLlamaV2: torch reports no CUDA device in this venv. "
            "A CPU fallback is not a GPU competitor row."
        )
        return
    probe.note(f"artifact={artifact} (EXL2 dump)")


def _probe_vllm(probe: _Probe, artifact: str) -> None:
    import torch

    try:
        import vllm
    except Exception as exc:  # noqa: BLE001
        probe.skip_reason = f"{SKIP} vLLM: import vllm raised {type(exc).__name__}: {exc}"
        return
    probe.note(f"vllm=={_version(vllm)}, torch=={torch.__version__}")
    if not torch.cuda.is_available():
        probe.skip_reason = (
            f"{SKIP} vLLM: torch reports no CUDA device in this venv. "
            "A CPU engine is not a GPU competitor row."
        )
        return
    probe.note(f"artifact={artifact}")


_PROBES = {
    "bitsandbytes-nf4": _probe_bitsandbytes,
    "gptq-marlin": _probe_gptq_marlin,
    "awq": _probe_awq,
    "llamacpp-q4": _probe_llamacpp,
    "exllamav2-exl2": _probe_exllamav2,
    "vllm": _probe_vllm,
}


def _worker_main(args: argparse.Namespace) -> int:
    """The isolated slot. Probes for real, writes ``competitor.json``, exits."""
    stack = stack_by_name(args.stack)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    probe = _Probe()
    try:
        _refuse_our_kernel()
        _PROBES[stack.name](probe, args.artifact)
        if not probe.skip_reason:
            probe.skip_reason = (
                f"{SKIP} {stack.label}: the stack loads and the artifact is on disk, but "
                "this slot has no inference body yet (K3 skeleton). Empty tok/s cells on "
                "purpose -- fill them only from a measured run on this 3080."
            )
            probe.note(
                f"probe passed with prompts={args.prompts}, "
                f"max_new_tokens={args.max_new_tokens}"
            )
    except Exception as exc:  # noqa: BLE001 -- an unloadable stack is the measurement
        probe.install = "missing"
        probe.skip_reason = f"{SKIP} {stack.label}: {type(exc).__name__}: {exc}"
    payload = probe.payload(stack)
    (dest / "competitor.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(payload["skip_reason"] or f"{stack.name}: measured", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gpu.lab.competitor",
        description=(
            "4-bit competitor harness on this 3080: bitsandbytes NF4, GPTQ+Marlin, "
            "AWQ, llama.cpp CUDA Q4, ExLlamaV2/EXL2, vLLM. End-to-end tok/s plus a "
            "named linear microbench [M,K]x[K,N]. A missing stack is a skip row, "
            "never a tok/s and never a CPU llama.cpp fill-in."
        ),
        epilog=(
            "This harness installs nothing and downloads nothing. Isolated venvs: "
            "docs/competitor-venvs.md (never the live torch-gpu env). Point "
            "DEEPFOLD_GPTQ / DEEPFOLD_AWQ / DEEPFOLD_GGUF / DEEPFOLD_EXL2 / "
            "DEEPFOLD_VLLM at artifacts you already have."
        ),
    )
    parser.add_argument("--lab", default="qwen25-3b", help="catalog slug; 3B first (both copies fit)")
    parser.add_argument(
        "--stacks",
        default="",
        help=(
            "comma-separated slot names, or 'all' / 'ours'. Default: the ChatGPT "
            "GPU stacks (not CPU llama.cpp)"
        ),
    )
    parser.add_argument("--out", default="", help="output dir; default $DEEPFOLD_RUNS/competitor-...")
    parser.add_argument(
        "--prompts",
        choices=PROMPT_SETS,
        default="smoke",
        help="smoke (Paris/Berlin/323) or the shared hard fixture. Separate tables.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-graphs", dest="graphs", action="store_false")
    parser.add_argument(
        "--python",
        dest="python_exe",
        default="",
        help="interpreter for the slot (its own venv under C:\\dev\\models\\venvs; never torch-gpu)",
    )
    parser.add_argument(
        "--detect", action="store_true", help="print the skip matrix and exit (no GPU, no imports)"
    )
    parser.add_argument(
        "--recipe",
        action="store_true",
        help="print the isolated-venv recipe path and exit (does not install)",
    )
    parser.add_argument("--list", action="store_true", help="print the slots and exit")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stack", default="", help=argparse.SUPPRESS)
    parser.add_argument("--artifact", default="", help=argparse.SUPPRESS)
    return parser


def _chosen(names: str) -> tuple[Stack, ...]:
    if not names:
        return COMPETITOR_STACKS
    if names == "all":
        return STACKS
    if names == "ours":
        return OUR_STACKS
    return tuple(stack_by_name(part.strip()) for part in names.split(",") if part.strip())


def _report(rows: Sequence[CompetitorRow]) -> None:
    print("\nstack               install   TTFT ms   tok/s   smi MiB   smoke")
    for row in rows:
        def cell(value: float | None, decimals: int = 0) -> str:
            return "-" if value is None else f"{value:,.{decimals}f}"

        smoke = "-" if row.smoke_ok is None else ("ok" if row.smoke_ok else "fail")
        print(
            f"{row.stack:<19} {row.install:<9} {cell(row.mean_ttft_ms):>7}  "
            f"{cell(row.mean_decode_tok_s, 1):>6}  {cell(row.smi_after_mib):>8}  {smoke:>5}"
        )
        if row.skip_reason:
            print(f"  {row.skip_reason}")
    measured = sum(1 for row in rows if row.measured)
    kernel = microbench_grid_for(rows)
    kernel_measured = sum(1 for cell_row in kernel if cell_row.measured)
    print(
        f"\n{measured}/{len(rows)} e2e slots produced a number. "
        f"{kernel_measured}/{len(kernel)} kernel cells produced a µs. "
        "A skip row is the result; an invented one is not. Do not rank a blank cell. "
        "CPU llama.cpp tok/s is not a GPU competitor."
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    if args.worker:
        if not args.stack:
            print("--worker needs --stack", file=sys.stderr)
            return 2
        return _worker_main(args)

    if args.recipe:
        path = recipe_path()
        print(path)
        print(f"venv root: {VENV_ROOT}")
        print("Do not install into the live torch-gpu env. This flag only prints.")
        if not path.is_file():
            print(f"recipe file missing: {path}", file=sys.stderr)
            return 1
        text = path.read_text(encoding="utf-8")
        print()
        print(text, end="" if text.endswith("\n") else "\n")
        return 0

    stacks = _chosen(args.stacks)

    if args.list:
        for stack in STACKS:
            kind = "ours" if stack.ours else "competitor"
            body = "wired" if stack.wired else "skeleton"
            print(f"{stack.name:<19} {kind:<11} {body:<9} {stack.artifact_env or '-'}")
            print(f"  venv {stack.venv_dir}")
            print(f"  {stack.note}")
        print(
            f"\nkernel grid: {len(MICROBENCH_LINEARS)} linears x {len(MICROBENCH_M)} M "
            f"x {len(COMPETITOR_STACKS)} GPU stacks; metric cells empty until measured."
        )
        return 0

    lab = lab_by_slug(args.lab)

    if args.detect:
        print(f"lab {lab.slug}  ({lab.title})")
        detections = detect_all(stacks, model_dir=lab.model_dir, chr_path=lab.chr_path)
        rows = []
        for detection in detections:
            stack = stack_by_name(detection.stack)
            state = "RUNNABLE" if detection.runnable else "SKIP"
            print(f"\n{detection.stack:<19} install={detection.install:<8} {state}")
            if detection.artifact:
                print(f"  artifact: {detection.artifact}")
            if detection.skip_reason:
                print(f"  {detection.skip_reason}")
            extra = ""
            if detection.runnable:
                extra = (
                    "Detect only: not launched (3080 left for the NF4 kernel). "
                    "A skip row with empty numeric cells is the result until a child runs."
                )
                rows.append(
                    CompetitorRow(
                        stack=stack.name,
                        install=detection.install,
                        skip_reason=f"{SKIP} {stack.label}: detect-only, generate not started",
                        notes=f"{stack.note} {extra}".strip(),
                    )
                )
            else:
                rows.append(CompetitorRow.skip(stack, detection))
        print("\nNo GPU touched, nothing installed, nothing downloaded.")
        kernel = microbench_grid_for(rows)
        print(
            f"microbench: {len(rows)} stacks x {len(MICROBENCH_LINEARS)} linears x "
            f"{len(MICROBENCH_M)} M = {len(kernel)} named cells; "
            "us/GB/s/TFLOP/s/occupancy/tensor/DRAM/regs/smem empty (SKIP). "
            "CPU llama.cpp tok/s is not a fill-in."
        )
        if args.out:
            dest = Path(args.out)
            if dest.suffix.lower() == ".csv":
                write_competitor_csv(dest, rows)
                micro_path = dest.with_name("microbench.csv")
                write_microbench_csv(micro_path, kernel)
                csv_path = dest
                source_dir = dest.parent
            else:
                csv_path, micro_path = write_matrix_csvs(dest, rows)
                source_dir = dest
            (source_dir / "SOURCE.txt").write_text(
                _matrix_source("WAVE 10 K3 detect-only matrix."),
                encoding="utf-8",
            )
            print(f"\nwrote {csv_path}")
            print(f"wrote {micro_path}")
        return 0

    out_dir = Path(args.out) if args.out else Path(RUNS_DIR) / time.strftime(
        f"competitor-{lab.slug}-{args.prompts}-%Y%m%d-%H%M%S"
    )
    print(
        f"lab {lab.slug}  prompts={args.prompts}  slots={[s.name for s in stacks]}  "
        f"out={out_dir}",
        flush=True,
    )
    rows = run_matrix(
        out_dir,
        stacks=stacks,
        lab=lab,
        prompt_set=args.prompts,
        max_new_tokens=args.max_new_tokens,
        graphs=args.graphs,
        python_exe=args.python_exe or None,
        verbose=not args.quiet,
    )
    _report(rows)
    print(f"\nwrote {out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

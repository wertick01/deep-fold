"""Write ``notebooks/06_hard_eval.ipynb`` and nothing else.

03 / 04 / 05 are executed artifacts of the smoke lab and must keep their
outputs. This builder only ever touches 06, and :func:`main` asserts that by
comparing the other notebooks' bytes before and after the write. Regenerating
06 can therefore never wipe a smoke run.

    python notebooks/build_hard_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

TARGET = "06_hard_eval.ipynb"
# Executed smoke notebooks. This builder must not change one byte of these.
PROTECTED = ("03_codec_lab.ipynb", "04_qwen25_14b_lab.ipynb", "05_internlm20b_lab.ipynb")


def _md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def _code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def build():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.cells = [
        _md(
            """
# Hard eval: 3B **and** 14B, uncompressed then compressed — not Paris / Berlin / 323

**Kernel → Restart Kernel, then Run All.** This kernel must never hold a
model. Every run below is a **child process** that loads, answers, unloads and
**exits** before the next one starts. 12 GB cannot hold two copies.

Four runs, in this order:

| # | codec | model | weights on disk | fits in 12 GB? |
|---|-------|-------|-----------------|----------------|
| 1 | BF16 dense | Qwen2.5-**3B**-Instruct | **~5,886 MiB** | yes |
| 2 | BF16 dense | Qwen2.5-**14B**-Instruct | **~28,172 MiB** | **no — spills** |
| 3 | NF4 driver | Qwen2.5-**3B**-Instruct | **~1,563 MiB** | yes |
| 4 | NF4 driver | Qwen2.5-**14B**-Instruct | **~7,483 MiB** | yes |

Uncompressed both models first, then compressed both models. The pre-flight
cell measures those four numbers off the filesystem, so a row labelled 14B can
be checked against **~28,172 MiB BF16 / ~7,483 MiB NF4** instead of being taken
on trust — a 14B that reports ~5,886 or ~1,563 MiB is a mislabelled 3B.

Notebooks 03 / 04 / 05 are **smoke**: a pass on capitals and `323` is not a
quality score. This plate asks a fixed 12-item set (GSM8K-style arithmetic, a
logic trap, mental code execution, a long-prefill needle) that 3B and 14B can
and do fail. Same prompts, greedy, `max_new_tokens = 256`.

**Honesty.** NF4 is lossy and a quality drop is in scope. Speed is two
different stacks (HuggingFace dense GEMM vs our fused NF4 loop): on 3B, where
both fit, **dense BF16 decodes faster**. `nvidia-smi` reports dedicated VRAM
and stops at 12288 MiB, so it cannot describe 14B BF16 — the honest number
there is the torch **CUDA working set** (~28 GiB), with the remainder in
Windows shared GPU memory. Do not quote two nvidia-smi peaks as the win. A
CUDA OOM or a missing file is a **recorded session with notes**, never a
crashed cell. Method: `docs/eval.md`.
"""
        ),
        _md(
            """
## Pre-flight: which models, which codecs, how many MiB

`gpu/lab/catalog.py` owns the paths; nothing here is typed twice. The
`weights on disk` column is measured now: the sum of the safetensor shards for
BF16, the `.chr` container for NF4. That is what makes 3B vs 14B checkable
before a single kernel launch.
"""
        ),
        _code(
            r"""
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(r"C:\dev\deep-fold")
sys.path.insert(0, str(REPO))

from gpu.lab.catalog import SUPPORTED_NF4, lab_by_slug
from gpu.lab.hard import (
    HARD_MAX_NEW_TOKENS,
    RUN_ORDER,
    expected_weight_mib,
    hard_max_seq,
    items_path,
    size_label,
)

RUN_HISTORY = False   # True = 4-turn growing-prefill protocol; independent is the default
RUN_20B = False       # optional negative control; off so 20B can never block 3B + 14B
RUNS = Path(r"C:\dev\models\runs")
MAX_NEW = HARD_MAX_NEW_TOKENS

# Required order: uncompressed both models, then compressed both models.
PHASE_A = (("qwen25-3b", "bf16"), ("qwen25-14b", "bf16"))
PHASE_B = (("qwen25-3b", "nf4"), ("qwen25-14b", "nf4"))
assert PHASE_A + PHASE_B == RUN_ORDER, RUN_ORDER

out_dir = RUNS / datetime.now().strftime("hard-matrix-%Y%m%d-%H%M%S")
CARD_MIB = 12288

print(f"{'run':<9} {'catalog':<11} {'weight file(s)':<34} {'MiB on disk':>11}  {'max_seq':>7}  state")
weights = {}
for slug, codec in RUN_ORDER:
    lab = lab_by_slug(slug)
    mib = expected_weight_mib(lab)[codec]
    label = f"{size_label(lab)} {codec.upper()}"
    weights[label] = mib
    if codec == "bf16":
        source = Path(lab.model_dir)
        present = source.is_dir() and any(source.glob("*.safetensors"))
        shown = f"{source.name}/*.safetensors"
    else:
        source = Path(lab.chr_path)
        present = source.is_file()
        shown = source.name
    state = "ok" if present else "MISSING -> recorded skip"
    if present and codec == "bf16" and mib and mib > CARD_MIB:
        state = f"ok, {mib / CARD_MIB:.1f}x the card -> will spill"
    print(
        f"{label:<9} {lab.slug:<11} {shown:<34} "
        f"{'n/a' if mib is None else f'{mib:>11,.0f}'}  {hard_max_seq(lab):>7}  {state}"
    )

print(f"\n14B BF16 is {weights['14B BF16'] / weights['3B BF16']:.1f}x the 3B BF16 weight set.")
print(f"14B NF4  is {weights['14B NF4'] / weights['3B NF4']:.1f}x the 3B NF4 weight set.")
print("If a run reports ~5.9k / ~1.6k MiB of weights, that run is 3B, not 14B.")
print("\nout_dir  ", out_dir)
print("fixture  ", items_path())
print("max_new  ", MAX_NEW, " history", RUN_HISTORY, " 20B", RUN_20B)
print("nf4 drivers supported:", sorted(SUPPORTED_NF4))
"""
        ),
        _md(
            """
## GPU snapshot

RTX 3080, **12288 MiB**. Do **not** `import torch` in this kernel — that opens
a CUDA context here and sits under all four workers. Workers own CUDA.
`nvidia-smi` `used` includes the desktop compositor. Do not subtract it.
"""
        ),
        _code(
            """
import subprocess

def nvidia_smi_query(fields: str) -> list[str]:
    cmd = [
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,nounits,noheader",
    ]
    out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace")
    return [x.strip() for x in out.strip().split(",")]


name, total, used, driver, temp, clock, pwr = nvidia_smi_query(
    "name,memory.total,memory.used,driver_version,temperature.gpu,clocks.sm,power.draw"
)
print(f"GPU: {name}")
print(f"Driver: {driver}")
print(f"Memory: {used} / {total} MiB (display is already inside used)")
print("Card limit: 12288 MiB")
print(f"Free roughly: {int(float(total)) - int(float(used))} MiB")
print(f"Now: {temp} °C, SM {clock} MHz, {pwr} W")
if float(used) > 4000:
    raise SystemExit(
        f"nvidia-smi is {used} MiB. Kernel → Restart Kernel, then Run All. "
        "This process is still holding a previous model."
    )
print("Idle enough to start. Four isolated workers, one after the other.")
"""
        ),
        _md(
            """
## The questions and their gold answers

12 independent items in `gpu/lab/data/hard_items.json`, committed, so tests
never download GSM8K. Override with `DEEPFOLD_HARD` pointing at a larger local
JSON. These same questions and golds are drawn on the infographic at the
bottom, so the plate says what was actually asked.
"""
        ),
        _code(
            """
from gpu.lab.hard import load_fixture, load_script, write_run_script

out_dir.mkdir(parents=True, exist_ok=True)
script_path = write_run_script(out_dir / "hard_script.json", history=RUN_HISTORY)
script = load_script(script_path, history=RUN_HISTORY)
ITEMS = script.items          # exactly what every run is asked
RUNS_DONE = []                # HardRun rows in run order; the plate draws these

print("script  ", script_path)
print(f"{script.conversation}: {len(ITEMS)} items, max_new_tokens={script.max_new_tokens}\\n")
print(f"{'#':>2}  {'item':<20} {'kind':<7} {'gold':<8} question")
for index, item in enumerate(ITEMS, start=1):
    question = " ".join(item.prompt.split())
    print(
        f"{index:>2}  {item.id:<20} {item.kind:<7} "
        f"{(item.gold or ','.join(item.needles)):<8} {question[:78]}"
    )
print("\\nOpen items (kind=open) are rated by two humans later and never auto-pass.")
"""
        ),
        _md(
            """
## Harness

`run_matrix` runs the pairs it is given **sequentially**, each in its own
`python -m gpu.lab.worker` child, and waits for `nvidia-smi` to come back down
in between. Per run it writes `<slug>-<codec>/` with the four frozen CSVs,
`hard_scores.csv`, and the worker's stdout/stderr in `<codec>/worker.log`.
`hard_matrix.csv` at the top is the cross-run index.

Quality is `hard_scores.csv`. Do not use panel C of the smoke plate — it still
labels Paris / Berlin / 323.
"""
        ),
        _code(
            """
from gpu.lab.bundle import LabBundle
from gpu.lab.hard import run_matrix
from gpu.lab.sampler import smi_used_mib

PREVIEW_REPLIES = 2   # full text of every reply is in messages.csv


def num(value, decimals=0):
    return "n/a" if value is None else f"{float(value):,.{decimals}f}"


def show_run(run):
    \"\"\"Printed as each child exits: which model, what it held, what it said.\"\"\"
    print(f"\\n--- {run.label}  {run.title}  ({run.codec})")
    print(
        f"    weights        {num(run.weight_mib)} MiB measured"
        f"   /  {num(run.expected_weight_mib)} MiB on disk"
    )
    print(
        f"    CUDA working   {num(run.working_set_mib)} MiB"
        f"   nvidia-smi peak {num(run.smi_peak_mib)} MiB (meter stops at 12,288)"
    )
    print(
        f"    TTFT {num(run.mean_ttft_ms)} ms   decode {num(run.mean_tok_s, 1)} tok/s"
        f"   accuracy {'n/a' if run.accuracy is None else f'{100 * run.accuracy:.0f}%'}"
        f"   replies {run.n_messages}/{len(ITEMS)}"
    )
    if run.notes:
        print(f"    notes: {run.notes}")
    if not run.ran:
        print("    no replies: recorded miss, not a crashed cell. See worker.log in", run.out_dir)
        return
    bundle = LabBundle.read(run.out_dir)
    for row in bundle.messages[:PREVIEW_REPLIES]:
        print("    " + "-" * 56)
        print(f"    item {row['message_id']}: {(row['prompt'] or '')[:90]}")
        print(f"    reply: {' '.join((row['response'] or '').split())[:300]}")
        print(
            f"    prompt_tokens={row['prompt_tokens']} new_tokens={row['new_tokens']} "
            f"ttft={row['prefill_ms']:.0f} ms {row['decode_tok_s']:.1f} tok/s "
            f"correct={row['quality_ok']}"
        )


def smi_now(label):
    used = smi_used_mib()
    print(
        f"nvidia-smi used {used:.0f} MiB  ({label})"
        if used is not None
        else f"nvidia-smi n/a ({label})"
    )
    return used


smi_now("idle, before the first worker")
print("artifacts →", out_dir)
"""
        ),
        _md(
            """
## Phase A — uncompressed BF16: 3B, then 14B

Two child processes, one after the other. HuggingFace `from_pretrained` +
`generate` on the dense shards.

3B fits with room to spare. **14B does not**: ~28 GiB of BF16 weights on a
12 GiB card, so CUDA spills into shared GPU memory and this run is slow —
expect it to dominate the notebook's wall clock. `nvidia-smi` will sit near
12288 because that is where the meter ends; the working set printed under each
run is the real number. If the driver refuses the allocation instead, the
session is recorded with `CUDA OOM` in its notes and the notebook carries on.
"""
        ),
        _code(
            """
RUNS_DONE = run_matrix(
    out_dir,
    pairs=PHASE_A,
    existing=RUNS_DONE,
    history=RUN_HISTORY,
    on_run=show_run,
)
smi_now("after both BF16 workers exited")
print("\\nphase A done:", [run.label for run in RUNS_DONE])
"""
        ),
        _md(
            """
## Phase B — compressed NF4: 3B, then 14B

Starts only after the BF16 **processes have exited**. Each child loads the
`.chr`, never the safetensor shards: `gpu.host.load_model` +
`gpu.loop.TokenLoop`, no `from_pretrained` on the weights and no
`transformers.generate` anywhere near them. A missing `.chr` is a recorded
skip with the compress command in its notes.
"""
        ),
        _code(
            """
RUNS_DONE = run_matrix(
    out_dir,
    pairs=PHASE_B,
    existing=RUNS_DONE,
    history=RUN_HISTORY,
    on_run=show_run,
)
smi_now("after both NF4 workers exited")
print("\\nall runs:", [run.label for run in RUNS_DONE])
"""
        ),
        _md(
            """
## Optional — InternLM 20B

Off by default. 20B is a negative control and is **not** part of this pass:
BF16 is ~37 GiB and the NF4 `.chr` is ~10 GiB, so both are slow or impossible
here. Set `RUN_20B = True` in the pre-flight cell to append it; it can never
block 3B + 14B, which have already run above. `trust_remote_code` needs
`einops` and `sentencepiece==0.1.99` — those are a 20B problem only, the two
Qwen models need neither.
"""
        ),
        _code(
            """
if RUN_20B:
    RUNS_DONE = run_matrix(
        out_dir,
        pairs=(("internlm20b", "bf16"), ("internlm20b", "nf4")),
        existing=RUNS_DONE,
        history=RUN_HISTORY,
        on_run=show_run,
    )
else:
    print("20B skipped (RUN_20B = False). No 20B numbers are claimed in this notebook.")
"""
        ),
        _md(
            """
## Tables

`hard_matrix.csv` is one row per run: model, codec, weights, memory, speed,
accuracy. `hard_scores.csv` in each run directory is that run's item-by-item
quality. The answer table joins them: every question, its gold, and what each
run extracted.
"""
        ),
        _code(
            """
import pandas as pd
from IPython.display import display

from gpu.lab.hard import answer_rows, read_matrix

pd.set_option("display.max_colwidth", 70)
pd.set_option("display.width", 200)

print("hard_matrix.csv — one row per (model, codec) run")
matrix = pd.DataFrame(read_matrix(out_dir / "hard_matrix.csv"))
display(
    matrix[
        [
            "label",
            "title",
            "codec",
            "weight_mib",
            "expected_weight_mib",
            "working_set_mib",
            "vram_peak_smi_mib",
            "mean_ttft_ms",
            "mean_decode_tok_s",
            "accuracy",
            "n_messages",
        ]
    ]
)

print("\\nanswer table — question, gold, and what each run answered")
rows = answer_rows(ITEMS, RUNS_DONE)
table = pd.DataFrame(
    [
        {
            "#": row["n"],
            "item": row["item"],
            "gold": row["gold"],
            **{label: cell.text for label, cell in row["cells"].items()},
            "question": row["question"][:60],
        }
        for row in rows
    ]
)
display(table)

for run in RUNS_DONE:
    acc = "n/a" if run.accuracy is None else f"{100 * run.accuracy:.0f}%"
    print(f"{run.label:<9} accuracy {acc:>4}   {run.out_dir / 'hard_scores.csv'}")
    if run.notes:
        print(f"          notes: {run.notes[:200]}")
"""
        ),
        _md(
            """
## Results infographic

One sheet for the whole matrix: CUDA working set, nvidia-smi peak (capped at
the card, so it is not the win), mean TTFT, decode tok/s and hard accuracy for
the four runs, with the model size on every tick label — then the answer table
underneath, so the questions, the golds and each run's answer are on the same
page as the bars.

The table renders from the fixture even with no live run. Once any run has
reported, the panels are padded out to all four ticks — `3B BF16`, `14B BF16`,
`3B NF4`, `14B NF4` — so a plate drawn while the 14B BF16 spill is still
grinding cannot be misread as a finished two-run comparison: the pairs still
queued say `not run yet` and their answer cells are dashes. Never a zero.
"""
        ),
        _code(
            """
%matplotlib inline
from gpu.lab.hard_plate import hard_plate, write_plate

fig = hard_plate(ITEMS, RUNS_DONE)
for path in write_plate(fig, out_dir / "hard-eval.png", REPO / "docs" / "img" / "hard-eval-qwen25.png"):
    print("wrote", path)
fig
"""
        ),
        _md(
            """
## How to read this

- **Which model.** Every bar tick is `3B` or `14B` over the codec, and
  `weight_mib` in `hard_matrix.csv` is next to the bytes measured on disk:
  **~5,886 MiB (3B BF16), ~28,172 MiB (14B BF16), ~1,563 MiB (3B NF4),
  ~7,483 MiB (14B NF4)**. If the measured and on-disk columns agree, the run
  is the model it claims to be — that is the check, not the row label.
- **Quality.** `hard_scores.csv` per run. GSM8K extraction is `#### N`, then
  “the answer is”, then the last number. A miss is a miss, and NF4 is allowed
  to score worse than BF16 — that is the measurement.
- **Speed.** TTFT is prefill. Decode tok/s is HuggingFace dense GEMM against
  the fused NF4 loop. On 3B, where both fit, **BF16 is the faster stack**. On
  14B, BF16 is crippled by the spill, so NF4 winning there is a statement
  about the card, not about GEMM.
- **Memory.** `nvidia-smi` is dedicated VRAM and stops at 12288 MiB.
  `working_set_mib` (torch) is the honest 14B BF16 number, and the part above
  the card is Windows shared GPU memory (system RAM). Two nvidia-smi peaks
  are **not** the compression win.
- **Misses.** A skipped or OOMed run keeps its row: blank bars, dashes in the
  answer table, and the reason in `summary.notes` plus `worker.log`. Never a
  zero, never a crashed cell, never an invented number.
- **Mid-queue.** A pair with no `summary.csv` yet is a grey `not run yet` slot,
  distinct from a run that reported nothing (red `no data`). `hard_matrix.csv`
  only ever holds real runs, so
  `python -m gpu.lab.hard --redraw <run-root>` picks up whatever has landed.
- **20B.** Not in this pass. `RUN_20B = True` appends it; nothing here claims
  a 20B result.
- **History protocol.** `RUN_HISTORY = True` re-encodes the whole chat each
  turn (growing prefill). It is not incremental KV reuse.

Regenerated by `notebooks/build_hard_eval.py`, which writes this notebook
only — 03 / 04 / 05 keep their executed outputs.
"""
        ),
    ]
    return nb


def main() -> Path:
    directory = Path(__file__).resolve().parent
    before = {
        name: (directory / name).read_bytes()
        for name in PROTECTED
        if (directory / name).is_file()
    }
    path = directory / TARGET
    nbf.write(build(), path)
    print("wrote", path)
    for name, payload in before.items():
        if (directory / name).read_bytes() != payload:
            raise SystemExit(
                f"{name} changed while writing {TARGET}. That must never happen: "
                "03/04/05 are executed smoke artifacts."
            )
    print(f"unchanged: {', '.join(before) or '(none on disk)'}")
    return path


if __name__ == "__main__":
    main()

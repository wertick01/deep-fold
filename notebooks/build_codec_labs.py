"""Generate one comparison notebook per downloaded model.

    python notebooks/build_codec_labs.py

Each notebook is the same protocol: BF16 (or a recorded CUDA OOM), unload,
then NF4 if the GPU driver speaks this architecture, then ``comparison_figure``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu.lab.catalog import LABS, LabModel  # noqa: E402


def _notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    return nb


def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def _code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


def _phase_b_markdown(lab: LabModel) -> str:
    if lab.nf4_driver != "qwen2":
        return f"""
## Phase B — compressed NF4

**Skipped on this model.** `TokenLoop` / `CompressedLinear` speak Qwen
`q_proj` / `k_proj` / `v_proj` / `o_proj`. InternLM2 fuses those into `wqkv`.
A `.chr` would still be a valid CHR0 file; the GPU loop cannot consume it yet.
Do not call `run_nf4` here.
"""
    if lab.nf4_fits:
        extra = (
            "NF4 is expected to fit on this card after BF16 has fully left VRAM."
            if not lab.bf16_fits
            else "Same three prompts, new worker, after VRAM is back near idle."
        )
        return f"""
## Phase B — compressed NF4

{extra} Starts only after the BF16 **process has exited**. Loads
`{Path(lab.chr_path).name}`, not the safetensor shards.
"""
    return f"""
## Phase B — compressed NF4

Budget says even NF4 weights plus CUDA exceed 12288 MiB. The cell will try
`run_nf4` only if `{Path(lab.chr_path).name}` exists, and a CUDA OOM is a
recorded miss, not a crash.
"""


def build(lab: LabModel) -> nbf.NotebookNode:
    chr_name = Path(lab.chr_path).name
    compress_cell = ""
    if lab.compress_cmd:
        compress_cell = f'''
COMPRESS_CMD = r"{lab.compress_cmd}"
print("compress (CPU, single-threaded, hours for 14B+):")
print(COMPRESS_CMD)
'''
    else:
        compress_cell = "print('CHR already on disk for this model.')\n"

    nf4_skip_reason = (
        "TokenLoop is Qwen-shaped (q/k/v/o). This architecture uses fused wqkv."
        if lab.nf4_driver != "qwen2"
        else ""
    )

    cells = [
        _md(
            f"""
# Codec lab: {lab.title} — BF16 vs our NF4 driver

**Kernel → Restart Kernel, then Run All.** This kernel must not already
hold a model. Each codec runs in a **child process** that exits before the
next one starts, so nvidia-smi actually comes back. Never two models at once.

Order:

1. Record idle `nvidia-smi`.
2. Load **uncompressed BF16** in a worker, three prompts, worker **exits**.
3. Wait until VRAM is back near idle.
4. Load **compressed NF4** in a new worker, same three prompts, worker exits.
5. Draw two VRAM graphs side by side (left uncompressed, right compressed),
   **same Y scale 0–12288 MiB**.

{lab.note}

**Honesty:** VRAM is the product metric. Decode tok/s is a different stack
(HF dense GEMM vs fused NF4). Display VRAM is inside `nvidia-smi`. A CUDA OOM
is written into `summary.notes` and the plate still draws; it is not a crashed
cell and it is not a fake number.
"""
        ),
        _md(
            """
## Paths

Weights are not in git. `sys.path` must include the repo so `import gpu.lab`
resolves. BF16 and NF4 never share the card.
"""
        ),
        _code(
            rf"""
# Paths on this machine. Weights are not in git.
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(r"C:\dev\deep-fold")
sys.path.insert(0, str(REPO))

MODEL_DIR = Path(r"{lab.model_dir}")
CHR = Path(r"{lab.chr_path}")
RUNS = Path(r"C:\dev\models\runs")
TITLE = {lab.title!r}
PLATE_TITLE = {lab.plate_title!r}
TRUST_REMOTE_CODE = {lab.trust_remote_code!s}
NF4_DRIVER = {lab.nf4_driver!r}
out_dir = RUNS / datetime.now().strftime("lab-{lab.slug}-%Y%m%d-%H%M%S")

print("REPO     ", REPO)
print("MODEL_DIR", MODEL_DIR, "exists" if MODEL_DIR.is_dir() else "MISSING")
print("CHR      ", CHR)
print(
    "CHR size ",
    f"{{CHR.stat().st_size / 1024**3:.2f}} GiB" if CHR.is_file() else "MISSING",
)
print("out_dir  ", out_dir)
print("driver   ", NF4_DRIVER)
{compress_cell}
"""
        ),
        _md(
            """
## Plotly + Jinja2

Quotes around version pins are required: an unquoted `>=` is a shell
redirect. Kaleido is optional (PNG for README); HTML does not need it.
"""
        ),
        _code(
            """
# Quotes are required: unquoted plotly>=5 is a shell redirect, not a version pin.
%pip install "plotly>=5" "jinja2>=3.1.0"
"""
        ),
        _md(
            """
## GPU snapshot

RTX 3080, **12288 MiB**. Do **not** `import torch` in this kernel — that would
open a CUDA context here and sit under both workers. Workers own CUDA.
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
print("Idle enough to start. Isolated workers: BF16 child, exit, then NF4 child.")
"""
        ),
        _md(
            """
## Harness

`run_bf16(..., isolated=True)` / `run_nf4(..., isolated=True)` each spawn a
child Python. The child loads the model, chats, unloads, **exits**. This
kernel never holds the weights. Read `nvidia-smi` between the two cells.

The plate is `comparison_figure`: VRAM is two graphs side by side, same Y.
"""
        ),
        _code(
            """
from gpu.lab import MESSAGES, comparison_figure, run_bf16, run_nf4, write_artifacts
from gpu.lab.bundle import LabBundle
from gpu.lab.sampler import smi_used_mib

print("MESSAGES (identical for both codecs, independent turns):")
for i, text in enumerate(MESSAGES, start=1):
    print(f"  {i}. {text}")


def print_turns(session) -> None:
    if not session.messages:
        print("no replies (OOM or skipped). notes:", session.summary.get("notes"))
        return
    for row in session.messages:
        print("=" * 60)
        print(f"codec={row['codec']}  message_id={row['message_id']}")
        print("prompt:", row["prompt"])
        print("--- reply ---")
        print(row["response"])
        print(
            f"prompt_tokens={row['prompt_tokens']}  new_tokens={row['new_tokens']}  "
            f"prefill_ms={row['prefill_ms']:.1f}  decode_tok_s={row['decode_tok_s']:.2f}  "
            f"quality_ok={row['quality_ok']}"
        )


def smi_now(label: str):
    used = smi_used_mib()
    print(f"nvidia-smi used {used:.0f} MiB  ({label})" if used is not None else f"nvidia-smi n/a ({label})")
    return used
"""
        ),
        _md(
            """
## Phase A — uncompressed BF16

Child process: HuggingFace dense weights. When the worker **exits**, VRAM
must come back. Do not start phase B until `nvidia-smi` is near idle.
"""
        ),
        _code(
            """
out_dir.mkdir(parents=True, exist_ok=True)
print("artifacts →", out_dir)
idle = smi_now("idle, before BF16 worker")

bf16 = run_bf16(
    out_dir=out_dir,
    model_dir=MODEL_DIR,
    trust_remote_code=TRUST_REMOTE_CODE,
    isolated=True,
)
print_turns(bf16)
print(bf16.summary)
smi_after_bf16 = smi_now("after BF16 worker exited")
peak_bf16 = bf16.summary.get("vram_peak_smi_mib")
print(f"BF16 peak smi {peak_bf16} MiB")
print("n_messages", bf16.summary.get("n_messages"))
"""
        ),
        _md(_phase_b_markdown(lab)),
        _code(
            f"""
import time

nf4 = None
skip_nf4 = {bool(nf4_skip_reason)!s}
skip_reason = {nf4_skip_reason!r}

if skip_nf4:
    print("NF4 skipped:", skip_reason)
elif not CHR.is_file():
    print("NF4 file missing:", CHR)
    print("Compress on CPU, then re-run this cell. Command:")
    print(globals().get("COMPRESS_CMD", "(no compress command for this notebook)"))
else:
    # Wait until the BF16 worker's VRAM is actually gone.
    deadline = time.time() + 45
    used = smi_now("before NF4 worker")
    while (
        used is not None
        and peak_bf16 is not None
        and used > float(peak_bf16) - 1500
        and time.time() < deadline
    ):
        time.sleep(1.0)
        used = smi_now("waiting for BF16 VRAM to drop")
    if (
        used is not None
        and peak_bf16 is not None
        and used > float(peak_bf16) - 1500
    ):
        raise SystemExit(
            f"nvidia-smi still {{used:.0f}} MiB after BF16 (peak {{float(peak_bf16):.0f}}). "
            "Restart the kernel so this process is not holding the dense model."
        )
    nf4 = run_nf4(
        out_dir=out_dir,
        model_dir=MODEL_DIR,
        chr_path=CHR,
        trust_remote_code=TRUST_REMOTE_CODE,
        isolated=True,
    )
    print_turns(nf4)
    print(nf4.summary)
    smi_now("after NF4 worker exited")

bundle = LabBundle()
bundle = bundle.merge(bf16.as_bundle())
if nf4 is not None:
    bundle = bundle.merge(nf4.as_bundle())
bundle.write(out_dir)
print("wrote CSVs to", out_dir)
print("codecs in bundle:", bundle.codecs)
"""
        ),
        _md(
            """
## One figure

VRAM is **two graphs side by side**: left uncompressed BF16, right compressed
NF4, **the same 0–12288 MiB Y scale**. If CUDA's working set is larger than
the card (14B BF16 does this: nvidia-smi sits at 12 GB, torch holds ~28 GB
in VRAM + shared GPU memory), a **second pair** is drawn directly underneath —
same Y on both columns, high enough to show the spill into system RAM.
Then budget, replies, TTFT, decode.
"""
        ),
        _code(
            """
# One interactive figure. Not five charts. Not SVG.
fig = comparison_figure(bundle, title=PLATE_TITLE)
write_artifacts(fig, out_dir)
fig.show()
print("lab.html", out_dir / "lab.html")
"""
        ),
        _md(
            """
## Tables

`messages.csv` is the chat log. `summary.csv` is one row per codec. Display
the frames; do not screenshot them. Numbers come from the harness — this
cell does not invent VRAM or tok/s.
"""
        ),
        _code(
            """
import pandas as pd
from IPython.display import display

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 160)

messages = pd.read_csv(out_dir / "messages.csv")
summary = pd.read_csv(out_dir / "summary.csv")

print("messages.csv")
display(messages)
print("summary.csv")
display(summary)
print("artifacts in", out_dir)
for name in (
    "timeline.csv",
    "events.csv",
    "messages.csv",
    "summary.csv",
    "lab.html",
    "lab.png",
):
    p = out_dir / name
    print(f"  {name}: {'ok' if p.is_file() else 'missing (allowed for lab.png if kaleido skipped)'}")
"""
        ),
        _md(
            f"""
## How to read this

VRAM is the claim. Tok/s is a different stack.

- **VRAM.** After load, NF4 must sit well below BF16 on the same card. If the
  two traces meet, the NF4 path materialized a layer into HBM — a bug, not a
  compression win. There is **no `[M,K]` BF16** weight matrix on the NF4
  path (`load_model` + `TokenLoop`, never `from_pretrained` on the shards).
  `nvidia-smi` is **dedicated VRAM only** and stops at 12288 MiB. If CUDA
  allocates more than that (14B BF16: torch ~28 GiB), the extra is Windows
  **shared GPU memory** (system RAM). The plate then adds a second pair of
  graphs under VRAM, same Y, so that spill is visible. Do not read a full
  12 GB nvidia-smi line as “the model fit.”
- **Tok/s.** BF16 is HuggingFace `generate` + dense GEMM. NF4 is our fused
  dequant-MMA loop. Report both numbers. Do not say “we are faster” unless
  the table says so.
- **TTFT.** Prefill. BF16 uses the HF prefill; NF4 uses `N≤16` chunks.
- **Display VRAM** is inside `nvidia-smi`. It is real. Do not subtract it.
- **Independent turns.** Each prompt is a clean prefill. KV is not carried
  across the three user messages, so TTFT stays comparable.
- **OOM.** An empty `messages.csv` row-set plus `CUDA OOM` in `summary.notes`
  is the 12 GB result for this variant. Do not invent a VRAM number for it.

This notebook is **{lab.title}** only. `{chr_name}` is the compressed file for
this model. Missing run → empty CSVs or a recorded skip, not fiction.
"""
        ),
    ]
    nb = _notebook()
    nb.cells = cells
    return nb


def write_all(out_dir: Path | None = None) -> list[Path]:
    directory = out_dir or Path(__file__).resolve().parent
    written: list[Path] = []
    for lab in LABS:
        path = directory / lab.notebook
        nbf.write(build(lab), path)
        print("wrote", path)
        written.append(path)
    return written


if __name__ == "__main__":
    write_all()

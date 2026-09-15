# Driver UX

This page is the path for people who already have a model on disk and want the
NF4 driver without a notebook.

**Shipped** (`gpu/cli/`). Neighbor install is [`install.md`](install.md);
short commands: [`quickstart.md`](quickstart.md). A one-model metrics dump for
another PC is `scripts/plate.ps1` / `scripts/plate.sh`.

Without installing anything from a checkout that already has CUDA torch:

```powershell
python -m gpu.cli doctor
python -m gpu.cli run --model C:\dev\models\Qwen2.5-3B-Instruct
```

After `pip install -e .` those are `deepfold doctor` and `deepfold run`.
Also shipped: `setup` (venv catch-up; refuses conda `torch-gpu`), `pull`
(allowlisted HuggingFace BF16 ids), `compress`, `chat` (TTY session),
`test`, and `from-ollama` (allowlisted Ollama tags → the same Hub ids;
never `~/.ollama`, never GGUF). Generate ships on Ampere-family CUDA:
**sm_86 is the measured plate** (RTX 3080).
A100 (`sm_80`) and Ada (`sm_89`) **generate as experimental** — allowed, not the
3080 tok/s. Turing, Hopper, Blackwell, ROCm, macOS generate, and CPU torch are
**refused**. The kernel image is `sm_80/sm_86/sm_89` plus PTX `compute_80`.

The lab comparison plate is separate: `python -m gpu.lab.run`, or
`gpu.host.load_model` + `gpu.loop.TokenLoop` by hand. See the
[README](../README.md).

## After a HuggingFace model is on disk

One command:

```text
python -m gpu.cli run --model C:\dev\models\Qwen2.5-3B-Instruct
```

`run` looks for a `.chr` in this order: `--chr`, `$DEEPFOLD_CHR` if that file
exists, `*.nf4.chr` inside the model directory, then siblings of that directory,
then `$DEEPFOLD_HOME/chr/<slug>.nf4.chr`. A sibling is accepted only when the
CHR0 header (`hidden_size` / `num_layers` / `vocab_size`) matches this model's
`config.json`. Ambiguous siblings are not resolved by sort order.
If nothing matches, `run` packs once with `chr compress` (CPU, minutes) and
then loads. `--no-compress` fails instead of packing. First-run compress is
not “two clicks to tokens,” and it is not two clicks on every OS.

The runtime always needs two things: a HuggingFace **directory**
(`config.json`, tokenizer; InternLM2 also remote-code Python) and one
**`.chr`** of packed NF4 weights. At generate time the safetensor shards
are not opened. After a successful compress you may delete the shards to
reclaim 6–40 GB; keep the config and tokenizer.

`chr` reads **BF16/FP16 safetensors**. GGUF is refused (the path is never
opened). Glue families: **llama_swiglu** (measured on Qwen2.5-3B/14B; also
Llama 3.x without qk-norm) and **internlm_gqa** (internlm2.5-20B). Gemma,
Phi-3, MoE, vision, and a live Mistral sliding window are named refusals.
This is not a general HuggingFace or GGUF runtime.

## If the model was pulled with Ollama

Ollama stores **GGUF**. This project does not load GGUF, does not read
`~/.ollama/models/blobs`, and does **not** ship an Ollama plugin.

`python -m gpu.cli from-ollama <tag>` maps **exact** allowlisted library
names (`qwen2.5:3b`, `qwen2.5:3b-instruct`, `qwen2.5:14b`,
`qwen2.5:14b-instruct`) to HuggingFace ids and downloads BF16 safetensors.
It never reads `~/.ollama` and never loads GGUF. `deepfold pull <hf_id>` is
the same table without an Ollama name (`Qwen/Qwen2.5-32B-Instruct` and
`internlm/internlm2_5-20b-chat` are pull/`--hf` only). `llama3.1:8b` stays
unknown until a measured 3080 Llama generate exists. A GGUF path is still
the WAVE 7 blob copy.

Confirm disk (`--yes` or a TTY) before `snapshot_download`. Extra:
`pip install "deepfold[hub]"`. If the tree is already at
`C:\dev\models\Qwen2.5-3B-Instruct`, nothing is pulled. Then:

```text
python -m gpu.cli run --model <that HF dir>
```

Dequantizing Ollama’s Q4/Q5 blob into fake BF16 and compressing that is
not an import path. Unknown tags and arbitrary GGUF files are refused.

## Install

Neighbor machines: [`install.md`](install.md) / [`install.ru.md`](install.ru.md)
(`scripts/setup.ps1` / `scripts/setup.sh`). This 3080 already has conda
`torch-gpu` — do not `deepfold setup` into that env.

No Jupyter. Conda/pip for PyTorch **with CUDA**, then `chr` (PATH Go 1.22+
or a portable Go 1.22 from go.dev during setup), then `doctor` and `run`. The
Ampere kernel
(`sm_80` / `sm_86` / `sm_89` plus PTX `compute_80`) still needs either a built
`chr_nf4_ext` or a host compiler (`cl.exe` / `g++`) plus `nvcc` for a one-time
JIT. On this 3080 the prebuilt sm_86 `.pyd` is enough until you rebuild.

```powershell
conda activate torch-gpu
go build -o chr.exe ./cmd/chr
pip install -e .
python -m gpu.cli doctor
```

On another PC (different NVIDIA Ampere/Ada card, Windows or Linux) the same
commands work without editing sources. Point at *that* machine's HuggingFace
tree; do not expect `C:\dev\models` or the 3080 tok/s.

```text
# Windows
set DEEPFOLD_MODELS=D:\weights
set DEEPFOLD_MODEL=D:\weights\Qwen2.5-3B-Instruct
go build -o chr.exe ./cmd/chr
python -m gpu.cli doctor
python -m gpu.cli run --model %DEEPFOLD_MODEL%

# Linux
export DEEPFOLD_MODELS=$HOME/models
export DEEPFOLD_MODEL=$HOME/models/Qwen2.5-3B-Instruct
go build -o chr ./cmd/chr
python -m gpu.cli doctor
python -m gpu.cli run --model "$DEEPFOLD_MODEL"
```

Doctor exit **3** because the card is not sm_86 is a bug of the old contract.
Ada / A100 must be **0** (or **2** if the install is broken), with
`generate: experimental`. Hopper, Turing, macOS generate remain **3**.
JIT of the fatbinary on a neighbor box takes about a minute the first time.

`pip install -e .` compiles no CUDA: the kernel is a prebuilt sidecar or a JIT
that doctor announces first. We do **not** install the NVIDIA driver, CUDA,
or Visual Studio Build Tools. Missing `chr` is filled by a portable Go 1.22
from go.dev (`$DEEPFOLD_HOME/toolchains`). The default PyPI `torch` is usually the
CPU wheel — doctor catches that and prints the CUDA 12.4 index URL. A red
doctor is the installer working, not a product bug.

`python -m gpu.cli doctor` exit codes: **0** run is possible; **2** this box
could run but the install is broken; **3** generate is refused by this
machine's class (no NVIDIA GPU, macOS, Turing, Hopper, Blackwell, ROCm)
while `chr compress` still works; **1** neither. **3 is not green generate.**
Ada (`sm_89`) and A100 (`sm_80`) are **experimental generate** (exit 0 if
the rest of the install works), not class-3.

On macOS `chr compress` is real and `python -m gpu.cli run` exits 1: there
is no CUDA kernel there. A Mac can pack a `.chr` for a CUDA Ampere/Ada
machine to run. Linux doctor can see a `.so`; there is **no published Linux
generate tok/s**.

A single `pip install` that also drops CUDA, Visual Studio, and Go is not
promised. “Two clicks, all OS” is a lie: click 1 is this package plus
`doctor`; the driver, CUDA torch, compiler, Go, and the HF tree are
user-provided. Ada generate is a fatbinary, not a second plate.

## Not in this repo

- HuggingFace id download of **arbitrary** repos — `pull` / `from-ollama` are an allowlist
- llama.cpp / GGUF drop-in
- OpenAI-compatible HTTP server
- Ollama plugin or runner for existing blobs

The machine of record for numbers remains one RTX 3080 12 GB. Packed
weights stay packed in VRAM for the whole run (that residency is prior art;
the claim is the stack). NF4 is 4.25 bits/weight. That part does not change
with the CLI.

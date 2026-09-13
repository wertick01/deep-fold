# Driver UX

This page is the path for people who already have a model on disk and want the
NF4 driver without a notebook.

**Shipped** (`gpu/cli/`). Without installing anything:

```powershell
python -m gpu.cli doctor
python -m gpu.cli run --model C:\dev\models\Qwen2.5-3B-Instruct
```

After `pip install -e .` those are `deepfold doctor` and `deepfold run`.
Also shipped: `compress` (wraps `chr compress --codec nf4`) and
`from-ollama` (the subcommand exists and **refuses** — Ollama stores GGUF).
There is no `deepfold pull`. Generate ships on Ampere `sm_86` only
(RTX 3080 class); other GPUs are refused with a reason, never a silent
fallback.

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
not “two clicks to tokens.”

The runtime always needs two things: a HuggingFace **directory**
(`config.json`, tokenizer; InternLM2 also remote-code Python) and one
**`.chr`** of packed NF4 weights. At generate time the safetensor shards
are not opened. After a successful compress you may delete the shards to
reclaim 6–40 GB; keep the config and tokenizer.

`chr` reads **BF16/FP16 safetensors**. GGUF is refused (the path is never
opened). Layouts: **qwen2** (Qwen2.5-3B/14B) and **internlm2** fused `wqkv`
(internlm2.5-20B). This is not a general HuggingFace or GGUF runtime.

## If the model was pulled with Ollama

Ollama stores **GGUF**. This project does not load GGUF, does not read
`~/.ollama/models/blobs`, and does **not** ship an Ollama plugin.

`python -m gpu.cli from-ollama <tag>` is a named refusal, not a conversion
and not a lookup: it does not guess a HuggingFace id. The honest path is
still by hand:

1. Take the HuggingFace id that library tag was built from.
2. Download that HuggingFace repo (full 16-bit safetensors — extra disk).
3. `chr compress` → `.chr`, or let `run` pack on first use.
4. `python -m gpu.cli run --model <that HF dir>`.

Dequantizing Ollama’s Q4/Q5 blob into fake BF16 and compressing that is
not an import path. Unknown tags and arbitrary GGUF files are refused.

## Install

No Jupyter. Conda/pip for PyTorch **with CUDA**, `go build` for `chr` until a
binary is attached to a release, then `doctor` and `run`. The Ampere kernel
(`sm_86`, RTX 3080 class) still needs either a built `chr_nf4_ext` or MSVC
Build Tools for a one-time JIT.

```powershell
conda activate torch-gpu
go build -o chr.exe ./cmd/chr
pip install -e .
python -m gpu.cli doctor
```

`pip install -e .` compiles no CUDA: the kernel is a prebuilt sidecar or a JIT
that doctor announces first. We do **not** install the NVIDIA driver, CUDA,
Visual Studio Build Tools or Go, and the default PyPI `torch` is usually the
CPU wheel — doctor catches that and prints the CUDA 12.4 index URL. A red
doctor is the installer working, not a product bug.

`python -m gpu.cli doctor` exit codes: **0** run is possible; **2** this box
could run but the install is broken; **3** generate is refused by this
machine's class (no NVIDIA GPU, macOS, Turing, ROCm, an unmeasured Ampere/Ada)
while `chr compress` still works; **1** neither.

On macOS `chr compress` is real and `python -m gpu.cli run` exits 1: there
is no CUDA kernel there. A Mac can pack a `.chr` for a Windows 3080 to run.

A single `pip install` that also drops CUDA, Visual Studio, and Go is not
promised.

## Not in this repo

- HuggingFace id download (`pull`)
- llama.cpp / GGUF drop-in
- OpenAI-compatible HTTP server
- Ollama plugin or runner for existing blobs

The machine of record for numbers remains one RTX 3080 12 GB. Packed
weights stay packed in VRAM for the whole run (that residency is prior art;
the claim is the stack). NF4 is 4.25 bits/weight. That part does not change
with the CLI.

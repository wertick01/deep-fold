<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/logo_dark_theme_empty_background.png">
    <img alt="deep-fold" src="img/logo_white_theme_empty_background.png" width="420">
  </picture>
</p>

GitHub social preview: [`img/logo_dark_theme_empty_background_1280_x_640.png`](img/logo_dark_theme_empty_background_1280_x_640.png) (1280×640).

# deep-fold

**Local LLM inference with compressed weights, from GPU-resident models to 32B on a 12 GB Ampere card.**

[Method](#1-method) · [Results](#2-results) · [Limits](#3-scope-and-limitations) · [Install](#4-installation) · [Quickstart](#5-quickstart) · [Contribute](#6-reproduce-and-contribute) · [Русский](README.ru.md)

**Status.** The project is in active development. Published generate kernels target **NVIDIA Ampere** (`sm_86`, measured on RTX 3080 12 GB). Ada, other Ampere-family cards, and SM120 (GeForce RTX 50, first remote SKU RTX 5070 Ti) can generate experimentally; Turing, Hopper, and SM100 generate are refused.

## Abstract

Running a language model on a consumer GPU is a problem of both capacity and data movement. Weights must fit alongside the attention cache and working buffers; weights held in system memory must cross a slower link during inference. Quantizing the file is not enough if inference then expands every linear layer back to a full 16-bit matrix.

**deep-fold** is an experimental stack around compressed weight storage. A Go compressor writes NF4 weights to the CHR0 container. Ampere CUDA kernels reconstruct values as they multiply and discard them. Embeddings (`PackedEmbed`) decode only the requested rows. Models that fit stay GPU-resident and use a full greedy-step CUDA graph. Larger models keep a resident subset and stream packed weights from pinned host memory.

On one RTX 3080 12 GB, InternLM2.5-20B hard-12 is **39.0 tokens/s** and **9/12** correct, versus **13.2** and **9/12** for Ollama Q4_K. That gap is mainly placement: deep-fold stays resident; Ollama offloads. On 3B and 14B, Ollama is faster. Qwen2.5-32B uses overflow CopyRing. A separate ignore-EOS 64-token plateau is the long-decode comparison with llama.cpp. The sections below give the method, evidence, and current limits.

## 1. Method

### Keep weights compressed until they are used

Packed codes and scales remain in GPU memory. Linear kernels reconstruct the values needed for the current tile without writing a full dense weight matrix back to device memory.

Each group of 64 weights stores four-bit NF4 codes and one FP16 scale:

$$
b_{\mathrm{weight}} = 4 + \frac{16}{64} = 4.25\ \text{bits}.
$$

For aligned matrices, the code-and-scale payload is about **3.76× smaller than BF16**. Metadata, alignment, norms, biases, activations, and KV cache add to the runtime footprint. Quantization is lossy.

### Use a different kernel for each workload

| Workload | Execution path |
|---|---|
| Prompt processing (*prefill*) | Ampere tensor-core NF4 GEMM (`mma.sync.m16n8k16`); reconstruction into BF16 register fragments; prompt chunks of up to 32 positions |
| Resident generation (*decode*) | N=1 CUDA-core NF4 GEMV inside a full greedy-step CUDA graph |
| Models exceeding the GPU budget | TokenLoop with CopyRing: selected packed matrices stream from pinned host memory through two reusable GPU slots |

Decode V2 keeps token position and valid KV length in device buffers. Its attention kernel reads the valid cache prefix. Before selecting V2, the CLI estimates weights, KV, working buffers, and runtime reserves. Those kernels are written for Ampere tensor cores; other architectures need their own ports.

![The deep-fold stack: CPU packing to CHR0, packed device weights, and tile-local reconstruction](scheme.png)

*Figure 1. Two execution paths share the compressed representation. Resident mode keeps packed matrices on the GPU. Overflow adds pinned host storage and two reusable device slots, then the same GEMM kernel. Embeddings decode requested rows on a separate path.*

### What the project contributes

The project connects a CPU compressor and inspectable file format to packed model loading, Ampere CUDA kernels, resident and overflow execution, a local chat CLI, and reproducible experiments. Chat reuses the KV prefix when the tokenized conversation prefix matches.

NF4 was introduced in [QLoRA](https://arxiv.org/abs/2305.14314). Fused quantized inference also appears in systems such as [Marlin](https://github.com/IST-DASLab/marlin). deep-fold's contribution is its implementation and integration; broader kernel rankings require matched measurements.

Implementation: [CHR0](docs/spec/chr0.md) · [NF4 layout](docs/spec/nf4.md) · [Ampere kernel](docs/kernel-ampere.md) · [GEMM](gpu/nf4/nf4_gemm.cu) · [GEMV](gpu/nf4/nf4_gemv.cu) · [Decode V2](docs/decode-v2.md) · [CopyRing](docs/plan-h2-ring.md).

## 2. Results

**Reference system:** one RTX 3080 12 GB (`sm_86`), Windows/WDDM. Author-reported measurements from this repository. Published results as of **18 September 2026**.

### Answer generation: the hard-12 fixture

Independent requests, a 2,048-token context limit, greedy generation, and up to 256 new tokens. Rates are arithmetic means of the per-request decode rates. Correctness is the number of answers accepted by the fixture's scorer.

| Model | deep-fold V2, tokens/s | Ollama Q4_K, tokens/s | Correct answers: V2 / Ollama |
|---|---:|---:|---:|
| Qwen2.5-3B-Instruct | 190.9 | 197.2 | 8/12 · 7/12 |
| Qwen2.5-14B-Instruct | 54.8 | 66.2 | 11/12 · 10/12 |
| InternLM2.5-20B-Chat | **39.0** | 13.2 | 9/12 · 9/12 |

On 20B the published mean is about **3×** Ollama on this fixture because the packed model stays on the card. Ollama is faster on the 3B and 14B rows. Twelve questions are a small regression check, not a general quality ranking of either quantization scheme.

The updated 20B run uses packed embedding rows: first-request time to first token is **434 ms**, mean first-token latency **552 ms**, after initialization. CHR loading takes **11.46 s**. These are separate timings, not a cold-start guarantee. The 3B and 14B rows are earlier V2 measurements and have not been republished as packed-embedding reruns.

![Hard-12 Decode V2 vs Ollama: correct answers and mean tok/s](docs/img/hard-v2-ollama.png)

*Figure 2. Same 12-item fixture on one RTX 3080. Decode V2 vs Ollama quality and mean decode rate. llama.cpp hard-12 is not published yet. Redraw: `python -m gpu.lab.hard_v2_plate --redraw`.*

Evidence: V2 [3B](docs/runs/hard-decodev2-3b/), [14B](docs/runs/hard-decodev2-14b/), [20B](docs/runs/hard-decodev2-20b/); Ollama [3B](docs/runs/ollama-hard-3b/), [14B](docs/runs/ollama-hard-14b/), [20B](docs/runs/ollama-hard-20b/). See the [lab log](docs/decode-v2-lab.md) for run history.

### Long decode (ignore-EOS 64) and llama.cpp

A different protocol — ignore-EOS, 64 new tokens, `max_seq=2048` — is the long-decode comparison with llama.cpp Q4_K on the same card. Decode V2 reports **197 / 57.5 / 40** tok/s on 3B / 14B / 20B. Exclusive Q4_K long (ctx 2048) is **~187 / 69.9 / 11.87**. Those numbers are not the hard-12 table; V2 still attends the full buffer. Details: [Decode V2 lab](docs/decode-v2-lab.md), [3080 comparison](docs/compare-3080.md).

![Decode V2 resident 3B/14B/20B on one RTX 3080](docs/img/decodev2-3080.png)

*Figure 3. Resident Decode V2 (GEMV graph + MMA prefill-32) next to exclusive Q4_K long plates. Headline V2: 3B **197**, 14B **57.5**, 20B **40**. Redraw: `python -m gpu.lab.decodev2_plate --redraw`.*

### Crossing the memory limit: 32B

Qwen2.5-32B uses about **16.2 GiB of packed weight payload**. In the recorded configuration, roughly 9.5 GiB of weights stay on the GPU and 6.7 GiB are held in pinned host memory. CopyRing stages the host matrices for computation on the GPU.

On the separate 64-token generation run, deep-fold reports **2.49 decode steps/s**, or **2.53 tokens/s** when normalized to the server-style token counter. Ollama and llama.cpp auto-fit each report **2.54 tokens/s**. On hard-12, the published means are **2.12** for deep-fold and **3.1** for Ollama, so the tie does not extend to every workload. [32B protocol](docs/eval-32b.md).

![Matched 32B/3B decode on one RTX 3080](docs/img/compare-3080.png)

*Figure 4. Quote 32B **long** plateaus. Ollama 2.54 vs llama.cpp auto-fit 2.54 vs H2 2.53 is CPU-suffix vs CPU-suffix vs PCIe CopyRing, not a kernel ranking. 3B Decode V2 **197** sits next to Q4_K **~187**. Redraw: `python -m gpu.lab.compare_plate --redraw`.*

## 3. Scope and limitations

- **Measured platform:** RTX 3080 12 GB, Windows. Ampere `sm_86` is the published generate path. `sm_80`, `sm_87`, and `sm_89` are accepted experimentally and unbenchmarked. SM120 (GeForce RTX 50, first remote SKU RTX 5070 Ti) is experimental generate — no plate yet. Turing, Hopper, and datacenter Blackwell (SM100) generate are refused. Linux install works; Linux tokens/s have not been published.
- **Generation:** greedy decoding. Context length consumes KV memory; the capacity estimate uses fixed reserves and cannot guarantee a fit under arbitrary competing GPU load.
- **Model coverage:** the public download allowlist contains Qwen2.5 3B, 14B, and 32B Instruct, plus InternLM2.5-20B-Chat. Other checkpoints require validation.
- **Input format:** compression starts from unquantized Hugging Face safetensors. GGUF import is not implemented. VQ 2-bit remains experimental and is excluded from automatic codec selection.
- **Evidence:** broad quality evaluation, full raw artifacts for every V2 run, repeated-session stability tests, and independent hardware replication remain open work.

## 4. Installation

Start with **Python 3.11 or 3.12, Git, and an NVIDIA driver**. The measured GPU capability is `sm_86`. CPU compression is separate and does not need a supported generate GPU.

| GPU / platform | Current status |
|---|---|
| RTX 3080 12 GB, `sm_86`, Windows | Measured configuration |
| Other `sm_86` hardware | Accepted capability; no matching performance claim |
| `sm_80`, `sm_87`, `sm_89` | Experimental generation |
| SM120 (GeForce RTX 50; first remote SKU RTX 5070 Ti) | Experimental generation; no named plate |
| Turing, Hopper, SM100 | Generation refused |
| AMD, macOS, CPU-only | No generate path; CPU compress is separate |

### Windows / PowerShell

```powershell
git clone https://github.com/wertick01/deep-fold.git
cd deep-fold
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
.\.venv\Scripts\Activate.ps1
deepfold doctor
```

### Linux / Bash

```bash
git clone https://github.com/wertick01/deep-fold.git
cd deep-fold
bash scripts/setup.sh
source .venv/bin/activate
deepfold doctor
```

Setup installs the CLI and CUDA PyTorch, prepares the Go compressor, and builds or locates the CUDA extension. A source build needs CUDA 12.4 (Ampere) or 12.8 (RTX 50) and a C++ toolchain; Windows setup can attempt their installation.

Full requirements and troubleshooting: [English installation guide](docs/install.md) · [Русская инструкция](docs/install.ru.md). If the command is unavailable, use `python -m gpu.cli` in the activated environment.

## 5. Quickstart

Qwen2.5-3B is the smallest supported starting point. Allow space for its approximately 6.2 GB source checkpoint plus the compressed file.

```text
deepfold pull Qwen/Qwen2.5-3B-Instruct --dir ./models/Qwen2.5-3B-Instruct --yes
deepfold compress --in ./models/Qwen2.5-3B-Instruct --out ./models/qwen25-3b.nf4.chr --codec nf4
deepfold chat --model ./models/Qwen2.5-3B-Instruct --chr ./models/qwen25-3b.nf4.chr
```

For a single response:

```text
deepfold run --model ./models/Qwen2.5-3B-Instruct --chr ./models/qwen25-3b.nf4.chr --prompt "Explain why the sky is blue." --max-new-tokens 128
```

`--executor auto` selects Decode V2 when the loaded model and estimated memory budget permit it; other supported cases use TokenLoop. `--max-seq` defaults to 512 for `run` and 2048 for `chat`. With `--agent` on a 12 GB resident 14B card the default is 4096 unless the flag is set.

Keep the model configuration, tokenizer, and any required remote-code files alongside the source model directory. The `.chr` supplies the packed weights. InternLM additionally requires `python -m pip install -e ".[internlm]"`.

In chat, `/help` lists commands, `/stats` shows timings, `/clear` resets the conversation, and `/chats` resumes saved conversations. Later turns prefill only the new suffix when the chat-template prefix matches. Optional `--agent` mode adds workspace tools (grep, patch, tests, allowlisted argv) with an approval policy. See [chat and agent documentation](docs/spec/agent.md) and [web-search configuration](docs/web-search.md).

More examples: [Quickstart](docs/quickstart.md) · [Быстрый старт](docs/quickstart.ru.md) · [Model notes](docs/models.md).

## 6. Reproduce and contribute

The project is early. The occupancy, graphs, and MMA/GEMV paths are Ampere-specific. **Contributors who can port those kernels to Ada, Hopper, or Blackwell, and people who can run the same plates on another GPU or on Linux, are the highest-leverage help right now.** Independent Ampere replication is equally useful.

Open an issue or pull request on [GitHub](https://github.com/wertick01/deep-fold). A useful report includes `deepfold doctor` output, the checkout commit, GPU name and CUDA capability, OS, and (when generate ran) the full plate directory — not a single tokens/s number.

To record a local installation and smoke run:

```powershell
# Windows
powershell -File scripts/plate.ps1 3b --out ./runs/local-3b
```

```bash
# Linux
bash scripts/plate.sh 3b --out ./runs/local-3b
```

The runner can download and compress the model. It records the machine, checkout, and smoke results; it does not reproduce the hard-12 comparison by itself. Keep the entire output directory.

For controlled comparisons, record the commit, checkpoint revision, engine versions, prompt and output token counts, context limit, placement, warmup, and timing definitions. Report startup, first-token latency, and decode speed separately. Do not fold CUDA-graph capture into the warmup timer. [Laboratory guide](docs/lab.md) · [Hard-12 evaluation](docs/eval-hard-qwen25.md) · [Committed runs](docs/runs/).

| Area | Start here |
|---|---|
| Architecture ports and doctor gates | [Ampere kernel](docs/kernel-ampere.md), [CLI probe](gpu/cli/main.py) |
| Compression and verification | [Go CLI](cmd/chr/), [CPU round trip](docs/cpu-roundtrip.md) |
| Resident inference | [Decode V2](gpu/decodev2/), [design](docs/decode-v2.md) |
| Overflow execution | [CopyRing](gpu/loop/ring.py), [recorded 32B data path](docs/runs/h2-qwen25-32b/data_path.md) |
| CUDA arithmetic | [NF4 kernels and checks](gpu/nf4/) |
| CLI and interaction | [CLI source](gpu/cli/), [UX notes](docs/ux.md) |

Numerical checks and end-to-end generation timings should accompany performance changes.

## License

[MIT](LICENSE). Downloaded models retain their own licenses.

Developed with [Cursor](https://cursor.com).

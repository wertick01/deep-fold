<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/logo_dark_theme_empty_background.png">
    <img alt="deep-fold" src="img/logo_white_theme_empty_background.png" width="420">
  </picture>
</p>

# deep-fold

**Abstract.** The sixteen-bit weights of a 14- or 20-billion-parameter language
model occupy about 28–38 GiB — two to three times a 12 GiB graphics card. Writing
the same weights in fewer bits does not, by itself, solve the problem. If each
layer is expanded back to sixteen bits before the matrix multiply, a full-size
copy of that layer exists in video memory at the peak, and the card's occupancy
is still governed by the original size. The overflow is served from system memory
over the bus; generation then slows to a crawl.

This work does not propose a new numerical code. The four-bit representation
used here (NF4: sixteen reconstruction levels, groups of 64, one scale factor
per group) belongs to the same family as bitsandbytes / QLoRA. Reconstructing
values inside the matrix multiply is also known, for example in Marlin and in
some AWQ kernels.

What is new here is not the code, but **where the weights live for the entire
run**. They are packed once, on the CPU, into a single `.chr` file. On the GPU
the only stored form of a linear layer is that packed table: the module has no
dense weight matrix, so nothing can silently recreate a full-precision layer.
During each multiply the processor reads a small tile of packed values,
reconstructs them in registers (the smallest and fastest storage on the chip),
multiplies, and discards the reconstructed numbers. In video memory the layer
remains packed from load to exit. The model's footprint on the card is
therefore the compressed size, not the sixteen-bit size. A complete layer never
exists on the device at any instant.

Measured on one RTX 3080 12 GB: Qwen2.5-14B-Instruct (~28 GiB of 16-bit
weights) generates at about 6.6 tokens per second; internlm2.5-20B (~38 GiB)
at about 5.0. The same 14B model in sixteen bits, spilling off the card, is 0.92.
When both copies fit (3B), the uncompressed path is faster. Compression pays when
the uncompressed model does not fit.

Русская версия: [README.ru.md](README.ru.md).

## The problem

A transformer is largely a stack of linear layers, and each linear layer is a
rectangular table of numbers: its weights. Model files normally store those
numbers in **BF16** — bfloat16, a 16-bit floating-point format, two bytes per
weight. A 14-billion-parameter model therefore needs roughly 28 GiB for its
weights alone, and a 20-billion-parameter model roughly 38 GiB. This card has
12 GiB. The weights miss the card by a factor of two to three.

Storing the weights in fewer bits is a necessary step but not a sufficient one.
The usual way to compute with compressed weights is to convert a layer back into
a full-size 16-bit table and then call an ordinary matrix multiply on it. At the
instant of that conversion the full-size table exists in video memory again, so
the peak memory the card has to supply is still governed by the 16-bit size, not
by the compressed size. When that peak exceeds the card, the driver satisfies it
from system memory across the PCI Express bus, and generation slows to a crawl
(measured below: 0.92 tokens per second on 14B).

## Contribution: what was built

A complete path from a 16-bit model file to generated tokens in which no
full-size copy of any layer is ever created on the card. It has four parts.

1. **A compressor that runs on the CPU.** `chr`, written in Go, reads a
   HuggingFace model directory of BF16 safetensors and writes one file with the
   extension `.chr`, whose container format is identified by the four characters
   **CHR0**. The weights are stored in **NF4**: each weight is replaced by one
   of 16 reconstruction levels, weights are handled in groups of 64, and each
   group keeps one scale factor of its own. Including the scale factors this
   costs about 4.5 bits per weight instead of 16.
2. **A replacement for PyTorch's `torch.nn.Linear` that cannot hold a dense
   weight matrix.** `gpu.host.CompressedLinear` has no `[out, in]` weight tensor
   among its parameters or buffers at all; the packed codes and the group scale
   factors are its only resident weight storage. This is a structural property,
   not a policy: `model.to("cuda")` has no full-precision weight to move, so no
   ordinary PyTorch operation can quietly bring a full-size layer into video
   memory.
3. **A matrix-multiply kernel for NVIDIA Ampere GPUs** (compute capability 8.6,
   which is what the RTX 3080 is) that multiplies directly against the packed
   representation. For each small block of the weight table — a *tile*, some
   kilobytes — the kernel reads the packed codes and the matching group scale
   factors, reconstructs those few thousand weight values in the processor's
   registers, which are the smallest and fastest storage on the chip (a few
   hundred bytes per thread), multiplies them against the incoming activations,
   adds the products into a running accumulator, and then discards the
   reconstructed values. The layer's copy in video memory is only ever read, so
   it remains packed from load to exit. Activations, the attention cache, the
   normalization layers, and the output projection stay in BF16: the weights are
   compressed, not the text being processed.
4. **A generation loop that uses those layers.** `gpu.loop.TokenLoop` drives the
   transformer's layer graph itself, processing the prompt in chunks of at most
   16 positions. On the compressed path it does not call
   `transformers.generate`.

**Why this matters.** Because no second, full-precision copy of a layer is ever
materialized, the memory a model occupies on the card is set by the size of its
*compressed* weights rather than by the size of its original 16-bit weights. A
14B model with 28,172 MiB of 16-bit weights and a 20B model with 37,882 MiB both
load onto a 12 GiB card and generate text there, at 7,483 MiB and 10,062 MiB of
packed weights respectively. Compression is not a free speedup: on a 3B model
both forms fit on this card, and there the uncompressed path is the faster one
(23.1 against 17.0 tokens per second). The result is about which models can run
at all on a given card, and at what rate once they do.

### The mental model to discard

The description above is often read as: unpack a layer into video memory, do the
arithmetic, then pack it again. That is not what happens, and it would defeat
the purpose, since the unpacked layer would be exactly the object that does not
fit. Reconstruction is local to the arithmetic of one tile of one matrix
multiply, it takes place in registers rather than in video memory, and its
results are thrown away when the tile is finished. The packed weights in video
memory are read-only for the whole run, and there is no point in time at which a
full-size copy of a layer exists on the card. There is also no swapping of
layers in and out over the bus.

## What is adopted from earlier work, and what is original here

Stating this precisely matters more than sounding novel.

**Adopted, and not claimed as new:**

- **The number format.** NF4 as used here is the same family as the format in
  `bitsandbytes` and QLoRA: 16 reconstruction levels, groups of 64, one scale
  factor per group, about 4.5 bits per weight. No new code was designed and no
  information-theoretic claim is made about it.
- **The idea of reconstructing weights inside the matrix multiply.** Marlin and
  several AWQ kernels do this too. No claim is made that this project is faster
  than those kernels; they were not benchmarked here.

**Original in this repository:**

- The `.chr`/CHR0 container and its Go compressor.
- `CompressedLinear`, which is constructed so that a dense weight matrix is not
  representable in it.
- The Ampere kernel itself.
- `TokenLoop`, and the measurements below.

The claim is the complete, working stack — container, host-side layer, GPU
kernel, and end-to-end measurements on a single 12 GiB card — under which packed
weights stay resident and packed. It is not a claim about the format.

**Compression is lossy.** NF4 is a four-bit approximation of the original
weights. The check below confirms that the compressed model still answers three
questions correctly; it is a smoke test, not a measurement of quality, and no
accuracy claim is derived from it.

## The same thing in implementation terms

For readers who want the names. `.chr` (container `CHR0`) stores NF4-quantized
weights: group size 64, one FP16 scale per group, 4.5 bits/weight effective.
`gpu.host.CompressedLinear` holds the packed code words and scales as its only
buffers; by construction it has no `[out, in]` BF16 parameter, so no dense copy
can be materialized by accident. `chr_nf4_gemm` is an `sm_86` kernel in which
reconstruction is part of the general matrix multiply (GEMM) rather than a
separate pass over memory: each block stages a tile of packed codes through
shared memory (the small per-block scratchpad on the chip), decodes it into BF16
fragments in registers, feeds those fragments to the tensor-core matrix
instructions, and accumulates in FP32. Reconstructed values live only in
registers for the lifetime of a tile and are never written back to global memory
(the card's main video memory), so the resident weight footprint equals the
packed footprint for the whole run. Activations, the KV cache, norms, and the LM
head stay BF16. `gpu.loop.TokenLoop` drives the layer graph directly, with
prefill in chunks of at most 16 positions; it does not call
`transformers.generate`.

## Measured on an RTX 3080 12 GB

Greedy decoding, `max_new_tokens = 64`, three fixed prompts, independent turns
(the attention cache is reset between prompts, so every measurement of the time
to the first token is a clean prompt-processing pass). Each codec runs in **its
own process**, which exits before the other starts, because 12 GB cannot hold
both copies. `nvidia-smi` figures include the CUDA context and the Windows
desktop; nothing is subtracted.

Two metrics in the tables need a definition. **Mean TTFT** is the mean
time-to-first-token: how long it takes from submitting a prompt until the first
generated token appears, which is the prompt-processing (prefill) stage. **Mean
decode tok/s** is the rate at which tokens are produced after that first one.

Weights and `.chr` files are not in git. The lab's CSVs and figures are, under
[`docs/runs/`](docs/runs/).

### Qwen2.5-3B-Instruct — both codecs fit, and dense BF16 is faster

![Qwen2.5-3B-Instruct: two video-memory graphs side by side, BF16 versus NF4, on the same 0–12288 MiB scale](docs/img/lab-qwen25-3b.png)

| | BF16 (HF `generate`) | NF4 (`CompressedLinear` + `TokenLoop`) |
|---|---:|---:|
| Weight MiB | 5,886 | 1,563 |
| nvidia-smi after load (MiB) | 7,477 | 3,142 |
| Peak nvidia-smi (MiB) | 7,535 | 3,286 |
| Mean TTFT (ms) | 52 | 212 |
| Mean decode tok/s | 23.1 | 17.0 |
| Smoke (Paris / Berlin / 323) | pass | pass |

Source: [`docs/runs/qwen25-3b/`](docs/runs/qwen25-3b/) — `summary.csv`,
`messages.csv`, `timeline.csv`, and the interactive
[`lab.html`](docs/runs/qwen25-3b/lab.html).

A 3B model fits either way on this card, and there dense BF16 wins on speed:
**23.1 against 17.0 tokens per second**. Compression only pays for itself when
the uncompressed model does not fit.

### Qwen2.5-14B-Instruct — BF16 spills off the card, NF4 stays on it

![Qwen2.5-14B-Instruct: the nvidia-smi pair plus a second pair of graphs for the CUDA working set including shared GPU memory](docs/img/lab-qwen25-14b.png)

| | BF16 (HF `generate`) | NF4 (`CompressedLinear` + `TokenLoop`) |
|---|---:|---:|
| Weight MiB | 28,172 | 7,483 |
| nvidia-smi after load (MiB) | 11,955 | 8,913 |
| Peak nvidia-smi (MiB) | 11,997 | 9,356 |
| torch after load (MiB) | 28,270 | 7,539 |
| Mean TTFT (ms) | 1,028 | 759 |
| Mean decode tok/s | 0.92 | 6.56 |
| Smoke (Paris / Berlin / 323) | pass | pass |

Source: [`docs/runs/qwen25-14b/`](docs/runs/qwen25-14b/).

Read this table carefully, because the obvious comparison is the wrong one.
`nvidia-smi` reports only dedicated video memory and therefore **stops at
12288 MiB**. So `11,955` for BF16 does not mean the model fit: it means the
meter ran out of scale. What BF16 actually occupied after load is the CUDA
working set torch reports, about **28,270 MiB**, some 16,300 MiB of which is
Windows shared GPU memory — system RAM reached over the bus. NF4 stayed inside
the card at about 7,539 MiB. **Do not quote 11,955 against 8,913 as the
result.** The result is panel F of the figure: about 28,270 MiB against about
7,539 MiB.

The speed numbers flip here for the same reason. BF16 decodes at 0.92 tokens
per second because most of every weight has to cross the bus for every token,
while the packed NF4 path reaches 6.56 — a bit over seven times faster. This
is not a kernel-versus-kernel result; it is what happens when one side fits and
the other does not.

### internlm2_5-20b-chat — 20B answers on a 12 GB card, BF16 has no baseline

![internlm2_5-20b-chat: the nvidia-smi pair plus a second pair of graphs for the CUDA working set including shared GPU memory](docs/img/lab-internlm20b.png)

| | BF16 (HF `generate`) | NF4 (`CompressedLinear` + `TokenLoop`) |
|---|---:|---:|
| Weight MiB | 37,882 | 10,062 |
| nvidia-smi after load (MiB) | 11,892 | 11,578 |
| Peak nvidia-smi (MiB) | 11,976 | 11,828 |
| torch after load (MiB) | 37,882 | 10,273 |
| KV cache MiB (preallocated, `max_seq=512`) | — | 96 |
| Load seconds | 54.0 | 19.3 |
| Mean TTFT (ms) | — | 605 |
| Mean decode tok/s | — | 5.01 |
| Smoke (Paris / Berlin / 323) | not run | pass |

Source: [`docs/runs/internlm20b/`](docs/runs/internlm20b/).

Read this table the same way as 14B. Both `nvidia-smi` traces sit near
12288 MiB. **Do not quote 11,976 against 11,828 as the result.** The result
is the CUDA working set after load: **37,882 MiB against 10,273 MiB**, that is
roughly 37 GiB against roughly 10 GiB. 11,976 only means the dedicated-VRAM
meter ran out of scale; the rest of BF16 sits in Windows shared GPU memory. NF4
decode is 5.01 tok/s with a smoke pass.

This is the first table here with a **`—` column instead of a baseline**, and
the dashes are the honest part. Both halves need reading separately.

**NF4 ran.** 48 layers of packed weights went on the card at 10,062 MiB and the
model answered all three prompts correctly — `Paris is the capital of France.`,
Berlin, `17 * 19 = 323` — at 605 ms mean prefill and 5.01 tokens per second. No
full-size 16-bit weight matrix exists anywhere on that path. Peak `nvidia-smi`
was 11,828 of 12,288 MiB, so a 20B model fits on this card with roughly 460 MiB
of headroom and nothing to spare. That headroom is why extra codecs were not
started; it is not a comparison against BF16's capped meter.

**BF16 has no decode numbers, and none are invented.** The load itself is a
real measurement: torch reserved **37,882 MiB** for the dense weights, about
26,000 MiB of that in Windows shared GPU memory, three times the card.
Generation never ran. `internlm2_5-20b-chat` ships `trust_remote_code` modules
written against transformers 4.41, and on transformers 5 its
`prepare_inputs_for_generation` raises `TypeError: can only concatenate tuple
(not "int") to tuple`. That is recorded in `summary.notes` as a miss.

It is left unpatched deliberately. Substituting the stock
`GenerationMixin.prepare_inputs_for_generation` does make `generate` run, but
transformers 5 no longer passes `cache_position`, so the model decodes fluent
repetition instead of an answer. That would be a fake baseline. The row stays
empty: recorded miss, not a crashed cell.

- InternLM2 keeps its query, key, and value projections in a single combined
  `wqkv` tensor instead of Qwen's separate `q`/`k`/`v`; the compressor
  classifies it as `kind=qkv` and `TokenLoop` splits it per KV head.
- HuggingFace remote code needs `einops` and `sentencepiece==0.1.99` (0.2.2
  cannot load InternLM's tokenizer), and transformers 5 must be steered to the
  slow tokenizer class.
- The stop set is read from the tokenizer, not hardcoded: InternLM2 keeps
  `eos_token` at `</s>` (2) while its chat template closes a turn with
  `<|im_end|>` (92542). Missing 92542 is why an early run talked past the end
  of every answer.

### Hard eval — 3B and 14B, twelve reasoning items

Smoke (Paris / Berlin / 323) is not a quality score. The figure below is a
separate plate: twelve independent reasoning items, greedy,
`max_new_tokens = 256`, the same prompts on both codecs, isolated workers.

![Hard eval on one RTX 3080 12 GB: CUDA working set, nvidia-smi peak, TTFT, decode tok/s, accuracy, and scored items for Qwen2.5-3B and 14B, BF16 versus NF4](docs/img/hard-eval-qwen25.png)

*Figure. Hard eval on one RTX 3080 12 GB — Qwen2.5-3B-Instruct and
Qwen2.5-14B-Instruct, dense BF16 against the NF4 driver. Order: BF16 3B →
BF16 14B → NF4 3B → NF4 14B. Panel A is the CUDA working set (14B BF16 at
28,490 MiB, above the card); panel B is `nvidia-smi`, which stops at 12,288 MiB
and is not the 14B result. Accuracy is 7/12, 9/12, 8/12, 10/12. Full
questions, gold answers, raw replies, what missed and why, and load / prefill
/ decode time per model and per message:
[`docs/eval-hard-qwen25.md`](docs/eval-hard-qwen25.md). CSVs:
[`docs/runs/hard-qwen25/`](docs/runs/hard-qwen25/).*

### The chat script

Identical for both codecs, in all three models:

```
Reply with one short sentence. What is the capital of France?
And the capital of Germany?
What is 17 times 19? Reply with the number only.
```

The check is a case-insensitive search for `paris` / `париж`, `berlin` /
`берлин`, and `323`. Three prompts and 64 tokens is a smoke test that the
compressed model still speaks and still does arithmetic. It is **not** a
quality benchmark, and no accuracy claim is made from it. Method:
[`docs/lab.md`](docs/lab.md).

## Honesty

- **Memory after load is the product metric.** Packed NF4 must sit well below
  dense BF16 once the model is loaded. If it sits near the BF16 line, the
  driver materialized a layer somewhere — that is a bug, not a win.
- **On 14B and 20B the `nvidia-smi` meter is capped** at the card's 12288 MiB.
  The honest figure is the CUDA working set (`torch.cuda.memory_reserved`),
  which counts dedicated video memory plus shared GPU memory. Panel F, not
  panel A, is the claim. On 20B both codecs sit near that cap: do not quote
  11,976 against 11,828. The 20B result is 37,882 MiB against 10,273 MiB, plus
  NF4 at 5.01 tok/s with a smoke pass. There is no BF16 speed baseline.
- **Decode tokens per second compare two different stacks:** HuggingFace
  `generate` with dense BF16 matrix multiplies on one side, our NF4 loop with
  reconstruction inside the multiply on the other. Both are reported, in both
  directions. On 3B, dense BF16 is faster. On 14B, NF4 is far ahead, and the
  reason is that BF16 has already spilled into system RAM. On 20B NF4 is
  5.01 tok/s; there is no BF16 generate, so there is no speed comparison.
- **Time-to-first-token is prompt processing,** and the two sides do it
  differently: BF16 uses the HuggingFace path, NF4 uses chunks of at most 16
  positions.
- **Display memory is inside the `nvidia-smi` reading.** It is real, it is on
  the same card, and it is not subtracted away here.
- **The lab can emit a synthetic figure** (`--dry-plot`), whose CSVs are
  stamped `FIXTURE`. Nothing in the tables above comes from that fixture.

## Build the compressor

Pure CPU, no GPU needed. The tests write tiny safetensors files of their own and
never download a model.

```bash
go test ./...
go build -o chr ./cmd/chr
```

Then compress a local HuggingFace tree — BF16 safetensors, not GGUF and not a
model that is already quantized:

```text
chr compress --in <model-dir> --out <model>.nf4.chr --codec nf4
```

Round-trip verification is described in
[`docs/cpu-roundtrip.md`](docs/cpu-roundtrip.md). Group scales so small that
they would flush to zero in float16 are encoded as a scale of `1`; see
[`internal/nf4`](internal/nf4/).

## Run the lab

Defaults point at the author's Windows layout and can be overridden:

| Variable | Default on this machine |
|---|---|
| `DEEPFOLD_MODEL` | `C:\dev\models\Qwen2.5-3B-Instruct` |
| `DEEPFOLD_CHR` | `C:\dev\models\qwen25-3b.nf4.chr` |
| `DEEPFOLD_RUNS` | `C:\dev\models\runs` |

```powershell
conda activate torch-gpu
cd <this-repo>
python -m gpu.lab.run
```

Notebooks: [`notebooks/03_codec_lab.ipynb`](notebooks/03_codec_lab.ipynb) for
3B, [`notebooks/04_qwen25_14b_lab.ipynb`](notebooks/04_qwen25_14b_lab.ipynb) for
14B, [`notebooks/05_internlm20b_lab.ipynb`](notebooks/05_internlm20b_lab.ipynb)
for 20B. Restart the kernel before Run All, and do not
`import torch` in the comparison kernel itself — the child processes load the
weights. CUDA's just-in-time compiler cannot find `cl.exe` when it is launched
from Jupyter, so `gpu.win_toolchain` injects `vcvars64.bat` into the process.

Without a GPU you can still draw the figure. The fixture numbers are not
publishable:

```powershell
python -m gpu.lab.run --out <dir> --dry-plot
```

Or redraw a committed plate from its CSVs:

```powershell
python -c "from gpu.lab import comparison_figure; comparison_figure(r'docs/runs/qwen25-3b')"
```

## Docs

The design notes under [`docs/spec/`](docs/spec/) are in Russian. This page and
the lab writeup are in English.

| | |
|---|---|
| Lab method and how to read the figure | [docs/lab.md](docs/lab.md) |
| Hard eval (3B/14B questions, replies, times) | [docs/eval-hard-qwen25.md](docs/eval-hard-qwen25.md) |
| Memory budget on a 3080 12 GB | [docs/vram-3080.md](docs/vram-3080.md) |
| CPU compress and verify | [docs/cpu-roundtrip.md](docs/cpu-roundtrip.md) |
| Which models to download | [docs/models.md](docs/models.md) |
| Russian README | [README.ru.md](README.ru.md) |

## License

[MIT](LICENSE).

This project was created with [Cursor](https://cursor.com).

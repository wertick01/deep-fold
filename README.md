<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/logo_dark_theme_empty_background.png">
    <img alt="deep-fold" src="img/logo_white_theme_empty_background.png" width="420">
  </picture>
</p>

# deep-fold

**Abstract.** The sixteen-bit weights of a 14-, 20-, or 32-billion-parameter
language model occupy about 28–65 GiB — two to five times a 12 GiB graphics card.
Writing the same weights in fewer bits does not, by itself, solve the problem. If
each layer is expanded back to sixteen bits before the matrix multiply, a
full-size copy of that layer exists in video memory at the peak, and the card's
occupancy is still governed by the original size. The overflow is served from
system memory over the bus; generation then slows to a crawl.

This work does not propose a new numerical code. The four-bit representation
used here (NF4: sixteen reconstruction levels, groups of 64, one scale factor
per group) belongs to the same family as bitsandbytes / QLoRA. Reconstructing
values inside the matrix multiply is also known, for example in Marlin and in
some AWQ kernels. Keeping the packed form resident for a whole run is not new
either: bitsandbytes' `Linear4bit` and the W4A16 kernels do that as well.

What this repository offers is not a first but a complete **stack**, organised
around **where the weights live for the entire run** and measured end to end on
one consumer card. Packed residency itself is prior art (`Linear4bit`, Marlin,
AWQ W4A16); the claim is the stack, not that idea. The weights are packed once,
on the CPU, into a single `.chr` file. On the GPU the only stored form of a
linear layer is that packed table:
the module has no dense weight matrix, so nothing can silently recreate a
full-precision layer.
During each multiply the processor reads a small tile of packed values,
reconstructs them in registers (the smallest and fastest storage on the chip),
multiplies, and discards the reconstructed numbers. In video memory the layer
remains packed from load to exit. The model's footprint on the card is
therefore the compressed size, not the sixteen-bit size. When that compressed
file is still larger than the card, only a resident subset plus two copy slots
sit in HBM; the tail stays packed in pinned host RAM. A complete dense layer
never exists on the device at any instant.

Measured on one RTX 3080 12 GB: Qwen2.5-14B-Instruct (~28 GiB of 16-bit
weights) generates at about 6.6 tokens per second; internlm2.5-20B (~38 GiB)
at about 5.0. The same 14B model in sixteen bits, spilling off the card, is 0.92.
Qwen2.5-32B packed NF4 is still ~16.6 GiB and does not fit; the overflow path
streams a pinned host tail and generates at 2.31 tokens per second. When both
copies fit (3B), packed decode is now faster on this card; time to the first
token is still slower. Compression pays when the uncompressed model does not fit.

![deep-fold: persistent packed weights for LLM inference — CPU packs NF4 into a CHR0 file, CompressedLinear holds packed codes and group scales in VRAM, each GEMM reconstructs a tile in registers and discards it](scheme.png)

*Figure. Weights are packed once on the CPU into a `.chr` file. On the GPU a linear layer holds only packed codes and group scales; each multiply reconstructs a tile in registers and discards it. A dense layer is never resident in video memory.*

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
   costs 4.25 bits per weight instead of 16 (4-bit codes plus one FP16 scale
   per group of 64: 4 + 16/64).
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
   32 positions. On the compressed path it does not call
   `transformers.generate`. When the packed table fits (3B, 14B, 20B), those
   weights stay on the card. When it does not (Qwen2.5-32B on 12 GB),
   `CopyRing` copies overflow matrices from a pinned host image into two static
   device slots; `chr_nf4_gemm` still reconstructs in registers. There is still
   no dense `[M,K]` copy in video memory. `--codec auto` never picks VQ.

**Why this matters.** Because no second, full-precision copy of a layer is ever
materialized, the memory a model occupies on the card is set by the size of its
*compressed* weights rather than by the size of its original 16-bit weights. A
14B model with 28,172 MiB of 16-bit weights and a 20B model with 37,882 MiB both
load onto a 12 GiB card and generate text there, at 7,483 MiB and 10,062 MiB of
packed weights respectively. A 32B model with ~16,599 MiB of packed NF4 still
misses the card; 9,716 MiB stay resident and 6,885 MiB stream from pinned host
memory, at 2.31 tokens per second. Compression is not a free speedup: on a 3B model
both forms fit on this card. Decode on the packed path is now ahead (28.7
against 24.8 tokens per second on a same-session pair); time to the first token is
still slower (92 against 48 ms). The result is about which models can run
at all on a given card, and at what rate once they do.

### The mental model to discard

The description above is often read as: unpack a layer into video memory, do the
arithmetic, then pack it again. That is not what happens, and it would defeat
the purpose, since the unpacked layer would be exactly the object that does not
fit. Reconstruction is local to the arithmetic of one tile of one matrix
multiply, it takes place in registers rather than in video memory, and its
results are thrown away when the tile is finished. The packed weights in video
memory are read-only for the whole run, and there is no point in time at which a
full-size copy of a layer exists on the card. On the all-resident path (3B, 14B,
20B) there is no swapping of layers over the bus. On 32B the packed table itself
does not fit: whole **packed** matrices move H2D into two slots. That is not
unpack-layer-into-VRAM. A dense layer is still never resident.

## What is adopted from earlier work, and what is original here

Stating this precisely matters more than sounding novel.

**Adopted, and not claimed as new:**

- **The number format.** NF4 as used here is the same family as the format in
  `bitsandbytes` and QLoRA: 16 reconstruction levels, groups of 64, one scale
  factor per group, 4.25 bits per weight. No new code was designed and no
  information-theoretic claim is made about it.
- **The idea of reconstructing weights inside the matrix multiply.** Marlin and
  several AWQ kernels do this too. No claim is made that this project is faster
  than those kernels; they were not benchmarked here.
- **Keeping the weights packed for the whole run.** `bitsandbytes.Linear4bit`,
  Marlin and AWQ W4A16 all hold packed weights in video memory and unpack
  inside the GEMM. This project is **not** the first to keep weights packed and
  does not claim to be.

**Original in this repository:**

- The `.chr`/CHR0 container and its Go compressor.
- `CompressedLinear`, which is constructed so that a dense weight matrix is not
  representable in it.
- The Ampere kernel itself.
- `TokenLoop`, the overflow `CopyRing` / pinned `HostImage`, and the
  measurements below.

The claim is the complete, working stack — container, host-side layer, GPU
kernel, and end-to-end measurements on a single 12 GiB card — under which packed
weights stay resident and packed, with a layer type that cannot grow a dense
weight matrix. It is not a claim about the format, and not a claim of priority
on packed residency.

**Compression is lossy.** NF4 is a four-bit approximation of the original
weights. The check below confirms that the compressed model still answers three
questions correctly; it is a smoke test, not a measurement of quality, and no
accuracy claim is derived from it.

## The same thing in implementation terms

For readers who want the names. `.chr` (container `CHR0`) stores NF4-quantized
weights: group size 64, one FP16 scale per group, 4.25 bits/weight effective.
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
prefill in chunks of at most 32 positions; it does not call
`transformers.generate`. When packed NF4 exceeds the card,
`gpu.host.HostImage` and `gpu.loop.CopyRing` stream HOST matrices into two
static slots; the HBM weight number is then `report.device_mib` (9,716 on 32B),
not the full packed file.

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

### Qwen2.5-3B-Instruct — both codecs fit; decode NF4 is ahead, TTFT still BF16

![Qwen2.5-3B-Instruct: two video-memory graphs side by side, BF16 versus NF4, on the same 0–12288 MiB scale](docs/img/lab-qwen25-3b.png)

*Figure. VRAM traces from the committed plate in `docs/runs/qwen25-3b/` (that
CSV still has the pre-split-K NF4 row, 17.0 tok/s / 212 ms). Do not read decode
speed off this picture. The markdown table below is a live same-session BF16+NF4
pair from `C:\dev\models\runs\qwen25-3b-paired-20260914`, not that committed folder.*

| | BF16 (HF `generate`) | NF4 (`CompressedLinear` + `TokenLoop`) |
|---|---:|---:|
| Weight MiB | 5,886 | 1,563 |
| nvidia-smi after load (MiB) | 8,722 | 4,382 |
| Peak nvidia-smi (MiB) | 8,781 | 4,525 |
| Mean TTFT (ms) | 48 | 92 |
| Mean decode tok/s | 24.8 | 28.7 |
| Smoke (Paris / Berlin / 323) | pass | pass |

Source: live paired wave in `C:\dev\models\runs\qwen25-3b-paired-20260914`
(`summary.csv`, `messages.csv`; `gpu.lab.run --codec both`, isolated worker
processes, NF4 `prefill_chunk=32`). That directory is outside git. The committed folder
[`docs/runs/qwen25-3b/`](docs/runs/qwen25-3b/) is the older plate and was not
overwritten. Packed weights still 1,563 MiB (4.25 bits/weight).

A 3B model fits either way on this card. Occupancy was the bottleneck: decode
used to launch one block per 128 output rows, **16 CTAs** for the query and
output projections and **2** for grouped key/value, against **70 streaming
multiprocessors**. After a 64-row tile and split-K those counts are **128**
and **64**. NF4 decode is **28.7 tok/s** against a same-session BF16 **24.8**.
A prior NF4-only WAVE 2 figure was 31.6 tok/s; this paired re-measure is lower,
still ahead of the BF16 row taken in the same session. This is **not** a
claim that the kernel is faster than Marlin, AWQ, bitsandbytes, or llama.cpp.
A live bitsandbytes NF4 smoke on the same three prompts was **22.8 tok/s /
57 ms** (`C:\dev\models\runs\competitor-qwen25-3b-20260913-bnb-e2e\`, isolated
venv, `Linear4bit` over the HF tree). Our paired NF4 is **28.7 tok/s / 92 ms**.
Report both; they are different stacks, not a kernel ranking. The committed
folder [`docs/runs/competitor-qwen25-3b/`](docs/runs/competitor-qwen25-3b/)
stays the SKIP matrix. Kernel µs on bitsandbytes are still 0/63 SKIP.
Time to first token is still worse: **92 against 48 ms**, so prefill is still
the next floor, less so than the n16 TokenLoop pair (139 against 45 ms). GEMM
counters (Nsight, L2-rotated weights, not live tok/s): on
3B `q_proj` decode, DRAM ~5%, tensor pipe ~1.4%, warp occupancy ~16%; prefill
N=16 occupancy ~29% — [`docs/runs/ncu/`](docs/runs/ncu/). TokenLoop
`LIVE_MAX_N` is **32**. n64 is plan-only. Compression still
pays when the uncompressed model does not fit. The 14B figures further down
are fit-versus-spill, not a kernel win, and they
are not a comparison against Marlin.

### What changed on this card

![Progress on one RTX 3080 12 GB: starved 3B kernel, occupancy fix, honest pair, 14B and 20B fit versus spill, hard eval, local GSM8K slice](docs/img/progress-3080.png)

*Figure. The arc on this card, each number labeled by the plate it came from.
Committed 3B (`docs/runs/qwen25-3b/`, VRAM figure above) is the starved kernel:
23.1 vs 17.0 tok/s — do not read decode speed off that picture. Occupancy-fix
NF4-only was 31.6 tok/s / 167 ms, not a BF16 pair. The table above is the
same-session pair, 24.8 vs 28.7 tok/s. 14B/20B working sets vs the 12,288 MiB
card line; hard eval 7/12, 9/12, 8/12, 10/12 on a separate Q&A sheet. Live
InternLM 20B NF4 hard is 8/12 with no BF16 pair — not that Q&A sheet, not a
quality headline. A local GSM8K slice of the first 200 of 1,319 main-test
items (greedy, `max_new_tokens = 256`) is on this picture only — not a
published GSM8K score. Footer F has true n32 vs two n16 on one `q_proj`
GEMM; TokenLoop now chunks at 32. One live bitsandbytes NF4 smoke row
(22.8 tok/s / 57 ms) is in the footer; that is a different stack, not a kernel
ranking. This plate summarizes; it does not replace `lab-qwen25-14b.png`
panel F or `hard-eval-qwen25.png`. Redraw:
`python -m gpu.lab.progress_plate --redraw`.*

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
meter ran out of scale. What BF16 actually asked for after load is the CUDA
working set torch reports — `torch.cuda.memory_reserved()`, the caching
allocator's reserved pool — about **28,270 MiB**, more than twice the card. A
reservation that size on a 12 GiB card is oversubscribed by definition: the
excess is served from system RAM over the bus, which is what Windows calls
shared GPU memory. NF4 stayed inside the card at about 7,539 MiB. **Do not
quote 11,955 against 8,913 as the result.** The result is panel F of the
figure: about 28,270 MiB against about 7,539 MiB.

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
is the CUDA working set after load — again the allocator's reserved pool:
**37,882 MiB against 10,273 MiB**, that is roughly 37 GiB against roughly
10 GiB. 11,976 only means the dedicated-VRAM meter ran out of scale; the rest
of BF16 is served from system RAM, which Windows reports as shared GPU memory.
NF4 decode is 5.01 tok/s with a smoke pass.

A later 12-item hard eval on the same NF4 weights, NF4 only, scored **8/12**
(misses: train, machines, sheep, bat-and-ball). Mean TTFT **1318 ms**,
**4.4 tok/s**, peak `nvidia-smi` **12067 MiB**, `max_seq=1024`. That is a
different plate from the smoke 605 ms / 5.01 tok/s at `max_seq=512`. CSVs
live in `C:\dev\models\runs\hard-internlm20b-nf4-20260914` — **not**
[`docs/runs/internlm20b/`](docs/runs/internlm20b/). Twelve items are a
regression, not WikiText / GSM8K / MMLU, and there is no BF16 pair.

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
real measurement: torch's caching allocator reserved **37,882 MiB** for the
dense weights, three times the card, so about 26,000 MiB of it could only have
come from system RAM. Generation never ran. `internlm2_5-20b-chat` ships
`trust_remote_code` modules written against transformers 4.41, and on
transformers 5 its
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

### Qwen2.5-32B-Instruct — packed NF4 does not fit; overflow speaks at 2.31 tok/s

![Qwen2.5-32B-Instruct NF4 overflow on RTX 3080 12 GB: packed 16.6 GiB vs resident 9.7 GiB vs streamed 6.9 GiB, product decode 2.31 tok/s, policy D CopyRing](docs/img/h2-qwen25-32b.png)

*Figure. Overflow plate, not a 14B/20B pair and not a kernel ranking.
Packed NF4 (16,599 MiB) crosses the 12,288 MiB card line; resident HBM weights
are 9,716 MiB. A pinned host tail of 6,885 MiB (96 matrices) is copied one
matrix at a time into two 71.72 MiB slots. Product decode is **2.31 tok/s**.
The pageable ~0.8 bar is a pin bug, not the design. Serial copy floor 3.6 tok/s
is 277 ms if wall equalled copy — measured wall is ~432 ms/tok. Redraw:
`python -m gpu.lab.h2_plate --redraw`.*

| | BF16 (HF `generate`) | NF4 overflow (`CopyRing` + `TokenLoop`) |
|---|---:|---:|
| Packed weight MiB | — | 16,599 |
| Resident HBM weights MiB | — | 9,716 |
| Streamed host MiB | — | 6,885 |
| nvidia-smi after load (MiB) | — | 11,268 |
| Peak nvidia-smi (MiB) | — | 11,933 |
| torch allocated after load (MiB) | — | 9,933 |
| KV cache MiB (preallocated, `max_seq=2048`) | — | 512 |
| Mean TTFT (ms) | — | 1,006 |
| Mean decode tok/s | — | 2.31 |
| Smoke (Paris / Berlin / 323) | not run | pass |

Source: live `C:\dev\models\runs\h2-qwen25-32b-20260914-234048`
(`python -m gpu.lab.h2_trace --no-timing`). A slim copy
(`data_path.md`, `messages.json`, `gate.txt`) is in
[`docs/runs/h2-qwen25-32b/`](docs/runs/h2-qwen25-32b/). The `.chr` is not in git.

This table has the same honest dashes as 20B, for a different reason. BF16 32B
was never loaded: ~65 GiB of sixteen-bit weights do not fit, and there is
nothing to compare against. Packed NF4 at 16,599 MiB **also** misses a 12 GiB
card. What ran is overflow: policy D keeps q/k/v/o, embed, and `lm_head` on
the device, streams every `down_proj` and the tail `gate`/`up` pairs (layers
48–63) from pinned host RAM, and still reconstructs only in `chr_nf4_gemm`
registers. **Do not quote 11,933 against 12,288 as leftover headroom on an
empty card.** Display memory is inside `nvidia-smi`. **Do not quote 16,599 as
the HBM footprint** — that is `report.device_mib` = 9,716.

**NF4 overflow ran.** Three independent turns answered
`The capital of France is Paris.`, `The capital of Germany is Berlin.`, and
`323` at 2.30–2.32 tok/s and ~1.0 s TTFT. `nvidia-smi` sat flat at
11,926–11,933 MiB across decode. The serial copy floor of the 6,885 MiB tape
is 277 ms/tok (~3.6 tok/s if wall were copy). Measured wall is ~432 ms/tok:
the floor is beaten, the ~3 tok/s HBM ceiling is not. 10 tok/s is not a claim.

**Hard-12 was not started.** Smoke 3/3 is not quality. A twelve-item hard eval
on this path is a separate, slow plate (`docs/eval-32b.md`).

Pageable H2D (~0.8 tok/s) was a footgun: `Tensor.is_pinned` is a method, so
`bool(arena.is_pinned)` was always true and the host image stayed pageable.
The product number is 2.31, not 0.8.

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
[`docs/runs/hard-qwen25/`](docs/runs/hard-qwen25/). Twelve items are a
regression, not WikiText / GSM8K / MMLU. NF4 matching or beating BF16 on
this set does not mean quantization is lossless.*

InternLM 20B was scored later, NF4 only: **8/12**, no BF16 pair, not drawn
on `hard-eval-qwen25.png`. Live dir
`C:\dev\models\runs\hard-internlm20b-nf4-20260914`. Same twelve prompts,
same extractor; still a regression fixture.

### The chat script

Identical for both codecs, in all four models:

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
- **On 14B and 20B the `nvidia-smi` meter is capped** at the card's 12288 MiB,
  and hitting that cap is not "the model fit". The figure plotted as the CUDA
  working set is `torch.cuda.memory_reserved()`: the **reserved pool of
  PyTorch's caching allocator**, not a driver field named "dedicated video
  memory plus shared GPU memory". On this WDDM machine that pool reaches about
  28 GiB on a 12 GiB card, which can only mean the allocation is
  oversubscribed — weights served from system RAM over the bus, which
  `nvidia-smi` cannot show. The conclusion stands; the counter is a PyTorch
  counter. Panel F, not panel A, is the claim. On 20B both codecs sit near that
  cap: do not quote 11,976 against 11,828. The 20B result is 37,882 MiB against
  10,273 MiB, plus NF4 at 5.01 tok/s with a smoke pass. There is no BF16 speed
  baseline.
- **On 32B packed NF4 still does not fit.** The product path is overflow, not
  “all 16,599 MiB in HBM”. Resident weights are 9,716 MiB; 6,885 MiB stream
  from pinned host RAM. Peak `nvidia-smi` 11,933 includes the desktop. Decode
  **2.31 tok/s** is CopyRing + TokenLoop + `chr_nf4_gemm`, not vs Marlin /
  llama.cpp / BF16 32B (never run). Smoke 3/3 is not quality; hard-12 is not
  started. Do not quote pageable ~0.8 as the H2 design. Serial copy 277 ms is
  not the measured 432 ms wall. `--codec auto` never picks VQ.
- **Decode tokens per second compare two different stacks:** HuggingFace
  `generate` with dense BF16 matrix multiplies on one side, our NF4 loop with
  reconstruction inside the multiply on the other. Both are reported, in both
  directions. On 3B, NF4 decode is now ahead on this card (28.7 tok/s against
  24.8) on a same-session pair in `C:\dev\models\runs\qwen25-3b-paired-20260914`
  (a prior NF4-only WAVE 2 figure was 31.6; that mixed 31.6-vs-23.1 claim is
  retired). TTFT is still BF16: 48 against 92 ms. On 14B, NF4 is far ahead, and
  there the reason is that BF16 has already spilled into system RAM. On 20B NF4
  is 5.01 tok/s; there is no BF16 generate, so there is no speed comparison.
  On 32B NF4 overflow is 2.31 tok/s; there is no BF16 generate and no
  all-resident NF4. **Do not write that deep-fold is faster than existing 4-bit engines.** A live
  bitsandbytes NF4 smoke (same three prompts, isolated venv) was 22.8 tok/s /
  57 ms; our paired NF4 is 28.7 tok/s / 92 ms. Those are different stacks
  (`Linear4bit` vs `CompressedLinear` + `TokenLoop`), not a kernel ranking.
  Marlin, AWQ, GPTQ/Marlin, ExLlamaV2, llama.cpp CUDA Q4, and vLLM were not
  timed. The committed folder
  [`docs/runs/competitor-qwen25-3b/`](docs/runs/competitor-qwen25-3b/) stays
  SKIP; live tok/s stay outside git. Isolated venvs:
  [`docs/competitor-venvs.md`](docs/competitor-venvs.md). Nsight counters in
  [`docs/runs/ncu/`](docs/runs/ncu/) are *our* kernel occupancy and pipes, not
  tok/s. Kernel-vs-CPU NF4 arithmetic lives in
  [`gpu/nf4/verify.py`](gpu/nf4/verify.py); split quantization vs kernel error is
  [`gpu/nf4/numerics.py`](gpu/nf4/numerics.py). Neither is a quality benchmark.
- **Time-to-first-token is prompt processing,** and the two sides do it
  differently: BF16 uses the HuggingFace path, NF4 uses chunks of at most 32
  positions. A true n32 GEMM on 3B `q_proj` is 1.63× faster than two n16
  launches (69 µs vs 113 µs); occupancy held. On live `q_proj` n32 matched
  two n16 bit-for-bit; the 0.058 maxabs spike is one element and a BF16
  half-ULP, not a tile bug. TokenLoop `LIVE_MAX_N=32`. Same-session pair
  2026-09-14 (`qwen25-3b-paired-20260914`): mean TTFT **48 vs 92 ms**, decode
  **24.8 vs 28.7 tok/s**. An NF4-only n32 plate the same day was 91 ms / 28.6
  tok/s. The older n16 pair was 45 against 139 ms. n64 is not live.
- **Display memory is inside the `nvidia-smi` reading.** It is real, it is on
  the same card, and it is not subtracted away here.
- **Generate is Ampere-family CUDA.** sm_86 (RTX 3080) is the measured plate.
  A100 (`sm_80`) and Ada (`sm_89`) generate as experimental. Turing / Hopper /
  Blackwell are a named refuse, not a port. “Two clicks, all OS” is not a
  claim: CUDA, the compiler, Go, and the HuggingFace tree are user-provided;
  Linux tok/s are unpublished.
- **The lab can emit a synthetic figure** (`--dry-plot`), whose CSVs are
  stamped `FIXTURE`. Nothing in the tables above comes from that fixture.
- **WikiText PPL is unpublished.** An NLL adapter exists; no corpus number
  is quoted here.

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

## Install

On another PC (not this 3080) use [`docs/quickstart.md`](docs/quickstart.md)
([русский](docs/quickstart.ru.md)) for the short command list, and
[`docs/install.md`](docs/install.md) ([русский](docs/install.ru.md)) for `-h`
dumps. Neighbor contract: a repo `.venv`, CUDA torch from the cu124 index,
`chr`, then `doctor` / `pull` / `chat`. Do not pip-install into conda env
`torch-gpu`. To send metrics back from another card:

```powershell
powershell -File scripts/plate.ps1 3b
```

```powershell
powershell -File scripts/setup.ps1
.\.venv\Scripts\Activate.ps1
deepfold pull Qwen/Qwen2.5-3B-Instruct --yes
deepfold chat --model <that directory>
```

## Run the lab

Generate from a packed `.chr` without a notebook. Ampere-family CUDA:
**sm_86 is the measured plate** (RTX 3080); A100 (`sm_80`) and Ada (`sm_89`)
generate as experimental. GGUF is refused; a sibling `.chr` is used only when
its CHR0 header matches this model (`hidden_size`, `num_layers`, `vocab_size`).

This is **not** two clicks on every OS. `doctor` / `run` do not install the
NVIDIA driver, a CUDA PyTorch wheel, MSVC/`nvcc`, Go, or a HuggingFace tree.
macOS can compress and cannot generate. Linux can load a `.so`; there is
**no published Linux tok/s**. Turing, Hopper, and Blackwell **refuse
generate**. The kernel image is `sm_80/sm_86/sm_89` plus PTX `compute_80`.

```powershell
conda activate torch-gpu
cd <this-repo>
python -m gpu.cli doctor
python -m gpu.cli run --model <HuggingFace-dir>
```

After `pip install -e .` those are `deepfold doctor` and
`deepfold run --model DIR`. Details: [`docs/ux.md`](docs/ux.md).

The BF16-vs-NF4 comparison plate is still the lab harness. Defaults point at
the author's Windows layout and can be overridden:

| Variable | Default on this machine |
|---|---|
| `DEEPFOLD_MODEL` | HuggingFace dir (`C:\dev\models\Qwen2.5-3B-Instruct` on the author box) |
| `DEEPFOLD_CHR` | packed file (`C:\dev\models\qwen25-3b.nf4.chr` there) |
| `DEEPFOLD_MODELS` | root for catalog trees; else `C:\dev\models` if present, else `$DEEPFOLD_HOME/models` |
| `DEEPFOLD_RUNS` | run dumps; else `<models>/runs` |
| `DEEPFOLD_HOME` | cache (`%LOCALAPPDATA%\deepfold` / `~/.cache/deepfold`) |

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
python -m gpu.lab.progress_plate --redraw
python -m gpu.lab.h2_plate --redraw
```

32B overflow smoke (Ampere, live weights outside git):

```powershell
python -m gpu.lab.h2_trace --no-timing
```

## Docs

The design notes under [`docs/spec/`](docs/spec/) are in Russian. This page and
the lab writeup are in English.

| | |
|---|---|
| Progress plate (first graphs through now) | [docs/img/progress-3080.png](docs/img/progress-3080.png) |
| 32B overflow plate | [docs/img/h2-qwen25-32b.png](docs/img/h2-qwen25-32b.png) |
| Lab method and how to read the figure | [docs/lab.md](docs/lab.md) |
| CLI (`doctor` / `run`) | [docs/ux.md](docs/ux.md) |
| Hard eval (3B/14B questions, replies, times) | [docs/eval-hard-qwen25.md](docs/eval-hard-qwen25.md) |
| 32B smoke + hard-12 sheet (hard not run) | [docs/eval-32b.md](docs/eval-32b.md) |
| H2 overflow ring | [docs/plan-h2-ring.md](docs/plan-h2-ring.md) |
| Nsight GEMM counters (not tok/s) | [docs/runs/ncu/](docs/runs/ncu/) |
| 4-bit competitor matrix (all SKIP) | [docs/runs/competitor-qwen25-3b/](docs/runs/competitor-qwen25-3b/) |
| Isolated competitor venvs (later) | [docs/competitor-venvs.md](docs/competitor-venvs.md) |
| Kernel vs CPU NF4 oracle | [gpu/nf4/verify.py](gpu/nf4/verify.py), [gpu/nf4/numerics.py](gpu/nf4/numerics.py) |
| Memory budget on a 3080 12 GB | [docs/vram-3080.md](docs/vram-3080.md) |
| CPU compress and verify | [docs/cpu-roundtrip.md](docs/cpu-roundtrip.md) |
| Which models to download | [docs/models.md](docs/models.md) |
| Russian README | [README.ru.md](README.ru.md) |

## License

[MIT](LICENSE).

This project was created with [Cursor](https://cursor.com).

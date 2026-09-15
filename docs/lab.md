# Origin lab: BF16 vs NF4 on an RTX 3080 12 GB

One comparison per downloaded model, not two stories. Same GPU, same three
English prompts, continuous `nvidia-smi` + torch metrics, then **one** Plotly
plate (`comparison_figure`). Notebooks:

- [`notebooks/03_codec_lab.ipynb`](../notebooks/03_codec_lab.ipynb) — Qwen2.5-3B
- [`notebooks/04_qwen25_14b_lab.ipynb`](../notebooks/04_qwen25_14b_lab.ipynb) — Qwen2.5-14B
- [`notebooks/05_internlm20b_lab.ipynb`](../notebooks/05_internlm20b_lab.ipynb) — internlm2.5-20B

Those three are **smoke**: Paris / Berlin / 323. They do not measure reasoning
quality. Hard eval (GSM8K-style / multi-step, quality **and** speed) is
[`eval.md`](eval.md) and
[`notebooks/06_hard_eval.ipynb`](../notebooks/06_hard_eval.ipynb).

Generate without a notebook: `python -m gpu.cli doctor` then
`python -m gpu.cli run --model DIR` (Ampere `sm_86` only; GGUF refused; a
sibling `.chr` is used only when its CHR0 header matches this model). See
[`ux.md`](ux.md).

The comparison plate is `python -m gpu.lab.run` (3B paths by default). Public
contract of the package: `from gpu.lab import run_both, run_bf16, run_nf4,
comparison_figure`.

Canonical pages under [`docs/`](.) are English. Russian twins exist only as
`*.ru.md` ([install](install.ru.md), [quickstart](quickstart.ru.md), and the
origin [README.ru.md](../README.ru.md)).

## What we claim

VRAM after load is the product metric. The NF4 path must keep weights packed
(`CompressedLinear` + `chr_nf4_gemm`) and never materialize a full `[M, K]`
BF16 layer in HBM. If NF4 sits near the BF16 line after load, that is a bug,
not a win.

Decode tok/s is a different stack on each side (HuggingFace `generate` + dense
GEMM vs our fused NF4 loop). Report both. Do not rank them as a kernel
benchmark.

## Method

12 GB cannot hold both models. Each codec runs in its **own process**, which
exits before the other starts. `empty_cache` in the same Jupyter kernel does
not return VRAM on Windows; the worker must actually exit.

1. Record idle `nvidia-smi`.
2. Load BF16 from `$DEEPFOLD_MODEL` via HuggingFace `from_pretrained`
   (`bfloat16`, `device_map` on GPU 0) **in a child process**.
3. Warmup, then the three prompts, greedy, `max_new_tokens = 64`, stop
   on EOS / Qwen `<|im_end|>` (151645). Worker unloads and **exits**.
4. Wait until `nvidia-smi` is back near idle. Do not start NF4 while the dense
   copy is still resident.
5. Load NF4 from `$DEEPFOLD_CHR` via `gpu.host.load_model` +
   `gpu.loop.TokenLoop` in a **new** child. No `from_pretrained` on the weight
   shards. No `transformers.generate`.
6. Same three prompts, then the worker exits.

Sampler runs from before load until after unload, poll interval ≤ 0.15 s.
`t_s` is seconds from **that session’s** `start`. The two VRAM graphs sit
**side by side** on one 0…12288 MiB Y scale (not overlaid). Turns are
independent: KV / HF cache is reset between user messages so each prompt is a
clean prefill and TTFT is comparable.

### Chat script (identical for both codecs)

```
Reply with one short sentence. What is the capital of France?
And the capital of Germany?
What is 17 times 19? Reply with the number only.
```

Quality needles (case-insensitive): `paris` / `париж`, `berlin` / `берлин`,
`323`. Record the raw text anyway. This is smoke, not a benchmark. Hard
reasoning belongs on [`eval.md`](eval.md), not here.

## How to read the figure

One `plotly.graph_objects.Figure` from `gpu.lab.comparison_figure` — a paper
plate, not a dashboard:

| Panel | Claim |
|---|---|
| A | Two VRAM graphs side by side: left uncompressed BF16, right compressed NF4, same 0…12288 MiB Y scale, card limit on both |
| F | Only if CUDA’s working set exceeds VRAM: a second pair under A, same Y, showing torch reserved (VRAM + shared GPU memory) |
| B | Where after-load VRAM goes (packed weights / rest of torch / outside torch / free) |
| C | The three replies and their needles |
| D | TTFT per turn (prefill) |
| E | Decode tok/s per turn (different stacks; not a kernel benchmark) |

Panel titles are generated from the CSVs. Lines and bars only — no event
markers. Hover is `x unified` on the VRAM panel. Colour **and** dash encode
the codec.

## Live plates in this repo

Copied from measured runs on the author’s 3080. Interactive HTML next to the
CSVs. **`lab-test` is a synthetic fixture** and is not here.

| Model | PNG | Run directory |
|---|---|---|
| Qwen2.5-3B-Instruct | [`img/lab-qwen25-3b.png`](img/lab-qwen25-3b.png) | [`runs/qwen25-3b/`](runs/qwen25-3b/) |
| Qwen2.5-14B-Instruct | [`img/lab-qwen25-14b.png`](img/lab-qwen25-14b.png) | [`runs/qwen25-14b/`](runs/qwen25-14b/) |
| internlm2.5-20B-chat | [`img/lab-internlm20b.png`](img/lab-internlm20b.png) | [`runs/internlm20b/`](runs/internlm20b/) |

### Qwen2.5-3B-Instruct

| | BF16 (HF `generate`) | NF4 (`CompressedLinear` + `TokenLoop`) |
|---|---:|---:|
| Weight MiB | 5,886 | 1,563 |
| nvidia-smi after load (MiB) | 7,477 | 3,142 |
| Peak nvidia-smi (MiB) | 7,535 | 3,286 |
| Mean TTFT (ms) | 52 | 212 |
| Mean decode tok/s | 23.1 | 17.0 |
| Smoke (Paris / Berlin / 323) | pass | pass |

These cells are the committed CSVs in [`runs/qwen25-3b/`](runs/qwen25-3b/)
(`summary.csv` NF4: 16.9953 tok/s, 212.422 ms). They were **not** overwritten.

**2026-09-13 — 3B NF4 decode after split-K.** Occupancy was the old floor:
one 128-row block per output tile, **16 CTAs** on `q`/`o` and **2** on GQA
`k`/`v` against **70 SMs**. After the 64-row tile and split-K those launches
are **128** and **64**. A live re-measure of the same 3B NF4 path
(`gpu.lab.worker`, same three prompts) was **31.6 tok/s** decode and **167 ms**
mean TTFT. The BF16 row (23.1 tok/s, 52 ms) was not re-run that day. Speed
cells in [`size-efficiency.md`](size-efficiency.md) use that re-measure; this
plate stays the git snapshot.

### Qwen2.5-14B-Instruct

On 14B, `nvidia-smi` is **capped** at 12288 MiB. BF16’s CUDA working set after
load is ~28,270 MiB (shared GPU memory on WDDM). NF4 stays on the card.

| | BF16 (HF `generate`) | NF4 (`CompressedLinear` + `TokenLoop`) |
|---|---:|---:|
| Weight MiB | 28,172 | 7,483 |
| nvidia-smi after load (MiB) | 11,955 | 8,913 |
| Peak nvidia-smi (MiB) | 11,997 | 9,356 |
| torch after load (MiB) | 28,270 | 7,539 |
| Mean TTFT (ms) | 1,028 | 759 |
| Mean decode tok/s | 0.92 | 6.56 |
| Smoke (Paris / Berlin / 323) | pass | pass |

### internlm2.5-20B-chat

On 20B, **both** `nvidia-smi` traces sit near 12288 MiB. Do not quote 11,976 vs
11,828 as the product win. BF16’s CUDA working set after load is ~37,882 MiB
(~26 GiB in Windows shared GPU memory). NF4 stays on the card at 10,273 MiB.
HuggingFace BF16 generate did not run (InternLM transformers-4.41 remote code
vs transformers 5: `TypeError: can only concatenate tuple (not "int") to
tuple`). Those TTFT / tok/s cells stay blank. Recorded miss, not a
patched-generate baseline. NF4 answered all three prompts at 5.01 tok/s.

| | BF16 (HF `generate`) | NF4 (`CompressedLinear` + `TokenLoop`) |
|---|---:|---:|
| Weight MiB | 37,882 | 10,062 |
| nvidia-smi after load (MiB) | 11,892 | 11,578 |
| Peak nvidia-smi (MiB) | 11,976 | 11,828 |
| torch after load (MiB) | 37,882 | 10,273 |
| Mean TTFT (ms) | — | 605 |
| Mean decode tok/s | — | 5.01 |
| Smoke (Paris / Berlin / 323) | not run | pass |

Source: [`runs/internlm20b/`](runs/internlm20b/). Live plate:
[`img/lab-internlm20b.png`](img/lab-internlm20b.png).

## Size vs efficiency (3B / 14B / 20B)

Compressed vs uncompressed **changes with model size**. Cross-model table
and the CUDA-working-set caveat: [`size-efficiency.md`](size-efficiency.md).

| Model | Weight MiB (BF16 / NF4) | nvidia-smi after load (BF16 / NF4) | CUDA working set (BF16 / NF4) | Mean TTFT ms (BF16 / NF4) | Mean decode tok/s (BF16 / NF4) | Smoke |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-3B-Instruct | 5,886 / 1,563 | 7,477 / 3,142 | 5,886 / 1,618 | 52 / 167 | 23.1 / 31.6 | pass / pass |
| Qwen2.5-14B-Instruct | 28,172 / 7,483 | 11,955† / 8,913 | 28,270 / 7,539 | 1,028 / 759 | 0.92 / 6.56 | pass / pass |
| internlm2.5-20B-chat | 37,882 / 10,062 | 11,892‡ / 11,578‡ | 37,882 / 10,273 | — / 605 | — / 5.01 | not run / pass |

† 14B `nvidia-smi` is capped at 12288 MiB. The win is CUDA working set **~28 GiB
vs ~7.8 GiB** (28,270 vs 7,539 MiB), not 11,955 vs 8,913. 3B NF4 decode/TTFT
in this table is the 2026-09-13 split-K re-measure (31.6 tok/s, 167 ms); the
committed plate CSVs still have 17.0 / 212. BF16 was not re-run that day.
On 14B packed NF4 wins on working set, TTFT, and decode because BF16 spills.

‡ 20B both `nvidia-smi` sit near 12288 MiB. Do not quote 11,976 vs 11,828. The
win is CUDA working set **~37.9 GiB vs ~10.3 GiB** (37,882 vs 10,273 MiB) and
NF4 decode 5.01 tok/s with smoke pass. BF16 generate failed.

## CSV artifacts

English headers, comma, UTF-8. One directory per lab:

| File | Role |
|---|---|
| `timeline.csv` | one row per poll (`t_s`, `codec`, smi + torch memory, util, power, clocks, `message_id`) |
| `events.csv` | `start`, `load_*`, `warmup_*`, `msg_send`, `first_token`, `msg_done`, `unload_*`, `stop` |
| `messages.csv` | prompt, response, token counts, prefill/decode timing, `quality_ok` |
| `summary.csv` | one row per codec — the README table |
| `lab.html` | the interactive figure |
| `lab.png` | static export (kaleido), same plate as the PNG in `docs/img/` |

`codec` is `bf16` or `nf4`. `message_id` is empty when not inside a user message.

## Honesty

- VRAM is the product metric. NF4 must sit well below BF16 after load. If it
  does not, the driver materialized a layer — that is a bug, not a win.
  `nvidia-smi` is dedicated VRAM and stops at 12288 MiB. A 14B or 20B BF16
  working set that spills into shared GPU memory (system RAM) is shown as a
  second pair of graphs under VRAM, same Y, not as “the model fit in 12 GB.”
  On 20B both codecs’ `nvidia-smi` sit near the cap; the claim is 37,882 vs
  10,273 MiB working set, not 11,976 vs 11,828.
- Decode tok/s is **not** the same code path: HF `generate` + dense GEMM vs our
  fused NF4 loop. Report both. Do not claim “we are faster” unless the numbers
  say so, and do not rank this against Marlin (not measured). Occupancy on
  small `M` was the old 3B decode floor (16 / 2 CTAs vs 70 SMs); that launch
  is landed as a 64-row tile plus split-K. Prefill (TTFT) is the next 3B floor.
  On 20B BF16 generate failed; those tok/s cells stay blank.
- TTFT is prefill. BF16 uses the HF prefill; NF4 uses `N≤16` chunks.
- Display VRAM is inside `nvidia-smi`. It is real. Do not subtract it away.

Notebooks `01_bf16_gpu_baseline.ipynb` and `02_nf4_gpu_driver.ipynb` are
archives (rotated SVG labels, two separate stories). Use `03_codec_lab.ipynb`.

## How to run

Generate (no notebook; Ampere-family CUDA: **sm_86 measured**, A100/Ada
experimental; Turing/Hopper/Blackwell refused; GGUF refused; sibling `.chr`
matched by CHR0 header):

```powershell
python -m gpu.cli doctor
python -m gpu.cli run --model <HuggingFace-dir>
```

The comparison plate still uses `DEEPFOLD_MODEL`, `DEEPFOLD_CHR`,
`DEEPFOLD_MODELS`, `DEEPFOLD_RUNS` (author-box defaults apply only when
`C:\dev\models` exists):

```powershell
conda activate torch-gpu
cd <this-repo>
python -m gpu.lab.run
```

Default output is `$DEEPFOLD_RUNS\lab-<timestamp>\`. PowerShell does not expand
`%Y%m%d`; pass `--out` only if you want a specific folder.

Or Run All on [`notebooks/03_codec_lab.ipynb`](../notebooks/03_codec_lab.ipynb).

`gpu.win_toolchain` injects `vcvars64.bat` into the current process so CUDA JIT
can find `cl.exe` from Jupyter or a Cursor terminal.

Fixture only (no CUDA, no 3B):

```powershell
python -m gpu.lab.run --out <dir> --dry-plot
```

Do not load both codecs at once. Close other fat processes on the 3080 before a
live run. Do not commit model weights or `.chr` files.

Tests (fixture only, never loads the 3B): `python -m gpu.lab.test_lab`.
Hard-eval scoring (no GPU): `python -m gpu.lab.test_hard`. See [`eval.md`](eval.md).

Notebook 05 (internlm2.5-20B) needs `einops` and **`sentencepiece==0.1.99`** in
`torch-gpu` (`0.2.2` rejects InternLM's `<0x00>` pieces). Transformers 5 always
picks InternLM's fast tokenizer; the harness loads the slow SentencePiece class
instead. Then `from_pretrained(..., trust_remote_code=True)` can import.

## Related

| | |
|---|---|
| Origin README | [../README.md](../README.md) |
| Hard eval (not smoke) | [eval.md](eval.md) |
| Hard eval appendix (full Q&A and times) | [eval-hard-qwen25.md](eval-hard-qwen25.md) |
| Size vs efficiency (3B / 14B / 20B) | [size-efficiency.md](size-efficiency.md) |
| CLI (`doctor` / `run`) | [ux.md](ux.md) |
| Codec specs | [spec/](spec/) |
| 3080 VRAM budget | [vram-3080.md](vram-3080.md) |
| CPU roundtrip | [cpu-roundtrip.md](cpu-roundtrip.md) |

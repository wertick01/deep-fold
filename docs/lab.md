# Origin lab: BF16 vs NF4 on an RTX 3080 12 GB

One comparison per downloaded model, not two stories. Same GPU, same three
English prompts, continuous `nvidia-smi` + torch metrics, then **one** Plotly
plate (`comparison_figure`). Notebooks:

- [`notebooks/03_codec_lab.ipynb`](../notebooks/03_codec_lab.ipynb) — Qwen2.5-3B
- [`notebooks/04_qwen25_14b_lab.ipynb`](../notebooks/04_qwen25_14b_lab.ipynb) — Qwen2.5-14B
- [`notebooks/05_internlm20b_lab.ipynb`](../notebooks/05_internlm20b_lab.ipynb) — internlm2.5-20B (not run)

Harness: `python -m gpu.lab.run` (3B paths by default). Public contract of the
package: `from gpu.lab import run_both, run_bf16, run_nf4, comparison_figure`.

Design notes under [`docs/`](.) are mostly Russian; this page and the origin
[README](../README.md) are English.

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
`323`. Record the raw text anyway. This is smoke, not a benchmark.

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

### Qwen2.5-3B-Instruct

| | BF16 (HF `generate`) | NF4 (`CompressedLinear` + `TokenLoop`) |
|---|---:|---:|
| Weight MiB | 5,886 | 1,563 |
| nvidia-smi after load (MiB) | 7,477 | 3,142 |
| Peak nvidia-smi (MiB) | 7,535 | 3,286 |
| Mean TTFT (ms) | 52 | 212 |
| Mean decode tok/s | 23.1 | 17.0 |
| Smoke (Paris / Berlin / 323) | pass | pass |

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
  `nvidia-smi` is dedicated VRAM and stops at 12288 MiB. A 14B BF16 working
  set that spills into shared GPU memory (system RAM) is shown as a second
  pair of graphs under VRAM, same Y, not as “the model fit in 12 GB.”
- Decode tok/s is **not** the same code path: HF `generate` + dense GEMM vs our
  fused NF4 loop. Report both. Do not claim “we are faster” unless the numbers
  say so. Current floor is occupancy on small `M` (16 blocks / 70 SMs).
- TTFT is prefill. BF16 uses the HF prefill; NF4 uses `N≤16` chunks.
- Display VRAM is inside `nvidia-smi`. It is real. Do not subtract it away.

Notebooks `01_bf16_gpu_baseline.ipynb` and `02_nf4_gpu_driver.ipynb` are
archives (rotated SVG labels, two separate stories). Use `03_codec_lab.ipynb`.

## How to run

Paths default to the author’s machine; override with `DEEPFOLD_MODEL`,
`DEEPFOLD_CHR`, `DEEPFOLD_RUNS`.

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

## Related

| | |
|---|---|
| Origin README | [../README.md](../README.md) |
| Codec specs | [spec/](spec/) |
| 3080 VRAM budget | [vram-3080.md](vram-3080.md) |
| CPU roundtrip | [cpu-roundtrip.md](cpu-roundtrip.md) |

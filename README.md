# deep-fold

Keep transformer weights **compressed in VRAM**. Unpack only the tile the GPU
is multiplying right now — in Ampere registers, on tensor cores — never as a
resident `[M, K]` BF16 matrix in HBM.

The measured hardware is an **RTX 3080 12 GB** (12288 MiB). This is not Marlin
and not `bitsandbytes`.

The repo ships:

- a Go compressor (`chr`) that writes `.chr` (CHR0, NF4 groups of 64)
- a PyTorch host + CUDA NF4 GEMM (`CompressedLinear` + `TokenLoop`)
- a lab that runs dense HuggingFace BF16 and the packed NF4 driver on the
  **same card, same three prompts**, then draws one figure

Weights and `.chr` files are **not** in git. Live lab CSVs and PNGs are, under
[`docs/runs/`](docs/runs/).

## Measured on an RTX 3080 12 GB

Greedy decoding, `max_new_tokens = 64`, independent turns (KV / HF cache reset
between prompts, so every time-to-first-token is a clean prefill). Each codec
runs in its **own process** so 12 GB never holds both copies. `nvidia-smi`
includes the CUDA context and the Windows desktop compositor; nothing is
subtracted.

### Qwen2.5-3B-Instruct — both codecs fit on the card

![Qwen2.5-3B-Instruct: two VRAM graphs side by side, BF16 vs NF4, same 0–12288 MiB scale](docs/img/lab-qwen25-3b.png)

| | BF16 (HF `generate`) | NF4 (`CompressedLinear` + `TokenLoop`) |
|---|---:|---:|
| Weight MiB | 5,886 | 1,563 |
| nvidia-smi after load (MiB) | 7,477 | 3,142 |
| Peak nvidia-smi (MiB) | 7,535 | 3,286 |
| Mean TTFT (ms) | 52 | 212 |
| Mean decode tok/s | 23.1 | 17.0 |
| Smoke (Paris / Berlin / 323) | pass | pass |

Source: [`docs/runs/qwen25-3b/`](docs/runs/qwen25-3b/) (`summary.csv`,
`messages.csv`, `timeline.csv`, interactive [`lab.html`](docs/runs/qwen25-3b/lab.html)).

### Qwen2.5-14B-Instruct — BF16 fills the card and spills; NF4 stays on-card

![Qwen2.5-14B-Instruct: nvidia-smi pair plus a second pair for CUDA working set / shared GPU memory](docs/img/lab-qwen25-14b.png)

`nvidia-smi` is dedicated VRAM and **stops at 12288 MiB**. After load, torch
reports a BF16 working set of **~28,270 MiB** (~16,362 MiB of that is Windows
shared GPU memory). NF4 stays inside the card (~7,539 MiB torch / 8,913 MiB
smi). Do not read 11,955 vs 8,913 on the smi meter as the product win — that
meter is capped. The product win is panel F: ~28 GiB vs ~7.8 GiB.

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

internlm2.5-20B is **not measured**. The notebook exists
([`notebooks/05_internlm20b_lab.ipynb`](notebooks/05_internlm20b_lab.ipynb));
TokenLoop is Qwen-shaped (`q`/`k`/`v`/`o`) and InternLM2 uses fused `wqkv`.

## Chat script (identical for both codecs)

```
Reply with one short sentence. What is the capital of France?
And the capital of Germany?
What is 17 times 19? Reply with the number only.
```

Needles (case-insensitive): `paris` / `париж`, `berlin` / `берлин`, `323`.
Smoke, not a benchmark. Method: [`docs/lab.md`](docs/lab.md).

## Honesty

- VRAM is the product metric. Packed NF4 must sit well below dense BF16 after
  load. If it does not, the driver materialized a layer — that is a bug, not a
  win.
- On 14B, `nvidia-smi` is capped at the card. The CUDA working set is
  `torch.cuda.memory_reserved` (VRAM + shared GPU memory).
- Decode tok/s is **not** the same stack: HuggingFace `generate` + dense GEMM vs
  a fused NF4 loop. Report both. On 3B, dense BF16 is faster. On 14B, NF4 is
  ahead because BF16 is already spilling.
- TTFT is prefill. BF16 uses the HF prefill; NF4 uses `N≤16` chunks.
- Display VRAM is inside `nvidia-smi`. It is real. Do not subtract it away.
- [`gpu/lab`](gpu/lab/) can emit a synthetic fixture (`--dry-plot`). Those CSVs
  are stamped `FIXTURE`. The tables above are **not** from that fixture.

## How to run the lab

Defaults point at the author's Windows layout and can be overridden:

| Env | Default on this machine |
|---|---|
| `DEEPFOLD_MODEL` | `C:\dev\models\Qwen2.5-3B-Instruct` |
| `DEEPFOLD_CHR` | `C:\dev\models\qwen25-3b.nf4.chr` |
| `DEEPFOLD_RUNS` | `C:\dev\models\runs` |

```powershell
conda activate torch-gpu
cd <this-repo>
python -m gpu.lab.run
```

Jupyter: [`notebooks/03_codec_lab.ipynb`](notebooks/03_codec_lab.ipynb) (3B),
[`notebooks/04_qwen25_14b_lab.ipynb`](notebooks/04_qwen25_14b_lab.ipynb) (14B).
Restart the kernel before Run All — do not `import torch` in the comparison
kernel; workers load the weights. CUDA JIT from Jupyter does not inherit
`vcvars64.bat`; `gpu.win_toolchain` injects it so `cl.exe` is on `PATH`.

No GPU (fixture figure only, do not publish those numbers):

```powershell
python -m gpu.lab.run --out <dir> --dry-plot
```

Redraw a committed plate without a GPU:

```powershell
python -c "from gpu.lab import comparison_figure; comparison_figure(r'docs/runs/qwen25-3b')"
```

## CPU compressor

No GPU. Tests write tiny safetensors; they do not download a model.

```bash
go test ./...
go build -o chr ./cmd/chr
```

Compress a local HuggingFace tree (BF16 safetensors, not GGUF):

```text
chr compress --in <model-dir> --out <model>.nf4.chr --codec nf4
```

Verify: [`docs/cpu-roundtrip.md`](docs/cpu-roundtrip.md). Tiny group scales that
flush to zero in float16 are encoded as scale `1` (see `internal/nf4`).

## Docs

Codec and GPU specs under [`docs/spec/`](docs/spec/) are in Russian. The origin
README and the lab are English.

| | |
|---|---|
| Lab writeup | [docs/lab.md](docs/lab.md) |
| 3080 VRAM budget | [docs/vram-3080.md](docs/vram-3080.md) |
| CPU compress / verify | [docs/cpu-roundtrip.md](docs/cpu-roundtrip.md) |
| Models to download | [docs/models.md](docs/models.md) |

## License

[MIT](LICENSE).

This project was created with [Cursor](https://cursor.com).

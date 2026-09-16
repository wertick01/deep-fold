# Matched 3080 comparison (2026-09-16)

One RTX 3080 12 GB, Windows/WDDM, greedy decode, `ctx=2048`, `n_predict=64`.
Qwen2.5 Instruct. Quote **long** ignore-EOS 64-token plateaus for 32B decode,
not Ollama’s short-EOS smoke mean (8+8+4 tokens → 3.18 tok/s).

Plate: [`docs/img/compare-3080.png`](img/compare-3080.png)
JSON: [`docs/runs/compare-3080/`](runs/compare-3080/)
32B protocol: [`docs/eval-32b.md`](eval-32b.md)

Redraw: `python -m gpu.lab.compare_plate --redraw`

## Numbers

| Stack | 3B long tok/s | 32B long tok/s | 32B smoke | smi after load, 32B |
|---|---:|---:|---:|---:|
| Ollama 0.34.0 `qwen2.5:*` Q4_K_M | **187.3** | **2.54** | 3.18 | 11559 |
| deep-fold NF4 TokenLoop | **35.2** | **2.49** | 2.31 | 11926 |
| llama.cpp b10964 Q4_K_M | **187.0** | **1.52** | 1.55 | 11520 |
| bitsandbytes NF4 | 22.2 *smoke* | — | — | — |
| AWQ / GPTQ+Marlin / ExLlamaV2 / vLLM | SKIP | SKIP | SKIP | SKIP |

SKIP is a row, not a borrowed tok/s. Isolated venvs:
[`docs/competitor-venvs.md`](competitor-venvs.md).

Commands:

```text
python -m gpu.lab.ollama_h2 --model qwen2.5:32b --size 32B
python -m gpu.lab.deepfold_long --size 32B
python -m gpu.lab.llamacpp_h2 --bench
python -m gpu.lab.compare --seed-known
```

## What Ollama actually runs

Ollama 0.34.0 is a scheduler around vendored `llama-server.exe` (llama.cpp
**b10760**). It is not a separate GEMM. Live argv from
`%LOCALAPPDATA%\Ollama\server.log` for `qwen2.5:32b`:

```text
--load-mode none --flash-attn auto -c 2048 -np 1 -b 512 -ub 512
```

`-ngl` is omitted (`NumGPU = -1`). llama-server auto-fits. On this 3080:

| | GPU | CPU / CUDA_Host |
|---|---|---|
| Layers | **33 / 65** (32 blocks + `lm_head`) | remaining **32** blocks |
| Weights | **9559 MiB** | **9367 MiB** |
| KV @2048 | 256 MiB | 256 MiB |

`sched_reserve: graph splits = 2` at `bs=1`. Hidden state (~10 KiB) crosses
the split once per token. Weights stay put. `--load-mode none` is forced on
Windows+CUDA (`disableMmapDefaultReason` → `windows_cuda`) so the CPU suffix
is in RAM, not mmap page faults. Flash Attention enabled. `USE_GRAPHS=1`.
16 AVX2 threads on the 5950X.

3B: **37/37 layers on GPU** → 187 tok/s.

Standalone llama.cpp in this repo used `-ngl 99`, which **aborts auto-fit**
(`n_gpu_layers already set by user to 99`). That is why 32B llama.cpp is
1.52, not 2.54. Same Q4_K_M family, worse placement.

## Overflow: layers vs matrices

**Ollama / llama.cpp.** First *N* transformer blocks fully on GPU. The rest
fully on CPU. Overflow costs CPU Q4_K GEMM plus one activation bounce.

**deep-fold H2 (policy D).** Every layer still runs on the GPU. Embed,
`lm_head`, and all `q/k/v/o` stay resident. Every `down_proj` and tail
`gate`/`up` (L48–63) live in pinned host memory. `CopyRing` copies **96**
packed matrices, **6885 MiB**, into two device slots each decode step.
Serial copy floor **277 ms/tok**; measured long wall **~402 ms → 2.49 tok/s**.
WDDM joins `e_copy` before the next prefetch (depth 1). Reconstruct is still
tile-local in `chr_nf4_gemm`.

The 32B long tie (2.54 vs 2.49) is two different ceilings landing in the
same place: half the net on the 5950X versus 6885 MiB over PCIe at ~24 GB/s.
It is **not** evidence that NF4 decode matches Q4_K mmvq.

## Why 3B is not a tie

Both 3B nets fit. Ollama/llama.cpp Q4_K fused CUDA (mmvq, graphs, FA) is
~187 tok/s. Our resident NF4 TokenLoop is 35.2. `LIVE_MAX_N=32`; decode N=1
does not feed the prefill tile. That gap is the kernel, not overflow.

## Hybrid CPU option (not in this commit)

A later branch can expose overflow policy as a generate option:

- `gpu` — current CopyRing (all compute on device, stream packed tails)
- `cpu-suffix` — Ollama-style: keep a GPU prefix, run leftover **layers**
  on CPU (activations move, weights do not)
- `hybrid` — a shorter CPU suffix than Ollama’s ~50%, rest CopyRing or
  resident, so the CPU does less than half and PCIe moves fewer than 6885 MiB

That is an experiment, not a measured row. Do not write a tok/s for it here.

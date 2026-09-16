# Codec i4c: affine INT4 for CPU decode (group 64)

CPU-suffix weights. Not NF4, not GGUF, not a llama.cpp wrap. Pack from BF16/FP32
(not from NF4). CHR0 stays one-codec-per-file: i4c lives as an in-memory /
sidecar blob for `home=cpu` layers. The `.chr` writer is unchanged.

N=1 decode on Zen3 AVX2 is the contract. Prefill `N>1` may stay a slower path.

---

## Why not NF4 on the CPU

NF4 is a 16-level **nonlinear** LUT. Decode is gather + float FMA. On a 5950X
that GEMV is DRAM-bound at ~15–25 GB/s and **~15 ms/layer** on Qwen2.5-32B.
Ollama’s CPU suffix is affine INT4 at ~6 ms/layer implied.

i4c is affine INT4 with the same FMA skeleton as the NF4 CPU kernel: dequant
nibble → float (`cvtepi8` + `cvtps` × scale), then `fmadd` with float `x`.
Accumulators stay in ymm until the row ends. Q8 activations with a per-group
hsum were a dead end (`down_proj` ~2× slower than NF4). Vector Q8 (no hsum)
is still slower than float-x FMA on this 5950X.

---

## Frozen constants

| Symbol | Value | Meaning |
|---|---:|---|
| `group_size` / `G` | **64** | Two AVX2 INT8 vectors along K. Same pad as NF4 so a sidecar can sit next to CHR0. |
| Levels | 16 | Nibble `0..15` means signed weight **`q - 8`** in `[-8, 7]`. |
| Weight scale | IEEE **binary16** | `s = absmax(group) / 8`, then round-trip through fp16. |
| Zero-point | none | Symmetric. No `zero` blob. |
| Nibble order | **low = even K** | Same as NF4 / compressor.md: `byte = high(k+1)<<4 \| low(k)`. |
| Matrix | `W[M, K]` row-major | Groups do not cross rows. Pad K to 64 with zeros. |

Forbidden in this codec: NF4 LUT, GGUF layouts, Q4_K superblocks, wrapping
ggml, writing `W_hat`, CUDA.

---

## Encode of one group of 64

Input: 64 float32 values `g[0..63]` (pad with 0).

1. Reject non-finite.
2. `a32 = max |g[i]|`. If `a32 == 0` → `a32 = 1`.
3. `s32 = a32 / 8`.
4. `s16 = float32_to_fp16_RNE(s32)`. Must be finite and `> 0`.
5. `s = fp16_to_float32(s16)` (the scale decode will use).
6. `q[i] = round(clip(g[i] / s, -8, 7)) + 8`, then clip to `0..15`.
   Ties-to-even on `round`. Smaller index on a remaining tie.
7. Pack 16 bytes: `data[c] = q[2c] | (q[2c+1] << 4)`.

Decode of one weight: `w = float(q - 8) * s`.

---

## GEMV N=1

`x` is float32 length K (or `[K,1]`). Never quantize `x`.

For each output row `m`, for each group of 64 along K:

1. Unpack 32 packed bytes → 64 signed int4 (`nibble - 8`).
2. Convert to float32, multiply by the row’s fp16 scale for that group.
3. `acc += w * x` via AVX2 FMA. Horizontal sum once at the end of the row.
   A K tail shorter than 64 is scalar.

Rows 4-wide in registers (8 ymm). Persistent thread pool, cap 16, same as
NF4 CPU. GIL released. Never materialize `W_hat`.

---

## Sizes

`K_pad = 64 * ceil(K / 64)`, `n_groups = K_pad / 64`.

| blob | dtype | shape |
|---|---|---|
| `data` | uint8 | `[M, K_pad/2]` |
| `scale` | fp16 | `[M, n_groups]` |

---

## Quality / speed

Target: **≤ 6 ms** for one Qwen2.5-32B decode layer (sum of seven linears,
N=1, 16 threads) so a 28-layer CPU suffix can beat the NF4 suffix (~15 ms/layer)
and sit next to Ollama’s Q4_K budget. Quality is signed INT4, not NF4 and not
Q4_K_M. Pack from BF16 at convert time.

A live `--compute hybrid --cpu-codec i4c` session may pack the sidecar from an
NF4 decode when BF16 is not in RAM. That is NF4∘i4c, not BF16∘i4c, and is not
written into the `.chr`. The AVX2 kernel is the same.

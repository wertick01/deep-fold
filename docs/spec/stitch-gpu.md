# Stitches: CPU `.chr` → GPU (wave 2)

If the kernel, the loader, or [kernel-ampere.md](../kernel-ampere.md) disagree with this page — **this page wins**.
We do not rewrite codecs or the container: [nf4.md](nf4.md), [vq.md](vq.md), [chr0.md](chr0.md), [stitch.md](stitch.md).

## Locked

| Topic | Decision |
|---|---|
| Who owns the bytes | The `.chr` already on disk. The GPU **reads**, it does not recode. |
| NF4 `group_size` | **64**, not 32 and not 128. |
| Nibbles | low 4 bits = `W[r, 2c]`, high = `W[r, 2c+1]`. Not CUDA bitsandbytes packing. |
| Scale | FP16. Dequant: `float32(LUT[nib]) * float32(scale)`, then pack BF16 into the A fragment. |
| NF4 LUT | 16 literals from [nf4.md](nf4.md) §1, bit-exact. |
| Blob layout | **row-major**, as on CPU. Ampere fragment-major is **not** this wave. Permute in registers. |
| 2¹⁶ codebook | do not read it, even if it someday appears in the file. |
| MMA | only `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. Not `wmma`, not `m8n8k4`, not INT4 MMA. |
| Decode tile | `BM=128`, `BN=8` (pad), `BK=256` (multiple of 64), `block=256`, stages=3. |
| Prefill tile (live) | `BM=64`, `BN=8` (N=2..8) / `BN=16` (N=9..16) / `BN=32` (N=17..32), `BK=128`, `block=256`, stages=3. `N=1` — decode launch. `N>32` the host splits into chunks ≤`LIVE_MAX_N`. |
| Prefill n32 / n64 (kernels) | Compiled: `chr_nf4_gemm_prefill_n32` (N=17..32, BN=32) and `prefill_n64` (N=33..64, BN=64), same `BM=64`/`BK=128`. Dispatch by tile width (`N≤8/16/32/64`), not by `kLiveMaxN`. **Live** cap: `kLiveMaxN = LIVE_MAX_N = 32`. n64 plan-only. ncu 2026-09-13, 3B `q_proj`: true n32 **69 µs** vs two n16 **113 µs** (**1.63×**). Numerics 2026-09-14: n32 bit-exact = 2×n16; peak 0.05847 — half-ULP BF16, floor `kernel_floor_ok`. Pair 2026-09-14 (`qwen25-3b-paired-20260914`, `prefill_chunk=32`): TTFT **48 vs 92 ms**, decode **24.8 vs 28.7 tok/s**. NF4-only n32 the same day: 91 ms / 28.6. Old n16 pair: 45 vs 139. n64 not measured. |
| `x` in the kernel | BF16, row-major **`[K, N]`**. `y` — BF16 **`[M, N]`**. HF `[..., K]` is transposed by the host. |
| Memory | Python owns the tensors. The kernel does not `cudaMalloc` on a token. |
| Scratch `W` | **none** in HBM. Dequant only in registers / the packed smem ring. |
| Orig safetensors | **do not open** on the GPU. The oracle is packed + LUT, not `from_pretrained`. |
| Go packages | do not change (`internal/*`, `cmd/chr`). Loader is Python/C++. |
| First model | `Qwen2.5-3B-Instruct`, file `C:\dev\models\qwen25-3b.nf4.chr`. |
| Card | RTX 3080 12 GB, **sm_86**. The display occupies VRAM — budget it in smi. |

## Size formulas (loader rejects a mismatch)

`K_pad = 64 * ceil(K / 64)`, `n_groups = K_pad / 64`.

| Blob | Bytes |
|---|---|
| NF4 `data` | `M * K_pad / 2` |
| NF4 `scale` | `M * n_groups * 2` |
| VQ `index` | `M * (K_pad_vq / 8) * 2` with `K_pad_vq = 8 * ceil(K / 8)` |
| VQ `codebook` | `2 * 256 * 8 * 2` = 8192 |

## Kernel error floor (floor 1)

On random `x` (RMSNorm-calibrated, rms ≈ 1) and real packed `W`:

Elementwise `|Y_gpu − Y_cpu| ≤ max(0.05, ½ ULP BF16(|Y_cpu|))` in float32, where `Y_cpu = dequant_nf4(packed) @ x` (float32 matmul). 0.05 is the floor when `|Y| ~ O(1)`. When `|Y| ∈ [16, 32)` half-ULP BF16 = 0.0625: a flat 0.05 threshold confuses `y` store rounding with a kernel bug. `gpu.nf4.numerics.kernel_floor_ok`.

This is a kernel bug, not quantization. Chat and KL are not this wave.

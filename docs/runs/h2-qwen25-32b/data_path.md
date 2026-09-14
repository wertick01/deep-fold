# H2 data path (this run)

Pinned host image of overflow NF4 (packed‖scale, one arena per matrix).
Two static device slots. One copy stream. Prefetch depth 1.
Dequant only in registers/smem of `chr_nf4_gemm`. No dense `[M,K]` in HBM.

## Machine

- PCIe calibration: 256 MiB / 10.31 ms = 24.2 GB/s (GiB/s) pinned H2D
- formula: `t_ms = size_MiB × 10.31 / 256`
- decide(): overflow=True codec=nf4 vram=12288 MiB
- cap: 9832.5625 MiB resident packed (NF4 packed weights 16599 MiB plus 1800 MiB runtime do not fit 12288 MiB VRAM (short 6111 MiB); NF4 overflow (H2), not VQ.)

## Residency (policy D)

- streamed matrices: 96 (6885.0 MiB)
- resident packed (plan): 9713.9 MiB
- slot (worst overflow matrix): 71.7 MiB → 2.89 ms H2D
- full overflow tape H2D: 6885.0 MiB → **277.3 ms** if serial and copy-bound
- down: DEVICE 0 (0.0 MiB), HOST 64 (4590.0 MiB)
- gate: DEVICE 48 (3442.5 MiB), HOST 16 (1147.5 MiB)
- up: DEVICE 48 (3442.5 MiB), HOST 16 (1147.5 MiB)
- q: DEVICE 64 (850.0 MiB), HOST 0 (0.0 MiB) / k / v / o stay DEVICE
- embed / lm_head: embed: DEVICE 1 (394.5 MiB), HOST 0 (0.0 MiB); lm_head: DEVICE 1 (394.5 MiB), HOST 0 (0.0 MiB)
- HOST down layers: 0..63 (n=64)
- HOST gate+up layers: 48..63 (n=16)

## Load

- HostImage arenas: 96 (6885.0 MiB), pinned=True (6885.0 MiB)
- report.device_mib (HBM weights, not 16601): 9716.041015625
- nvidia-smi after load: 11268 MiB
- torch allocated after load: 9933.0419921875

## Token path (one forward)

1. `CopyRing.arm(host_tape)` — HOST GEMMs in consume order.
2. `prefetch()` — H2D of tape[0] into slot 0 on `copy_stream` (overlaps embed).
3. Each layer: DEVICE q/k/v (CUDA graph if captured) → RoPE/SDPA → DEVICE o → gate/up (DEVICE graph or HOST slot) → down (usually HOST).
4. HOST GEMM: `wait e_copy[s]` → `chr_nf4_gemm(slot views)` → `record e_gemm[s]` → prefetch next tape entry into the other slot.
5. Prefill: this whole tape once **per chunk** (LIVE_MAX_N=32), not per column.
6. Decode N=1: same tape once per token.

- groups: 257 (graphed 177, eager 80)
- GEMMs: DEVICE 353, HOST 96
- graph_mode=linears error=None
- KV: 512.0 MiB at max_seq=2048

## Measured generate

### message 1

- prompt_len=42 prefill=1004.4151000038255 ms (2 chunks)
- decode 2.31339632953258 tok/s over 7 steps (3026 ms, 432.3 ms/tok)
- H2D 61965.0 MiB in 864 copies, 9 forwards (expect 61965.0 MiB = tape × forwards)
- CUDA copy events: None ms (None/0 if timing off)
- serial copy floor ≈ 277.3 ms/tok; wall 432.3 ms/tok. Copy-bound, little overlap.
- quality=True needle=('paris', 'париж')
- smi during/after: 11926.0

### message 2

- prompt_len=35 prefill=992.6561999891419 ms (2 chunks)
- decode 2.324118050318896 tok/s over 7 steps (3012 ms, 430.3 ms/tok)
- H2D 61965.0 MiB in 864 copies, 9 forwards (expect 61965.0 MiB = tape × forwards)
- CUDA copy events: None ms (None/0 if timing off)
- serial copy floor ≈ 277.3 ms/tok; wall 430.3 ms/tok. Copy-bound, little overlap.
- quality=True needle=('berlin', 'берлин')
- smi during/after: 11931.0

### message 3

- prompt_len=45 prefill=1020.9357000130694 ms (2 chunks)
- decode 2.301167788961582 tok/s over 3 steps (1304 ms, 434.6 ms/tok)
- H2D 34425.0 MiB in 480 copies, 5 forwards (expect 34425.0 MiB = tape × forwards)
- CUDA copy events: None ms (None/0 if timing off)
- serial copy floor ≈ 277.3 ms/tok; wall 434.6 ms/tok. Copy-bound, little overlap.
- quality=True needle=('323',)
- smi during/after: 11933.0


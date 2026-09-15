# Ampere kernel: fused dequant-MMA on RTX 3080 12 GB

Hardware (numbers are locked): GA102, **sm_86**, **70 SM**, 8960 CUDA cores, 280 Tensor Cores (4/SM), boost 1.71 GHz, **912 GB/s** GDDR6X, L2 **5 MiB**, registers 64 Ki registers/SM, threads/SM **1536**, warps/SM **48**. Shared: **100 KB/SM**, **99 KB/block** (101376 B). Has `cp.async`. No TMA, no TMEM, no WGMMA, no Blackwell DE.

Contract from [schema.md](schema.md): weights stay compressed in GDDR; tile `(layer, matrix, row_tile, col_group)` → registers → MMA → discard. Do not materialize a full BF16 layer. Two codecs, one skeleton.

Codecs:

| | A: group-wise INT4 / NF4 | B: codebook per matrix |
|---|---|---|
| Symbol | nibble + `scale` per group of 32 or 128 along K | group of 8 weights = 1 or 2 indices |
| Codebook | NF4: LUT 16×BF16 = 32 B | **256×8 BF16 = 4096 B** (keep). **2¹⁶×8 = 1 MiB — drop** (§3) |
| Bits/weight | 4 + 16/G (G=32 → 4.50; G=128 → 4.125) | 1 index×8 bit → 1.00; 2×8 bit → **2.00**; 1×16 bit into 2¹⁶ → 2.00, but the codebook does not fit |

---

## 1. Which MMA, and how dequant feeds Tensor Cores

**Instruction (one for both codecs):**

```
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

SASS: `HMMA.16816.F32.BF16`. Shape 16×8×16. Accumulate in FP32.

Fragments per thread (warp-synchronous, 32 threads):

| Operand | PTX registers | Contents |
|---|---|---|
| A (weights, row) | 4×`.b32` `{a0,a1,a2,a3}` | 8×BF16 |
| B (activations, col) | 2×`.b32` `{b0,b1}` | 4×BF16 |
| C/D | 4×`.f32` `{d0,d1,d2,d3}` | 4×FP32 |

Fallback if BK is not a multiple of 16: `mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32` (A: 2×`.b32`, B: 1×`.b32`). On sm_86 it is weaker — **not the baseline**.

**What not to use**

| PTX | Why not |
|---|---|
| `mma.sync.aligned.m8n8k4.*` | On Ampere this is **FPU**, not Tensor Core |
| `wmma.mma.sync.aligned.m16n16k16.*` | Old path, 2× HMMA.16816 + extra shuffles |
| `mma.sync.*.s32.s4.s4.s32` (`m16n8k32` / `k64`) | Needs INT4 **activations**. Our x is BF16, scales are per group |
| `mma.sync.*.s32.s8.s8.s32` (`m16n8k16` / `k32`) | Only makes sense if we quantize x on the fly (SmoothQuant). Not v1 |
| `wgmma.mma_async` / TMA / TMEM | Hopper / Blackwell |

**How dequant lands in the A fragment (not in HBM):**

```
GDDR packed  --cp.async.cg-->  smem[stage]  --dequant ALU-->  A_regs (BF16)
GDDR x BF16  --cp.async.ca-->  smem_x[stage] --ldmatrix.x2/x4--> B_regs
A_regs, B_regs  --mma.sync.m16n8k16-->  acc FP32
acc stays in registers for the whole K; epilogue: BF16 store to y
```

Codec A, group G, weight `w` in row `m`, column `k`:

1. Packing: 8 nibbles per `uint32`, 32 nibbles per 16 B — `cp.async` granule is 16 B.
2. `nib = (word >> (4*(k&7))) & 0xF` — 2 shifts + AND.
3. INT4: `bf16(nib − 8) * scale[m, k/G]`. NF4: `lut[nib] * scale[...]`, LUT 16×BF16 = **32 B** in smem (or 8×`.b32` in constant registers).
4. The result is packed immediately into `.b32` pairs for `{a0..a3}`. Do **not** write an intermediate BF16 tile to smem (Marlin-style). Otherwise 64×128 BF16 = 16 KB extra per stage.

Codec B, group of 8:

1. 1 index (`uint8`) or 2 indices (`uint8`+`uint8`) per 8 weights.
2. `v = book[i0]` (16 B = 8×BF16). If two indices: `v = book0[i0] + book1[i1]` componentwise in FP32, pack back to BF16 **or** two MMA passes, one vector each (2× more TC, but no add). Recommendation: **add in FP32 → pack BF16**, one MMA.
3. 8 BF16 = exactly half of the per-thread A fragment at k16; two groups → full `{a0..a3}`.

Loading B after x is in smem:

```
ldmatrix.sync.aligned.x2.m8n8.shared.b16   // 1× m16n8k8  B
ldmatrix.sync.aligned.x4.m8n8.shared.b16   // better: enough for k16 at once, 4×.b32 if we take A from smem
```

A after register dequant — **without** `ldmatrix`. If dequant into smem is ever needed: `ldmatrix.sync.aligned.x4.m8n8.shared.b16`.

On GEMV (N=1), 7 of 8 B columns are zeros. Treat this as a TC tax (~1–2 µs on `gate_proj`, §6). The bus matters more. Do not switch to a CUDA-core GEMV: that would break the shared skeleton with prefill.

---

## 2. Tiles that fit in 99 KB with double/triple buffer

Matrix convention: `W[M, K]` row-major compressed, `x[K, N]`, `y[M, N]`.  
`gate_proj`: **M=14336, K=4096, N=1…32**.  
`row_tile` = block of rows `BM`. `col_group` = block along K: `BK`, a multiple of G (32/128) and of 8 (codebook) → **BK ∈ {64, 128, 256}**.

The baseline tile for **decode (N=1, pad BN=8)** and **prefill N≤16 (BN=16)** is the same packed rectangle `BM×BK`.

### Recommended shapes

| Mode | BM | BN | BK | stages | threads | Why |
|---|---|---|---|---|---|---|
| Decode GEMV | **128** | 8 (pad) | **256** | **3** | 256 | we will not have too few blocks along M; 16 K-iterations |
| Prefill N≤16 | **64** | **16** | **128** | **3** | 256 | 8 warps: 4 along M × 2 along N |
| Prefill N≤32 | **64** | **32** | **64** | **2** | 256 | 4 warps along N (4× m16n8) |
| Small matrix (k_proj M=1024) | 64 | 8 | 128 | 3 | 128 | + split-K, see §8 |

### Bytes of one stage (without codebook)

**INT4 / NF4, G=32** (worst scale budget):

| Buffer | Formula | BM=128, BK=256 | BM=64, BK=128 | BM=64, BK=64 |
|---|---|---|---|---|
| packed W | `BM·BK / 2` | **16384 B** | 4096 B | 2048 B |
| scales BF16 | `BM·(BK/32)·2` | **2048 B** | 512 B | 256 B |
| x BF16 | `BN·BK·2` | N=8: 4096 B | N=16: 4096 B | N=32: 4096 B |
| Stage total | | **22528 B** | 8704 B | 6400 B |

G=128: scales 4× smaller (512 B on 128×256). Take the G=32 numbers as the ceiling.

**Codebook, 2 indices × uint8** (2 bit/weight):

| Buffer | Formula | 128×256 | 64×128 |
|---|---|---|---|
| indices | `BM·(BK/8)·2` | **8192 B** | 2048 B |
| x | `BN·BK·2` | 4096 B | 4096 B |
| Stage total (without codebook) | | **12288 B** | 6144 B |

1 index × uint8: exactly half the index buffer.

### smem budget, limit 101376 B

Decode INT4, BM=128, BK=256, stages=3, BN=8, G=32:

```
3 × (16384 + 2048)     = 55296   packed+scales, ring
2 × (8 × 256 × 2)      =  4096   x, 2 stages are enough (x is hot, small)
NF4 LUT                =    32
padding / barrier      =   256
────────────────────────────────
total                    59680 B   (58.3 KiB)     headroom 41 KiB
```

Three stages of x also fit (`+2048`), not needed.

Prefill INT4, BM=64, BK=128, stages=3, BN=16:

```
3 × (4096 + 512)       = 13824
2 × (16 × 128 × 2)     =  8192
────────────────────────────────
                         22016 B   (21.5 KiB)
```

Decode codebook-256, two codebooks (additive), BM=128, BK=256, stages=3:

```
book0 + book1          =  8192   (2 × 4096), live for the whole kernel
3 × 8192               = 24576   indices
2 × 4096               =  8192   x
────────────────────────────────
                         40960 B   (40.0 KiB)
```

**Ceiling that still fits in 99 KB:** INT4 `BM=128, BK=256, stages=4` → `4×18432 + 4096 ≈ 78 KiB`. A fourth stage will not win on a 3080 (GDDR latency is already hidden at 3). `BM=256, BK=256, stages=3` INT4: packed 32 KiB ×3 = 96 KiB **without** scales and x — does not fit. So **128×256×3 is the largest useful tile**.

Align `cp.async` 16 B: `BM` even, `BK` a multiple of 32. Swizzle smem 128 B (XOR col) against bank conflict on `ldmatrix`.

---

## 3. The 256×8 codebook and the 2¹⁶ problem on 3080

### 256 × 8 BF16 = 4096 B — **keep in smem**

Not in L1 as a “global cache”, and not smeared across registers.

| Place | Verdict | Why |
|---|---|---|
| **block smem, 4 KiB (or 8 KiB for two codebooks)** | **yes** | One `__ldg`-like loader at start: 256 threads × 16 B = 4 KiB exactly in 1 round of `cp.async` / `ld.global.L2::128B`. After that every matrix tile hits the same codebook. Broadcast + 32 banks. |
| L1 (28–100 KiB per SM, shared with smem) | no as primary | On sm_86 L1 is not coherent across SMs; `cp.async.cg` bypasses it. If the codebook is left in GDDR, 70 SMs randomly gather 16 B — L1 does not glue that together. |
| Registers | not the whole thing | 4096 B / 256 threads = 16 B/thread = 8 BF16. That is **one** codebook row, not all of them. Storing 256 rows in registers = 2048 B/thread → 1024 registers. Impossible (limit 255). |

Bank conflict on lookup: vector 8×BF16 = 16 B = 4 consecutive banks. 32 threads, random indices → on average ~4–8-way serialize. That is ~20–40 cycles, not GDDR. With two codebooks — two lookups + 8 adds.

Kernel start: the codebook is loaded **once** into smem before the K-loop. Not in the stage ring.

### 2¹⁶ × 8 BF16 = 1 048 576 B — **drop on 3080**

Budgets:

- smem: 1 MiB > 99 KB. Does not live in the block.
- L1: 1 MiB > 100 KB. Does not live on the SM.
- L2: 1 MiB < 5 MiB. Formally “fits” if one such codebook is hot on the card.
- VRAM: 7 matrices/layer × 32 layers × 1 MiB ≈ **224 MiB** of codebooks alone. Space exists. The problem is not gigabytes.

What breaks in the skeleton:

Each group of 8 weights does an **uncoalesced** `ld.global` of 16 B from 1 MiB. On `gate_proj`: 58.72e6 / 8 = **7.34e6** gathers × 16 B = **117.4 MiB** of L2 traffic on top of 14.7 MiB of indices. That is **8×** the useful payload, and it is a gather, not `cp.async` 128 B.

70 SMs hit the same 1 MiB: L2 hit is almost 100% after warmup, but

- no TMA multicast;
- L2 sector is 32 B, we take 16 B → 2× sector overfetch ≈ **234 MiB** of touches;
- `cp.async` does not apply here (address comes from an index, not a linear tile);
- two 2¹⁶ codebooks = 2 MiB, L2 is 5 MiB, plus x/y/other kernels — eviction.

Estimate for `gate_proj` with a 2¹⁶ codebook (after L2 hit, ~2 TB/s L2 on GA102 — order of magnitude): 117 MiB / 2000 GiB/s ≈ 59 µs plus indices 16 µs ≈ **75–120 µs**. 4–5× worse than codebook 256. On a full 8B token we still fit in 10 tok/s, but the **“tile → smem → MMA” skeleton falls apart**: dequant becomes a random global load.

**Decision:** on the 3080, only codebook **256×8**; at 2 bit/weight — **two** such codebooks (AQLM additive, 2×uint8). Leave the 2¹⁶ mode as an offline “quality” flag, not as an inner kernel. If a 16-bit index is ever needed — slicing the codebook into 256-vector pages and keeping the **current page** in smem will not work: indices within a tile jump across the whole 1 MiB.

Bottom line: **keep 256, drop 65536.**

---

## 4. Threads and warps: GEMV N=1 vs GEMM N=16

One warp = one `m16n8k16` instruction per beat of the shape, i.e. it owns **16 rows of M × 8 columns of N × 16 along K**.

### Decode, N=1, pad BN=8, block 256 threads = 8 warps

```
block: BM=128, BN=8, BK=256

warp (wx, wy) = (0..7, 0):   all 8 warps stand along M
  warp w computes rows [16w .. 16w+15], all 8 “columns” of N
  the live column is only n=0; n=1..7 = 0 in the B fragment

thread lane:
  A: dequants its 8 BF16 of the current k-subtile (k16), writes {a0..a3}
  B: lanes 0–3 hold x[k .. k+15] broadcast (shuffle / ldmatrix from smem_x)
     the other N-columns are zeros — one `mov.b32 b, 0` for the dead ones
```

A coverage: 8 warps × 16 rows = 128 = BM.  
K-loop: 256/16 = 16 MMA per warp per stage.  
TC tax: 8× along N, 1× useful → 12.5% TC. At 61 TFLOPS dense-BF16 that is still ~2 µs vs ~36 µs of bus (§6).

The alternative of “8 output elements on CUDA cores” is **forbidden by the contract**: it would break prefill and the two codecs.

### Prefill N=16, block 256 threads = 8 warps

```
block: BM=64, BN=16, BK=128

warp (wm, wn), wm∈{0,1,2,3}, wn∈{0,1}:
  rows [16·wm .. +15], N columns [8·wn .. +7]

K-loop: 128/16 = 8 MMA per warp per stage
TC utilization along N: 100% (two m16n8 cover 16)
```

N=9..15 — the same kernel, epilogue mask. N=17..32 — a grid along N=2 (two BN=16) **or** a BN=32 block, 8 warps as 2×M × 4×N, BK=64, stages=2.

In this tree the plan (`gpu/nf4/plan.py`) keeps **the same BM=64 / BK=128 / stages=3** as the live n16, and adds BN=32 and BN=64. Live dispatch is n8/n16; TokenLoop slices by 16. The BK=64 / stages=2 row above is an old sketch; do not raise `kLiveMaxN` without the oracle and ncu. The next TTFT bottleneck is N≥17 (167 vs 52 ms); ncu on the current GEMM is ~5% DRAM.

### Who dequants which `col_group`

Linear address of the input:  
`base = ((layer, matrix) → W_ptr) + row_tile * stride_row + col_group * stride_col`.

INT4 G=32, BK=256: one `col_group` = 256 weights = 8 scale groups per row. Thread `(warp, lane)` computes:

- `row = row_tile*BM + warp*16 + (lane % 16)` — the actual A mapping in m16n8k16: 2 rows per thread are not consecutive, as in PTX (`lane%4`, `lane/4`). Put packed into smem **in fragment layout** so dequant does not do cross-lane. Practical: store packed as 8 nibbles convenient for this thread’s `a0..a3` (an offline repack, or one at `cp.async`, is not needed if the compressor writes fragment-major).

Recommendation to the compressor: inside `(row_tile, col_group)` store **Ampere fragment order**, not row-major nibbles. Otherwise +permute ~8 `prmt.b32` / thread / k16.

Codebook: `col_group` is a multiple of 8. The thread reads 1 or 2 `uint8` and 16 B from the smem codebook. Two groups of 8 → one A fragment k16.

### K reduction

Accumulator 4×FP32 per thread for the whole K. Split-K only if there are too few blocks (§8). Epilogue: `__float2bfloat16` + `st.global.v2` of 4/8 B, not atomic if split-K=1.

---

## 5. Prefetch: how many tiles ahead via `cp.async`

On sm_86 there is no TMA. We program the ring by hand.

**3 stages for weights, 2 for x. Prefetch = 2 tiles ahead along K.**

```
prolog:
  cp.async packed[0], col_group 0
  cp.async packed[1], col_group 1
  commit; wait_group 1          // packed[0] ready, [1] in flight
  dequant packed[0] → A_regs[0]
  cp.async packed[2], col_group 2

loop s = 0 .. nK-3:
  wait_group 1                  // s+1 ready; s+2 in flight
  dequant packed[(s+1)%3] → A_regs[(s+1)%2]
  mma A_regs[s%2]               // compute tile s
  cp.async packed[(s+3)%3], col_group s+3
  commit

epilog: finish the tail, wait_group 0
```

PTX (16 B granule, bypass L1 — leave L1 for the codebook / x):

```
cp.async.cg.shared.global.L2::128B [%smem], [%gmem], 16;
cp.async.commit_group;
cp.async.wait_group 1;    // ≤1 group in flight → depth 2 = stages-1
```

x (activations, N small, reused by all blocks of one col_group):

```
cp.async.ca.shared.global.L2::128B [%smem_x], [%x], 16;
```

`ca` — let L1 hold x: the same `x[k:k+BK]` is read by every `row_tile` block (on `gate_proj`, 112 blocks × one vector). This is the **only** place L1 is useful. Packed W — `cg`, do not poison L1 with one-shot INT4.

**Why not 2 weight stages.** GDDR6X latency ~300–400 ns ≈ 500–700 cycles @ 1.71 GHz. Dequant of a 128×256 tile: 32768 weights / 256 threads = 128 nibbles/thread ≈ 0.6–1.0 µs ALU. Double-buffer often fails to hide wait+dequant on the first tile of a wave. CUTLASS Ampere default = 3. A fourth stage (prefetch 3) is +16 KiB, does not pay off on a 3080.

**Why not 1 tile ahead on codebook 256.** Lookup is cheaper than INT4-scale, 2 stages already hide 8 KiB of indices. Keep **the same 3-stage skeleton** so the codec is an `#ifdef` inside dequant, not different pipelines.

Barriers: `cp.async.wait_group` + `__syncthreads()` after wait, **before** another warp group reads the same smem stage. Not `cp.async.bulk` (it does not exist). Not `ld.global.nc` into a register + `st.shared`: that is +1 pass and no async overlap.

Commit-group depth: no more than 4 (hardware limit on outstanding `cp.async` groups per SM — stay ≤3).

---

## 6. Time estimate for `gate_proj` 14336×4096 on 3080

Elements: **58 720 256**.  
Peak bus: 912 GiB/s. Dense BF16 TC: ~**61 TFLOPS** (half of 122 with sparsity). FP32 ALU: ~**30 TFLOPS**.

GEMV FLOP: `2·M·N·K = 1.174e8` (N=1) → 1.9 µs on TC. For N=16: 30.6 µs. Everywhere below memory, except an artificially bad dequant.

### Weight traffic

| | INT4 G=32 | INT4 G=128 | 2-bit, 2×uint8 + 2×4 KiB codebooks | 2-bit, u16 index + 2¹⁶ codebook |
|---|---|---|---|---|
| packed / indices | 29.360 MiB | 29.360 MiB | **14.680 MiB** | 14.680 MiB |
| scales / codebook | 3.670 MiB | 0.917 MiB | 8 KiB (once) | codebook 1 MiB + **117.4 MiB gather** |
| x + y (N=1) | 0.008 + 0.027 MiB | same | same | same |
| **sum from bus / L2** | **33.07 MiB** | 30.31 MiB | **14.72 MiB** | ~132 MiB eff. |
| roofline 912 GiB/s | **36.2 µs** | 33.2 µs | **16.1 µs** | indices 16 µs + L2 ~60–90 µs |

### Dequant ALU (whole `gate_proj`, N=1)

INT4: ~6 integer ops/weight + 1 BF16 mul by scale.

`58.72e6 × 7 / 30e12 ≈ 14 µs` if this were FP32 peak. In reality shifts/AND are ~half of peak → **25–40 µs** if counted sequentially. In a 3-stage pipeline this is **hidden** behind 16 K-iterations of `cp.async` (each carries 112 blocks × 18 KiB ≈ 2.0 MiB / 912 GiB/s = 2.2 µs, stage dequant ~2 µs). Result: **+15–25% above roofline**, not 2×.

Codebook 256: 7.34e6 lookups × ~40 conflict cycles / (70×4×1.71e9) ≈ **6 µs**. Hidden completely.

### Total for one `gate_proj`, batch=1

| Codec | Roofline | Realistic (60–75% GDDR + hidden dequant) | Share of 100 ms @ 10 tok/s |
|---|---|---|---|
| INT4 G=32 | 36 µs | **45–55 µs** | 0.05% |
| INT4 G=128 | 33 µs | **42–50 µs** | 0.05% |
| 2-bit codebook 256×2 | 16 µs | **22–30 µs** | 0.03% |
| 2-bit codebook 2¹⁶ | ~75 µs | **90–140 µs** (gather) | still small, but the kernel is bad |

N=16 prefill: the same W bytes, x = 16×4096×2 = 128 KiB (noise). TC 31 µs. INT4: **50–65 µs** (still memory).

### Whole Llama-3 8B token (check ≥10 tok/s)

Weights ~7B. INT4+scales ≈ **4.5 GiB** / 912 ≈ **4.9 ms** peak → **~200 tok/s**. On decode at 55–70% of the bus → **110–140 tok/s**. That is the llama.cpp Q4 caliber (80–110). Our kernel must not be worse than **0.6×** of this; 10 tok/s is **~10×** headroom, needed for 32B@2-bit, not 8B.

`gate_proj` ≈ 27% of layer weights, layer ≈ 1/32 of a token: 55 µs × (1/0.27) × 32 ≈ **6.5 ms** per token if every matrix were like gate. Agrees with 4.9 ms × 1.3 overhead.

2-bit 8B: ~2.0–2.2 GiB / 912 ≈ 2.3 ms peak → **>200 tok/s** with a live kernel.

---

## 7. Traps that are specifically sm_86

1. **No TMA.** No `cp.async.bulk.tensor`, no multicast of x to every block, no automatic swizzle. Tile descriptors are ordinary `ptr + row_tile*s + col_group*t`. Alignment errors will not be caught by a hardware descriptor.

2. **No TMEM / WGMMA.** You cannot “put dequant in TMEM and forget”. Every `mma.sync` requires **this warp** to hold A,B,C in registers. Occupancy is killed by accumulators: 8 warps × 4 float × (several N tiles) — budget registers to 96, not 255.

3. **`cp.async` ≠ `ld.global`.**  
   - Sizes **4 / 8 / 16 B** per thread, not 32.  
   - Dest is smem only, not a register (that is Hopper `ldgsts` vs Ampere).  
   - An unaligned 16 B address = silent fault / truncation. Pad packed INT4 and indices to 16.  
   - `wait_group N` waits until in-flight **groups** ≤ N, not bytes. One group = everything after the previous `commit`.  
   - `cp.async` does not issue `__syncthreads`. Forget the sync — you read someone else’s stage.

4. **L1 on GA10x is small and shared with smem.** Set 99 KB smem → L1 ≈ 28 KB. A 4 KiB codebook in smem is more correct than hoping for L1. `cp.async.cg` for W is mandatory, otherwise 16 KiB of packed evicts the codebook from L1 (if it were there).

5. **1536 threads/SM, not 2048 like A100.** Block 256 × 96 regs = 24576 regs → **2 blocks/SM** by registers (65536). By smem 59 KiB → also 1–2. Occupancy 512/1536 = 33%. Fine for memory-bound. Do not raise the block to 1024: A/B/acc registers will not fit, spilled acc will kill the 45 µs.

6. **No async barrier / mbarrier.** Only `bar.sync` + `cp.async.wait_group`. Do not copy a Hopper pipeline 1:1 from CUTLASS 3.x.

7. **`m8n8k4` is a Volta-tutorial trap.** On sm_86 this is CUDA cores. Profile SASS: it must be `HMMA.16816`, not `FFMA`.

8. **`ldmatrix` bank conflict.** Without XOR-swizzle, 8 warps read one column of x — 32-way. x is small, but dequant of A from linear packed without fragment layout is the same 32-way on nibbles.

9. **Split-K atomics on GDDR6X.** `atomicAdd` BF16/FP32 into y hits the same 14336 rows from 70 SMs. On small M (k_proj=1024) without split-K there will be 16 blocks on 70 SMs. With split-K — atomics. Better FP32 partials in a workspace `[split][M]` + a separate 0.01 ms reduce than `atomicAdd` in BF16.

10. **No DE.** Huffman/ANS is not hidden “for free” in this kernel. That is why they are not in v1 (schema).

11. **sm_86 vs sm_80:** the same MMA and `cp.async`, but **100 KB** smem, not 164. A 256×256×3 tile from A100 does not transfer here.

---

## 8. Launch config

### Decode, `gate_proj` INT4 / codebook-256

```
__launch_bounds__(256, 2)          // 2 blocks/SM target
grid  = dim3(M / 128, 1, 1)        // 14336/128 = 112 blocks
block = dim3(256, 1, 1)            // 8 warps
smem  = 59680                      // INT4, §2; codebook: 40960
dyn_smem: yes, one kernel for both codecs
split_k = 1                        // 112 ≥ 70, 1.6 waves — ok
```

112 blocks × 70 SM: a wave of 70 + a tail of 42. The tail is not ideal, but BK=256 is long — do not cut split-K (atomics cost more than the tail).

### Prefill N=16

```
grid  = dim3(M / 64, 1, 1)         // 224 blocks
block = dim3(256, 1, 1)
smem  = 22016
```

N=32: `grid.y = 2` (along BN=16) **or** `BN=32`, `grid.x = M/64`, same 256.

### Small matrices (k/v_proj M=1024)

```
BM=64, BK=128, block=128
tiles_m = 1024/64 = 16 < 2*70
split_k = 8                        // 16*8 = 128 blocks
workspace = FP32[8][1024]          // 32 KiB, not atomic into y
```

Rule: `if (ceil(M/BM) < 140) split_k = next_pow2(ceil(140 / tiles_m))`, else 1.

### Registers and occupancy

| | target | ceiling |
|---|---|---|
| registers/thread | 80–96 | 128 (otherwise 1 block/SM) |
| smem/block | 40–60 KiB | 99 KiB |
| blocks/SM | 2 | 1 still lives (memory-bound) |
| warps/SM | 16 | 48 max |

`cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, 65536)` — on sm_86 the ceiling is 99 KiB, we ask for 64 KiB.

Cluster / `cg::grid_group`: no. Persistent kernel: not needed at 112 blocks.

### Sketch < 40 lines (skeleton, not product code)

```cuda
// sm_86, BM=128, BK=256, BN=8, stages=3, block=256
__global__ void dq_mma(const uint8_t* W, const half* scale,
                       const uint8_t* idx, const bf16* book,
                       const bf16* x, float* acc_out, int K) {
  extern __shared__ char sm[];          // packed ring + book
  bf16* book_s = (bf16*)sm;             // 256*8, load once
  uint8_t* pk[3];                       // 16 KiB or 8 KiB indices
  // cp.async.cg packed[0], packed[1]; wait_group 1;
  float d0,d1,d2,d3;                    // C fragment
  for (int cg = 0, s = 0; cg < K; cg += 256, s++) {
    // cp.async.cg packed[(s+2)%3] next col_group
    // cp.async.ca x_tile
    // cp.async.wait_group 1; __syncthreads();
    uint32_t a0,a1,a2,a3, b0,b1;
    // codec A: nibbles * scale -> pack a*
    // codec B: v = book[i0] (+ book[i1]); pack a*
    // ldmatrix B from smem_x
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }
  // epilogue: d* -> y[row_tile, n]
}
```

---

## Handoff to other modules

- The compressor writes tile `(layer, matrix, row_tile, col_group)` in **fragment-major** for `m16n8k16`, not pure row-major.
- A 2¹⁶ codebook may be placed in the format file; the v1 kernel does not read it.
- The 12 GB planner does not allocate a BF16 layer: peak smem 60 KiB + split-K workspace in kilobytes.
- Caliber: `gate_proj` INT4 **≤55 µs** on 3080. If >80 µs — pipeline/conflicts, not “the format is at fault”.

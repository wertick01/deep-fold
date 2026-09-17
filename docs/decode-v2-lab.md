# Decode V2 lab log (RTX 3080, 2026-09-17)

One Windows 3080 12 GB, WDDM, greedy, CHR0 NF4. This page is the experiment
log: what the stack is, what we tried, what matched, what missed, and what is
still open. The frozen contract (shapes, gates, non-goals of the *slice*) is
[`decode-v2.md`](decode-v2.md).

**TokenLoop stays the product default** (`gpu/loop/generate.py`, MMA
`chr_nf4_gemm`). Decode V2 is a second resident executor. Do not quote its 3B
tok/s as a product switch. Do not wrap llama.cpp / dual GGUF. Do not edit
`gpu/nf4/nf4_gemm.cu`. Do not mix `demo.py` into git.

Product bar on this card: llama.cpp Q4_K_M 3B long ≈ **187 tok/s**, +10% ≈
**206**. Live Decode V2 GEMV graph: **195** host / **201** device-window
(`C:\dev\models\runs\decodev2-3b-gemv-fuse9-20260917`). Greedy ids match
TokenLoop sequential N=1.

## 1. What the stack is

Python does tokenize and I/O. One greedy decode token is a CUDA graph replay
against static arena pointers. Position and KV length live in GPU buffers, not
Python ints captured into the graph.

CHR/NF4 weights stay packed. N=1 linears are CUDA-core GEMV (`chr_nf4_gemv`),
not tensor-core GEMM. MMA remains available (`CHR_NF4_DECODE=mma`) and is what
TokenLoop uses.

### One 3B decode step (CUDA GEMV path)

```text
x = embed[token]
for layer:
    RMS  (1-CTA, in-place)          → h
    fused QKV + q/k/v bias          → q, k_act, v_act
    flash-decode (RoPE + KV write
      from k_act/v_act, not cache)  → attn
    o_proj GEMV  (add = residual)
    RMS
    fused SwiGLU (gate+up, silu*u)
    down GEMV    (add = residual)
RMS
lm_head GEMV                        → logits
next_token = argmax
commit: token, position += 1
```

Prefill is still the same step, token-at-a-time. That is not the TokenLoop
chunked TTFT. Host tok/s includes a per-token `.item()`; device-window is the
same graph with one host copy after the window. WDDM scatters the host number;
quote both.

### Code map

| piece | path |
|---|---|
| Synthetic specs / oracle / eager / graph | `gpu/decodev2/` |
| 3B CHR load + ids-gate | `gpu/decodev2/load.py`, `run_3b.py` |
| N=1 CUDA kernels | `gpu/nf4/nf4_gemv.cu` |
| ABI | `gpu/include/chr_gpu.h` (`chr_nf4_gemv`, `_qkv`, `_swiglu`, `_attn`, `_rms`, `_rope_kv`) |
| Python bindings | `gpu/nf4/bindings.cpp`, `gpu/nf4/__init__.py` |
| Isolated GEMV bench | `python gpu/nf4/bench_gemv.py` |
| 3B plate | `python gpu/decodev2/run_3b.py --backend gemv` |

### Live kernel rules (GEMV)

- 4 warps split K, NF4 LUT in **smem** (a `__constant__` table serializes when
  every lane hits a different nibble).
- `M >= 4096`: **4 output rows per CTA**, `x` loaded once per 16-wide K-tile
  (`lm_head`, SwiGLU, 32B-rep gate). Override: `CHR_NF4_GEMV_NR=1|2|4|8`.
- 3B `q` / `o` / `down` (`M=2048`) stay **one row**. NR=2/4 underfills the
  68 SMs and slowed `down`.
- Optional `rms_w` on GEMV/QKV/SwiGLU exists for small-M tests. The 3B decode
  path does **not** pass it (see miss below).
- Flash-decode: `n_q × 32` CTAs, 4 warps, two-launch online-softmax merge.
  Cooperative `this_grid().sync()` is off (wrong medium greedy ids on WDDM).

### How to check

```text
python gpu/decodev2/test_oracle.py
python gpu/decodev2/test_state.py
python gpu/decodev2/test_step.py
python gpu/decodev2/test_graph.py
python gpu/nf4/test_gemv.py
python gpu/decodev2/run_3b.py --backend gemv
```

Python: `C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe`. JIT rebuilds
`chr_nf4_ext` when `nf4_gemv.cu` / bindings move. Do not pip into that env.
Do not start a plate while another GPU job is running.

## 2. Live 3B numbers

`max_seq=512`, ignore-EOS 64, prompt_len=48. Greedy match ids
`[2121, 358, 69431, 389, 419, 11618, 11, 358]`.

| path | decode tok/s | prefill ms | notes |
|---|---:|---:|---|
| TokenLoop product plate | **35.2** | ~94 | `docs/compare-3080.md`; MMA |
| TokenLoop, same GEMV process | 8.4 | 358 | `graph=linears`; not the 35.2 plate |
| Decode V2 MMA graph | 30.6 | 10658 | N=1 prefill |
| Decode V2 GEMV, constant LUT | 34 | 6138 | ~50 GB/s class |
| Decode V2 GEMV, smem LUT + split-K | 113 | 7209 | host 113 / device 114 |
| concat gate/up into a second buffer | 108 | 6633 | extra ~0.8 GiB; worse |
| concat QKV into a second buffer | 100 | 5870 | extra copies; worse |
| fused QKV/SwiGLU/RoPE, **no qkv bias** | 129 | 3363 | **wrong ids** (`220` spam) |
| fused + qkv bias + residual, SDPA | 128 | 2973 | host 128 / device 133 |
| naive 1-warp/head attn | 108 | 3298 | slower than SDPA |
| flash-decode + in-place RMS | **182** | 1516 | host 182 / device 195 |
| 4-warp flash + RoPE-in-attn | **178** | 1532 | host 178 / device 196; WDDM host scatter |
| RMS fused into every GEMV CTA | 111 | 1137 | ids matched; **miss** |
| NR=4 on `lm_head` + SwiGLU | **195** | 918 | host 195 / device **201** |
| llama.cpp Q4_K_M | **187** | — | not this run; bar +10% ≈ 206 |

Plates under `C:\dev\models\runs\`:

- fuse9 (live): `decodev2-3b-gemv-fuse9-20260917`
- fuse7b (178/196): `decodev2-3b-gemv-fuse7b-20260917`
- fuse8 RMS miss (111/112): `decodev2-3b-gemv-fuse8-20260917`
- fuse8b restore: `decodev2-3b-gemv-fuse8b-20260917`
- fuse6 (182/195): `decodev2-3b-gemv-fuse6-20260917`
- mmvq 113: `decodev2-3b-gemv-mmvq-20260917`
- MMA: `decodev2-3b-20260917-172115`

Isolated GEMV (`bench_gemv.py`, this hour, WDDM windows jump on small M):

| kind | M | K | gemv µs | GB/s | NR |
|---|---:|---:|---:|---:|---|
| down_proj | 2048 | 11008 | 69 | 173 | 1 |
| lm_head | 151936 | 2048 | **393** | **421** | 4 |
| lm_head same tensors, NR=1 | 151936 | 2048 | 494 | — | 1 |
| 32B-rep gate | 27648 | 5120 | 164 | 458 | 4 |
| 32B-rep down | 5120 | 27648 | 173 | 436 | 4 |

3080 HBM peak is ~760 GB/s. `lm_head` at 421 is the fattest N=1 we have; 3B
`down` at ~170–186 is still far from the bus. Do not claim end-to-end tok/s
from an isolated GEMV.

## 3. What we tried, in order

1. **Decode V2 MMA graph.** Same N=1 step as GEMV, tensor cores. ~31 tok/s.
   Lesson: unused MMA columns on N=1 do not buy 3B decode. GEMV is the
   candidate, not an 8× slogan.

2. **CUDA-core GEMV, LUT in `__constant__`.** ~34 tok/s, ~50 GB/s. Every lane
   hits a different nibble → constant memory serializes.

3. **smem LUT + 4-warp split-K (mmvq ncols=1).** ~113 tok/s. First real jump.
   Staging all of `x` in smem on 3B `down` (K=11008, 22 KiB) kills occupancy;
   do not.

4. **Concat packed gate/up or q/k/v into a second device buffer.** 100–108
   tok/s and extra VRAM. Miss. Fusion must share `x` without copying weights.

5. **Fused `gemv_qkv` + `gemv_swiglu` + standalone RoPE.** Speed up, then
   **space-token `220`** from token 0. Qwen2.5 q/k/v **bias** has to be added
   inside `gemv_qkv`. After bias + residual `add` on o/down: ~128 / 133, SDPA.

6. **Naive attention, one warp per head.** 108 tok/s. 16 warps on ~70 SMs.
   SDPA was faster. Miss.

7. **Flash-decode** (`n_q × 32` CTAs) + **in-place RMS**. 182 / 195. Attention
   finally fills more of the card. RMS is a 1-CTA kernel; do not fuse it into
   152k `lm_head` CTAs.

8. **4-warp flash (128 threads) + RoPE inside attn.** Device-window 196, host
   178 (WDDM scatter vs 182). GQA live slot must come from `k_act`/`v_act`.
   Writing rotated K then reading it back from cache races sibling q-heads.
   `__syncthreads` before the cache store (second half of the head).

9. **Cooperative merge (`this_grid().sync()`).** Wrong medium-llama greedy
   ids on WDDM. Two-launch merge stays. Do not turn coop back on.

10. **Optional `rms_w` inside every GEMV CTA.** 3B 111 / 112. Each of ~152k
    `lm_head` CTAs repeats the same 2048-wide RMS. Reverted; RMS stays 1-CTA.
    `__ldg` on smem is illegal — fused-RMS path uses a plain load flag.

11. **NR=2 on M=2048 (q/o/down).** Isolated `down` 64 µs → ~90 µs. Too few
    CTAs. Reverted.

12. **NR=4 when `M >= 4096`.** `lm_head` 494 → 393 µs same tensors. 3B plate
    195 / **201**, ids still match. q/o/down stay 1-row.

## 4. CUDA 70–85% — open, not closed

Task Manager CUDA util while 3B Decode V2 was generating, same card:

| era | what the user saw |
|---|---|
| early fused (holes obvious) | ~60–70% |
| after flash-decode | 65–85%, mean a bit above 75%, **narrower** graph (model finished faster) |
| after NR=4 `lm_head` | **70–85%** |

That is **not** SM occupancy from Nsight, and it is **not** “15% more CUDA ⇒
15% more tok/s”. WDDM’s CUDA % mixes engine idle, copy, and compute. A
1-CTA RMS and a 16-CTA merge look like holes next to a fat GEMV wave; they
are also cheap compared with `lm_head`. Painting more skinny kernels did not
move the device window (RMS-in-GEMV made it worse).

What we believe, unmeasured until the next session:

- During GEMV the SMs are busy. Remaining holes are **between** graph nodes
  (RMS ×2/layer, attn merge, maybe WDDM graph replay) plus **weight bandwidth**
  inside GEMV (170–421 GB/s vs ~760 peak).
- 201 vs 206 is ~0.12 ms/token. `lm_head` is ~0.39 ms of a ~5.0 ms step.
  Eating the last skinny launches cannot be the whole 5 tok/s unless Nsight
  shows they actually stall the engine that long.
- Next instrument is **Nsight Systems on one captured greedy step** (or
  CUDA graph node timestamps), not another “one more 1-CTA fuse”. Skip
  WikiText. Full Nsight only when we sit down for this.

Do not optimize from Task Manager alone. Use it as a “holes still exist”
flag, then measure which nodes are idle.

## 5. Conclusions we will not re-litigate

- N=1 belongs on CUDA-core GEMV, not MMA. TokenLoop’s product path is still
  MMA until a later switch.
- NF4 LUT in smem, not `__constant__`.
- Share `x` in-kernel (QKV grid, SwiGLU pair, NR rows). Do not concat packed
  weights into a second buffer.
- Qwen q/k/v bias is mandatory.
- Flash-decode + RoPE from `k_act`/`v_act`. No cooperative grid sync on WDDM.
- RMS stays a 1-CTA kernel on the decode path.
- Multi-row only when the grid still fills the SMs (`M >= 4096` → NR=4).
- Host tok/s on WDDM is scatter; always pair with device-window.
- Prefill N=1 is not a TTFT claim.
- Isolated GEMV GB/s is not end-to-end tok/s.

## 6. Next: utilization research, then 14B / 20B

Pavel’s next session: **why CUDA sits at 70–85%**, then **Decode V2 plates on
14B and 20B** for interest. Not in this commit.

### Utilization

- Nsight Systems (or CUPTI / graph node times) on fuse9’s captured greedy
  step. Name the idle gaps. Then decide whether the lever is launch/graph,
  merge, RMS, or GEMV weight throughput (vectorized dequant / more outstanding
  loads) — not another guess.
- Keep `CHR_NF4_GEMV_NR` for A/B on the same tensors.

### 14B (Qwen2.5-14B-Instruct)

- Catalog: `gpu.lab.catalog` slug `qwen25-14b`,
  `C:\dev\models\qwen25-14b.nf4.chr`, hidden **5120**, intermediate 13824,
  GQA 40/8 × 128, vocab ~152k.
- TokenLoop NF4 product plate is **6.56 tok/s** (resident). Decode V2 has **not**
  been run.
- `run_3b.py` is hardcoded to the 3B slug. Need a slug/max-seq runner (or a
  `run_14b.py`) that still **matches greedy ids vs TokenLoop sequential N=1**.
- `load_chr` materializes a **dense** embed table. 14B: vocab × 5120 × 2 B ≈
  **1.5 GiB** extra on top of ~7.5 GiB packed weights + KV. Should still fit
  at `max_seq=512`; measure `memory_allocated`, do not quote `nvidia-smi` as
  “it fit”.
- More linears hit NR=4 (`q` is 5120). CUDA % may look higher because more of
  the step is fat GEMV. That is a measurement, not a promise.
- Do not start 14B BF16. Do not interrupt an active GPU job.

### 20B (internlm2.5-20B-chat)

- Catalog slug `internlm20b`, `C:\dev\models\internlm2_5-20b.nf4.chr`.
  Fused **wqkv** (`internlm_gqa` / InternLM packing). Synthetic Decode V2
  already has that family; checkpoint `plan.family` must match `ArchSpec`.
- TokenLoop NF4 is **5.01 tok/s**, weights ~10 GiB, smi near the 12 GiB cap
  (headroom hundreds of MiB). Decode V2 dense embed + arena may **OOM**.
  First action is a dry load + allocated-bytes, not a 64-token plate.
- `trust_remote_code=True`. Same ids-gate vs TokenLoop. No BF16 20B generate
  baseline on this box.
- Do not start BF16 32B. 32B Decode V2 stays overflow/H2, not this graph.

### Still forbidden unless a later message says otherwise

- Product switch away from TokenLoop MMA
- Editing `nf4_gemm.cu`
- llama.cpp / GGUF dual path
- Claiming 201 tok/s as the shipped 3B number
- WikiText / full Nsight until we sit on utilization
- `demo.py` in git

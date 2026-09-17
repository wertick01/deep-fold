# Decode V2 lab log (RTX 3080, 2026-09-17)

One Windows 3080 12 GB, WDDM, greedy, CHR0 NF4. This page is the experiment
log: what the stack is, what we tried, what matched, what missed, and what is
still open. The frozen contract (shapes, gates, non-goals of the *slice*) is
[`decode-v2.md`](decode-v2.md).

**CLI `--executor auto`** picks Decode V2 on resident NF4 (3B/14B/20B).
TokenLoop remains overflow / CopyRing / VQ. Do not wrap llama.cpp / dual GGUF.
Do not edit `gpu/nf4/nf4_gemm.cu`. Do not mix `demo.py` into git.

llama.cpp Q4_K_M 3B long ≈ **187 tok/s** (ctx 2048). Live Decode V2 ignore-EOS
at `max_seq=2048`: **197** host / **199** device-window, prefill **86 ms**
(`decodev2-qwen25-3b-gemv-maxseq2048-20260918`). The 2026-09-17 `max_seq=512`
plate was **196** host / **200** device-window (`decodev2-3b-gemv-prefill-20260917`).
V2 still attends the full buffer; do not say faster than Ollama. Greedy ids
match TokenLoop sequential N=1. Figure:
[`docs/img/decodev2-3080.png`](img/decodev2-3080.png).

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

Prefill is chunked MMA (`LIVE_MAX_N=32`): 3B prompt_len=48 is **86 ms** at
`max_seq=2048` (127 ms was `max_seq=512`). Same-process TokenLoop was 86 ms
on the 512 plate. Decode stays the
N=1 GEMV graph. Host tok/s includes a per-token `.item()`; device-window is
the same graph with one host copy after the window. WDDM scatters the host
number; quote both.

### Code map

| piece | path |
|---|---|
| Synthetic specs / oracle / eager / graph | `gpu/decodev2/` |
| 3B CHR load + ids-gate | `gpu/decodev2/load.py`, `run_3b.py` |
| N=1 CUDA kernels | `gpu/nf4/nf4_gemv.cu` |
| ABI | `gpu/include/chr_gpu.h` (`chr_nf4_gemv`, `_qkv`, `_swiglu`, `_attn`, `_rms`, `_rope_kv`) |
| Python bindings | `gpu/nf4/bindings.cpp`, `gpu/nf4/__init__.py` |
| Prefill (eager, not graphed) | `gpu/decodev2/prefill.py` |
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
python gpu/decodev2/test_prefill.py
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
| NR=4 on `lm_head` + SwiGLU | **195** | 918 | host 195 / device **201**; N=1 prefill |
| MMA chunked prefill (N=32) | **196** | **127** | host 196 / device 200; ids matched |
| llama.cpp Q4_K_M | **187** | — | not this run; ctx 2048; not a bake-off vs 196 |

Plates under `C:\dev\models\runs\`:

- fuse9 (live decode, N=1 prefill 918 ms): `decodev2-3b-gemv-fuse9-20260917`
- prefill chunks (127 ms): `decodev2-3b-gemv-prefill-20260917`
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

13. **MMA chunked prefill (`LIVE_MAX_N=32`).** Prompt is no longer 48 GEMV
    steps. 3B prefill **918 → 127 ms**, greedy ids still match. Same-process
    TokenLoop prefill was 86 ms. Decode graph unchanged (~196 / 200). Do not
    capture prefill. CopyRing still out of scope.

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
- Prefill is MMA chunks of `LIVE_MAX_N` (32), not the N=1 GEMV graph.
  Token-at-a-time prefill is not a TTFT claim.
- Isolated GEMV GB/s is not end-to-end tok/s.

## 6. 14B and 20B plates; Nsight is Coming soon

14B and 20B Decode V2 GEMV are measured at `max_seq=2048` (below).
**Coming soon:** why 3B CUDA sits at 70–85% (Nsight). CLI `--executor auto`
uses Decode V2 on resident NF4. Do not start a plate while another GPU job is
running — overlapping 20B writes are not numbers.

### Utilization (3B, Coming soon)

- Nsight Systems (or CUPTI / graph node times) on fuse9’s captured greedy
  step. Name the idle gaps. Then decide whether the lever is launch/graph,
  merge, RMS, or GEMV weight throughput — not another guess.
- Keep `CHR_NF4_GEMV_NR` for A/B on the same tensors.

### 14B (Qwen2.5-14B-Instruct) — 2026-09-17

`python gpu/decodev2/run_3b.py --slug qwen25-14b --backend gemv`
`max_seq=512`, ignore-EOS 64, prompt_len=48.

- CHR: 337 NF4 linears 7088 MiB + packed embed 395 MiB, **untied** lm_head,
  total 7484 MiB.
- Spec: `llama_swiglu`, 48 layers, hidden 5120, GQA 40/8 × 128, vocab 152064.
- Dense embed for graph gather: **1485 MiB** (packed rows were 395).
- VRAM after Decode V2 load: **9130 MiB** allocated / 9838 reserved / peak 9688.
  TokenLoop ctor +97 MiB. Headroom on 12 GB is real; do not quote `nvidia-smi`.
- Greedy ids matched TokenLoop sequential N=1:
  `[5050, 11618, 1526, 279, 4746, 315, 6993, 323]`.
- TokenLoop this process: **7.44 tok/s** host (`graph=linears`, prefill 315 ms).
  Product 14B plate remains **6.56**.
- Decode V2 GEMV graph, MMA prefill chunks of 32: **55.6 tok/s** host /
  **55.3** device-window, prefill **333 ms** (was 798 ms token-at-a-time),
  `max_seq=512`. CUDA graph captured. Ids still match.
- Exclusive Q4_K long, ctx 2048 (2026-09-18): Ollama `qwen2.5:14b` **58.9**,
  llama.cpp auto-fit **69.9**. Overlapping Ollama **5.95** is not a number.
- Matched-capacity Decode V2, `max_seq=2048` (2026-09-18 exclusive):
  **57.5 tok/s** host / **58.1** device-window, prefill **322 ms**. Still
  under Ollama 58.9 / llama.cpp 69.9. V2 attends the full buffer.
  Plate: [`docs/runs/decodev2-14b-maxseq2048/`](runs/decodev2-14b-maxseq2048/).
- Plates: chunked prefill `decodev2-qwen25-14b-gemv-prefill-20260917`;
  N=1 prefill `decodev2-qwen25-14b-gemv-20260917`;
  load-only `decodev2-qwen25-14b-load-20260917`.
- Q4_K long: [`docs/runs/ollama-h2-14b/`](runs/ollama-h2-14b/),
  [`docs/runs/llamacpp-h2-14b/`](runs/llamacpp-h2-14b/).

### 20B (internlm2.5-20B-chat) — 2026-09-17

`python gpu/decodev2/run_3b.py --slug internlm20b --backend gemv --expect-ids …`
`max_seq=512`, ignore-EOS 64, prompt_len=29. TokenLoop ctor in the same process
does not fit; ids dumped first (`--tokenloop-ids-only`).

- CHR: 241 NF4 linears 9774 MiB + packed embed 288 MiB, **untied** lm_head,
  total 10063 MiB. Dense embed for graph gather: **1084 MiB**.
- Spec: `internlm_gqa`, 48 layers, hidden 6144, GQA 48/8 × 128, vocab 92544,
  fused wqkv, `trust_remote_code=True`.
- VRAM after Decode V2 load: **11463 MiB** allocated / 12254 reserved / peak
  12153. Headroom is real but thin; do not quote `nvidia-smi`.
- Greedy ids matched TokenLoop sequential N=1:
  `[2421, 260, 33540, 519, 395, 11771, 560, 713]`.
- Product TokenLoop smoke remains **5.01** tok/s (`max_seq=512`). Long
  travelogue is **4.59** (`max_seq=1024`).
- Q4_K long, ctx 2048: Ollama `internlm2.5:20b-q4km` **11.53**, llama.cpp
  auto-fit **11.87**. Decode V2 `max_seq=512` was **41.0** host.
- Decode V2 GEMV graph, exclusive card, MMA prefill chunks of 32, `max_seq=2048`
  (2026-09-18): **40.0 tok/s** host, prefill **220 ms**. Device window **20.7**
  is the second 64-token pass at reserved 12256 / 12288 MiB — WDDM scatter,
  same class as 25.6, not decode. Quote host **40**. Plate:
  [`docs/runs/decodev2-20b-maxseq2048/`](runs/decodev2-20b-maxseq2048/).
- The 2026-09-17 exclusive `max_seq=512` plate was **41.0 tok/s** host, prefill
  **215 ms** (was 566 ms N=1). Device window **25.6** is the second 64-token
  pass at 11463/12254 MiB. N=1-prefill exclusive was **43.4 / 41.6**.
  CUDA graph captured. 8-id gate matches; do not treat overlapping 7–10 tok/s
  plates as this number.
- Plates: chunked `decodev2-internlm20b-gemv-prefill-clean-20260917`;
  N=1 exclusive `decodev2-internlm20b-gemv-20260917b`;
  ids dump `decodev2-internlm20b-tl-ids-20260917`;
  load-only `decodev2-internlm20b-load-20260917`.
- Overlapping reruns (7.37/8.81, 0.90/1.49, 10.5/14.6, 7.7/8.8) are WDDM
  paging. Do not quote them.
- Not a product switch. Do not start BF16 20B/32B.

### CLI (`--executor auto`) — 2026-09-17

Default `deepfold run` / `chat` is `--executor auto`: resident NF4
(3B/14B/20B) uses `gpu/decodev2/session.py`; overflow / VQ / CopyRing stay
on TokenLoop. Force MMA with `--executor tokenloop`; force V2 (or refuse)
with `--executor decodev2`. **In progress:** TTY chrome / agent layout
(neighbor chat). Packed embed and Nsight are still later. Do not start a GPU
plate from this wiring.

### Hard-12 vs Ollama (2026-09-18)

Independent turns, `max_new=256`, `max_seq=2048`, same fixture as
`gpu/lab/hard.py`. Not the ignore-EOS 64 plateau.

| Model | Decode V2 | Ollama Q4_K | llama.cpp |
|---|---|---|---|
| 3B | **8/12** · **190.9** tok/s · TTFT **175** ms | **7/12** · **197.2** tok/s | Coming soon |
| 14B | **11/12** · **54.8** tok/s · TTFT **693** ms | Coming soon | Coming soon |
| 20B | **9/12** · **35.2** tok/s | **9/12** · **13.2** tok/s | Coming soon |

Ollama tok/s is the mean of 12 `decode_tok_s` in `plate.json` (3B median **184.5**, 20B median **11.6**). 20B V2 **35.2** is not ignore-EOS **40**. Table: [`docs/img/hard-v2-ollama.png`](img/hard-v2-ollama.png). Evidence: [`docs/runs/hard-decodev2-3b/`](runs/hard-decodev2-3b/),
[`docs/runs/hard-decodev2-14b/`](runs/hard-decodev2-14b/),
[`docs/runs/hard-decodev2-20b/`](runs/hard-decodev2-20b/),
[`docs/runs/ollama-hard-3b/`](runs/ollama-hard-3b/),
[`docs/runs/ollama-hard-20b/`](runs/ollama-hard-20b/).

### Coming soon

- Ollama 14B hard-12
- llama.cpp hard-12 (3B / 14B / 20B)
- 3B Nsight utilization 70–85%

### In progress

- CLI TTY chrome / agent layout (neighbor chat)

### Still forbidden unless a later message says otherwise

- Product switch of overflow 32B away from CopyRing
- Editing `nf4_gemm.cu`
- llama.cpp / GGUF dual path
- Quoting overlapping 20B jobs or 20B device-window 25.6 / 20.7 as decode
- WikiText / full Nsight until we sit on 3B utilization
- `demo.py` in git

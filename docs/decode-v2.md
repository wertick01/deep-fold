# Decode V2 — resident executor (synthetic first)

Lab log (what we tried, plates, CUDA 70–85% open, 14B/20B next):
[`decode-v2-lab.md`](decode-v2-lab.md).

Status: synthetic stack is green. Qwen2.5-3B `.chr` loads into Decode V2.
Greedy ids match TokenLoop on sequential N=1 (prompt 48, 8 new tokens), MMA
and GEMV. Ignore-EOS 64 at `max_seq=512`: Decode V2 graph **195 tok/s** host /
**201** device-window (fused QKV+SwiGLU+RoPE-in-attn+4-warp flash-decode+in-place
RMS + NR=4 GEMV on `lm_head`/SwiGLU). Host 195 is WDDM scatter vs the earlier
178/182 plates; the device window moved 196 → 201. Prefill is still
token-at-a-time (~0.9 s), not the TokenLoop chunked TTFT. TokenLoop remains
the product default. Product bar is still ~206 (+10% vs llama.cpp ~187).
Today's TokenLoop recheck in the GEMV process was ~8.4 tok/s
(`graph=linears`); the published 3B plate is still **35.2**. Task Manager CUDA
while 3B ran is still **70–85%**; that is an open measurement, not a closed
kernel story ([lab log §4](decode-v2-lab.md#4-cuda-70–85--open-not-closed)).

The ~50 GB/s GEMV wall was a `__constant__` NF4 LUT: every lane hits a
different nibble, so constant memory serializes. The live kernel copies the
16-float table into smem once per CTA, four warps split K (llama.cpp mmvq
`ncols=1` layout). `M >= 4096` packs **4 output rows per CTA** so `x` is
loaded once per K-tile (`lm_head`, SwiGLU, 32B-rep gate). 3B `q`/`o`/`down`
(`M=2048`) stay one row: NR=2/4 underfills the SMs. Isolated 3B `down_proj`
is **69 µs / 173 GB/s** this hour (earlier 1-row plate 64.5 / 186). Isolated
`lm_head` 151936×2048 is **393 µs / 421 GB/s** at NR=4 vs **494 µs** at NR=1
same tensors. Do not stage all of `x` in smem (22 KiB on 3B down kills
occupancy). MMA windows on WDDM still jump; quote GEMV microseconds, GB/s,
and the same-process ratio. `CHR_NF4_GEMV_NR=1|2|4|8` overrides the pick.

Upstream note: `f891996`. Plan A graphs in `gpu/loop/graph.py` cover NF4
GEMM groups only and copy each group's input into a static buffer. Decode V2
does not extend that plan. It is a second resident executor with its own
state, arena, and step graph. `TokenLoop` stays the production path until a
later switch.

## 0. Goals

1. One decode token, batch=1, greedy. Prefill is a sequence of the same
   write/attend steps (or a later dedicated plan). Not multi-request
   throughput.
2. Python does tokenization and I/O. The next token id must not wait on
   Python reading the previous id.
3. Position and KV length are **GPU buffer contents**, not Python ints
   captured into a CUDA graph.
4. Weights stay CHR/NF4. This slice uses in-memory NF4 blobs from
   `gpu.tests.nf4_oracle.encode_nf4`. No second codec. No GGUF. Do not
   edit `gpu/nf4/nf4_gemm.cu` here.
5. Overflow / CopyRing / parallel-MLP / speculative decoding are out of
   scope until the resident synthetic step is correct.

Acceptance for this slice: CPU oracle and GPU eager (and GPU graph, when
CUDA graphs capture) agree on greedy ids for seeded synthetic models.
Logits vs the float32 NF4 oracle may differ by bf16/kernel rounding;
argmax must match on the published seeds, and max-abs vs oracle is
reported with a fixed cap.

## 1. Package

`gpu/decodev2/`. Do not import `TokenLoop` to run a step. Reuse formulas
via `gpu.decodev2.rope` (copied from `gpu.loop.generate._rope`, not
re-derived). Reuse `gpu.tests.nf4_oracle` for encode/decode/matmul.
Reuse `gpu.loop.generate.split_internlm_wqkv` / `split_concat_qkv` for
fused QKV. Reuse `gpu.nf4.nf4_gemm` only inside `gpu.decodev2.linear`
when CUDA is up.

Standalone tests, same style as `gpu/loop/test_decode_tax.py`:

```text
python gpu/decodev2/test_oracle.py
python gpu/decodev2/test_state.py
python gpu/decodev2/test_step.py
python gpu/decodev2/test_graph.py
```

Absent CUDA is `SKIP`, never a silent pass (`gpu.tests.skips`).

## 2. Synthetic models

`ArchSpec` in `gpu/decodev2/plan.py`. Two frozen correctness shapes plus a
timing-only toy (`MEDIUM_LLAMA`, still no checkpoint):

| name | family | layers | hidden | q/kv/hd | intermediate | vocab | max_seq |
|---|---|---:|---:|---|---:|---:|---:|
| `TINY_LLAMA` | `llama_swiglu` | 2 | 64 | 4/2/16 | 128 | 32 | 16 |
| `TINY_INTERNLM` | `internlm_gqa` | 2 | 64 | 4/2/16 | 128 | 32 | 16 |
| `MEDIUM_LLAMA` | `llama_swiglu` | 4 | 512 | 8/2/64 | 1024 | 128 | 32 |

`hidden == n_q * head_dim`. `n_q % n_kv == 0`. `hidden` and `intermediate`
are multiples of 64 (NF4 groups). Embeddings are dense bf16 `[vocab,
hidden]`. Linears are NF4. RMS weights are bf16 `[hidden]`. Optional
bias is allowed and default **off**.

`llama_swiglu`: seven matrices per layer — q, k, v, o, gate, up, down.
`internlm_gqa`: fused `wqkv` with InternLM packing
(`split_internlm_wqkv`), then o, gate, up, down.

Seeded factory: `gpu.decodev2.synth.build(spec, seed) -> SynthModel`.
Finite weights only. NF4 encode is the spec path (`encode_nf4`).

## 3. CPU oracle

`gpu.decodev2.oracle.step_logits(model, token_ids, pos0) -> f32 [n, vocab]`
teacher-forced: each position uses the **given** id, not its own argmax.

Math, all float32:

- embed gather
- RMSNorm: `x * rsqrt(mean(x^2)+eps) * w` (Qwen-style, no bias)
- NF4: `decode_nf4` then `matmul_f32` (`Y = W_hat @ x` with x `[K,N]`)
- RoPE: `gpu.decodev2.rope` (rotate-half)
- attention: GQA **without** repeating KV. Scores use
  `scale = head_dim**-0.5`. Invalid tail (`t >= valid_len`) is **masked
  to -inf** before softmax. Zeros in the tail are not a mask.
- residual adds, SwiGLU `silu(gate)*up` then down. SiLU after the full
  gate GEMM, never on a split-K partial.
- final RMS + lm_head (untied)

Greedy ids: `argmax` of those logits.

## 4. GPU state

`DecodeState` owns device tensors (shape `[1]` unless noted):

| buffer | dtype | role |
|---|---|---|
| `token` | int64 | current input id |
| `position` | int64 | slot to write this step |
| `valid_len` | int32 | KV prefix length **after** this step's write, used by attention |
| `next_token` | int64 | greedy id, written by the step |
| `finished` | uint8 | EOS or cancel |

`GraphSafeKV`:

- storage `[n_layers, max_seq, n_kv, head_dim]` bf16, like `KVCache`
- attention sees the **full** `max_seq` axis plus an additive mask from
  `valid_len`. No `[:,:,:python_seq]` on the hot path.
- write: `index_copy_` (or equivalent) indexed by the **GPU** `position`
- reset of a request: zero `position`/`valid_len`/`finished`; do **not**
  require zeroing KV. Attention must ignore a dirty tail.

`Arena`: every activation and logit buffer allocated at load. No
`cudaMalloc`, `.item()`, `.cpu()`, or host sync between layers.

RoPE tables `[max_seq, head_dim]` built once. A step does
`index_select(table, 0, position)` — `position` is the GPU buffer.

## 5. Resident step (eager, then graph)

Two entry points (no `.item()` on the hot path):

`forward_decode(state, weights)` — teacher-force / body:

1. `x = embed.index_select(0, token)`
2. for each layer: RMS → QKV NF4 → RoPE → KV write at `position`.
   After the first write, `valid_len = position + 1` (tensor ops) so
   attention sees the new slot. Masked GQA SDPA → o_proj → residual →
   RMS → gate/up → SiLU* → down → residual
3. final RMS → lm_head → `next_token = argmax(logits)`
4. leave `token` and `position` unchanged

Teacher-force a prompt id: `load_token(id); forward_decode; position += 1`.

`greedy_decode(state, weights)`: `forward_decode` then `commit_step`
(`token = next_token`, `position += 1`, `finished |= eos`).

Python may copy `next_token` to host **after** the step for display.
The following greedy step must not depend on that copy.

CUDA graph: capture `greedy_decode` against the static arena/state
pointers. Replay must produce the same `next_token` as eager for a
recorded sequence of positions `0..max_seq-2`. Changing `valid_len` by
mutating the tensor (not recapturing) is required. Bucket recapture is
forbidden in this slice (`max_seq` is the static capacity).

Do not copy activations into per-GEMM static inputs the way
`GraphedGemmGroup` does. Adjacent ops share arena addresses.

## 6. Linear primitive

`gpu.decodev2.linear.nf4_linear` with CUDA backends:

- ``mma`` (default): existing `chr_nf4_gemm` tensor cores. TokenLoop stays here.
- ``gemv``: `chr_nf4_gemv` CUDA-core split-K, smem LUT, FP32 LUT×scale×x.
  Select with `set_linear_backend("gemv")` or `CHR_NF4_DECODE=gemv`.

A specialized GEMV is an ablation, not an 8× promise from unused MMA columns.
Pick by `(M,K)` microbench, not a slogan. Do not edit `nf4_gemm.cu` for this.

3080 median µs / launch (`python gpu/nf4/bench_gemv.py`, 5 windows × 40 iters).
`gemv_lo`/`gemv_hi` are the min/max window means. GB/s is packed+scale+x over
median GEMV time. Small `q`/`k`/`v`/`o` stay launch-bound; fat K/M approach
HBM:

| kind | M | K | mma_us | gemv_us | gemv/mma | GB/s | gemv_lo | gemv_hi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q_proj | 2048 | 2048 | 155.6 | 58.4 | 0.38 | 38.2 | 53.8 | 80.7 |
| k_proj | 256 | 2048 | 129.2 | 63.8 | 0.49 | 4.4 | 56.5 | 95.4 |
| v_proj | 256 | 2048 | 100.6 | 56.3 | 0.56 | 5.0 | 53.9 | 82.3 |
| o_proj | 2048 | 2048 | 157.8 | 54.5 | 0.35 | 40.9 | 53.6 | 79.0 |
| gate_proj | 11008 | 2048 | 276.4 | 78.2 | 0.28 | 153.2 | 60.1 | 90.1 |
| up_proj | 11008 | 2048 | 273.4 | 93.6 | 0.34 | 128.0 | 71.6 | 97.5 |
| down_proj | 2048 | 11008 | 338.2 | 69.4 | 0.21 | 172.8 | 68.0 | 87.0 |
| lm_head_rep | 16384 | 2048 | 365.1 | 104.9 | 0.29 | 169.9 | 88.9 | 108.3 |
| lm_head | 151936 | 2048 | 2434.0 | 392.7 | 0.16 | 421.0 | 378.6 | 401.3 |
| 32b_q | 5120 | 5120 | 417.6 | 87.0 | 0.21 | 160.2 | 84.1 | 109.6 |
| 32b_k | 1024 | 5120 | 161.8 | 65.4 | 0.40 | 42.8 | 55.6 | 77.2 |
| 32b_gate | 27648 | 5120 | 1407.0 | 164.3 | 0.12 | 457.7 | 161.6 | 169.4 |
| 32b_down | 5120 | 27648 | 1480.0 | 172.7 | 0.12 | 435.8 | 168.3 | 176.4 |

`MEDIUM_LLAMA` Decode V2 median ms / greedy step, prompt excluded
(`python gpu/decodev2/bench_step.py`): MMA eager 13.8 / graph 0.817; GEMV
eager 13.0 / graph 0.698. Graph vs eager is Python launch tax on a 4-layer
toy, not a 3B tok/s number. N=1 linears pass a `[K]` view into the kernel
instead of `.t().contiguous()`.

QKV: `chr_nf4_gemv_qkv` is one grid over `q.M+k.M+v.M` rows sharing `x` (no
packed copy). Qwen2.5 q/k/v **bias is added in that kernel**; dropping it
emits space-token `220` from token 0. Gate/up: `chr_nf4_gemv_swiglu` (NR
gate+up pairs share `x` when `M >= 4096`, else one pair per CTA, writes
`silu(g)*u`). CPU RoPE+KV write stays
`chr_nf4_rope_kv`; CUDA folds it into flash-decode.
o/down residual: `chr_nf4_gemv` with `add` aliased to `y`. RMS writes
in-place (`chr_nf4_rms`). Attention is flash-decode: `n_q * 32` CTAs of
**4 warps** (128 threads, one pass at `hd=128`), each walks
`t = split, split+32, ...` then a second-launch merge of online-softmax
partials. CUDA-path RoPE+KV write is fused into that kernel: every GQA q-head
rotates the current K/V from `k_act`/`v_act` (the live slot is not read back
from cache — sibling heads would race). Cooperative `this_grid().sync()`
merge returned wrong medium-llama greedy ids on WDDM; do not turn it back on.
One warp per head was a miss (108 tok/s): 16 warps on 70 SMs. Concatenating
packed q/k/v or gate/up into a second buffer is also a miss (100–108).

## 7. Correctness gates

CPU (no CUDA):

- LUT / encode / decode still via `nf4_oracle.selfcheck` import or a
  one-liner that calls it
- oracle greedy is deterministic on `(TINY_LLAMA, seed=0)` and
  `(TINY_INTERNLM, seed=1)` for a fixed 8-token prompt
- dirty-tail: KV slot `max_seq-1` filled with large values, `valid_len=1`,
  oracle attention ignores it (argmax / weights)

CUDA (skip if no device):

- eager GPU greedy ids == oracle greedy ids on those two seeds (prompt
  length 4, then 8 decode steps, or teacher-forced 8 positions — both)
- teacher-forced max-abs logit vs oracle is recorded; fail if any
  position's argmax disagrees
- graph replay == eager `next_token` and `token` for 8 steps
- `position` / `valid_len` after n steps equal n (int, after one sync)
- no allocated-bytes growth across 32 extra steps (`torch.cuda.memory_allocated`)
- reset then a second request: ids match a fresh `DecodeState`
- EOS: `finished` becomes 1 when next id is `eos_id`; runner stops
- do not write KV past `max_seq-1`

## 8. Explicit non-goals (this slice)

- `gpu/loop/generate.py` production switch (TokenLoop stays MMA)
- editing `nf4_gemm.cu`
- CopyRing / HOST groups inside the graph
- parallel MLP split
- FlashInfer
- C++ executor
- 32B Decode V2 / BF16 32B

14B and 20B Decode V2 plates are **next research**, not this slice's
acceptance. Notes and VRAM traps: [lab log §6](decode-v2-lab.md#6-next-utilization-research-then-14b--20b).

## 9. Files

| path | role |
|---|---|
| `plan.py` / `rope.py` / `state.py` / `kv.py` / `arena.py` | frozen contract |
| `synth.py` / `oracle.py` / `test_oracle.py` | tiny NF4 models + f32 oracle |
| `linear.py` / `ops.py` / `step.py` / `test_step.py` | eager step (`mma` or `gemv`) |
| `graph.py` / `runner.py` / `test_graph.py` | full greedy CUDA graph |
| `load.py` / `run_3b.py` | Qwen2.5-3B `.chr` ids-gate + ignore-EOS |
| `bench_step.py` | `MEDIUM_LLAMA` MMA vs GEMV, eager + graph |
| `gpu/nf4/nf4_gemv.cu` / `test_gemv.py` / `bench_gemv.py` | CUDA-core N=1 candidate |
| [`docs/decode-v2-lab.md`](decode-v2-lab.md) | experiment log, CUDA util open, 14B/20B next |

```text
python gpu/decodev2/test_oracle.py
python gpu/decodev2/test_state.py
python gpu/decodev2/test_step.py
python gpu/decodev2/test_graph.py
python gpu/nf4/test_gemv.py
python gpu/decodev2/run_3b.py
```

## 10. Qwen2.5-3B (first checkpoint)

`python gpu/decodev2/run_3b.py`. Loads `qwen25-3b.nf4.chr` via `load_model`,
aliases packed linears, materializes a dense embed table (~594 MiB) so gather
is graph-safe. RoPE tables come from the same `rotary_emb` TokenLoop uses.
`tied_embed` is allowed; `lm_head` is still the NF4 linear (shared packed).

Live 2026-09-17, `max_seq=512`, ignore-EOS 64, prompt_len=48:

| path | decode tok/s | prefill ms | notes |
|---|---:|---:|---|
| TokenLoop, GEMV-session recheck | 8.4 | 358 | `graph=linears`, same process; not the 35.2 plate |
| TokenLoop product plate | **35.2** | ~94 | `docs/compare-3080.md`, not this hour |
| Decode V2 MMA graph | 30.6 | 10658 | N=1 prefill |
| Decode V2 GEMV graph (old LUT) | 34.1 | 6138 | ids matched; ~50 GB/s class |
| Decode V2 GEMV graph (split-K + smem LUT) | 113 | 7209 | ids matched; host 113 / device-window 114 |
| Decode V2 GEMV + concat gate/up | 108 | 6633 | extra ~0.8 GiB weights; worse |
| Decode V2 GEMV + concat QKV | 100 | 5870 | extra copies of q/k/v; worse |
| Decode V2 GEMV fused QKV/SwiGLU/RoPE (no qkv bias) | 129 | 3363 | **wrong ids** (`220` spam) |
| Decode V2 GEMV fused + qkv bias + residual | 128 | 2973 | ids matched; host 128 / device-window 133; SDPA |
| Decode V2 GEMV + naive 1-warp/head attn | 108 | 3298 | ids matched; slower than SDPA |
| Decode V2 GEMV flash-decode attn + in-place RMS | **182** | 1516 | ids matched; host 182 / device-window 195 |
| Decode V2 GEMV 4-warp flash + RoPE-in-attn | **178** | 1532 | ids matched; host 178 / device-window 196; WDDM host scatter |
| Decode V2 GEMV RMS fused into qkv/swiglu/lm_head | 111 | 1137 | ids matched; **miss** — each of 152k lm_head CTAs re-does RMS |
| Decode V2 GEMV NR=4 on lm_head + SwiGLU | **195** | 918 | ids matched; host 195 / device-window 201 |
| llama.cpp Q4_K_M | **187** | — | not this run |

Greedy ids `[2121, 358, 69431, 389, 419, 11618, 11, 358]` matched. Prefill is
not the product TTFT. Task Manager CUDA 70–85% during 3B is **open** (lab log
§4): engine idle + skinny RMS/merge + GEMV below HBM peak, not a closed “paint
one more kernel” story. Flash-decode uses a static `n_q*32` grid of 4-warp
CTAs (512×128 threads on 3B). RoPE is no longer a separate 18-CTA launch on
CUDA. Fusing RMS into the following GEMV looks tempting (one less launch) but
every CTA repeats the same 2048-wide reduction; on `lm_head` that is ~152k
copies of the work. Leave RMS as the 1-CTA kernel. Optional `rms_w` on
`nf4_gemv`/`nf4_qkv`/`nf4_swiglu` stays for small-M tests only. Multi-row GEMV
(NR=4, `M>=4096`) cuts isolated `lm_head` ~494 → 393 µs; q/o/down stay 1-row.
Host 195 vs llama.cpp 187; device-window 201. Product bar remains +10% vs a
fresh competitor plate (~206 if 187 holds). Plate:
`C:\dev\models\runs\decodev2-3b-gemv-fuse9-20260917`. Earlier fuse7b
(178/196): `C:\dev\models\runs\decodev2-3b-gemv-fuse7b-20260917`. RMS-fuse miss:
`C:\dev\models\runs\decodev2-3b-gemv-fuse8-20260917`. Earlier fuse6:
`C:\dev\models\runs\decodev2-3b-gemv-fuse6-20260917`. Earlier:
`decodev2-3b-20260917-172115` (MMA), `decodev2-3b-gemv-mmvq-20260917` (113),
`decodev2-3b-gemv-fuse5-20260917` (128, SDPA), `fuse4` (naive attn, 108).

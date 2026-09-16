# CPU/GPU hybrid overflow (32B)

**Branch:** `exp/cpu-hybrid-overflow`. This file is the design. It is not a
measured row. Do not write a tok/s for `cpu-suffix` or `hybrid` until the
same long travelogue as [`docs/compare-3080.md`](compare-3080.md) exists.

**Wave goal:** a user-facing generate option that chooses *where overflow
layers compute* on Qwen2.5-32B-Instruct NF4 (RTX 3080 12 GB, Windows/WDDM).
Default stays today’s product: all 64 repeating layers on the GPU, policy D
CopyRing. This option is for **overflow 32B**. It does not close the 3B kernel
gap (Ollama 187.3 vs our resident NF4 35.2). That gap is `chr_nf4_gemm` vs
fused Q4_K mmvq, not layer placement.

Do not import GGUF into `.chr`. Do not wrap llama.cpp. SKIP stays SKIP.
Ampere `sm_86` only for generate claims. CPU math must still run on this
Windows box (5950X AVX2) without a GPU in the unit tests.

## Frozen (inherit H2)

From [`docs/plan-h2-ring.md`](plan-h2-ring.md), still true:

- 2 CopyRing slots, one H2D stream, H2D unit = whole packed+scale matrix.
- WDDM: CPU-join `e_copy` **before** prefetch. Depth > 1 was **0.01 tok/s**.
  A CPU suffix must not reintroduce that (no extra in-flight H2D, no
  `copy_{i+1}` queued before the join).
- `LIVE_MAX_N=32`. Decode `N=1`. Prefill chunks `N≤32`.
- `auto` never takes VQ. No dense `[M,K]` in HBM. Reconstruct stays
  tile-local in `chr_nf4_gemm` on the GPU path.
- Pin-set for **gpu** mode: embed + `lm_head` + all qkvo DEVICE. Overflow
  is every `down`, then tail `gate+up` (policy **D**).
- Chat `max_seq=2048` (KV 512 MiB if all layers stay on GPU). 4096 is a
  separate mode.
- `Tensor.is_pinned` is a method. Pin checks use `cpu_is_pinned()`.

## Measured basis (quote these, nothing else)

Same card, greedy, `ctx=2048`, ignore-EOS 64-token plateau
([`docs/compare-3080.md`](compare-3080.md), [`docs/eval-32b.md`](eval-32b.md)):

| Stack | 32B long tok/s | What the time is |
|---|---:|---|
| Ollama 0.34.0 Q4_K_M | **2.54** | GPU prefix **32 repeating + `lm_head`** (33/65), **32** repeating on 5950X. Weights stay put. Hidden ~10 KiB crosses once. |
| deep-fold NF4 TokenLoop | **2.49** | All 64 layers on GPU. Policy D streams **96** matrices, **6885 MiB**, every token. Serial copy floor **277 ms**. |
| llama.cpp b10964 `-ngl 99` | **1.52** | Auto-fit aborted. Not the algorithm to copy. |

32B smoke (Paris / Berlin / 323) is a talk check. Quote **long**, not Ollama’s
short-EOS 3.18. 3B is not this experiment.

Ollama argv (from its server log): `--load-mode none --flash-attn auto`
(`-ngl` omitted). Windows+CUDA disables mmap. CUDA0 **9559 MiB** +
CUDA_Host **9367 MiB**. KV @2048: **256 + 256 MiB**. `graph splits = 2` at
`bs=1`. 16 AVX2 threads.

Packed NF4 32B is **16599 MiB** (4.25 bit). Q4_K_M GGUF is ~**18.5 GiB**.
Denser NF4 is why a GPU prefix *longer than 32 repeating layers* can still
fit. That is the hybrid hypothesis. It is not a tok/s.

---

## 1. Three modes + numeric layer control

One flag family, same on `deepfold run` and `deepfold chat`. Default does
not change the 2.49 path.

| `--compute` | Repeating layers on GPU | CPU suffix | CopyRing | When |
|---|---|---|---|---|
| `gpu` (**default**) | 64 / 64 | none | policy D (96 matrices, 6885 MiB) | current product |
| `cpu-suffix` | **32** / 64 unless overridden | 32 | **none** | Ollama-matched cut, our NF4 |
| `hybrid` | **36** / 64 first try (auto may later pick 36–37) | 28 | **none** in v1 | shorter CPU than Ollama; GPU prefix fully resident |

Numeric overrides (repeating **transformer** blocks, not Ollama’s 65-count):

```text
--gpu-layers N          # 0..num_hidden_layers; source of truth
--cpu-layers N          # sugar: gpu_layers = n_layers - N
--gpu-frac F            # 0..1 of repeating layers, floor, then clamp
```

`--gpu-layers` / `--cpu-layers` / `--gpu-frac` require `--compute cpu-suffix`
or `--compute hybrid`. `--gpu-layers 64` with `--compute gpu` is a no-op.
`--gpu-layers` and `--cpu-layers` together must sum to `num_hidden_layers`.

`--residency` stays the **matrix** policy for `--compute gpu` (default `D`).
It is ignored when the layer split is on: v1 does not mix CopyRing tape with
a CPU suffix (both taxes at once; WDDM risk). If the chosen GPU prefix does
not fit as whole resident layers, **move more layers to the CPU**. Do not
start streaming `down` of the prefix in v1.

`--max-resident-mib` remains the HBM cap for GPU-resident NF4. Auto-fit
uses it. Explicit MiB still wins over the 12 GB formula.

**Ollama conversion (docs / stderr only):** their `33/65` = **32 repeating +
output**. Ours `--gpu-layers 32` + `lm_head` DEVICE = that cut.
`--gpu-layers 36` = **37/65** in their counting. `embed_tokens` is not a
layer in either count. `lm_head` stays DEVICE in all three modes (Ollama
kept the output layer on GPU).

3B / 14B / 20B: `--compute gpu` only. `cpu-suffix` / `hybrid` **refuse**
unless an explicit `--gpu-layers` is passed for a canary (correctness, not
speed). Do not silently split a net that already fits.

---

## 2. Placement algorithm

New planner, CPU-only, no `.chr` required for tests. Shapes from
`descs_from_qwen` / `descs_from_header` (already in
[`gpu/host/residency.py`](../gpu/host/residency.py)). Do not overload
`ResidencyPlan.cpu` (that is host-embed, packed embed on CPU, not a
transformer suffix).

```text
ComputePlan
  compute: gpu | cpu-suffix | hybrid
  n_gpu: int                 # repeating layers [0, n_gpu)
  n_cpu: int                 # repeating layers [n_gpu, n_layers)
  lm_head: device            # v1
  embed: device              # v1
  kv: split | gpu-all
  ring: none | D             # v1: none unless compute=gpu
```

Suffix is contiguous and tail-only. Refuse CPU prefix, interleaved CPU
layers, or `lm_head` on CPU in v1.

### 2.1 Qwen2.5-32B NF4 sizes (from `nf4_nbytes`, no live GPU)

| Piece | MiB |
|---|---:|
| `q` / `o` | 13.28125 |
| `k` / `v` | 2.65625 |
| `gate` / `up` / `down` | 71.71875 each |
| one repeating layer | **247.03125** |
| `embed` / `lm_head` (untied) | 394.453125 each |
| packed total | **16598.9** |
| policy D resident / streamed | 9713.9 / **6885.0** (96 tape entries) |
| KV @2048, all 64 layers | **512** (8 MiB / layer) |
| `RUNTIME_OVERHEAD_MIB` | 1800 |
| policy D cap (slots in) | 9832.56 |

Budget for a **whole-layer** GPU prefix, no CopyRing slots:

`used(N) = embed + lm_head + N × 247.03125 + KV(N)`

must stay ≤ `(12288 − 1800) MiB` = 10488 MiB. That is the same overhead
constant as `resident_cap_bytes`. It is not nvidia-smi. Current H2 decode
smi is **11926** with resident 9716 + slots + KV 512; the 1800 pad is the
WDDM/CUDA/desktop remainder. Treat **N that make `used+1800 > 12288` as
unsafe** even if `used ≤ 10488`.

| N GPU repeating | weights | KV all-GPU | used all-KV | used+1800 | KV split | used split | used+1800 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 (Ollama-matched) | 8693.9 | 512 | 9205.9 | 11006 | 256 | 8949.9 | 10750 |
| **36 (first hybrid)** | 9682.0 | 512 | 10194.0 | **11994** | 288 | 9970.0 | **11770** |
| 37 | 9929.1 | 512 | 10441.1 | **12241** | 296 | 10225.1 | 12025 |
| 38 | 10176.1 | 512 | 10688.1 | 12488 | 304 | 10480.1 | **12280** |

N=32 whole layers of NF4 fit with a lot of air (Ollama needed 32 of the
*fatter* Q4_K_M). N=36 sits next to today’s H2 smi. N=37 with all KV on GPU
is past 12 GB once overhead is added. N=38 with split KV is on the 12288
line. **Do not ship 38 as the first live try.**

CPU suffix weights stay in RAM, pageable (not `cudaHostRegister`). For
`--gpu-layers 36` that is **28 × 247.03 ≈ 6917 MiB**. For `cpu-suffix` N=32
it is **7905 MiB**. Ollama already held 9367 MiB CUDA_Host on this box; RAM
is not the new constraint. Pinning the suffix would steal lockable pages
from the gpu-mode ring and from the OS. **Do not pin compute-on-CPU
matrices.**

### 2.2 KV: split vs all-on-GPU

| Policy | GPU KV | CPU KV | Code | v1 |
|---|---|---|---|---|
| **split** (Ollama) | 8 MiB × `n_gpu` | 8 MiB × `n_cpu` | two `KVCache` (or per-layer device) | **yes** |
| gpu-all | 512 MiB | — | today’s one `KVCache` on CUDA | only if CPU attn H2Ds K/V every layer — **refuse** |

CPU layers produce K/V on the CPU. Shipping those into a GPU cache, then
reading the full prefix back for SDPA, is more PCIe than the one hidden
bounce. **KV lives where the layer computes.** GPU layers: GPU cache. CPU
layers: CPU cache. Same `max_seq`. `bytes_per_token` in stderr should
print both.

v1 default for `cpu-suffix` / `hybrid`: **split**. First live hybrid still
uses **N=36**, which also fits all-GPU KV, so an OOM is not “we guessed
split wrong.”

Auto-fit (later, after N=36 is not-OOM):

```text
slack_mib = 256
cap = (vram_mib - 1800) * MiB - slack_mib
max N s.t. embed + lm_head + N*layer + (8 MiB * N if split else 512 MiB) <= cap
clamp to [1, n_layers]
```

On 12 GB / 2048 / split / slack 256 that lands on **37**. Do not make 37
the first measurement. Product `--compute hybrid` without `--gpu-layers`
is **36** until a recorded smi for 37 exists.

`--compute cpu-suffix` without `--gpu-layers` is **32** (Ollama-matched),
even if 36 would fit. The named mode is the comparison cut, not auto-fit.

### 2.3 What stays DEVICE vs CPU

For each repeating layer `i`:

- `i < n_gpu`: all seven NF4 matrices DEVICE; RMS1/RMS2 DEVICE; K/V GPU.
- `i >= n_gpu`: all seven NF4 packed+scale on CPU (pageable); RMS1/RMS2
  CPU copies (tiny bf16); K/V CPU. Qwen2 q/k/v **bias** goes with the
  linear (CPU add).

Always DEVICE in v1: `embed_tokens`, `lm_head`, `model.norm` (final RMS),
RoPE table used by the GPU prefix. CPU suffix gets a **512 KiB CPU clone**
of `cos`/`sin` (or a D2H of the slice — either is fine; clone once at
init).

`plan_residency` / CopyRing / `SlotPair`: **not constructed** unless
`--compute gpu`. `load_chr_nf4(..., max_resident_bytes=)` today always
builds slots when a cap is set. Layer-split load must not allocate 2×72 MiB
arenas it will not use.

---

## 3. TokenLoop changes

Today [`gpu/loop/generate.py`](../gpu/loop/generate.py) walks **every**
layer on device (`_decode_layers` / `_prefill_layers`), and
[`gpu/loop/ring.py`](../gpu/loop/ring.py) H2Ds HOST GEMMs. CPU suffix is a
split in that walk, not a second process.

### 3.1 Decode `N=1`

```text
x = embed(ids)                         # GPU, as now
for li in 0 .. n_gpu-1:
    GPU layer body (rms, qkv, rope, kv_gpu.write, sdpa, o, swiglu)
    CopyRing only if compute=gpu
torch.cuda.synchronize()              # prefix done; do not join a copy stream
h = x.to("cpu")                        # one D2H, [1, 5120] bf16 = 10 KiB
for li in n_gpu .. n_layers-1:
    CPU layer body (same order, cpu NF4, kv_cpu, cpu sdpa)
x = h.to(device)                       # one H2D, 10 KiB
hidden = rms(x, final_norm)            # GPU
logits = lm_head(hidden)              # GPU, graphs allowed
```

Ollama’s `graph splits = 2` is this bounce. Do **not** bounce per CPU
layer. Do **not** start CopyRing `prefetch_next` during the CPU suffix.
If `compute=gpu`, today’s path is unchanged (no bounce, tape 6885 MiB).

CUDA graphs: plan A on DEVICE groups of the **GPU prefix** only, as now.
CPU groups are never captured. `HostSlotGemm` is unused when there is no
tape. `graph_error` must not become `"overflow"` for a mixed CPU session
(same rule as mixed HOST today).

WDDM: the 10 KiB bounce uses the compute/default stream **after** the
prefix join. It is not an overflow H2D and must not go through
`CopyRing.copy_stream`. Depth stays 0 for hybrid/cpu-suffix.

### 3.2 Prefill `N≤32`

Same split, once per chunk (and once per superchunk if `prefill_mode=hold`
on the GPU prefix — hold is CopyRing-only, so **hybrid v1 is `chunk`**).
Bounce shape `[N, 5120]` bf16 = **N × 10 KiB** (320 KiB at N=32). Still
negligible next to 6885 MiB.

CPU suffix prefill: same `LIVE_MAX_N=32` chunks. Attention stays
left-to-right inside a layer (layer i token t needs layer i KV ≤ t). Do
not invert the CPU stack into weight-stationary without a separate plan.

`prefill_mode=hold` remains a `--compute gpu` flag. Hybrid does not
implement hold in v1.

### 3.3 `CompressedLinear.home`

Today (`gpu/host/linear.py`): `packed.numel()>0` ⇒ `home="device"`, else
`host_image` ⇒ `"host"`. `attach()` of a CPU `ChrMatrix` would lie and say
`device`, then `nf4_gemm` would blow up.

v1:

```text
home: "device" | "host" | "cpu"
attach()           -> device CUDA packed
attach_host()      -> host, pinned, CopyRing only
attach_cpu()       -> packed+scale on CPU, pageable, no pin, no H2D
forward()          -> cpu gemm if home=cpu; refuse silent H2D
Gemm.of            -> home="cpu", packed/scale CPU tensors, host_image=None
```

`GemmGroup.run` for a cpu group calls `nf4_gemm_cpu`. Mixed DEVICE/CPU
inside one group is a load bug (a layer is atomic).

### 3.4 Glue on CPU

Reuse the Python body, not a second architecture:

- RMS: `F.rms_norm` / `rms_norm_exact` on CPU bf16 weights.
- RoPE: existing `_rope` on CPU `q,k` with CPU `cos/sin`.
- SDPA: `F.scaled_dot_product_attention` on CPU. If CPU build lacks
  `enable_gqa`, expand K/V by `n_rep` (32B: 40/8, `n_rep=5`). Tests cover
  that branch without GPU.
- SwiGLU: `silu(gate)*up` in bf16/fp32 as now, then CPU down-proj.

Do not `torch.compile` the CPU suffix (same CopyRing/graph reason).

---

## 4. CPU NF4 matvec / GEMM

We have CUDA `chr_nf4_gemm` and a CPU **oracle**
([`gpu/tests/nf4_oracle.py`](../gpu/tests/nf4_oracle.py): decode packed →
float32 `W_hat`, then `W_hat @ x`). We do **not** have ggml Q4_K. A CPU
suffix needs NF4 × activation on the 5950X.

### 4.1 Smallest honest v1

**File:** [`gpu/host/cpu_linear.py`](../gpu/host/cpu_linear.py) (next to
`dequant_nf4_rows` in [`gpu/host/embedding.py`](../gpu/host/embedding.py)).
Do not put this in `gpu/tests/`. Do not import the CUDA extension.

```text
nf4_gemm_cpu(packed, scale, x, M, K, K_pad, *, row_chunk=256) -> y
  packed: uint8  [M, K_pad/2]  CPU
  scale:  fp16   [M, n_groups] CPU
  x:      bf16/fp32 [K, N]     CPU, N in 1..32
  y:      bf16 [M, N]
```

Row-chunked so a 32B `gate_proj` `[27648, 5120]` never materializes a 540 MiB
float32 `W_hat`:

1. For rows `[lo:hi]`, nibble-unpack + LUT × group scale in float32
   (same SS1/SS3/SS5 as the oracle; reuse `dequant_nf4_rows` with
   `dtype=float32` or a sibling that already returns fp32).
2. `y[lo:hi] = W_chunk @ x` in float32.
3. Cast to bf16 at the end (GPU path is bf16 activations; greedy needles
   are not bitwise logits).

`N=1` is a matvec. `N=2..32` is a thin GEMM with the same decode. No new
`.cu`. No AVX2 in v1.

Numeric tests (no GPU): packed toy + 3B-shaped slices vs
`gpu.tests.nf4_oracle.decode_nf4` + `matmul_f32`. Gate = existing oracle
maxabs, not the CUDA kernel’s 0.05. If CPU disagrees with the oracle, the
CPU path is wrong; if it disagrees with CUDA greedy tokens, say so in the
run notes (bf16 tensor cores vs fp32).

### 4.2 Expected slowness (qualitative, no tok/s)

llama.cpp Q4_K mmvq on this 5950X is hand AVX2. Our v1 is Python + a
chunked dequant + a BLAS gemm. **Per layer it will be slower than Ollama’s
CPU suffix.** The only way hybrid can still win the *wall* is if:

- the suffix is shorter (28 vs 32 repeating), and
- the GPU prefix is fully resident (zero 6885 MiB / 277 ms copy floor),

and those two savings beat the slower NF4 CPU kernel. That is a
measurement, not a prediction. A numpy/torch CPU NF4 that is many times
slower than Q4_K mmvq will lose even at `--gpu-layers 36`.

**Microbench before TokenLoop (required):** one `down_proj` and one
`q_proj` of 32B shapes, `N=1` and `N=32`, 5950X, wall ms. Write the ms in
the run folder. If one decode layer is already tens of milliseconds above
Ollama’s implied budget (~6 ms/layer if one naively halves 394 ms/tok —
that split is not measured; do not treat it as a gate), **stop and plan
an AVX2 kernel** instead of burning a 32B long plateau. AVX2 NF4 is a
scoped follow-up in `gpu/cpu/` or a small C++ extension. Still not
llama.cpp, not GGUF.

GIL: numpy/torch CPU GEMM can release the GIL; the Python chunk loop does
not. `torch.set_num_threads` default `min(16, cpu_count)` to match Ollama’s
16 on this box; document `OMP_NUM_THREADS`. Do not oversubscribe 32 HT
workers against a live CUDA prefix. v1 is **serial**: GPU prefix, join,
CPU suffix, bounce, `lm_head`. No “CPU suffix overlapped with CopyRing.”

---

## 5. Interface sketch

### 5.1 CLI (`gpu/cli/main.py` `_add_runtime_flags`)

Same flags on `run` and `chat`. `--help` names the family once.

```text
--compute {gpu,cpu-suffix,hybrid}
    Overflow compute policy (default: gpu = all layers on device,
    policy D CopyRing). cpu-suffix / hybrid are 32B overflow
    experiments; they do not speed up 3B.

--gpu-layers N
    Repeating transformer layers on GPU (0..num_hidden_layers).
    lm_head stays on GPU. Requires --compute cpu-suffix|hybrid.
    Default: 32 with cpu-suffix, 36 with hybrid.

--cpu-layers N
    Repeating layers on CPU; sets --gpu-layers to n_layers-N.

--gpu-frac F
    Fraction of repeating layers on GPU, floored.
```

`--residency` help text: “matrix overflow for `--compute gpu` only
(default D).” `--max-resident-mib` unchanged.

Stderr after load (hybrid example):

```text
compute=hybrid gpu_layers=36/64 cpu_layers=28 lm_head=device
kv gpu=288 MiB cpu=224 MiB  (split, max_seq=2048)
resident 9682 MiB  cpu_weights 6917 MiB pageable  ring=off
```

Refuse GGUF as today. `cpu-suffix`/`hybrid` on a fully-fitting net without
`--gpu-layers`: explicit error, not silent split.

### 5.2 Chat slash

Load-time placement. Switching `--compute` needs a reload. **No**
`/gpu-layers` that pretends to move weights mid-session.

`/stats` (already exists) prints `compute=`, `gpu_layers=`, `cpu_layers=`,
ring on/off. `/help` line can mention that compute is a launch flag.

### 5.3 Lab

[`gpu/lab/deepfold_long.py`](../gpu/lab/deepfold_long.py) grows `--compute`
and `--gpu-layers`. Separate compare ids, do not overwrite
`deepfold-nf4-32B-overflow`:

```text
python -m gpu.lab.deepfold_long --size 32B --compute gpu
python -m gpu.lab.deepfold_long --size 32B --compute cpu-suffix
python -m gpu.lab.deepfold_long --size 32B --compute hybrid
python -m gpu.lab.deepfold_long --size 32B --compute hybrid --gpu-layers 36
```

Ollama long stays `python -m gpu.lab.ollama_h2 --model qwen2.5:32b --size 32B`.
Do not change that runner to take our flags.

`--plan-only` on deepfold_long prints the `ComputePlan` (N, bytes, kv split)
without loading CUDA.

### 5.4 Tests without GPU

| File | What |
|---|---|
| `gpu/host/test_residency.py` (or new `gpu/host/test_compute.py`) | 32B shapes, 12 GB, `cpu-suffix` → N=32; `hybrid` → 36; `gpu` → n_cpu=0 + policy D tape 96/6885; N=38 split+slack refuses or clamps; `--gpu-layers` + `--cpu-layers` sum; 3B refuse without explicit N |
| `gpu/cli/test_cli.py` | parse `--compute`, `--gpu-layers`, `--cpu-layers`, `--gpu-frac`; help strings; conflict errors |
| `gpu/host/test_cpu_linear.py` (or `gpu/cpu/test_nf4.py`) | oracle match N=1 and N=32, toy + small M/K; `home=cpu` on a fake linear; no pin |
| `gpu/loop/test_ring.py` | **unchanged contract** for `--compute gpu`; do not relax join-before-prefetch |
| chat `classify_slash` | still no compute slash |

3B canary (GPU, optional): `--gpu-layers 8` on `qwen25-3b.nf4.chr` to
exercise bounce + CPU suffix correctness (Paris/Berlin/323). Not a 32B
speed number. Not a 187 tok/s claim.

---

## 6. Risks

| Risk | What goes wrong | Mitigation |
|---|---|---|
| WDDM CopyRing depth | 0.01 tok/s if CPU work is mixed with `copy_{i+1}` before join | v1: **no ring** on cpu-suffix/hybrid; bounce is 10 KiB on the compute stream after prefix sync |
| Pin of CPU suffix | 7–8 GiB lockable pages, gpu-mode pin 6885 MiB fights the OS | `attach_cpu` pageable only; assert `not cpu_is_pinned` in tests |
| Threads / GIL | 32 HT workers + CUDA; chunk loop holds GIL | 16 threads; serial GPU→CPU; microbench before plateau |
| Correctness vs GPU | fp32 CPU vs bf16 tensor cores; greedy tokens may differ | GEMM vs oracle; smoke needles; do not require bitwise logits |
| `home="device"` lie | CPU packed launched as CUDA | `attach_cpu` + explicit home; `Gemm.of` tests |
| Final RMS / lm_head | forgetting the bounce-back | `lm_head` always DEVICE; test that CPU suffix does not run the head |
| Prefill TTFT | CPU GEMM N=32 is heavier than decode | quote long decode; TTFT is a separate cell; may regress |
| RAM + pagefile | 7.9 GiB suffix + 12 GB VRAM mapping | same class as Ollama CUDA_Host; fail loud on OOM |
| Graphs + bounce | capture includes a `to("cpu")` | never capture across the split; prefix groups only |
| Policy D leftover | hybrid still streams 96 matrices | load path must skip `plan_residency` when `n_cpu>0` |
| 3B accidental split | “hybrid” on 3B tanks 35 tok/s | refuse unless explicit `--gpu-layers` canary |
| N=37/38 OOM | display + CUDA + activations | first try **36**; smi after load before generate |

---

## 7. Measurement plan

Quote **long**, ignore-EOS, 64 tokens, same travelogue as
[`docs/compare-3080.md`](compare-3080.md) /
[`gpu/lab/deepfold_long.py`](../gpu/lab/deepfold_long.py) `LONG_PROMPT`.
Do not quote smoke 2.31 / Ollama 3.18 as the hybrid result.

Order:

1. CPU NF4 microbench (no TokenLoop). Write ms for `q_proj` and
   `down_proj`, N=1 and N=32. If this is hopeless vs a CopyRing-free GPU
   prefix, stop.
2. 3B canary `--gpu-layers 8` (or similar): needles 3/3, not a speed row.
3. 32B `--compute gpu` repeat: expect long **~2.49** (control; same as
   recorded). If this moved, the branch is dirty.
4. `--compute cpu-suffix` (N=32). Long plateau, smi after load, TTFT,
   `compute=` dump. Compare to Ollama **2.54** on the same cut, different
   codec/kernel. Do not invent a number.
5. `--compute hybrid` `--gpu-layers 36`. Same plate. Then optionally 37
   if smi after 36 is not against the wall.
6. Upsert new compare.json ids. Do not reuse `deepfold-nf4-32B-overflow`.
   SKIP rows stay SKIP. No Marlin/AWQ fill-in.

Commands:

```text
python -m gpu.lab.deepfold_long --size 32B --compute gpu
python -m gpu.lab.deepfold_long --size 32B --compute cpu-suffix
python -m gpu.lab.deepfold_long --size 32B --compute hybrid --gpu-layers 36
python -m gpu.lab.ollama_h2 --model qwen2.5:32b --size 32B
```

Report: long tok/s, decode_steps, prefill_ms, smi after load, gpu_layers,
cpu_layers, ring on/off, pin MiB, whether CPU weights are pinned (must be
no). Hard-12 32B: **not run** until Pavel says “run it.”

---

## 8. Out of scope

- Beating 3B **187 tok/s**. Resident 3B stays the kernel problem
  (`LIVE_MAX_N=32`, decode N=1). Hybrid is not a 3B flag.
- Marlin / AWQ / ExLlamaV2 / vLLM. SKIP remains SKIP.
- GGUF, Ollama blobs, `llama.cpp` as a library, Q4_K tables inside `.chr`.
- AVX2 / C++ NF4 in v1 (follow-up only if the microbench says Python cannot
  win).
- CopyRing **plus** CPU suffix in one forward (v1).
- CopyRing depth > 1, third slot, `DEEPFOLD_COPY_JOIN=0` on WDDM.
- `lm_head` on CPU; host-embed combined with hybrid.
- `max_seq=4096`, 70B, VQ.
- Speculative decode / draft model on the same 12 GB.
- Claiming a hybrid tok/s in README or compare-3080 before a long file
  exists.

---

## Waves (implementation, later chats)

Units without GPU after a code wave:

```text
python gpu/cli/test_cli.py
python gpu/host/test_residency.py
python gpu/host/test_compute.py          # new
python gpu/host/test_cpu_linear.py    # new
python gpu/loop/test_ring.py
```

| Wave | DoD |
|---|---|
| CH-0 | this document |
| CH-1 | `ComputePlan` + CLI parse + load wiring (`attach_cpu`, no ring when `n_cpu>0`). No generate. |
| CH-2 | `nf4_gemm_cpu` vs oracle, N=1 and N≤32. Microbench 32B shapes. |
| CH-3 | TokenLoop split + bounce. 3B canary needles. Ring tests still PASS. |
| CH-4 | Live 32B long: gpu / cpu-suffix / hybrid-36. Files under `docs/runs/`. No tok/s in this plan. |

---

## Recommended default hybrid (first try)

**`--compute hybrid` → `--gpu-layers 36`** (layers 0–35 DEVICE, 36–63
CPU, `lm_head` DEVICE, embed DEVICE, **split KV**, **no CopyRing**).

Why 36, not 32 and not 38:

- Ollama auto-fit **32 repeating + output** of *Q4_K_M* (~18.5 GiB). NF4
  is **16599 MiB**; 32 whole NF4 layers + embed + `lm_head` are only
  **8694 MiB** of weights — we can keep more than 33/65 on the GPU.
- 36 repeating + embed + `lm_head` = **9682 MiB** weights. With split KV
  (288 MiB) + 1800 overhead ≈ **11770 MiB**, in the same smi band as
  today’s H2 **11926**. 28 CPU layers (**6917 MiB** pageable) vs Ollama’s
  32.
- 37 with all-KV on GPU is **12241** estimated smi — past 12 GB with the
  display. 38 with split KV is **12280**. Those are later probes, not the
  first live try.
- v1 GPU prefix is **fully resident**. That deletes the 6885 MiB / 277 ms
  tape. CPU does **28/64** of the net, not half. Whether that beats 2.54
  depends on CPU NF4 vs AVX2 Q4_K and is **not claimed here**.

Product default remains `--compute gpu` (CopyRing, 2.49 long). Hybrid is
opt-in until CH-4 has a file.

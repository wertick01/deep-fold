# Token loop on an RTX 3080 12 GB

This is module 5 of [schema.md](schema.md): how a token actually walks the model.

**Product CLI (2026-09-17).** Resident NF4 (3B / 14B / 20B) uses Decode V2
(`gpu/decodev2`, `--executor auto`): N=1 CUDA-core GEMV graph + MMA prefill
chunks of 32. This page describes **TokenLoop** (`gpu/loop/generate.py`) —
still the overflow / CopyRing / VQ / `--executor tokenloop` path. Decode V2
numbers: [`decode-v2-lab.md`](decode-v2-lab.md).

Neighbors we rely on and do not argue with:

- kernel — [kernel-ampere.md](kernel-ampere.md): one skeleton `mma.sync.m16n8k16`, decode tile `128×256`, codebook only **256×8**;
- 12 GB budget — [vram-3080.md](vram-3080.md): arenas, leftover, KV in KiB/token.

Here only **who holds memory, in what order we compute a layer, how to sit inside HuggingFace, which launch to call, whether CUDA graphs are needed, where this breaks, and how to accept it on a live 3080**.

Hardware is the same: Ampere sm_86, 70 SMs, 912 GB/s, no TMA/WGMMA. There is `cp.async`.
Compressed weights are resident in GDDR. A tile dequantizes into registers (not HBM), goes into MMA, the scratch is discarded. We do not pack back. While `col_group` `i` is computing, `cp.async` is already pulling `i+1` and `i+2` (three stages, as in the kernel).

Driver is Python (PyTorch). Our CUDA extension is **linear layers only**. Norms, RoPE, SiLU, residual, attention — ordinary PyTorch, attention via `F.scaled_dot_product_attention`.

Two phases, **two launches of one skeleton**, the same weight format:

```
prompt ──► prefill (long GEMM) ──► KV filled
              │
              ▼
         decode one token at a time (GEMV), until EOS
```

---

## 0. Picture of the loop

After load, VRAM holds only compressed plus KV plus a thin scratch for activations. There is no dequantized layer. The tile ring lives in **block smem**, not two buffers in GDDR (that is an outdated wording of the schema; the kernel holds 3 packed stages + dequant straight into registers).

```
                     VRAM (lives the whole time)
  ┌──────────────────────────────────────────────────────────┐
  │  WeightArena: indices + scales / codebooks 256×8           │
  │  KV: [layer, K|V, T_max, kv_heads, d]   (opt. Q8)          │
  │  scratch: residual, normed, qkv, attn, ffn  (decode ~48 MB)│
  └──────────────────────────────────────────────────────────┘
           │ pointers, not copies
           ▼
  one CUDA extension, two launches (§4)
           │
           │  smem: codebook 4–8 KB + ring packed[3]
           ▼
      nibble×scale / book[i] → A_regs BF16
      mma.m16n8k16 → acc FP32 → forget
```

Matrix convention — **as in the kernel**, not as in `F.linear`:

```
W[M, K] compressed    M = out_features (gate_proj 14336)
x[K, N] BF16          N = sequence length (decode: 1, pad BN=8)
y[M, N]
tile: (layer, matrix, row_tile, col_group)
```

On a token, Python does not load weights, does not write them to disk, and does not call `empty_cache`. It only feeds the kernel pointers and computes the layer “glue”.

Budget from the schema: 10 tok/s = 100 ms per token. On 8B INT4 the bus gives ~5 ms (≈200 tok/s peak; the kernel aims at 110–140). 10 tok/s is headroom for 32B 2-bit, not for 8B.

---

## 1. Who owns memory

In short: **Python owns all long-lived buffers. The kernel only looks.**
`cudaMalloc` inside the extension is forbidden, except a tiny one-shot workspace at init if there is no other way. Otherwise the 12 GB scheduler (module 3) loses control, and on this card that is immediate OOM.

### 1.1. Resident, not touched on a token

| Object | Where | Who allocates | Shape (idea) | Who reads |
|---|---|---|---|---|
| Packed indices | GDDR | `WeightArena` at load | `uint8`/`uint16`, one big tensor for the whole model or per matrix | kernel, tile `(layer, matrix, row_tile, col_group)` |
| Scales (stage A, INT4) | GDDR | same arena | `fp16`, one scale per group 32–128 | kernel |
| Codebooks (stage B, 2-bit) | GDDR; at kernel start — into smem | same arena | **256×8 BF16**, one or two per matrix (additive). Kernel v1 does not read a 2¹⁶ book | kernel; 4 or 8 KB in smem for the whole K-loop |
| Offset table | GDDR + host copy | arena | `int32` start/length of each matrix | driver picks the pointer, kernel already gets a slice |
| `embed_tokens` | GDDR | HF module, our loader | 8B: BF16 is OK (~1 GB). 14B/32B: **INT4**, otherwise one Qwen embedding pair ≈ 2970 MB | `F.embedding` (+ row dequant, not MMA) |
| `lm_head` | GDDR | HF module or `CompressedLinear` | 8B: BF16 OK. 14B/32B: INT4, as in the VRAM budget. Via the extension — yes, it is Linear | decode `N=1` / prefill of the last position |
| RMSNorm γ | GDDR | HF module | BF16, tens of KB for the whole net | PyTorch |
| RoPE `inv_freq` | GDDR | computed once on CUDA | small | PyTorch |

The codebook is a matrix key (`q`, `k`, `v`, `o`, `gate`, `up`, `down`), not an archive per tile. All pieces of the matrix poke it with indices. That is the contract from the schema.

Stage A and stage B live in one arena with the same tile address. The payload changes: nibble+scale or index+codebook. The token loop does not notice — the kernel itself knows the format from a tensor flag.

What we **do not** store:

- a dequantized BF16 layer (0.4–1.7 GB) — forbidden, that is the hole in 12 GB;
- a second copy of weights “just in case”;
- per-token temporary `idx`/`scale`.

### 1.2. KV — a separate owner

KV grows with conversation length; weights do not. Owner is `KVCache`, not `CompressedLinear` and not `transformers.DynamicCache`.

Why not DynamicCache: on every decode it `cat`s along length. That is an allocation + copy of the whole cache + holes in the allocator. On 12 GB that dies after a couple hundred tokens, even though “the numbers all fit”.

Lay it out like this:

```
k_cache: [n_layers, max_seq, n_kv_heads, head_dim]   # BF16 or Q8
v_cache: same
seq_len: scalar per request
```

Or paged in 16/32 blocks, if the scheduler so decides. The token loop does not care: it writes `cache[layer, t, :, :] = k` and hands SDPA the slice `[:, :t+1]`.

Optional Q8: store `int8` + `fp16` scale per `(token, head)` or per group. PyTorch SDPA does not eat INT8 KV. So for attention we **dequantize KV into a preallocated FP16 buffer** of the same `max_seq`, run SDPA, reuse the FP16 scratch. Do not hold two “real” caches. Decode cost: an extra pass over KV (on 8B at 4k that is ~16–32 MB, small next to weights). Makes sense when 14B/32B hit the 12 GB ceiling, not on 8B.

Numbers — from the VRAM budget, not “about 40 layers”:

| Model | Layers | KV FP16 / token | 2k FP16 | 4k FP16 | 4k Q8 |
|---|---|---|---|---|---|
| Llama-3.1-8B | 32 | **128 KiB** | 256 MB | 512 MB | 256 MB |
| Qwen2.5-14B | 48 | **192 KiB** | 384 MB | 768 MB | 384 MB |
| Qwen2.5-32B | 64 | **256 KiB** | 512 MB | 1 GB | 512 MB |

Q8 = exactly half plus ~6% for scales. 8B Q8 is not needed (leftover is gigabytes). 32B 2-bit leftover ~3 GB — Q8 only if they push 8k+ and a fat CUDA as well. Preallocate `max_seq` at session start, do not grow.

### 1.3. Activation scratch — a ring per token, not per layer

MMA tiles live in kernel smem (3 packed stages, dequant into registers). In GDDR the scheduler reserves **48 MB** for decode `b=1` (not a layer). Prefill asks for more — its own buffers, allocated once for the promised `max_prefill`:

```
buf_a: [max_batch, max_prefill, hidden]   # residual / block input
buf_b: [max_batch, max_prefill, hidden]   # norm output / attn output
qkv:   [max_batch, max_prefill, q_dim + kv_dim + kv_dim]
ffn:   [max_batch, max_prefill, intermediate]   # gate or up, can be in turn
```

On decode `max_prefill` is not needed — same buffers, just `seq=1`. Allocate **once** for the worst prefill we promise (e.g. 2048). Not `torch.empty` on every layer: on Ampere PyTorch’s caching allocator is kind while there are no holes, and mean once KV is sitting next to them.

Peak activations on 8B prefill, S=2048, BF16: hidden 4096 → 16 MB per tensor; `intermediate` 14336 → 56 MB. That is fine. Forbidden to materialize `gate` and `up` at the same time if we can compute `silu(gate)*up` in pieces or straight into one buffer. On 32B / 2048 `intermediate` is already ~100+ MB — still not a weight layer, but we do not hold two such buffers for nothing.

### 1.4. Who is on which side of the API

```
WeightArena.ptr(layer, matrix) -> (idx, scale_or_book, meta)
KVCache.view(layer, seq)       -> (k, v)     # already on CUDA
Scratch.layer_bufs()           -> named tuple

ext.linear_prefill(x, idx, aux, meta) -> y
ext.linear_decode (x, idx, aux, meta) -> y
```

The extension does not know about Llama and does not hold global pointers to weights. Otherwise CUDA graphs and reloading the model become a lottery.

---

## 2. Llama-3 layer order

The block is the same as `LlamaDecoderLayer`. Compressing and reordering ops for a “pretty kernel” is not allowed: residuals and norms will shift, logits will drift from BF16 even with honest weights.

One block, batch=1 (chat). Who computes — in parentheses.

```
x = residual0

h = RMSNorm(x, γ_attn)                         # PyTorch
q, k, v = Linear_qkv(h)                        # CUDA ext  (see packing below)
q, k = RoPE(q, k, position)                    # PyTorch
KVCache.write(layer, t, k, v)                  # view-assign, no cat
a = SDPA(q, K_cache[:t], V_cache[:t], causal)  # PyTorch SDPA / Flash on Ampere
h = Linear_o(a)                                # CUDA ext
x = x + h                                      # PyTorch, in-place into residual

h = RMSNorm(x, γ_mlp)                          # PyTorch
g = Linear_gate(h)                             # CUDA ext
u = Linear_up(h)                               # CUDA ext  (or together with gate)
h = silu(g) * u                                # PyTorch
h = Linear_down(h)                             # CUDA ext
x = x + h                                      # PyTorch
```

Then the next layer. After the last: `RMSNorm` + `lm_head` + sample on CPU.

### 2.1. QKV: three matrices or one packed

In HuggingFace these are three `nn.Linear`: `q_proj`, `k_proj`, `v_proj`. Llama-3-8B shapes are `4096×4096`, `1024×4096`, `1024×4096` (GQA 8/32). Computing them separately is honest and easier to stitch to checkpoint names.

Packing into one matrix `M = 4096+1024+1024` (kernel convention `W[M,K]`) is better: input `x` is read **once**, one launch, one pass over the codebook. On decode the bus is everything. Makes sense to do it immediately, not as “later”.

Careful: `k_proj`/`v_proj` separately have `M=1024` — the kernel then turns on split-K (fewer blocks than 2×70 SMs). Glued QKV (`M=6144`) does not need split-K. Another reason to glue.

How not to drift from HF:

- in the compressed-weight file store either three tensors with HF names, or one `qkv` plus `n_q, n_kv` in the header;
- in the module tree replace three Linears with one `PackedQKV`, and patch `LlamaAttention.forward` so it calls it;
- or leave three modules, but in the block `forward` call `ext.linear_*(h, packed_qkv)` and split the output. The second road is easier for the loader: names as in Meta, the kernel eats a glued pointer that the arena glues **offline** (q,k,v indices consecutive along `M`).

Do not glue `o_proj` to QKV: it sits after SDPA, the input is different.

### 2.2. SiLU and gate/up

SwiGLU cannot be hidden inside MMA: `down(silu(gate(x)) * up(x))` is two GEMMs out to one width and one back. Glue **gate+up** into one matrix (same input `h`) — yes, same story as QKV. Leave SiLU and the product as a PyTorch elementwise on the packed kernel’s output.

Do not fuse `silu` into the first kernel at the cost of a second format. The win is microseconds, the cost is two code paths.

### 2.3. What cannot be reordered

- Norm and linear layer (RMSNorm does not commute with quantized GEMM in a way that would let us “skip”).
- Write KV **after** RoPE, as HF. Otherwise decode will see different angles.
- Compute `o_proj` before residual add “in place on x” if the old residual is still needed.

Matrix order on the bus for one 8B layer, roughly:

| Matrix | INT4 bytes | 2-bit bytes | When |
|---|---|---|---|
| qkv packed | ~12 MB | ~6 MB | before attention |
| o | ~8 MB | ~4 MB | after attention |
| gate+up | ~56 MB | ~28 MB | FFN |
| down | ~28 MB | ~14 MB | FFN |

FFN is two thirds of traffic. If something is optimized first, it is gate/up/down, not attention.

---

## 3. How to sit inside HuggingFace without loading BF16

Goal: `LlamaModel` / `LlamaForCausalLM` as a skeleton (norms, RoPE, masks; we can skip `generate` — our own loop is more reliable), and **no** `nn.Linear` materializes `[out, in]` in BF16. On a 3080 even 8B BF16 is 16 GB; the loader dies before the first token. 32B is not even discussable.

Do not call `from_pretrained` on the original weight repo. It will download and start placing tensors.

The working path is the same as AWQ/HQQ/Quanto:

```python
from transformers import AutoConfig, AutoModelForCausalLM
import torch
import torch.nn as nn

def load_skeleton(model_id: str):
    cfg = AutoConfig.from_pretrained(model_id)          # config.json only, ~KB
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            cfg, torch_dtype=torch.bfloat16
        )
    # Here every Linear .weight is on meta, bytes in VRAM ≈ 0
    replace_linears(model)
    return model
```

`accelerate.init_empty_weights()` does the same, can use it. Important: **skeleton first, then replace, then our tensors**. Not `from_pretrained(..., device_map="auto")` and not `model.cuda()` over still-alive meta-Linears: `.cuda()` will try to allocate full BF16.

Replace:

```python
SKIP_LINEAR = set()  # 8B: can leave lm_head BF16. 14B/32B — CompressedLinear too

def replace_linears(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name not in SKIP_LINEAR:
            # child.weight is meta. Not .cpu(), not .to("cuda"), not .float().
            setattr(module, name, CompressedLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
            ))
        else:
            replace_linears(child)
```

`CompressedLinear` does **not** register a `weight` parameter of shape `[out, in]`. Only buffers `idx`, `scale` or `book` — and those empty until the loader fills them. Otherwise `model.to("cuda")` will birth BF16 again.

HF and third-party utilities often do `module.weight.dtype`. To not crash, a property is enough:

```python
@property
def weight(self):
    # 0 bytes, but correct device/dtype for checks
    return torch.empty(0, dtype=torch.bfloat16, device=self.idx.device)
```

Loading the compressed file (our safetensors, not Meta’s `model-00001-of-00004.safetensors`):

```python
def materialize(model, packed_path: str):
    with safe_open(packed_path, framework="pt", device="cuda") as f:
        for name, module in model.named_modules():
            if isinstance(module, CompressedLinear):
                module.idx = f.get_tensor(f"{name}.idx")
                if stage_A:
                    module.scale = f.get_tensor(f"{name}.scale")
                else:
                    module.book = f.get_tensor(f"{name}.book")
            elif name in {embed, norm, lm_head}:
                assign(module, f.get_tensor(...))
```

Rules without which the loader lies that it “does not load BF16”, and does:

1. Do not open the original `model.safetensors` even for one matrix key. The offline compressor already walked the layer from disk and wrote our file.
2. `tie_weights()` after replace: if `lm_head` is left Linear and the embedding is already INT8/BF16 — either assign explicitly or untie. Blindly calling `model.tie_weights()` on meta is not allowed.
3. Do not use `device_map` / `hf_device_map`. One card, everything resident, as in the schema.
4. After `replace` do not `load_state_dict` from a BF16 checkpoint: `*.weight` keys will not be found or will explode the shape.
5. Do not turn on `torch.compile(model)` in v1 — custom op and graph capture get along badly; debugging on a 3080 turns into shamanism.

Attention patch is minimal: leave `LlamaAttention`, swap only the projections. SDPA is already inside HF (`attn_implementation="sdpa"`). Check that the config did not wander into eager matmul `[T,T]`. On Ampere in recent PyTorch, SDPA = FlashAttention-2; that is enough.

Write our own `generate` loop anyway. HF `generate` spawns tensors, sometimes copies KV, sometimes moves to CPU. For tok/s acceptance it is the enemy.

---

## 4. Prefill vs decode: which launch

One weight format, one extension, **two launch configs** of the same skeleton (`mma.m16n8k16` + `cp.async` 3 stages). A separate CUDA-core GEMV is forbidden by the kernel: it would break the shared path with prefill. The TC tax on decode (`N=1`, pad `BN=8`, 1 of 8 columns alive) is 1–2 µs on `gate_proj` vs ~45 µs of bus. We pay it.

Threshold in the driver. Here `N` is sequence length (kernel convention), not `out_features`:

```python
def linear(x, w):
    # x: [K, N]   (after permute from HF [B, S, H])
    n = x.shape[-1]
    if n == 1:
        return ext.linear_decode(x, w.idx, w.aux, w.meta)   # BM=128, BN=8, BK=256
    return ext.linear_prefill(x, w.idx, w.aux, w.meta)      # BM=64,  BN=16, BK=128
```

Prompt prefill: `N = len(prompt)`. Decode: `N = 1`. Chunk / speculative: `N < 16` → the same decode launch (pad to 8/16); `N ≥ 16` → prefill. Grids and smem are set by the kernel; the driver does not invent them.

Small matrices (`k_proj`/`v_proj`, `M=1024`): the kernel itself sets split-K and asks for an FP32 workspace `[split][M]` from the scheduler’s **preallocated** pool (kilobytes, not a layer). The token loop only passes the pointer.

### 4.1. Prefill

- `y[M, N] = W[M, K] @ x[K, N]`, `N ≤ 16` in one block along N, `N=17..32` — `grid.y=2` or `BN=32`.
- Codebook 256×8 is loaded into smem **once per block**, not in the stage ring.
- Attention: one SDPA per layer, `is_causal=True`. Write K/V into the cache as a **slice** `t = 0..N-1`, not a loop over tokens.
- On 8B `N=16` `gate_proj` INT4 the kernel promises **50–65 µs** — still memory, not FLOP. Do not write our own flash-attn.
- Do not build an intermediate BF16 `W`. Debug dequant is a separate function, not the chat path.

### 4.2. Decode

- `N=1`, pad `BN=8`. Grid `M/128` blocks of 256 threads. 8B INT4 `gate_proj`: **45–55 µs**, kernel caliber ≤55.
- Attention: `q` of length 1, `K,V` = cache slice `[:seq]`. SDPA on Ampere pulls it. Do not write our own flash-decoding until 4k becomes a noticeable share next to weights (on 8B it will not).
- RoPE: position `seq-1`, not zero. The first decode after prefill most often breaks here, not in MMA.
- KV write: `cache[layer, seq-1] = k,v` — assign into a preallocated slot.

### 4.3. What is shared and what is forbidden

Both launches:

- read `(layer, matrix, row_tile, col_group)` fragment-major, as the compressor wrote;
- do not write a dequantized tile to GDDR;
- do not compress back;
- do not touch KV.

Driver on a request:

```
1. embed(prompt)                             # permute to [K, N]
2. for layer: block in prefill mode (N = S)
3. logits = lm_head(norm(x of last position))
4. token = sample(logits)
5. until stop:
      embed(token)                           # N=1
      for layer: block in decode mode
      logits = lm_head(...)
      token = sample(...)
```

No “prefill every new token whole”. That is a common own-loop mistake: it works, but tok/s drops tens of times, KV is written in a circle.

---

## 5. CUDA graphs on a 3080: worth it?

**Yes, on decode it is worth it. On prefill — no.**

Not because “vLLM does it that way”, but because on Ampere the kernel launch cost is about 5–10 µs. Count launches per token if QKV and gate/up are already packed:

| What | Launches per layer | ×32 (8B) | ×64 (32B) |
|---|---|---|---|
| linear (qkv, o, gateup, down) | 4 | 128 | 256 |
| RMSNorm ×2, residual ×2, SiLU, RoPE | ~6 small | ~192 | ~384 |
| SDPA | 1 | 32 | 64 |
| **roughly total** | | **~350** | **~700** |

350 × 8 µs ≈ **2.8 ms**. 700 × 8 µs ≈ **5.6 ms**.

- 8B INT4: bus ~5 ms on weights (4.3 GB / 912 GB/s), live llama.cpp ~80–110 tok/s → 9–12 ms per token. **2.8 ms of launch is a quarter of the budget.** Without a graph our kernel will look “as if twice worse than Marlin”, even though MMA is honest.
- 14B / 32B toward ≥10 tok/s (100 ms): 6 ms is 6%, not death. But 32B also has more Python in the layer loop. The graph takes that off too.

Prefill: `N` is different every time; the graph would have to be captured for each length. Little benefit, capture is expensive. Let it go eager.

How not to set our own feet on fire with a graph:

1. Warm up eager first (see §6): kernels, SDPA, RoPE `inv_freq`, allocator.
2. Capture a **decode step with static addresses**. Inputs are preallocated buffers into which the current token is copied before `replay`.
3. KV grows — a full step with SDPA has a shape that depends on `t`. Three working exits, by increasing pain:

   **A. Graph of linears only.** Python still does norms, RoPE, SDPA. Removes 4 launches × N layers, KV of any length. This is the first step; it is enough to see whether the kernel is alive.

   **B. Full decode, SDPA bucketed.** Buckets `T ∈ {256, 512, 1024, 2048, 4096}`. Five graphs. Attention reads padding — on a 3080 for 8B that is a cheap tax; for 32B at 4k it is already noticeable, but weights are still the main thing.

   **C. Do not.** Recapture the graph every token, `cudaGraphExec` with node patching. Too thin for v1.

4. During capture, must not allocate. If SDPA suddenly asks for a workspace of a new shape — that must be allocated at warmup.
5. Do not hijack `transformers.generate`. Our loop, then a wrapper.

Graph memory is a few megabytes. On 12 GB that is not an argument against.

Acceptance bottom line: **8B without a graph can be shown as “kernel is alive”; the tok/s number for comparison with llama.cpp should be taken with a graph (at least plan A).** 32B — a graph is desirable, but not a blocker for the 10 tok/s goal.

---

## 6. How it breaks

Three diseases that on 12 GB look like “the schema does not work”, even though the kernel has nothing to do with it.

### 6.1. VRAM fragmentation

Symptom: `torch.cuda.memory_allocated()` 7 GB, `nvidia-smi` 11.6 GB, next token — OOM. Or the reverse: smi still tolerates, `CachingAllocator` cannot find an 80 MB hole.

Where holes come from:

- stacks of small tensors (32×7 matrices × `{idx, scale, book}` separately). Cured by **one-to-three arenas** (indices / scales / codebooks), offsets in a table;
- `torch.cat` on KV every token;
- prefill allocated `intermediate[B,S,I]`, decode then wants a long KV — the hole does not glue;
- CUDA graph capture and SDPA workspace after the model is already sitting;
- load: for a second we materialized a BF16 layer “we’ll convert and delete” — a 0.5–1 GB hole, `empty_cache` does not always help.

What to do in the driver: allocate all large pieces **before** the first token, in size-descending order (weights → KV at `max_seq` → prefill scratch). During generation not a single `tensor = tensor.cat(...)`. Turn on `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (PyTorch 2.2+). After load check there is no tensor of size `hidden * intermediate * 2`.

Card with a monitor: the driver will give 300–500 MB to the desktop. The schema budget is not 12.00 GB, but ≈11.2. The scheduler must know. CUDA context in the budget is **768 MB**, not “small change”.

### 6.2. First token “eternal”

These are **three different** pauses; they must not be folded into one “tok/s” number.

| Pause | Who is at fault | How long to wait | When we pay |
|---|---|---|---|
| Process start | CUDA context, extension import, JIT | seconds | once per process |
| Warmup / graph capture | first SDPA shapes, `inv_freq`, capture | 0.2–2 s | once per shape |
| Prompt prefill | honest compute of `S` tokens | ~0.3 s for 512 tok. 8B; seconds on 2k/32B | every request |

Prefill order of magnitude: FLOP ≈ `2 * N_params * S`. 8B × 512 on a 3080 (~30 TFLOPS FP16 Tensor Core) — about 0.3 s plus memory. If the first 8B/32-word token thinks for 8 seconds — that is a bug (most likely they run prefill with the decode kernel, or copy every layer to BF16). If 0.4 s — that is how it should be.

Do not print tok/s that includes prefill and capture. Print three lines: `load_s`, `prefill_ms`, `decode_tok_s`.

### 6.3. Warmup

A 3080 under load jumps in frequency (boost vs drop from 320 W). The first ten tokens are faster or slower than the next. Acceptance:

1. `cudaSynchronize`, 16–32 decode tokens into nowhere.
2. If graph — capture **after** that.
3. Then measure 64–128 tokens, `synchronize` at the boundaries, not on every layer.
4. Do not measure under Nsight and do not measure right after a cold `nvidia-smi dmon`.

Small things that give a “quietly wrong” first token, not a slow one:

- RoPE `inv_freq` was built on CPU, then moved — divergence from BF16 only on decode;
- `cp.async` without `cp.async.commit_group` / `wait_group` — a race, numbers drift, sometimes only on a long prefill;
- graph captured with `seq=1` in SDPA, replay at `seq=400` — garbage in attention, not a crash.

---

## 7. Acceptance on a live 3080

User’s card: RTX 3080 **12 GB** (70 SMs, 912 GB/s), not 10 GB. If `nvidia-smi` shows 10 GB — different numbers, do not blame the schema. Driver 550+, CUDA 12.x, PyTorch 2.4+, the card may be a display card — then write in the protocol “occupied by screen X MB”.

BF16 reference: does not live whole on this card. Compute **layer by layer from disk** (like the offline compressor) or on CPU / another machine and save a logit dump. Compare not “model vs model in VRAM”, but “our run vs dump”.

Two floors of error. People mix them — and fix the wrong thing.

1. **Kernel vs our own dequantized matmul.** Dequant the tile to FP16 on host / a separate kernel, do `F.linear`. Divergence = MMA/`cp.async` bug, not quantization. Hard here.
2. **End of the net vs original BF16.** Divergence = INT4 or 2-bit price. Different thresholds here.

### 7.1. Speed

After warmup, batch=1, greedy sample (so we do not argue about the sampler kernel).

| Scenario | What we measure | Target | How rec |
|---|---|---|---|
| 8B INT4, one `gate_proj` | Nsight / CUDA events | **≤ 55 µs** (kernel) | > 80 µs — pipeline, not format |
| 8B INT4, decode | 128 tokens after warmup 16 | **≥ 50 tok/s** (0.6× llama.cpp Q4); good = 80–110 | < 20 — no graph or BF16 layer |
| 8B INT4, prefill 512 | ms to first token | **< 1.0 s** | > 3 s — they spin prefill with decode-launch one token at a time |
| 14B INT4, ctx 2k, decode | 128 tokens | **≥ 10 tok/s** | schema goal, leftover ~3.5 GB |
| 32B 2-bit 2×8, ctx 2k | 128 tokens | **≥ 10 tok/s** | stage B; weights ~8.3 GB |
| Leak | smi at 16 and 128 tokens | growth ≈ 0 | growth = `cat` KV |

8B INT4 ceiling: 4.3 GB / 912 ≈ 4.7 ms ≈ 210 tok/s. The kernel writes 110–140 at 55–70% of the bus. 50 is the floor. 80+ — can go to 14B.

### 7.2. VRAM, specifically `nvidia-smi`

`torch.cuda.max_memory_allocated()` lies optimistic (does not see CUDA context, graph, display). Into the protocol:

```
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits
```

Take: after load, mid 2k prefill, mid decode. Not after `empty_cache`.

Guide = weights + fixed 880 MB (CUDA 768 + scratch 48 + reserve 64) + KV:

| Scenario | smi, guide | leftover after weights+fix | Ceiling |
|---|---|---|---|
| 8B INT4 + KV 2k | ~5.4 GB + display | 7100 MB | **< 8 GB** |
| 14B INT4 + KV 2k | ~9.2 GB + display | 3484 MB | **< 11.5 GB** |
| 32B 2-bit + emb INT4 + KV 2k | ~9.7 GB + display | 3118 MB | **< 11.5 GB** |
| 32B 2-bit + KV 4k | +256 MB KV | 2094 MB | still yes; 8k FP16 leftover 1070 — tight |

In the load log there must not flicker an allocation of ~`2 * hidden * hidden` per layer (that is a full BF16 matrix). Can hang a `torch.cuda.memory_stats` assert.

### 7.3. Logits vs BF16

Fix the prompt. Two are enough:

- short: `The capital of France is`
- longer: 256–512 tokens from WikiText, to catch accumulation

Take: logits of the **last prefill position** and the first 8 decode steps (a RoPE/KV bug often surfaces only on decode).

**Floor 1 — kernel (both INT4 and 2-bit).** On random `X` and real compressed `W`:

- decode `N=1` and prefill `N=512`;
- `max |Y_kernel − Y_dequant_linear|` ≤ **0.05** on typical layer activations (RMSNorm output ~1);
- if it is already bad here — do not go to the end of the net.

**Floor 2 — net vs BF16 dump.** Metrics on the softmax distribution, not “bit for bit”.

| Metric | INT4 (stage A) | 2-bit codebook (stage B) |
|---|---|---|
| Mean KL(`p_bf16` ‖ `p_ours`) per position | **< 0.01** | **< 0.08** |
| Share of positions where the greedy token matched (128 tok.) | **≥ 85%** | **≥ 60%** |
| Top-5 overlap | ≥ 4 of 5 on average | ≥ 3 of 5 |
| Mean \|Δlogit\| over vocab | < 0.2 | < 0.8 |
| WikiText PPL minus BF16 (if they compute it) | < 0.15 | < 0.8 |

INT4 that passes floor 1 and fails KL 0.01 — a bad offline quant (scales/groups), not the token loop. 2-bit that matches BF16 like INT4 — either the codebook is too fat, or we are looking in the wrong place.

Greedy chat “did not fall apart”: on 8B INT4 the France-capital answer contains Paris. On 2-bit — the same, but already as smoke, not as proof.

Do not require a match to BF16 over a whole 256-token sequence: after the first divergence trajectories drift, and that is not a kernel bug. Hence a **128-token** window and a separate KL on the **same** prefix (teacher-forced on BF16 tokens) — that one is honest.

### 7.4. Minimal protocol to bring back from the card

```
gpu: 3080 12GB, smi_total=12288, display_mb=...
build: stage A | stage B
model: Llama-3.1-8B | Qwen2.5-14B | ...
load_s, vram_after_load_mb
prefill_512_ms, vram_prefill_mb
decode_tok_s (128, after warmup 16), vram_decode_mb
kernel_vs_dequant_maxabs  N=1 / N=512
kl_vs_bf16  prefill_last / decode_8
greedy_match_128
graph: off | linears | full_bucket
```

If 8B INT4: `decode_tok_s ≥ 50`, `vram < 8000`, `kl < 0.01`, `kernel_maxabs < 0.05` — the loop can be glued to 14B. If tok/s is alive and KL is not — the compressor is at fault. If KL is alive and tok/s < 20 — kernel and graph are at fault, not “12 GB is too little”.

---

## Seam with the other modules

- **Ampere kernel** gives one skeleton, two launches (`linear_decode` / `linear_prefill`) and a tile address `(layer, matrix, row_tile, col_group)`. The loop does not know nibble vs codebook, only `meta.dtype`. Do not feed a 2¹⁶ codebook.
- **12 GB scheduler** gives arenas, the 880 MB fix, and `KVCache(max_seq, q8=...)`. The loop does not allocate weights and does not cut the cache. Do not load 32B without INT4 embeddings — leftover 539 MB, that is not our hatch.
- **Offline compressor** writes a fragment-major file that `materialize()` reads. The loop does not quantize on the fly and does not open a BF16 checkpoint.

The token loop is considered assembled when a 3080 has passed the §7.4 protocol for 8B INT4. 14B and 32B are the next checkboxes of the same program, not another architecture.

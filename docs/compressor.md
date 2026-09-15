# Offline compressor for an RTX 3080 12 GB

Seam with the schema: [schema.md](schema.md). Chunk address is the same: `(layer, matrix, row_tile, col_group)`. Here — how to get a file from HuggingFace safetensors **without loading 14B/32B BF16 whole**.

The compressor runs on the same card as inference. 8B BF16 ≈ 16 GB, 14B ≈ 28 GB, 32B ≈ 64 GB — none of them fit in 12 GB. What fits is **one matrix** (max ~0.3 GB Linear, ~1.6 GB embed/lm_head).

---

## Recommendation in one sentence

**v1:** stream tensor by tensor → **NF4, group 64, no calibration** (codec A). In parallel we can write a **2×8 codebook, group 8, k-means on weight MSE** (codec B, M=2). 8B on a 3080 — **20–40 minutes**, not two hours. 32B with the same procedure — **1.5–3 hours**. Full GPTQ/AQLM is not promised in v1.

Calibration (AWQ/GPTQ/codebook fine-tune) is a separate pass, once v1 already runs the kernel.

---

## 1. What fits: one matrix, not a model

Typical Linears in BF16 (one matrix on GPU, the rest on disk):

| Model | Q | gate / up | down | embed / lm_head |
|---|---|---|---|---|
| Llama-3 8B (d=4096, ff=14336) | 34 MB | **117 MB** | 117 MB | **1.05 GB** |
| Qwen2.5 14B (d=5120, ff=13824) | 52 MB | 142 MB | 142 MB | ~1.3 GB |
| Qwen2.5 32B (d=5120, ff=27648) | 52 MB | **283 MB** | 283 MB | **1.56 GB** |

Peak weights are not FFN, but `embed_tokens` / `lm_head`. Still <2 GB. We do not quantize norms (RMSNorm); they are kilobytes, leave them BF16.

The killer is not W, but the **activation cache** at calibration. 128 sequences × 2048 tokens:

| What | 8B, d_in=4096 | 8B, down d_in=14336 | 32B, down d_in=27648 |
|---|---|---|---|
| X in FP16, all at once | 2.1 GB | **7.5 GB** | **14.5 GB** |
| Hessian H=XXᵀ, FP32 | 67 MB | 822 MB | **3.1 GB** |
| Channel means \|x\| only | 8 KB | 28 KB | 54 KB |

Hence the table “can we calibrate on 12 GB”:

| Method | 8B | 14B | 32B |
|---|---|---|---|
| Weight-only NF4 / INT4 / k-means | yes, minutes | yes | yes, hours |
| Imatrix (accumulate x² per channel, X in batches on CPU) | yes | yes | yes |
| AWQ: only s_X + α grid, X in chunks | yes | yes | yes, if layer by layer |
| AWQ/GPTQ: keep the whole layer X on GPU | tight / no | no | no |
| GPTQ: model on CPU, layer on GPU, H += X_bᵀX_b | yes, ~1 h | yes, 2–4 h | yes, overnight, if we do not accumulate quant on GPU |
| AQLM: beam + block fine-tune | not as v1 | no | no (authors: 7B ≈ a day on A100) |

GPTQ on 12 GB for 8B — **yes**, if we do not `from_pretrained` the whole model into VRAM (that is the classic AutoGPTQ OOM). TheBloke recipe: weights on CPU/disk, on the card only the current layer + Hessian. 32B: H for down_proj 3.1 GB + Cholesky another ~2–3 GB — fits, **if there is nothing else on the card**.

AWQ is easier than GPTQ: no Hessian, per-channel magnitudes and a grid of ~20 α values ([Lin et al., MLSys 2024](https://arxiv.org/abs/2306.00978)). The X cache is still needed while searching the scale — stream it in batches or take a token subset (128–512).

---

## 2. v1 that will actually finish in <2 h on 8B

One pass over disk. No calibration.

```
for each shard in model.safetensors.index.json:
  open the file (mmap / safe_open), LRU = 1 file
  for each tensor in the shard:
      if RMSNorm / bias / rotary → copy BF16 as-is
      if nn.Linear / embed / lm_head:
          read ONLY this tensor → GPU
          codec A: NF4-g64
          codec B (optional, same W): residual k-means M=2
          append the blob to model.chr
          delete W, empty_cache()
```

Do not call `AutoModelForCausalLM.from_pretrained` without `device_map` + empty weights. Do not assemble a `state_dict`. Read via `safetensors.safe_open` / `get_slice` by name from `index.json`.

**Why NF4, not INT4-RTN and not GPTQ.** QLoRA: NF4 is a grid for N(0,σ); on weights it beats FP4/INT4 without calibration ([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)). The gap vs AWQ/GPTQ at 4 bits is usually **+0.2–0.5 PPL** to BF16 for all three; for a first chat NF4 is enough. Calibration does not speed up the kernel, only quality.

**Why a codebook at M=2 immediately, not M=1.** M=1, B=8, g=8 is **1 bit/weight**. Even full AQLM 1×8 on Llama-2-7B gives WikiText **7.85** vs FP16 **5.12**. Without training the codebook it will be worse. M=2 → 2 bit/weight, which is exactly schema stage B and AQLM format `2x8` ([Egiazarian et al., 2024](https://arxiv.org/abs/2401.06118)).

v1 time, 3080, NVMe:

| Step | 8B | 14B | 32B |
|---|---|---|---|
| Read safetensors + NF4 | 5–15 min | 10–25 min | 20–45 min |
| + k-means M=2, GPU | +15–25 min | +30–50 min | +1–2 h |
| Total v1 both codecs | **20–40 min** | ~1 h | **1.5–3 h** |

---

## 3. Codec A — group-wise NF4

Like bitsandbytes / QLoRA, group along the **input** (K axis), so tile `(row_tile, col_group=8)` takes the same scale on 8 neighboring weights.

**Steps on matrix W ∈ R^{n_out × n_in}:**

1. `n_in` must be divisible by 64 (if not — pad with zeros, original `n_in` in the header).
2. Slice W into groups `[n_out, n_in/64, 64]`.
3. `s = max(|g|)` along the last axis. If s=0 → s=1.
4. `u = g / s`, clip to [−1, 1].
5. Each u → nearest level of the **fixed** NF4 table (16 quantiles of N(0,1), as in QLoRA; we do not train a codebook).
6. Pack two nibbles into a byte. Layout: `[n_out, n_in/2]` uint8, row-major. A 64×8 weight tile = 64×4 bytes, a contiguous row-wise chunk.
7. Scales: `[n_out, n_in/64]` FP16.
8. Double-quant of scales (another −0.4 bit in QLoRA) we **do not** do in v1 — the saving is tiny, the kernel is harder.

Asymmetric INT4 + zero-point (GPTQ/AWQ-pack) — the same container, different `codec` and an optional `zeros` field. One kernel: nibble → dequant(`scale`, `zero`, level).

**Calibration later, not instead.** If memory allows (8B — yes):

- **AWQ-lite:** run 128×512 tokens layer by layer; on CPU accumulate `s_X[j] = mean(|X[:,j]|)`; grid α ∈ {0, 0.05, …, 1}; pick α minimizing ‖X Wᵀ − X Q(s⊙W)ᵀ‖ on a **subsample** of X (not all 256k tokens). Then ordinary NF4/INT4 on already multiplied channels. We bake the inverse scale into the next norm / do not store it separately, as in AWQ.
- **GPTQ:** H = 2XᵀX by accumulating GEMM in batches (`H.addmm_(Xb.T, Xb)`), drop Xb immediately. Then column by column with the H⁻¹ correction ([Frantar et al., 2022](https://arxiv.org/abs/2210.17323)). On a 3080 for 8B it fits. For 32B down_proj — only this recipe, no X cache.
- **Do not** do `desc_act` / act-order in v1: it breaks sequential tile access along K.

---

## 4. Codec B — codebook, group 8

Format as AQLM / additive quantization ([Babenko & Lempitsky, 2014](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Babenko_Additive_Quantization_for_2014_CVPR_paper.html); [Egiazarian et al., 2024](https://arxiv.org/abs/2401.06118)):

```
group of 8 weights ≈ C₁[i₁] + C₂[i₂]     # M=2
C_m ∈ R^{256 × 8}                       # B=8
indices i — uint8
```

Bit/weight = M·B / 8 = **2.0** at M=2. Codebook per matrix: 2×256×8×2 B = 8 KB. Per layer 7 matrices ≈ 56 KB. In kernel smem — the same 4–8 KB per matrix as in the schema.

### 4.1. v1: residual k-means, MSE(W) only

No Hessian, no X. This is AQLM **initialization** (residual k-means, Chen et al. 2010), without their beam search and block fine-tune.

```
vectors V ← W.reshape(n_out * n_in/8, 8)     # do not copy extra, view
R ← V
for m = 1..M:
    C_m, idx_m ← k-means(R, k=256, iters=20)
    R ← R − C_m[idx_m]
write C[0:M], idx[0:M]
```

Lloyd on GPU:

1. Codebook 256×8 in smem / cache (4 KB).
2. Assignment in **chunks** of 256k–1M vectors: `dist = ‖x‖² + ‖c‖² − 2 x Cᵀ` (GEMM), argmin.  
   **Must not** materialize `(N, 256, 8)` — for a fat matrix that is ~60 GB and instant OOM.
3. Update: sum and count over 256 bins (two passes or `index_add`). Empty clusters — resplit the fattest.
4. 15–25 iterations. k-means++ only on a 64k-vector subsample.

Mini-batch k-means ([Sculley, 2010](https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf)) — fallback if full Lloyd suddenly hits a wall; quality a bit worse.

### 4.2. What one 4096×14336 costs on a 3080

- Vectors: 4096×14336 / 8 = **7.34M**, dim=8, k=256.
- One assignment = GEMM (7.34M × 8) × (8 × 256) ≈ **30 GFLOP** + argmin.
- 3080: 29.8 TFLOP/s FP32, 912 GB/s. Theory: **~1 ms** compute, **<1 ms** memory.
- Practice with Python/sync overhead: **1–5 s** per matrix with a proper CUDA k-means (faiss-gpu / own kernel). 20 iterations + k-means++.
- M=2: ×2, plus subtraction → **3–10 s**.
- sklearn on CPU: **2–10 min** for the same matrix. On 8B that is already hours — CPU-only is not our v1.

8B ≈ 7×10⁹ Linear weights ≈ 120 such “fat equivalents”. GPU k-means M=2: 120×6 s ≈ **12 min** of pure compute, with I/O **15–25 min**.

32B ×4 in parameters: **1–2 h** k-means. A night is not needed; a night is reserve for calibration.

Full AQLM (beam + fine-tune codebooks on activations): 7B ≈ **1 day on A100**, 70B ≈ 10–14 days on one GPU ([AQLM, app. D](https://arxiv.org/abs/2401.06118)). On a 3080 that is not overnight.

### 4.3. Codebook calibration, if memory allows

Order by cost:

1. **Initialization as in §4.1** — always.
2. **2–3 Lloyd passes already with channel weight** `w_j = mean(x_j²)` (imatrix / Fisher diagonal, like SqueezeLLM). We do not store X: one forward layer by layer, on CPU `acc[j] += x_j²`, then weighted k-means. Fits in 12 GB on 8B/14B/32B.
3. **Per-layer MSE(XW)**: freeze indices, learn C from `min ‖X (W − decode(C,idx))ᵀ‖` — a linear problem in C. X in batches from CPU. On 8B — yes. On 32B down_proj X is heavy: batch 1–2 sequences.
4. Beam + block fine-tune AQLM / GPTVQ ([van Baalen et al., 2024](https://arxiv.org/abs/2402.15319)) — not v1.

---

## 5. How not to catch OOM

Hard compressor rules:

1. **Stream from disk.** One shard is open. One tensor is read. After the codec — `del W; torch.cuda.empty_cache()`.
2. **The whole model must not live in VRAM, nor necessarily in RAM.** 32B BF16 = 64 GB host RAM; if RAM is less — mmap the shard only, not `load_state_dict`.
3. **Layer by layer for calibration.** Load layer i (BF16), run a batch, activation **straight to CPU** (pin memory), delete layer i, load i+1. A prefix already in quant can sit on disk and be loaded as INT4 if we need a “like GPTQ” accumulated input; for v1 there is no calibration, the question does not arise.
4. **X not whole.** Hessian: `H += X_bᵀ X_b` over 1–4 sequences. AWQ search: random 1–2k tokens, not 256k.
5. **Quant straight to disk.** Do not accumulate the compressed model in VRAM during compression (on 32B that is ~8 GB and the Hessian is already tight).
6. **k-means in chunks.** See §4.1.
7. **embed / lm_head separately.** If 1.6 GB + codebook + distance chunk > comfort — compute NF4/k-means **in row stripes** (e.g. 4096 rows), shared codebook: first pass accumulate on a row subsample, second — assign indices.
8. **Do not hold two BF16 layers** “just in case prefetch” during compression. Prefetch is for runtime, not for the compressor.

Peak v1 on 32B, roughly: CUDA context ~0.5 GB + lm_head 1.6 GB + k-means distance chunk (1M×256×2 ≈ 0.5 GB) ≈ **3 GB**. Headroom is huge.

---

## 6. `.chr` file format (CHR0)

Like safetensors: JSON header + raw blobs. The kernel reads by offset, does not parse the whole model.

```
offset 0        uint64  header_nbytes     # JSON length, little-endian
offset 8        UTF-8   header_json
offset 8+N      pad to 64 bytes
then            payloads, each aligned to 64
```

### 6.1. Header

```json
{
  "magic": "CHR0",
  "version": 1,
  "arch": "llama",
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_layers": 32,
  "vocab_size": 128256,
  "tile": { "row": 64, "col_group": 8 },
  "tensors": {
    "model.layers.0.self_attn.q_proj": {
      "layer": 0,
      "kind": "q",
      "codec": "nf4",
      "shape": [4096, 4096],
      "group_size": 64,
      "data":  [4096, 8200],
      "scale": [8200, 16400]
    },
    "model.layers.0.mlp.down_proj": {
      "layer": 0,
      "kind": "down",
      "codec": "vq",
      "shape": [4096, 14336],
      "group_size": 8,
      "n_codebooks": 2,
      "codebook_bits": 8,
      "codebook": [20000, 28192],
      "index": [28192, 200000]
    },
    "model.norm.weight": {
      "kind": "norm",
      "codec": "bf16",
      "shape": [4096],
      "data": [200000, 208192]
    }
  }
}
```

Fields `data` / `scale` / `zero` / `codebook` / `index` are `[start, end)` **from the start of the file**, not from the end of the header. That makes `mmap` easier.

`kind`: `q k v o gate up down embed lm_head norm other`.

`codec`: `bf16` | `nf4` | `int4` | `vq`.

Both codecs can live in one file (A on an 8B run, B on the same names with a suffix, or a separate `.chr`). For kernel comparison it is more convenient to have **two files** from one pass: `model.nf4.chr` and `model.vq2.chr`.

### 6.2. Blob layout

**nf4 / int4**

- `data`: uint8, shape `[n_out, n_in/2]`. Byte `data[r, c]` = nibble `W[r, 2c]` in the low 4 bits, `W[r, 2c+1]` in the high. Level 0..15 — index into the NF4 table or shifted INT4.
- `scale`: FP16, `[n_out, n_in/group_size]`, group_size=64.
- `zero`: optional FP16 of the same shape (for asymmetric `int4`). NF4 has none.
- Tile `(row_tile=i, col_group=j)`: rows `[64i, 64i+64)`, weight columns `[8j, 8j+8)` → bytes `data[64i:64i+64, 4j:4j+4]`, scale `scale[64i:64i+64, floor(8j/64)]`.

**vq**

- `codebook`: FP16, `[M, 256, 8]`.
- `index`: uint8, `[n_out, n_in/8, M]`. For M=2 two bytes per group.
- Tile `(i, j)`: `index[64i:64i+64, j, :]` → gather from the codebook, add M vectors of 8, that is the 64×8 BF16 in registers.

**bf16:** raw little-endian, as in safetensors.

Tensor names — HuggingFace (`model.layers.{L}.mlp.down_proj.weight` without the `.weight` suffix or with it — freeze: **without `.weight`**, bias if present — `.../bias`, codec `bf16`).

End of file: nothing magic. Length = max(end). Checksum is not required in v1; if needed — `sha256` of the whole file in a neighboring `model.chr.sha256`.

---

## 7. Minimally live quality: NF4 + k-means 256 without Hessian

**NF4 without GPTQ — yes, OK for a first test.** This is stock 4-bit QLoRA. On Llama-class 7B/8B WikiText is usually within ~0.3–0.6 of BF16, chat is coherent. Worse than AWQ/GPTQ at the same bit, but not a “broken model”. This is the right caliber: kernel and format, not a hunt for 0.1 PPL.

**K-means 256, g=8, no calibration — OK as a stage-B diagnostic, not as 32B chat.**

- Full AQLM 2×8 (trained on X) on Llama-2-7B: WikiText **7.61**, with extra fine-tune **6.57**, FP16 **5.12** (paper table 12).
- Residual k-means is only their init. Without steps 2–3 from §4.3 it will be noticeably worse than 7.61. Expectation: the model still speaks in sentences, gets dumber, sometimes drifts. That is enough to see that the codebook kernel runs and PPL is not 50.
- Do **not** put M=1 (1 bit) in the first pass as a quality target: even a trained 1×8 is already 7.85; weight-only will be garbage.
- A GPTQ Hessian on the codebook (GPTVQ) is not needed in v1: it is expensive and does not answer “will the format eat 12 GB”.

Practical v1 acceptance:

1. 8B NF4: WikiText does not go to space (guide: no worse than +1.0 to BF16). Chat on 10 questions does not rave.
2. 8B VQ 2×8 weight-only: take and record PPL, chat — “alive / dead”. If dead — do not fix the codebook with a Hessian until there is imatrix.
3. Kernel vs llama.cpp Q4 on the same 8B — a separate acceptance; the compressor has nothing to do with it.

---

## 8. Path to 32B overnight

One card, one disk, no multi-GPU.

| Night | What we run | Why | Fits in 12 GB? |
|---|---|---|---|
| First (evening) | v1 NF4 on 8B, run chat + kernel | pipeline caliber | yes |
| Same evening | v1 VQ 2×8 on 8B, take PPL | see codebook damage without X | yes |
| Night 1 | v1 NF4 + VQ on **32B**, same flags | file for schema stage B | yes, 1.5–3 h |
| Night 2, if 2-bit chat is bad | imatrix: 128×512, layer by layer, X on CPU; recompute k-means with channel weight | cheap calibration, like llama.cpp imatrix / SqueezeLLM diagonal | yes |
| Night 3, if 4-bit 14B/32B matters | AWQ-lite on 14B, then 32B | stage A quality without Hessian | yes if X is streamed |
| Someday | GPTQ-stream on 8B, compare to NF4 | is dragging H worth it | yes on 8B; 32B — yes, but longer than a night if suboptimal |
| Not promised on a 3080 | full AQLM / PV-Tuning | SOTA 2 bit | no by time |

32B NF4 ≈ 16 GB / 4 = 4 GB of weights? No: 32B × 0.5 byte ≈ **16 GB**, will not fit in 12 GB. Stage A on 32B is not a schema goal (schema: 32B at **~2 bits** ≈ 8 GB). Overnight it makes sense to write **VQ 2×8 on 32B**; NF4 on 32B — only if we offload later. 14B NF4 ≈ 7–8 GB — fits, that is stage A.

Overnight 32B total: **one §2 pass with codec B**. In the morning there is `model.vq2.chr` ~8 GB. If PPL is scary — night 2 with imatrix, not GPTQ.

---

## 9. Implementation order (no wall of code)

1. Reader `index.json` + `safe_open`, print names/shapes, peak RSS/VRAM. Check: 32B index walks without loading weights.
2. CHR0 writer: header in memory, blobs append, at the end rewrite JSON and `header_nbytes`.
3. Codec A on one `4096×4096`, check decode vs W (MSE, max err).
4. 8B run → `*.nf4.chr`, time and peak VRAM (should be <4 GB).
5. Codec B: chunked k-means on one `4096×14336`, time in seconds — check against §4.2.
6. 8B VQ, PPL. Then 32B VQ overnight.

v1 dependencies: `torch`, `safetensors`, optionally `faiss-gpu`. No transformers model in memory.

---

## Sources

- [Frantar et al. GPTQ. arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- [Lin et al. AWQ. MLSys 2024 / arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- [Dettmers et al. QLoRA / NF4. arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- [Egiazarian et al. AQLM. arXiv:2401.06118](https://arxiv.org/abs/2401.06118) · [code](https://github.com/Vahe1994/AQLM)
- [Babenko & Lempitsky. Additive Quantization. CVPR 2014](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Babenko_Additive_Quantization_for_2014_CVPR_paper.html)
- [Han et al. Deep Compression. arXiv:1510.00149](https://arxiv.org/abs/1510.00149)
- [Kim et al. SqueezeLLM. arXiv:2306.07629](https://arxiv.org/abs/2306.07629)
- [van Baalen et al. GPTVQ. arXiv:2402.15319](https://arxiv.org/abs/2402.15319)
- [Sculley. Web-scale k-means. 2010](https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf)
- [TheBloke on AutoGPTQ memory](https://github.com/PanQiWei/AutoGPTQ/issues/179): model on CPU, layer quant on GPU
- llama.cpp imatrix: activation x² as quantization weight, without a full Hessian

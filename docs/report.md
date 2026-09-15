# Can you fit a large model into VRAM the way DNA fits into a chromosome?

In human language, from the end (will the piece decompress in time): [ot-konca.md](ot-konca.md).

---

**Short answer.** Yes, but not with ZIP and not “10×” without a cost. Scalar lossless squeezes ~**30%** — that is a weak decoder. With a **key** (codebook, basis, latent) you can go to 2 bits/weight and even to a kilobyte on toy tasks. “As much as you want” then hits not the numbers as such, but the pigeonhole principle plus a time limit on decompression: either it is no longer the same model, or generating the weights costs more than the compute itself. 70B in BF16 is 141 GB, lossless — 95 GB, RTX 4090 — 24 GB. Quantization, a vector codebook, and distillation put the model on the card, not AES.

---

## 1. What is actually being proposed

The scheme sounds like this:

1. Compress the entire model “like zip” and put it wholly into VRAM.
2. As the token walks the layers, unpack only the active stretch.
3. Immediately compress the finished stretch back.

This is very similar to how RNA polymerase II passes a nucleosome: DNA is locally unwound, read, and the already-transcribed stretch is wound back onto histones ([Kujirai et al., Science 2019](https://www.science.org/doi/10.1126/science.aau9904); [Filipovski et al., Science 2022](https://www.science.org/doi/10.1126/science.abo3851)).

In transformer inference there really is a “polymerase”: the token walks layers strictly in order, layer N+1 is not needed until layer N is computed. So **you do not have to keep the entire net expanded at once**. FlexGen, llama.cpp, and DeepSpeed already exploit this fact: they stream layer weights from RAM/SSD.

Inside a layer the picture is different. The GPU computes `Y = WX` in tiles, not “left to right through the file”. Classic ZIP is a sequential stream without proper random access. A matrix kernel needs a codec that can jump to a specific tile and decompress it in constant time. So working systems compress not “the model file” but **the exponents of the numbers** and decode **the tile straight into Tensor Core registers**.

One more correction to step 3. Re-compressing a finished layer is pointless: the compressed copy already sits in HBM. You just drop the unpacked buffer. “Recompaction” is needed in biology because the compact form *is* the source. On a GPU the source is the compressed tensor; the expansion is a consumable.

```
HBM (compressed weights, always)
        │  load tile
        ▼
SRAM / registers  ── decompress ──► Tensor Core GEMM
        │
        └── drop the buffer; the compressed copy stays
```

This is literally ZipServ’s slogan: *load-compressed, compute-decompressed* ([Fan et al., ASPLOS 2026](https://arxiv.org/abs/2603.17435)).

---

## 2. Where the DNA analogy is exact, and where it breaks

| Chromatin | Dense LLM | What follows |
|---|---|---|
| Genome compressed in the nucleus ~10⁴× | BF16 weights are already close to their entropy limit | 10⁴× from zip will not happen |
| Most genes are silent | In a dense model **every** weight is read on **every** token | You cannot “leave most of the net un-unwound” |
| Gene expression is a rare event | MoE: a small share of experts is active | MoE is real “gene expression” |
| Hot genes in euchromatin, cold in heterochromatin | PowerInfer: hot neurons in VRAM, cold in RAM | Selective residency matters more than archiving |
| Polymerase walks along the gene | Layers walk along the token | Layer streaming is a valid idea |
| Inside a nucleosome, access is almost linear | GEMM reads weights in tiles, not as a stream | You need a tile-codec, not DEFLATE |
| FACT/Spt6 help recompaction | Prefetch the next layer while the current one is computed | Overlap I/O and compute |
| DNA alphabet is 4 letters, many repeats | BF16 mantissa and sign are almost noise | An LZ77 dictionary finds no repeats |

The main gap: DNA can be compressed so hard because **most of the sequence is not being read at a given moment**. A dense LLM reads all parameters on every token. Compression is then bounded not by “how cleverly you pack the file” but by **how much information is in the numbers at all**.

Closest to biology is not zip, but three things:

1. **Mixture-of-Experts.** Mixtral-8×7B stores **46.7B**, computes **12.9B** (2 of 8 experts). DeepSeek-V3: 671B on disk, **37B active** per token (~5.5%). Memory is still needed for almost all experts if they are resident; compute drops. That is “do not transcribe the whole genome”.
2. **PowerInfer and LLM in a Flash.** Neuron activations follow a power law: a small core of “hot” ones is almost always on. Hot ones go in VRAM, cold ones are computed on CPU ([Song et al.](https://arxiv.org/abs/2312.12456)). Apple goes further: from flash they read ~**2%** of FFN per token, the model can be **2× larger than DRAM** ([LLM in a Flash](https://arxiv.org/abs/2312.11514)).
3. **Online tile decompression.** DFloat11, ZipServ, NeuZip, Unweight — local nucleosome unwinding before compute.

---

## 3. Why ordinary ZIP is almost useless

The intuition “weights are a pile of numbers, zip will compress them” hits Shannon’s theorem.

BF16 is 1 sign bit + 8 exponent bits + 7 mantissa bits. Measurements on models from fractions of a billion to a trillion parameters:

- sign ≈ 1 bit of entropy out of 1 (fair coin);
- mantissa ≈ 7 out of 7 (almost the maximum);
- exponent ≈ **2.6 bits out of 8**.

Total ~**10.6 bits of information per 16 bits of storage** — a third of the budget is empty, and all of that emptiness sits in the exponent. Weight magnitudes after training cluster tightly around 2⁻⁷…2⁻⁶, so of 256 possible exponent bytes only a handful of values are live: IBM reports that **12 values cover 99.9%** ([ZipNN, IBM Research](https://research.ibm.com/blog/Zip-NN-AI-compression); [DFloat11](https://arxiv.org/abs/2504.11651); [Fergus Finn, 2026](https://fergusfinn.com/blog/weight-entropy/)).

LZ77 (the heart of zip/gzip/zstd as a “dictionary”) looks for repeating *byte sequences*. Trained weights have almost none. NVIDIA writes it outright: on dense checkpoints LZ4 and Bitcomp give **~1.00×**, and the win comes only from entropy codecs (Huffman / ANS) — about **1.14–1.18×** on a mixed checkpoint and about **1.3×** if you split off the exponent ([nvCOMP + PyTorch checkpoints](https://developer.nvidia.com/blog/cut-checkpoint-costs-with-about-30-lines-of-python-and-nvidia-nvcomp/)). ZipNN: dictionary methods “hardly achieve any data reduction”; you need Huffman over the exponent byte ([arXiv:2411.05239](https://arxiv.org/abs/2411.05239)).

Hence the practical ceiling **lossless for BF16: ~30%**, i.e. the model occupies ~70% of the original size. This is not one paper’s heuristic — it is Shannon on the empirical distribution. You can go higher only if:

- the model is “clean” (after rounding/conversion the mantissa is sparse too) — ZipNN saw **>50%** on those;
- you compress the *delta* of two similar models (checkpoints, LoRA) — many more repeats;
- you quantize first, then entropy-pack the code indices again.

On INT4/FP4 the theoretical gap to Shannon is sometimes drawn as “another 6–10×” ([Approaching Shannon Bound](https://arxiv.org/html/2606.15789v1)). That is the entropy of *already quantized symbols*, often from zeros and histogram skew. Realizing 10× on a GPU tile without killing decode speed has not happened yet; a realistic add-on on top of quantization is **another ~10–30%**.

---

## 4. What is already built — do not write from scratch

The idea “stored compressed, unwound as you compute” became its own line of work in 2024–2026. Ready bricks below.

### 4.1. Lossless, weights stay compressed on the GPU

| System | What it does | Savings | How it unpacks | Link |
|---|---|---|---|---|
| **DFloat11** (NeurIPS 2025) | Huffman on the BF16 exponent → ~11 bits/weight. Output is **bit-identical**. 70B: 141→95 GB; 405B: 812→551 GB | ~30% (68% of size) | Unpack a **block** into HBM, then GEMM. At batch=1 ≈ **2× slower** than native BF16 until the kernel is fused | [paper](https://arxiv.org/abs/2504.11651), [code](https://github.com/LeanModels/DFloat11) |
| **ZipServ** (ASPLOS 2026) | Fixed bitmap scheme TCA-TBE instead of variable-length Huffman | up to 30% | **ZipGEMM**: decode straight into Tensor Core registers, no intermediate HBM | [paper](https://arxiv.org/abs/2603.17435), [code](https://github.com/HPMLL/ZipServ_ASPLOS26) |
| **NeuZip** | ANS on the exponent; there is a lossy mantissa-trim variant | training Llama-3 8B: 31 → <16 GB; inference — even stronger in lossy | GPU-parallel ANS | [paper](https://arxiv.org/abs/2410.20650), [code](https://github.com/BorealisAI/neuzip) |
| **Unweight** (Cloudflare) | Huffman of MLP exponents; tile assembled in shared memory before WGMMA | ~30% MLP, ~20% of the whole model | fused kernel on Hopper | [report](https://research.cloudflare.com/papers/unweight-2026.pdf) |
| **ANS tiles “toward Shannon”** | rANS/tANS per tile, back-to-back with GEMM, even on top of INT4/AWQ | to 0.01–0.1 bit of the bound; Mixtral-176B: batch 20 → 95 | decode into shared memory for each GEMM tile | [paper](https://arxiv.org/html/2606.15789v1) |
| **bf16_huffman_infer** | fused Huffman-GEMV in the spirit of DFloat11 | ~25% VRAM | 80–90% of BF16 speed, on a 4060 Ti sometimes *faster* than BF16 (the bottleneck is the bus) | [code](https://github.com/lszxb/bf16_huffman_infer) |

DFloat11 is the most direct answer to the original question. The compressed model lives on the GPU. Before matmul the layer is expanded, after — discarded. Llama 3.1 405B (810 GB BF16) thus fits on **one 8×80 GB node** instead of two. Versus CPU-offload of the same pieces that would not fit: **2.3–46×** higher generation speed.

ZipServ closes DFloat11’s main hole: a separate unpack into global memory takes 1.56–3.44× the GEMM itself. Fusion with Tensor Core not only saves memory but also **speeds up** inference (up to 2.21× kernel vs cuBLAS, ~1.22× end-to-end vs vLLM) — because fewer bytes travel the bus, and decode on the decode phase (batch=1) hides behind memory wait.

### 4.2. ZIP-like codecs — for disk and network, not for VRAM

**ZipNN** (IBM + BU/MIT/Dartmouth/TAU) — Huffman on the split-off exponent, up to 80 GB/s unpack on CPU, ~33% on BF16. Goal: Hugging Face, checkpoints, traffic. GPU version “on the way”. This is the right zip for *delivering* the model, not for living in HBM ([paper](https://arxiv.org/abs/2411.05239), [code](https://github.com/zipnn/zipnn), [IBM](https://research.ibm.com/blog/Zip-NN-AI-compression)).

**nvCOMP** — NVIDIA’s hardware stack (LZ4, ZSTD, GDeflate, ANS, Bitcomp). On Blackwell there is even a dedicated **Decompression Engine**: up to hundreds of GB/s, zero load on SMs, for LZ4/Snappy/Deflate ([NVIDIA blog](https://developer.nvidia.com/blog/speeding-up-data-decompression-with-nvcomp-and-the-nvidia-blackwell-decompression-engine/)). The irony: the codec that hardware squeezes for free **does not compress weights**. For weights you need ANS/Huffman, which streaming multiprocessors still compute. Hardware is already heading toward “store compressed in HBM”, but you have to pick the codec that has entropy, not a dictionary.

### 4.3. Not zip, but “not all genes at once”

| System | Idea | When it wins |
|---|---|---|
| **llama.cpp / GGUF** | mmap the file + `-ngl` layers in VRAM, the rest in RAM/SSD | 70B Q4 on 24 GB: some layers on GPU, ~8–14 tok/s instead of 30+ |
| **FlexGen** | Scheduler: weights/KV/activations across GPU–CPU–SSD; 4-bit | High throughput with large batches, not interactive decode |
| **DeepSpeed ZeRO-Infinity** | Training: parameters, gradients, optimizer states on NVMe | Training, not a single chat |
| **DeltaZip** | Base in VRAM, compressed fine-tune delta is loaded | Many adapters on one base, 6–8× on the delta | [arXiv:2312.05215](https://arxiv.org/abs/2312.05215) |
| **MoE-Infinity / FloE** | Cache hot experts, the rest in RAM | Mixtral FP16 ~94 GB, of which ~67 GB are *sleeping* experts |
| **PowerInfer / PowerInfer-2** | Hot/cold neurons; on phone — clusters + UFS | Up to 11.7× vs llama.cpp on a 4090; PowerInfer-2 up to 27.8× on OnePlus 12 |
| **LLM in a Flash** (Apple) | FFN from flash, DRAM holds a window of active neurons | Model up to 2× DRAM; ~2% FFN from storage per request |

Offload loses to compression-in-VRAM for one reason: PCIe 4.0 ×16 ≈ 32 GB/s, HBM is hundreds of gigabytes–terabytes per second. DFloat11 beats offload by tens of times for exactly that reason. SSD as “heterochromatin” is fine for rare experts and cold neurons, not for every layer of a dense model on every token.

### 4.4. Quantization — the real “zip you also compute in”

This is no longer about unwinding. INT4/AWQ/GPTQ/GGUF Q4 and BitNet store few bits **and compute in that format**. No need to restore BF16.

| Format | 70B, weights | On 24 GB |
|---|---|---|
| BF16 | ~140 GB | no |
| DFloat11 / ZipNN lossless | ~95 GB | no |
| INT8 / Q8 | ~70–75 GB | no |
| Q4_K_M | ~42–43 GB | only with offload |
| INT4 weights + KV separate | ~35–38 GB | barely on 48 GB, not on 24 |
| BitNet 1.58 (if you *train* that way) | theoretically ~14 GB | yes, but that is a different model |

BitNet b1.58 stores weights in {−1, 0, +1}, packs 4 values into int8, unpacks in SRAM in the kernel and computes with adds ([technical report](https://arxiv.org/html/2504.12285v2)). Same “pack–load–unpack–compute” pattern, but compression is **lossy and baked into training**. You cannot do this from a finished BF16-70B without retraining.

Practical takeaway for local hardware in 2026: people run 70B on 24 GB not with zip, but with **Q4 + partial offload**. Lossless-30% is a way to squeeze an *unquantized* 405B onto one 8×80 node or to win batch size in a service where bit-identical match to BF16 matters legally.

---

## 5. The second stomach: KV-cache

Even a perfect weight zip does not solve all memory. KV-cache grows with context length and request count. For Llama 3 70B (GQA) that is on the order of **0.3 MB per token**; at 128K that is tens of gigabytes, comparable to the model itself. Weights are static and shared across users; KV is not.

So the “model chromosome” and the “conversation chromosome” are different objects. For KV, other methods: PagedAttention (vLLM), KV quantization, windows (StreamingLLM), selective storage (H2O, SnapKV), MLA in DeepSeek. LEXI separately squeezes activation and cache exponents on a chiplet bus ([arXiv:2603.15589](https://arxiv.org/html/2603.15589)).

---

## 6. “These are just numbers, a key will compress as much as you want”

The first “~30%” estimate is about a very weak decoder: each number independent, exponent histogram. If you allow a **key** (codebook, basis, small generator net), the bound is different. But “as much as you want” also breaks, and not on the archiver’s greed.

### Three different “keys” — they get mixed up

**Crypto key (AES).** With it you can hide the numbers and get them back. Size does not drop: ciphertext = source. That is not compression.

**Kolmogorov key.** The shortest program that prints these weights. For any finite string such a program exists, and it can be short. For Llama a recipe formally works: dataset + trainer code + seed. The description is short (“this GitHub and this torrent”). But:

- if the key must be **self-contained**, you have to put the training data inside. Llama 3: on the order of 15T tokens, that is **tens of terabytes of text** — more than 810 GB of weights. The weights are already a compression of the internet, not the reverse;
- if the key is a **reference** (“download Llama-3-70B”), you can compress to a filename. The data lives at Meta, not in your VRAM;
- “decompressing” such a key = retraining the model. That is not inference.

**Working key: a decoder that sits alongside and finishes in milliseconds.** Codebook, SVD factors, seed + latent, hypernetwork. Compressed size = |key| + |indices|. Then the pigeonhole principle applies: a k-bit key yields at most 2^k different models. To hit **exactly** this trained 70B, that point must lie in the decoder’s image. A random PRNG from a short seed gives noise, not Llama. So either the decoder is rich (the key is large), or the hit is approximate — and that is no longer the same model.

The right limit is not “byte entropy” but **time-bounded Kolmogorov complexity** (Levin complexity): a program is allowed to be short only if it also finishes quickly. Inference needs milliseconds. Retraining and searching over programs drop out.

### The vector way — it exists, and it is not zip

A weight is not a scalar in a vacuum. A matrix row, a group of 8 neighboring numbers, a GEMM tile — that is a vector. If the vectors live in a small dictionary, you store an index, not coordinates.

That is what people do:

- **Deep Compression** (Han et al., 2015): pruning + a shared codebook (k-means) + Huffman. Prototype of the “dictionary key”.
- **AQLM** ([Egiazarian et al., 2024](https://arxiv.org/abs/2401.06118)): a group of 8–16 weights = a sum of vectors from learned books. Classic additive / product quantization from nearest-neighbor search. In practice ~**2–3 bits/weight**, at 2 bits — the best Pareto among PTQ. The index *is* the key into the codebook.
- **GPTVQ, QuIP#:** vector / lattice quantization; the book is fixed (E8 lattice) or learned.
- **SVD-LLM, LoRA, Tensor-Train:** key = low-rank factors. `W ≈ UV`. At inference you can **not expand** W: compute `(xU)V`. That is the ideal “vector zip” for a GPU.
- **Kilobyte Models** ([Dhayalkar, 2026](https://arxiv.org/html/2608.00860)): model = **seed + quantized latent**. A random basis is restored from the seed, weights = f(z, seed). On MNIST a 4-bit net in **2 KB**. This is literally the “right key”.

Low **intrinsic dimension** of the landscape is not fantasy. Li et al. trained nets in a random subspace and compressed >100× on small tasks. Aghajanyan et al.: a RoBERTa fine-tune task fits in **~200** random coordinates (90% of MRPC quality). VeRA, NOLA, GaLore — same family: frozen random basis + short key.

### Why not “as much as you want”

Because three requirements cannot be satisfied at once.

1. **It must be the same model (bit-identical).** Then vector lossless almost does not win: unique 8-dimensional groups in 70B are almost as many as groups. A collision-free codebook ≥ the weights themselves. Huffman 30% is the ceiling precisely for *exact* recovery with a fast decoder.
2. **Quality can be given up a little.** Then “as much as you want” becomes a rate–distortion knob. 4 bits — almost the same 70B. 2 bits (AQLM) — already the edge, but alive. 1.58 bits (BitNet) — a different model, that is how they *train*. Distillation 70B→8B is also a key: a small net. Further, quality drops not because “we cannot compress numbers” but because **the weights hold information about the world**, and a leaky projection throws it away.
3. **You have to decompress on the fly, peak VRAM = key + one layer.** Here the key can be tiny, but **transcription costs FLOPs**. Kilobyte scheme: θ = tanh(W₀z + b₀), W₀ is not stored, it is seeded in blocks from the seed. To materialize a layer of ~0.9B parameters from a latent of dimension d, you need ~0.9B·d multiplies. At d = 16k that is ~10¹³ operations **per layer**, per token, over 80 layers — hours, not chat. Biology can wait for polymerase; a user cannot. The way out: do not generate W, compute in factored form `(xU)V` — and you hit rank again, not key magic.

Fine-tune (MRPC, LoRA) has tiny intrinsic dimension because the base already contains the world. **Pretraining** *is* compressing text into weights. Compressing another three orders of magnitude losslessly = compressing the compressor better than its content allows. Overparameterization gives slack (lottery ticket, low rank, 2-bit VQ), not infinity.

### A key from the model itself, in pieces, then stitch

The training text has nothing to do with this. The “dataset + seed” recipe was only a Kolmogorov thought-limit. A working key is taken **from an already computed W**, piece by piece, without retraining on the internet.

The scheme is exactly what you described:

```
W  →  slice into pieces (layer / channel group / 8–128 neighboring weights)
   →  fit a key K_i to the piece:  D(K_i, indices) ≈ W_i
   →  store K_i + indices
inference: take the needed piece by the connection map, decompress, compute, drop
```

The connection map already exists: it is just tensor indices `(layer, row, column)`. GEMM already stitches tiles in one place. You do not need to invent a separate “seam graph”.

This is already done, and the key is indeed learned from the model:

- **AQLM** — key = the layer’s codebooks, piece = 8–16 neighboring weights. Books are fit to this matrix (k-means / AQ), indices are written into each group. At inference the group is assembled as a sum of book vectors.
- **XFP** — a library of 32 books per layer; a group of 128 weights picks which key from the library fits it.
- **AAAC** — two books per layer, a group takes one of the two.
- **Neural Weight Compression** ([Ryu et al., 2025](https://arxiv.org/html/2510.11234v3)) — a neural codec: analysis/synthesis trained **on a dataset of weight pieces**. The paper’s question is literally: *can weight compression be learned from data, where the data are the weights themselves.*
- **NeRN** — a small MLP on the coordinate `(layer, filter, channel)` restores a piece. Key = the weights of that MLP, learned from the pretrained net.

Splitting a layer into groups is fine. There is no “will stitch / will not stitch” difference: a linear layer is the sum of the groups’ contributions. If the numbers in the groups are close to the original, `Wx` will converge.

Another matter — **splitting itself does not reduce the sum of bits.**

Full size = Σ|K_i| + Σ|indices_i|. Slice into a thousand pieces with a thousand independent keys — you can **inflate** storage: each book takes space. The win appears when the key is **shared** and the pieces only pick from it (one ribosome, many codons). So in prod it is not “its own autoencoder per piece” but **a library per layer + short indices**. Too fine a cut → book bloat. Too coarse (the whole layer as one SVD) → a high-rank layer, the key is large again.

Peak VRAM really does drop to “key + current piece” if you decompress-compute-drop. Inside a dense layer, on one token you still need **all** groups: `y = Wx` touches every column. Slicing gives a pipeline and a fused tile (AQLM/ZipServ), not the right to skip half the layer. You can skip a piece only if the corresponding input is zero (sparsity of x) or the piece is someone else’s MoE expert.

Also: a key learned only as `D(K) ≈ W` stores digits. It is often better to train so that `D(K) x ≈ Wx` on a short calibration. That is not “attach Wikipedia as a key”, but a few thousand sentences to see which errors in W spoil the output. Seams then converge on **function**, not bits.

Bottom line: several keys on pieces + an index map is the right architecture, the same as DNA (ribosome shared, genes local). The difference is not in the seams, but in that the sum of keys and indices still has to carry the pieces’ information. A shared key per layer does that; a unique key per byte does not.

---

---

## 7. How real this is — an honesty scale

**Already works.** Store a BF16 model compressed in VRAM and unwind a layer/tile before GEMM. Savings ~30%, quality = original. Code: DFloat11, ZipServ, NeuZip, Unweight.

**Works, but it is not zip.** 4-bit quantization gives ~4× and computes without a full unpack to BF16. For local 70B this is the main path.

**Works as gene expression.** MoE and hot/cold neurons: most parameters can be left untouched on a given token. Memory is still almost full if everything is resident; the win is compute and the ability to keep cold in RAM.

**A bad idea.** A whole-file zip stream of the entire model + sequential unzip + re-compress of the finished stretch. No random access, LZ77 does not compress, recompress burns SMs for nothing, an intermediate BF16 layer buffer eats the win.

**Hard ceiling of lossless with a fixed fast decoder.** For an ordinary BF16 LLM and scalar Huffman — about a third. With a key (book, basis, latent) lossless still hits the pigeonhole principle; “as much as you want” starts only as lossy. See §6.

**Hardware is going the same way.** Blackwell Decompression Engine — NVIDIA’s bet on “HBM stores compressed”. For now the free block is tuned for analytical codecs. The next logical industry step is fused decompress-GEMM and, possibly, ANS in fixed logic.

---

## 8. If you are assembling a system, not a paper

Do not start with zip. A stack from ready parts:

1. **Storage format.** Huffman/ANS on the BF16 exponent (DFloat11 / ZipNN) or TCA-TBE (ZipServ) if you need speed. For already quantized weights — tiled ANS on top of GGUF/AWQ.
2. **Execution.** Fused decompress-GEMM. Unpack into shared memory / registers, not into a new tensor in HBM. Prefetch the next layer.
3. **Architecture.** If the goal is a locally large model, take MoE and put routed experts in RAM (llama.cpp tensor override, MoE-Infinity, FloE), attention and the shared expert — in VRAM.
4. **Loss by consent.** Q4/AWQ if 30% is not enough. BitNet — only if you are ready to train from scratch.
5. **Do not touch KV with the same zip.** Separate budget.

Rough arithmetic “will it fit”:

- 8B BF16 16 GB → DFloat11 10.9 GB → comfortable on 16–24 GB together with a short KV.
- 70B BF16 141 GB → DFloat11 95 GB → still does not fit even in 80 GB. Q4 ~43 GB → a 48 GB card or 2×24. On a single 24 GB — only Q4 + offload. Streaming 95 GB over PCIe 4.0 (~32 GB/s) theoretically gives fractions of a token per second.
- 405B BF16 812 GB → DFloat11 551 GB → one 8×80 GB node (640 GB) is tight, ~90 GB left for KV, not for 128k context. The official FP8 is more practical.
- 671B MoE (DeepSeek-V3): compute 37B, store hundreds of gigabytes; without expert quantization and their offload a consumer GPU is useless.

---

## 9. Bottom line in one sentence

The idea is right in the mechanics (resident compressed copy + local unpack as the “polymerase”-layer walks) and already exists under the names DFloat11/ZipServ/NeuZip. Scalar zip yields ~30%. A **key** (codebook, basis, latent) beats that ceiling, but not “as much as you want”: either you lose accuracy, or you pay FLOPs to generate weights, or the key itself grows to a model. The live vector path is AQLM/GPTVQ, SVD without materializing W, and distillation, not AES and not gzip.

---

## Sources

**Lossless weights and entropy**

- [Zhang et al. DFloat11. arXiv:2504.11651](https://arxiv.org/abs/2504.11651) · [GitHub](https://github.com/LeanModels/DFloat11)
- [Hershcovitch et al. ZipNN. arXiv:2411.05239](https://arxiv.org/abs/2411.05239) · [GitHub](https://github.com/zipnn/zipnn) · [IBM Research](https://research.ibm.com/blog/Zip-NN-AI-compression)
- [Fan et al. ZipServ. arXiv:2603.17435](https://arxiv.org/abs/2603.17435)
- [Hao, Cao, Mou. NeuZip. arXiv:2410.20650](https://arxiv.org/abs/2410.20650) · [GitHub](https://github.com/BorealisAI/neuzip)
- [Tan et al. Approaching Shannon Bound with Lossless LLM Weight Compression](https://arxiv.org/html/2606.15789v1)
- [Nikulin. Unweight (Cloudflare, 2026)](https://research.cloudflare.com/papers/unweight-2026.pdf)
- [Finn. In search of wasted bits (2026)](https://fergusfinn.com/blog/weight-entropy/)
- [LEXI: lossless exponent coding](https://arxiv.org/html/2603.15589)

**GPU codecs and hardware**

- [NVIDIA nvCOMP](https://developer.nvidia.com/nvcomp)
- [Cut checkpoint costs with nvCOMP](https://developer.nvidia.com/blog/cut-checkpoint-costs-with-about-30-lines-of-python-and-nvidia-nvcomp/)
- [Blackwell Decompression Engine](https://developer.nvidia.com/blog/speeding-up-data-decompression-with-nvcomp-and-the-nvidia-blackwell-decompression-engine/)

**Offload, mmap, sparse**

- [FlexGen (ICML 2023)](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf) · [GitHub](https://github.com/FMInference/FlexGen/)
- [PowerInfer. arXiv:2312.12456](https://arxiv.org/abs/2312.12456)
- [PowerInfer-2. arXiv:2406.06282](https://arxiv.org/abs/2406.06282)
- [Alizadeh et al. LLM in a Flash. arXiv:2312.11514](https://arxiv.org/abs/2312.11514)
- [DeltaZip. arXiv:2312.05215](https://arxiv.org/abs/2312.05215)
- [MoE-Infinity. arXiv:2401.14361](https://arxiv.org/abs/2401.14361)
- [FloE: on-the-fly MoE. arXiv:2505.05950](https://arxiv.org/abs/2505.05950)
- [llama.cpp MoE offload guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)

**Quantization, vectors, and the key-decoder**

- [BitNet b1.58 2B4T](https://arxiv.org/html/2504.12285v2)
- [bitnet.cpp](https://aka.ms/bitnet)
- [Egiazarian et al. AQLM. arXiv:2401.06118](https://arxiv.org/abs/2401.06118)
- [GPTVQ. arXiv:2402.15319](https://arxiv.org/abs/2402.15319)
- [Han et al. Deep Compression. arXiv:1510.00149](https://arxiv.org/abs/1510.00149)
- [Wang et al. SVD-LLM. arXiv:2403.07378](https://arxiv.org/abs/2403.07378)
- [Li et al. Intrinsic dimension. arXiv:1804.08838](https://arxiv.org/abs/1804.08838)
- [Aghajanyan et al. Intrinsic dimensionality of LM fine-tuning](https://aclanthology.org/2021.acl-long.568/)
- [Ryu et al. Neural Weight Compression. arXiv:2510.11234](https://arxiv.org/html/2510.11234v3)
- [Ashkenazi et al. NeRN. arXiv:2212.13554](https://arxiv.org/abs/2212.13554)
- [XFP adaptive codebooks](https://arxiv.org/html/2605.14844v1)

**Chromatin (for the analogy)**

- [Kujirai et al. Nucleosome transition during Pol II passage. Science 2019](https://www.science.org/doi/10.1126/science.aau9904)
- [Filipovski et al. Nucleosome retention during elongation. Science 2022](https://www.science.org/doi/10.1126/science.abo3851)
- [Chromatine transcription elongation, review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11649447/)

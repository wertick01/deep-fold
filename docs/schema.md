# Schema for an RTX 3080 12 GB

Hardware: Ampere, 70 SMs, **912 GB/s**, 12 GB, 3rd-generation Tensor Cores. No Hopper WGMMA and no Blackwell unpacker. There is `cp.async` — we can prefetch the next chunk while the current one is computing.

10 tok/s is **100 ms per token**. An 8B has 32 layers → **~3 ms per layer**. A 32B has ~64 layers → **~1.6 ms per layer**. That is 3–5× more relaxed than “20 tok/s on a 70B”. The schema is viable.

## What we do not promise

There will be no 70B on 12 GB. Even at 2 bits that is ~18 GB of weights alone. On this card the space ceiling is **14B at 4 bits comfortably** (leftover ~3.5 GB) or **32B at 2 bits** (leftover ~3.1 GB, INT4 embeddings, no BF16 layer). Exact MB: [vram-3080.md](vram-3080.md).

## One pipeline, two quality stages

One runtime. Only the chunk format changes.

```
disk (safetensors)
    → offline, layer by layer (a full 32B BF16 will not fit in 12 GB)
    → file: layer codebook + chunk indices

load: everything compressed into VRAM
    + KV and a thin scratch
    + tile ring — 3 stages in block smem, not a layer in HBM

token:
  for each layer:
    for each tile:
      while tile i is computing, cp.async is already pulling i+1
      index → codebook / 4-bit nibble → numbers in registers
      Tensor Core MMA
      do not write the scratch to GDDR and do not compress it back
```

**Stage A (first).** 4 bits, group 32–128, a scale per group. This is the familiar GPTQ/AWQ world. Target: Llama 8B and Qwen 14B, **≥10 tok/s** on a 3080. 8B Q4 already runs that way from llama.cpp (~80–110 tok/s) — that is not a win, it is a **kernel caliber**: our kernel must not be much worse.

**Stage B (what the schema is for).** A codebook per layer: 256 or 65536 vectors of length 8. Each chunk of 8 weights = one or two indices into the codebook. ~2–2.5 bits. Target: **Qwen/Llama 32B on 12 GB**, ≥10 tok/s, quality does not fall apart in chat.

Huffman/zip is **not taken** in v1: on Ampere, without fusing into MMA, it loses, and writing Huffman-MMA from scratch takes longer than a codebook. Huffman enters the budget only as an option on 8B, if we do not keep a BF16 layer in HBM.

## Chunks and “seams”

A chunk is an **Ampere MMA tile**: 8 or 16 weights wide (like the codebook), 16/32/64 rows tall (like `mma.m16n8k16`).

Chunk address: `(layer, matrix, row_tile, col_group)`. The seam = that address. A separate link graph is not needed.

One key per matrix (Q, K, V, O, gate, up, down). All chunks of the matrix poke it with indices. This is the layer’s ribosome, not an archive per byte.

## Memory on 12 GB = 12288 MB

Exact budget from HF shapes: [vram-3080.md](vram-3080.md). Here — working rows. CUDA+decode+reserve = **880 MB**. There is no layer scratch in HBM: smem holds only 2 compressed tiles + 1 BF16.

| Model × codec | Weights | Books | Leftover | 8k FP16 / Q8 | max ctx FP16 |
|---|---:|---:|---:|---|---:|
| 8B INT4 4.5 bit | 4308 | 0 | 7100 | yes / yes | 57k |
| 8B 2-bit 2×8 | 2228 | 4 | 9176 | yes / yes | 73k |
| 8B Huffman ~11 bit | 10534 | 0 | 874 | **no** / yes | 7.0k |
| 14B INT4 4.5 bit | 7924 | 0 | 3484 | yes / yes | 18.6k |
| 14B 2-bit 2×8 | 3987 | 7 | 7414 | yes / yes | 39.5k |
| 14B Huffman ~11 bit | 19373 | 0 | −7964 | no | no |
| 32B INT4 4.5 bit | 17577 | 0 | −6169 | no | no |
| **32B 2-bit 2×8 + emb INT4** | **8277** | **12** | **3118** | **yes 1070 / yes 2030** | **12.5k** |
| 32B Huffman ~11 bit | 42968 | 0 | −31560 | no | no |

32B 2-bit **fits** (INT4 embeddings, no BF16 layer). PCIe is not needed. Huffman only makes sense on 8B and only without decompressing a layer into GDDR.

A tile in SRAM: 64×64 BF16 = 8 KB; 2 compressed INT4 ≈ 4.5 KB. Ampere block limit ≈ 99 KB. Codebook 2×8 = 8 KB (smem). Codebook 1×16 = 1 MB/matrix — in HBM.

## Loop on a token

1. The compressed tile is already in GDDR.
2. `cp.async` puts it into shared memory.
3. Threads fetch indices, look up the codebook (the layer codebook in smem; for 256×8 BF16 that is **4 KB**).
4. Tensor Core computes.
5. Registers forget. The compressed tile in GDDR stays where it was.

## How to check on a 3080

1. One linear kernel vs PyTorch BF16 and vs llama.cpp Q4 on 8B. If we are much slower than Q4 — the kernel is bad, the format is not at fault.
2. Assemble a full 8B, measure tok/s and VRAM.
3. 14B 4-bit, context 2k, target ≥10 tok/s.
4. 2-bit codebook on 8B, watch quality damage.
5. If it is alive — 32B on 12 GB.

Offline compression of 32B on the same 3080: only **layer by layer from disk**. Do not load full BF16.

## Modules

1. Chunk and codebook format — `.chr` in [compressor.md](compressor.md), MMA tile in [kernel-ampere.md](kernel-ampere.md)
2. Ampere kernel — [kernel-ampere.md](kernel-ampere.md): `mma.sync.m16n8k16` BF16, tile 128×256, codebook only **256×8**
3. 12 GB scheduler and KV — [vram-3080.md](vram-3080.md)
4. Offline compressor — [compressor.md](compressor.md): tensor stream, NF4 and k-means 2×8
5. Token-loop assembly (prefill / decode) — [token-loop.md](token-loop.md)

Seam: everyone speaks `(layer, matrix, row_tile, col_group)` and nobody proposes compressing a layer back.

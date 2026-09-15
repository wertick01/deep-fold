# From the end: how to finish decompressing a chunk and “put it away”

Imagine the token is already in flight. The card must decompress the needed chunk, compute, put it away. If this loop does not fit in a fraction of a millisecond — the schema is dead, however pretty it looks on paper.

## How much time there is

A good chat is about **20 tokens per second**. That is **50 milliseconds per token**.

Llama 70B has about **80 layers**. So for a whole layer (decompress + compute + put away) there is about **0.6 ms**. An 8B has 32 layers; an RTX 4090 delivers ~50 tok/s in FP16 — per layer that is the same **half millisecond**. This is not a “wish”; it is how the bus works: an 8B layer in FP16 weighs ~0.5 GB, a 4090 reads memory at ~1000 GB/s, the read is exactly ~0.5 ms.

A 70B layer in “raw” BF16 is **1.71 GB** (8B — 436 MB). Just reading a 70B layer from a 4090: **~1.7 ms**. Already over the 20 tok/s budget. That is why a large model in full BF16 does not fly on a single consumer card.

From here every method is measured with this ruler: **did it fit in ~0.5 ms or not**.

## “Put it back” is not compress again

Weights do **not** change at inference. The compressed copy already sits in memory. The decompressed chunk is a scratch.

Put it away = drop the buffer or forget the registers. That is zero nanoseconds.

Compressing a layer again is separate work, and it is **slower** than decompressing. Huffman on GPU (Unweight) packs an 8B layer in about **20 ms**; LZ4 on an A100 — about **5 ms** for the same 436 MB. Layer budget is 0.3–0.6 ms. Even “fast” packing is tens of times late. On a 70B token that becomes seconds. For chat that is an immediate no.

If we “return” to disk or RAM over PCIe (~32 GB/s), a 70B layer travels **~50 ms**. The token will think for seconds. Also no.

The correct loop from the end: decompress into scratch → compute → drop the scratch. Even better: there is no scratch in big memory at all; numbers appear in registers and go straight into the multiply.

## Options, worst to best

### 1. Classic zip of the whole layer

Decompress the layer into video memory, then compute, then compress back.

Decompression + writing a large buffer + reading again for the multiply — you drive the bus **twice**, plus the zip itself. On fat numbers LZ4 barely compresses, and it eats time. ZipServ measured this directly: standalone decompression takes **1.5–3.5× longer** than the multiply itself. On a 4090 such schemes give about **0.17–0.28** of ordinary BF16 speed, i.e. 3–6× slower.

For chat: no.

### 2. Haul a chunk from RAM or SSD

Like “heterochromatin”. Over PCIe a 70B layer is those same **55 ms**. Even if decompression is free, the trip kills the budget a hundredfold.

For an offline batch of tokens (FlexGen) it is still tolerable. For dialogue — no.

### 3. Cipher (AES)

Decrypt a chunk with a key. On a GPU AES can be fast (tens–hundreds of GB/s in synthetic). The layer decrypts, say, in milliseconds.

But the file is the same size. In video memory it did not shrink. We pay time and do not win space.

For our task: miss.

### 4. Assemble a layer from a tiny key (neural generator)

Pretty idea: a 2 KB key, we draw weights from coordinates.

To draw a layer of ~a billion numbers, the key must be multiplied by a huge random basis. With a wide key that is trillions of operations per layer — **seconds and hours**, not milliseconds. For chat, no. It only makes sense if we **do not assemble** the weights, but compute immediately in short form (see item 7).

### 5. Decompress the whole layer into video memory, but with a “numbers” codec (Huffman / DFloat11)

The compressed model sits on the GPU. Before the layer we decompress it into a ~1.8 GB scratch, compute, drop the scratch. We write nothing back.

This is already a live scheme, and Llama 405B is stuffed onto an eight-card node this way. But the scratch is large, we stroke the bus an extra time. On a 4090 on average **~3.5× slower** than ordinary BF16. For a server “as long as it fits” — yes. For snappy chat — weak.

### 6. Decompress not a layer but a small tile, straight into registers (ZipServ)

The card never holds a decompressed layer. It takes a compressed piece, in registers assembles a 64×64 square (that is **8 KB**, fits comfortably in the block’s fast memory), multiplies, forgets.

“Put it back” = end of the kernel tick. Zero cost.

On a 4090 that multiply is **faster** than ordinary BF16 (on average **~1.3×**, peak ~1.7×): fewer bytes travel the bus, assembly hides behind memory wait. One large 8B layer: **0.195 ms** vs **0.215 ms** for ordinary BF16 on an A100. 70B slims from 132 GB to **94 GB**. Quality = original.

This is the best answer if we cannot spoil the numbers.

### 7. Do not decompress to BF16 at all: compute in 4 bits or from a codebook

Here “decryption” is looking up a scale or a vector from a codebook, for 4–16 numbers, not a gigabyte.

- **INT4 / Marlin.** 4× less on the bus. On decode almost **4× faster** than BF16. 70B weighs ~40 GB. Quality a bit lower; for chat usually OK.
- **AQLM (codebook on a group of 8 weights).** ~2 bits. In chat **1.3–3× faster** than FP16, depending on the codebook format. The codebook is that same model key; the chunk picks an index and is “stitched” in place.
- **BitNet.** Even denser, but the model has to be trained that way, not compress a ready Llama in an evening.

There is nothing to “put back”: there was no large BF16 chunk.

For local hardware this is the working path.

## Clear numbers on one card (RTX 4090)

Per one 70B layer (~1.8 GB in BF16). Chat target — **under ~0.6 ms**.

| How we do it | What happens to the chunk | Time order | Put back | Chat? |
|---|---|---|---|---|
| Compress with zip again after the layer | full archiver cycle | ~20 ms compression alone | expensive | no |
| Bring the layer over PCIe | 1.8 GB trip | ~55 ms | even worse | no |
| Zip → big memory → compute | decompress the whole layer | decompression 1.5–3.5× longer than the multiply, overall 3–6× slower than BF16 | drop the scratch | almost no |
| Huffman layer on GPU (DFloat11) | layer scratch, then drop | ~3.5× slower than BF16 | drop | fits, but slow |
| Tile in registers (ZipServ) | 8 KB, forgot | **faster** than BF16 ~1.3× | forget registers | yes, lossless |
| INT4, compute immediately | we do not assemble BF16 | **~4× faster** | nothing to put away | yes |
| AQLM codebook ~2 bit | index → 8 numbers | **1.3–3× faster** than FP16 | nothing to put away | yes, a bit coarser |

For comparison, the same 4090 in real life: Llama 8B Q4 ≈ **130 tok/s**, 8B FP16 ≈ **54 tok/s**, 70B Q4 on two 4090s ≈ **19 tok/s**. Speedup is almost “fewer bytes on the bus”. Whoever hauls less — talks faster.

## What follows if we go from the end

First we pick a **motion that makes the half-millisecond**. That is not a zip file and not “compress back”. It is either a tile in registers, or compute in 4/2 bits.

Then we look at what to fill video memory with:

- cannot lie about a single bit → ZipServ / DFloat11, we live at ~70% of size;
- can lie a little → INT4, and that is both smaller and faster;
- really tight → codebook on groups (AQLM) or skip some experts.

Chunking is needed so we decompress a **tile**, not a model. It stitches itself by address `(layer, row, column)`. A shared key per layer, short indices on groups — ribosome and codons. A unique archiver per chunk and reverse packing after work — that is like rebinding the book after every sentence.

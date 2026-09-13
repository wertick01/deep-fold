# `gpu/host` — the PyTorch seat for CHR0 weights

Two codecs, one rule: **no `[out, in]` BF16 weight ever exists.** A module holds
the file's bytes and hands them to a kernel; `weight` is a 0-numel property.

| codec | module | loader | kernel | acceptance |
|---|---|---|---|---|
| `nf4` (g=64) | `CompressedLinear` | `gpu.chr0.materialize_nf4` | `gpu/nf4` | `verify.py` |
| `vq` (2×8) | `CompressedVqLinear` | `vq_blobs.materialize_vq` | `gpu/vq` | `verify_vq.py` |

`gpu/chr0` materializes `nf4` only (gpu-abi.md §8) and is not ours to extend, so
the VQ blobs are read in `vq_blobs.py` with the same discipline `blobs.py` uses
for BF16: one positioned read per matrix, one host buffer that dies right after
the host-to-device copy, views into a single device allocation.

## VQ 2×8 in three lines

```python
import sys; sys.path.insert(0, r"C:\dev\deep-fold")
from gpu.host import load_chr_vq

layer = load_chr_vq(r"C:\dev\models\qwen25-3b.vq2.chr", "model.layers.0.mlp.gate_proj")
y = layer(x)          # x is [..., K] bf16 with prod(leading dims) == 1
```

Shapes come from the header. `book` is FP16 `[2, 256, 8]` (8192 bytes, one book
per matrix), `index` is uint8 `[M, K_pad/8, 2]` with `K_pad = 8*ceil(K/8)`, and
the reconstruct is `g = C1[i1] + C2[i2]` in float32 (vq.md §7.2). The kernel is
decode-only: `N != 1` raises `NotImplementedError` instead of quietly computing
one column.

`materialize_vq` / `VqMatrix` are the lower level if you want the bytes without
a module; `reconstruct_vq` is the CPU oracle (it materializes `M × K` float32,
so it is for checks, never for the token path).

## Acceptance

```bat
python gpu\host\verify.py
python gpu\host\verify_vq.py
```

`verify_vq.py` needs no model: it runs the golden 1×16 layout of vq.md §6.2/§7.3,
synthetic matrices at the real Qwen2.5-3B projection shapes, and a tiny
`chr.exe --codec vq` fixture it builds in `%TEMP%`. It picks up
`C:\dev\models\qwen25-3b.vq2.chr` automatically when that file exists (`--chr`
to point elsewhere); until then the real-file check reports `SKIP` and does not
block the verdict.

If the VQ extension has not been built yet, run the command from a VS x64 prompt
or after

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

so the JIT build in `gpu/vq/__init__.py` can find `cl.exe`.

## Producing `qwen25-3b.vq2.chr`

The Go codec is the only encoder (residual k-means is CPU-side and
single-threaded by spec, vq.md §3.4):

```bat
chr.exe compress --in C:\dev\models\Qwen2.5-3B-Instruct ^
                 --out C:\dev\models\qwen25-3b.vq2.chr ^
                 --codec vq --seed 0 --iters 20
```

Budget hours, not minutes: 20 Lloyd iterations × 2 books ≈ 42 assignment passes
over 3.09 G weights, one core, and `embed_tokens` alone is 311 M of them. The
whole model is buffered in RAM and the `.chr` appears only at the end, so a
partial file is not a thing. Do not run two of these at once, and do not
`git add` the output.

Optional cross-check of the encoder itself (weights vs reconstruct, CPU only):

```bat
chr.exe verify --orig C:\dev\models\Qwen2.5-3B-Instruct ^
               --chr C:\dev\models\qwen25-3b.vq2.chr ^
               --fail-rmse 0.50 --fail-maxabs 8.0 --json > C:\dev\models\qwen25-3b.vq2.verify.json
```

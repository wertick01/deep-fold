# GPU ABI: `.chr` → device tensors (wave 2)

What the `gpu/chr0/` loader hands the kernel, and under which conditions. Contested points are decided by [stitch-gpu.md](stitch-gpu.md); container — [chr0.md](chr0.md); codec — [nf4.md](nf4.md). No LUT here, no dequant, no `.cu`.

---

## 1. `ChrMatrix` — frozen

```python
@dataclass(frozen=True)
class ChrMatrix:
    name: str              # CHR0 name, without ".weight"
    M: int                 # logical n_out = shape[0]
    K: int                 # logical n_in  = shape[1]
    K_pad: int             # 64 * ceil(K / 64)
    packed: Tensor         # uint8,   [M, K_pad // 2],  row-major, contiguous
    scale: Tensor          # float16, [M, K_pad // 64], row-major, contiguous
```

Do not change field names or order: agent 2 reads them via `chr_nf4_dev_t`, agent 3 — directly.

`packed` and `scale` live on the same `device`. Both are views into **one** per-matrix device allocation (§4), so `packed.data_ptr()` and `scale.data_ptr()` stay valid for the lifetime of the `ChrMatrix` itself.

Mapping to the C header `gpu/include/chr_gpu.h` (layout is frozen; the loader does not change it):

| C field | Python |
|---|---|
| `int32_t M` | `m.M` |
| `int32_t K` | `m.K` |
| `int32_t K_pad` | `m.K_pad` |
| `const uint8_t *packed` | `m.packed.data_ptr()` |
| `const uint16_t *scale` | `m.scale.data_ptr()` — binary16 **bits**, not an “fp16 object” |

`scale` is `torch.float16`, i.e. the same 16 bits as in the file. The kernel reads them as `uint16` / `__half` and expands to float32 **before** multiplying by `LUT[nib]` ([nf4.md](nf4.md) §3).

---

## 2. API

```python
load_header(path)                                   -> Header
iter_linears(header)                                -> Iterator[str]   # codec == "nf4"
materialize_nf4(path, name, device="cuda", *, header=None) -> ChrMatrix
```

- `load_header` opens the file **read-only** (`open(path, "rb")`), parses and **fully validates** the header (§5), then closes the file. It does not read blobs.
- `iter_linears` does not touch disk: names with `codec == "nf4"` in header key order (lexicographic UTF-8 from the writer). `embed_tokens` and `lm_head` appear here if they are nf4 in the file — filtering by `kind` is the host’s job, not the loader’s.
- `materialize_nf4` — one matrix. `header=` lets an already-parsed header be reused when loading many matrices (parsing 65 KB of JSON × 300 is the only reason for this parameter).
- The original `safetensors` is **never** opened by the loader.

`Header` carries `magic/version/arch/hidden_size/intermediate_size/num_layers/vocab_size/tile`, `file_size`, and `tensors: Mapping[str, TensorInfo]`. `TensorInfo` is `kind`, `codec`, `shape`, `layer`, `group_size`, blobs `{key: (start, end)}`, and derived `M / K / K_pad / n_groups` for rank-2 nf4.

---

## 3. Byte formulas (loader rejects a mismatch)

`K_pad = 64 * ceil(K / 64)`, `n_groups = K_pad / 64`. From [stitch-gpu.md](stitch-gpu.md):

| Blob | dtype | shape | `end − start` |
|---|---|---|---|
| nf4 `data` | uint8 | `[M, K_pad/2]` | `M * K_pad / 2` |
| nf4 `scale` | FP16 LE | `[M, n_groups]` | `M * n_groups * 2` |
| bf16 `data` | BF16 LE | `shape` | `2 * Π shape` |
| vq `codebook` | FP16 LE | `[2, 256, 8]` | `8192` |
| vq `index` | uint8 | `[M, K_pad_vq/8, 2]` | `M * (K_pad_vq/8) * 2`, `K_pad_vq = 8*ceil(K/8)` |

Offsets `[start, end)` are **from the start of the file**, `start % 64 == 0`.

Size checks run in `load_header` for **every** tensor in the file, including vq/int4 slots: their sizes are validated, and `materialize_nf4` on them raises (§5, `CodecError`). Wave 2.0 materializes `nf4` only.

Nibbles (low = `W[r, 2c]`), LUT, codebooks — **not the loader’s concern**. It neither reads nor rearranges a single bit: `packed[r, c]` is exactly the file byte at offset `data.start + r*(K_pad/2) + c`.

---

## 4. How bytes get into VRAM

**Chosen strategy — one: per-matrix arena, one HtoD per matrix.**

```
seek(data.start)
readinto(bytearray(data.nbytes + scale.nbytes))   # one pread: data ‖ scale
arena = torch.empty(total, uint8, device)          # one device allocation
arena.copy_(torch.frombuffer(host))                # one cudaMemcpy HtoD
packed = arena[:len_data].view(M, K_pad//2)
scale  = arena[len_data:].view(torch.float16).view(M, n_groups)
host buffer is released
```

The CHR0 writer always lays `data` then `scale` back-to-back (field order §2.3 chr0.md), so the fused path is the common case: `scale.start == data.end` for any matrix where `M * K_pad / 2` is a multiple of 64, i.e. for all real ones. If there is pad or a foreign blob between the blobs, the loader silently takes the fallback: two `pread` + two HtoD **into the same** single device buffer. The result shape is identical.

The disk is read **before** the first device touch: a short file fails without allocating VRAM.

Consequences this locks:

- **Host-RAM peak = blobs of one matrix** (for 3B `gate_proj` — 11.42 MiB), not the file and not the model. No mmap of the whole `.chr`, no `read()` of the entire file, no `dict[str, bytes]`.
- **One device allocation per matrix**, not one per blob: 434 tensors of 3B → ≤ 434 allocations from the torch caching allocator, not “200 tiny `cudaMalloc`s” per blob.
- While loading the full model, host and device copies do **not** coexist: the host buffer for matrix `i` is dead before matrix `i+1` is read.
- `torch.empty(M, K, dtype=bfloat16)` is never called — not “for a check”, not as an intermediate. There is no scratch `W` in HBM ([stitch-gpu.md](stitch-gpu.md)).

Measured on a 3080 (`model.layers.0.mlp.gate_proj`, `M=11008, K=2048`): blobs `11 272 192 + 704 512` B = **11.42 MiB**, `torch.cuda.memory_allocated` +**12.00 MiB** (the caching allocator rounds a large block to 2 MiB), `nvidia-smi memory.used` +**12 MiB** above the already-created CUDA context. A BF16 matrix would be 43 MiB, F32 — 86 MiB. A delta above ~14 MiB is a bug.

Writes to `.chr` are impossible by construction: the only call is `open(path, "rb")`.

---

## 5. Errors

All are subclasses of `Chr0Error`. All header checks run in `load_header`, i.e. **before** any `to(device)` and any `cudaMemcpy`.

| Class | When |
|---|---|
| `TruncatedError` | file < 8 bytes; `8+N > size`; `end > size` on any blob; `readinto` returned fewer bytes |
| `HeaderError` | `N ∈ {0,1}`; `N > 100_000_000`; first JSON byte ≠ `{`; invalid UTF-8/JSON; not an object; tail is not whitespace; JSON number with a fractional part; extra/missing key; `magic ≠ "CHR0"`; `version ≠ 1`; `tile ≠ {row:64,col_group:8}`; empty `tensors`; duplicate name; name with a control byte; `layer` does not match `layers.<n>`; mixed codecs in the Q-set |
| `AlignmentError` | `start % 64 ≠ 0` |
| `SizeMismatchError` | `end ≤ start`; `end − start` ≠ the §3 formula |
| `OverlapError` | any two `[start,end)` ranges in the file intersect |
| `CodecError` | `codec` outside `{bf16,nf4,int4,vq}`; `materialize_nf4` on a non-`nf4` tensor; key set not matching the codec |
| `GroupSizeError` | `group_size ≠ 64` for nf4 (and `≠ 8` for vq) |
| `TensorNotFoundError` | name not in `tensors` |

The reader does **not** repair the header: it does not search for `{` further in the file, does not trim `N`, does not ignore unknown keys. Extra file tail past `Align64(max_end)` is not an error ([chr0.md](chr0.md) §1.5).

---

## 6. Who transposes what

**The host transposes, not the loader and not the kernel.**

- In the file, `W` is row-major `[M, K]`, axis 1 = Linear input. The loader hands these bytes through as-is.
- The kernel expects `x` in BF16 row-major **`[K, N]`** and writes `y` BF16 `[M, N]` ([stitch-gpu.md](stitch-gpu.md)).
- HuggingFace gives activations `[..., N, K]`. Bringing them to `[K, N]` is `CompressedLinear`’s job (agent 3): `x.reshape(-1, K).t().contiguous()` before the call and the inverse reshape of `y` after. The loader does not take part and never sees `x`.
- No fragment-major and no permute of `packed` on disk or at load: permute is in the kernel’s registers.

---

## 7. Check

```
python gpu/chr0/test_chr0.py
```

The happy-path fixture is built in place: `write_safetensors` (64×128 and 2×65 F32 + a norm) → `chr.exe compress --codec nf4` → `materialize_nf4(device="cpu")` → compare `packed`/`scale` with `file[start:end]` byte for byte. Malformed cases (`end > filesize`, size ≠ formula, overlap, `start % 64 ≠ 0`, `group_size=32`, `codec=vq`, broken JSON) are assembled by hand: the writer does not produce them.

The “before `cuda`” check: in truncated-file tests `torch.empty` is replaced with a throwing stub; `materialize_nf4(..., "cuda")` must fail with `TruncatedError`, not on the stub.

Memory discipline (separate run, 60 matrices of 3B in a row, 346.9 MiB of blobs): `device_delta = 360.5 MiB`, `host RSS delta = 0.0 MiB`, largest host buffer at once — 11.42 MiB.

---

## 8. What the loader does not do

It does not dequant, does not compute `W_hat`, does not open safetensors, does not write `.chr`, does not mmap for write, does not cache blobs across calls, cannot materialize `int4`/`vq` (parses the slot and rejects), does not pick a `stream`, and does not allocate anything on a token.

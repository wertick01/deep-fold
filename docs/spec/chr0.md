# CHR0 v1: `.chr` container and safetensors stream (CPU, Go without CGO)

Interface: [compressor.md](../compressor.md) §6 (file layout) and §9 step 2 (writer). The NF4/VQ codecs are **not** defined here — only slots for their blobs. Inference and GPU are out of this slice.

Second-wave packages: `internal/chr0` (this format) and `internal/safetensors` (HF reading). Implementation: standard library, `encoding/binary`, `os.File`, **no CGO and no mmap**.

---

## Forks (summary — only these choices from here on)

| Question | Choice | Why |
|---|---|---|
| How to write JSON vs blobs | Two passes over **metadata**, JSON is written **immediately at its final size**, then blobs. No blob sidecar, no “hole” for JSON | Blob sizes depend only on `shape`+codec, not on weight values. JSON is known before reading weights; a slot cannot “fail to fit” |
| mmap vs `ReadAt` | Only `os.File.ReadAt` / sequential `Write` | Windows/WSL; mmap is not needed for CPU-verify |
| Non-weights (`inv_freq`, rotary, …) | **Skip** (not in `.chr`) | RoPE is reconstructed from `theta`; CLI test `TestVerifySkipInvFreq`. Do not quantize. |
| Multiple codecs in one file | Forbidden for Linear/embed/lm_head | Requirements: one codec per file; the mixed nf4+vq example in compressor.md §6.1 is **not** the v1 norm |
| `tensors` key order | Lexicographic UTF-8 sort | `encoding/json` serializes `map[string]T` that way; do not write a custom marshaler |
| **Blob** order on disk | Shard/tensor traversal order (not JSON keys) | One open shard, one tensor; offsets in JSON bind the name to the bytes |
| `int4` | Present in the read schema; the v1 writer **does not** create it | Requirements |
| Checksum in the file | None | `verify` computes metrics itself |

---

## 1. Byte by byte

### 1.1. File map

All multi-byte integers in the binary part are **little-endian** (`binary.LittleEndian`). JSON is UTF-8 text without BOM.

```
offset 0                uint64le   header_nbytes = N     # JSON length in bytes, excluding these 8 bytes and excluding pad
offset 8                N bytes    header_json           # exactly one JSON object
offset 8+N              P bytes    pad                   # zeros, P = Align64(8+N) − (8+N), 0 ≤ P ≤ 63
offset Align64(8+N)     …           payloads              # each blob starts on a 64-byte boundary
```

There is no end-of-file magic. File length after a successful write = `Align64(max_end)`, where `max_end` is the maximum of all `end` values in the header.

`Align64(x)` for a non-negative integer `x`:

```
Align64(x) = x,                 if x % 64 == 0
           = x + (64 − x % 64), otherwise
```

On two’s complement / Go `uint64`: `(x + 63) &^ 63`. Count **from the start of the file**, not “from the start of the section”.

### 1.2. Offset 0: `header_nbytes`

Read exactly 8 bytes. Interpretation: `uint64` LE.

| Condition | Reader action |
|---|---|
| File shorter than 8 bytes (including empty) | Error `truncated` |
| `N == 0` or `N == 1` | Error `header_nbytes` (no valid object `{…}`) |
| `N > 100_000_000` | Error `header_too_large` (same ceiling as safetensors) |
| `8+N > size(file)` | Error `header_nbytes` (it “lies”: JSON extends past EOF) |

Writer: `N = len(compact_json)`, no trailing `\n`.

### 1.3. Offset 8: JSON

Read exactly `N` bytes. This is **not** a C string: a zero byte inside is part of the JSON, and it is **forbidden** (see below).

Reader:

1. If `N≥1` and the byte at offset 8 ≠ `0x7B` (`{`) — error `json_invalid` (including BOM `EF BB BF`).
2. `json.Decoder` / `Unmarshal` into an object. Use `UseNumber()` (or parse offsets via `json.Number`): offsets must not pass as `float64` with a fractional part.
3. After one top-level value inside these `N` bytes, only ASCII whitespace is allowed: `0x20`, `0x09`, `0x0A`, `0x0D`. Any other tail (second object, `NUL`, garbage) — error `json_invalid`.
4. Invalid UTF-8, a truncated surrogate, non-JSON — error `json_invalid`. Do not “fix” it and do not search for `{` further in the file.
5. If the JSON is valid but is an array / number / `null` — error `json_invalid` (an object is required).

If `header_nbytes` is **larger** than the real JSON and the tail of the `N` bytes contains pad or the start of a blob: either a parse error or a non-whitespace tail → still an error. If `N` is **smaller** than the real object — parse error. The reader **does not** scan the file looking for `}`.

`DisallowUnknownFields` on the root and on each tensor: an unknown key is an error. In v1 there is no “ignore for forward-compat”.

### 1.4. Pad after JSON

`P = Align64(8+N) − (8+N)`.

The writer writes `P` bytes of `0x00`.

The reader **does not** check that the pad is zeros, and does not check the bytes of “holes” between blobs. It only skips them by the formula. (Checking pad contents does not give weight integrity.)

If `8+N` is already a multiple of 64, `P=0`, and the first blob starts immediately after the JSON.

Numeric example (the same file — §2.4): `N=516` = `0x204`, bytes `0..7`:

```
04 02 00 00 00 00 00 00
```

`8+516=524`, `Align64(524)=576`, `P=52` zeros. The first blob is at offset **576**.

Another blob-pad example (not from the toy model): blob `[start,end)=[576,582)` (6 bytes BF16 × 3), next start = `Align64(582)=640`. The writer writes 6 data bytes and 58 zeros.

### 1.5. Blobs

Each payload:

- `start % 64 == 0`
- `end > start`
- `end − start` = exact content size **without** pad
- bytes `[start, end)` are the contents; `[end, Align64(end))` is pad (zeros from the writer)

Offsets `[start, end)` are counted **from the start of the file** (not from the end of the header). This is the main difference from safetensors.

Overlap of any two `[start,end)` is an error (reader and writer).

Writer: after the last blob, if `max_end % 64 ≠ 0`, append zeros up to `Align64(max_end)`. Reader: `size(file) ≥ max_end` is enough; extra bytes in the tail are **ignored** (not an error), a shortfall is error `truncated`.

### 1.6. Endianness of blob contents

| Contents | Byte order |
|---|---|
| `uint8` (NF4 nibbles, VQ indices) | byte as-is |
| FP16 / BF16 | 16-bit code unit, little-endian |
| JSON numbers | text, not LE/BE |

The implementation platform is x86_64 / Windows WSL: native = LE, but still write via `binary.LittleEndian`, not via host-casting a padded struct.

---

## 2. Header JSON schema

### 2.1. Root (all fields required, no other keys)

| Key | JSON type | Constraints |
|---|---|---|
| `magic` | string | Exactly `"CHR0"` (case is fixed) |
| `version` | integer | Exactly `1`. Anything else — error `unsupported version` |
| `arch` | string | Non-empty, UTF-8. Source — §5.6. Not an enum |
| `hidden_size` | integer | `≥ 1` |
| `intermediate_size` | integer | `≥ 0` (`0` is allowed if the file has no MLP) |
| `num_layers` | integer | `≥ 0` |
| `vocab_size` | integer | `≥ 0` (`0` is allowed without embed/lm_head) |
| `tile` | object | Exactly `{"row":64,"col_group":8}`. Other numbers in v1 — error |
| `tensors` | object | Not empty. Key — CHR0 tensor name |

Root key order on disk (writer, struct tags):

`magic`, `version`, `arch`, `hidden_size`, `intermediate_size`, `num_layers`, `vocab_size`, `tile`, `tensors`.

`tile`: only `row` and `col_group`, both integer, both required. This is a slice constant (the Ampere 64×8 tile from the schema); CPU-verify does not interpret it, but the reader **checks** the values.

Root numbers are JSON integers without `.` and `e`. Do not write a `+` sign. The reader rejects `"4096"` (string) and `4096.0`.

### 2.2. Tensor name (key in `tensors`)

- As in HuggingFace, **without** the `.weight` suffix.
- Bias: full name **with** `.bias`, e.g. `model.layers.0.self_attn.q_proj.bias`.
- Case and dots as in the source. Name comparison is byte-wise, case-sensitive.
- Name length 1…1024 UTF-8 bytes. `U+0000` and ASCII control `0x00–0x1F` are forbidden.
- Keys of the `tensors` object on disk: **sorted by increasing UTF-8 bytes** (like `encoding/json` for a map).

Two different HF names that, after the §5.5 rule, yield one CHR0 name — error on write.

### 2.3. Tensor object

There are four keys “common to all codecs”; the rest are per-codec. Unknown keys are forbidden. Extra keys of another codec are forbidden (`bf16` has no `group_size`).

**Always:**

| Key | Type | Rule |
|---|---|---|
| `kind` | string | Exactly one of: `q`, `k`, `v`, `o`, `qkv`, `gate`, `up`, `down`, `embed`, `lm_head`, `norm`, `other` |
| `codec` | string | `bf16` \| `nf4` \| `int4` \| `vq` |
| `shape` | array of integer | Length 1 or 2; each element `≥ 1`; rank 3+ is forbidden |
| `layer` | integer | See below |

`layer`:

- If the name contains a `layers.<n>` segment as a whole dotted component (the first such from left to right, `n` unsigned decimal) — `layer` is **required** and equals that `n` (`0 ≤ layer < num_layers`).
- Otherwise (`model.norm`, `model.embed_tokens`, `lm_head`, …) the `layer` key is **absent**.
- `layer: 0` is **written explicitly**. The writer **does not** use `json:",omitempty"` on `int` (otherwise layer zero would disappear).

`shape` is **logical** (without `n_in` padding). For a matrix `[n_out, n_in]` axis 0 = rows = `n_out`, axis 1 = `n_in` (like `nn.Linear.weight` in PyTorch). For a vector `[n]`.

A `[start, end)` pair is a JSON array of exactly two integers, `0 ≤ start < end ≤ 2^63-1`, `start % 64 == 0`, `end − start` equals the blob-size formula (§7, §10).

Key order inside a tensor (those present):  
`layer`, `kind`, `codec`, `shape`, `group_size`, `n_codebooks`, `codebook_bits`, `data`, `scale`, `zero`, `codebook`, `index`.

#### codec `bf16`

`data` is required. Forbidden: `group_size`, `scale`, `zero`, `codebook`, `index`, `n_codebooks`, `codebook_bits`.

`end − start = 2 * Π shape[i]` (each element is 2 bytes of BF16).

Allowed for any `kind`. For `kind=norm` and names with suffix `.bias` and for `kind=other` — the **only** allowed codec.

#### codec `nf4`

Required: `group_size` (exactly `64`), `data`, `scale`.  
Forbidden: `zero`, `codebook`, `index`, `n_codebooks`, `codebook_bits`.  
`shape` only rank 2.

#### codec `vq`

Required: `group_size` (exactly `8`), `n_codebooks` (exactly `2`), `codebook_bits` (exactly `8`), `codebook`, `index`.  
Forbidden: `data`, `scale`, `zero`.  
`shape` only rank 2.

#### codec `int4` (read-only for foreign files)

Like `nf4`, plus optional `zero` of the same length as `scale`. The v1 writer **never** sets `codec=int4`. The container reader accepts the object and returns the raw blobs; there is no int4 decoder in this slice.

#### Invariant “one codec per file”

Let `Q` be the set of `codec` values of tensors with `kind ∈ {q,k,v,o,qkv,gate,up,down,embed,lm_head}` whose name **does not** end with `.bias`.

- If `Q` is non-empty, then `Q` is a singleton `{nf4}` or `{vq}` or `{int4}`. A mix or `bf16` in `Q` is a reader error.
- Norms, bias, `other` are not in `Q` and are always `bf16`.

### 2.4. Example: toy “model” of 3 tensors, `--codec nf4`

Safetensors source (tensor order in the ST header = blob order):

| HF name | shape | dtype |
|---|---|---|
| `model.layers.0.self_attn.q_proj.weight` | `[64,64]` | BF16 |
| `model.layers.0.mlp.down_proj.weight` | `[32,128]` | BF16 |
| `model.norm.weight` | `[64]` | BF16 |

Alongside `config.json`: `model_type=toy`, `hidden_size=64`, `intermediate_size=128`, `num_hidden_layers=1`, `vocab_size=0`.

Blob sizes (`n_in` padding not needed: 64 and 128 are already multiples of 64):

| CHR0 name | blob | bytes |
|---|---|---|
| `model.layers.0.self_attn.q_proj` | `data` | `64 * (64/2) = 2048` |
| same | `scale` | `64 * (64/64) * 2 = 128` |
| `model.layers.0.mlp.down_proj` | `data` | `32 * (128/2) = 2048` |
| same | `scale` | `32 * (128/64) * 2 = 128` |
| `model.norm` | `data` | `64 * 2 = 128` |

Compact JSON (this is the **byte-level header reference**, `N=516`):

```
{"magic":"CHR0","version":1,"arch":"toy","hidden_size":64,"intermediate_size":128,"num_layers":1,"vocab_size":0,"tile":{"row":64,"col_group":8},"tensors":{"model.layers.0.mlp.down_proj":{"layer":0,"kind":"down","codec":"nf4","shape":[32,128],"group_size":64,"data":[2752,4800],"scale":[4800,4928]},"model.layers.0.self_attn.q_proj":{"layer":0,"kind":"q","codec":"nf4","shape":[64,64],"group_size":64,"data":[576,2624],"scale":[2624,2752]},"model.norm":{"kind":"norm","codec":"bf16","shape":[64],"data":[4928,5056]}}}
```

The same object, readable (offsets are the same; **do not** put this on disk with spaces — `N` and all `[start,end)` would change):

```json
{
  "magic": "CHR0",
  "version": 1,
  "arch": "toy",
  "hidden_size": 64,
  "intermediate_size": 128,
  "num_layers": 1,
  "vocab_size": 0,
  "tile": { "row": 64, "col_group": 8 },
  "tensors": {
    "model.layers.0.mlp.down_proj": {
      "layer": 0,
      "kind": "down",
      "codec": "nf4",
      "shape": [32, 128],
      "group_size": 64,
      "data": [2752, 4800],
      "scale": [4800, 4928]
    },
    "model.layers.0.self_attn.q_proj": {
      "layer": 0,
      "kind": "q",
      "codec": "nf4",
      "shape": [64, 64],
      "group_size": 64,
      "data": [576, 2624],
      "scale": [2624, 2752]
    },
    "model.norm": {
      "kind": "norm",
      "codec": "bf16",
      "shape": [64],
      "data": [4928, 5056]
    }
  }
}
```

File map:

| Region | `[start, end)` | contents |
|---|---|---|
| `header_nbytes` | `[0, 8)` | `04 02 00 00 00 00 00 00` |
| JSON | `[8, 524)` | 516 bytes of UTF-8 |
| JSON pad | `[524, 576)` | 52 × `00` |
| q `data` | `[576, 2624)` | 2048 × uint8 |
| q `scale` | `[2624, 2752)` | 128 × FP16 |
| down `data` | `[2752, 4800)` | 2048 × uint8 |
| down `scale` | `[4800, 4928)` | 128 × FP16 |
| norm `data` | `[4928, 5056)` | 128 × BF16 |
| EOF | `5056` | `5056 % 64 == 0` |

Keys in JSON: `down_proj` before `q_proj` (lexicographically). Blobs: `q_proj` first, because that is how it sits in safetensors. That is fine.

### 2.5. The same three tensors, `--codec vq` (fields, not a second file reference)

Blobs in traversal order:

| name | blob | bytes | `[start,end)` at `N=593` |
|---|---|---|---|
| q_proj | `codebook` | `2*256*8*2=8192` | `[640, 8832)` |
| q_proj | `index` | `64*(64/8)*2=1024` | `[8832, 9856)` |
| down_proj | `codebook` | 8192 | `[9856, 18048)` |
| down_proj | `index` | `32*(128/8)*2=1024` | `[18048, 19072)` |
| norm | `data` | 128 | `[19072, 19200)` |

`N=593`, `8+593=601`, `Align64(601)=640`. q_proj fragment:

```json
"model.layers.0.self_attn.q_proj": {
  "layer": 0,
  "kind": "q",
  "codec": "vq",
  "shape": [64, 64],
  "group_size": 8,
  "n_codebooks": 2,
  "codebook_bits": 8,
  "codebook": [640, 8832],
  "index": [8832, 9856]
}
```

The codebook is **per matrix**, not per layer: `q` and `down` have **different** `codebook` ranges.

---

## 3. Write algorithm

### 3.1. Why not a sidecar and not a “hole”

compressor.md §9: “header in memory, blobs append, rewrite JSON at the end”. That works only if you leave a hole ≥ the final JSON in advance. If after the blobs the JSON grew (different offset digits) and did not fit — the hole is useless.

**Choice:** all blob sizes are known from `shape`+codec **before** reading weights. The writer:

1. Pass A: only safetensors headers + `index.json` + `config.json` (kilobytes).
2. Stabilize `N` and offsets (loop below).
3. Write `uint64`+JSON+pad.
4. Pass B: one shard, one tensor, encode, write blobs at already-known offsets, release the source buffer.

There is no temporary file of raw blobs. Atomicity: write to `<out>.part` in the same directory, `Close`, then `os.Rename` to `<out>`. On error delete `.part`. (This is not a blob sidecar: `.part` *is* the output.)

If the JSON “grew” — that is step 2 **before** any weight; the loop recomputes offsets. There is no situation “blobs already written, JSON does not fit”. If the loop does not converge in 8 iterations — error `header_not_stable` (a stabilization bug, not “make the hole bigger”).

### 3.2. Pass A — tensor list

1. Build the shard plan (§5.4).
2. For each shard: open, parse **only** the ST header, close.
3. For each tensor of the shard that is in the plan:
   - dtype ∈ {`F32`,`F16`,`BF16`}, otherwise error;
   - CHR0 name = §5.5;
   - `kind`+`codec` = §6;
   - `layer` = §2.3;
   - remember `shape`, shard file, ST `data_offsets`, dtype.
4. Check uniqueness of CHR0 names. Empty list — error `no tensors`.
5. Compute `arch`, model sizes (§5.6).
6. For each tensor a list of blobs in **fixed field order**:
   - `bf16`: `data`;
   - `nf4`: `data`, `scale`;
   - `vq`: `codebook`, `index`.
   Lengths — §7 / §10.

### 3.3. Header stabilization

Offsets go into the JSON, `N` depends on the number of digits, blob starts depend on `N`. Loop:

```
N ← 0
repeat at most 8 times:
    pos ← Align64(8+N)
    for tensors in traversal order (pass B):
        for each blob of the tensor in field order:
            start ← pos
            end ← start + nbytes
            remember [start,end)
            pos ← Align64(end)
    json ← CompactJSON(root + tensors with these offsets)
    if len(json) == N: done
    N ← len(json)
    if N > 100_000_000: error header_too_large
else: error header_not_stable
```

`CompactJSON`: no spaces, no HTML-escape (`SetEscapeHTML(false)`), UTF-8, `tensors` keys sorted, root is a struct (not a map). `layer:0` is present.

In practice it converges in 2 iterations (as in the example: 515→516).

`file_size = Align64(max_end)` after stabilization. You can `Truncate(file_size)` on `.part` before writing blobs.

### 3.4. Pass B — weights

Invariant: **≤ 1 shard open, in RAM ≤ 1 source tensor + its encoded blobs + JSON**.

```
open .part, Truncate(file_size)
WriteAt(0, uint64le(N))
WriteAt(8, json)
WriteAt(8+N, P zeros)

current_shard ← none
for each tensor in traversal order:
    if shard ≠ current_shard:
        close previous shard
        open new (os.Open)
        current_shard ← this one
    raw ← ReadAt of the shard at ST data_offsets        # native F32/F16/BF16
    blobs ← Encode(codec, kind, raw, dtype, shape)  # see below
    release raw (do not put it in a “all tensors” slice)
    for each blob:
        if len(bytes) ≠ end-start: error
        WriteAt(start, bytes)
        append zeros up to Align64(end), if this is not a hole before the next (with dense packing the next WriteAt will overwrite the pad — still write the pad explicitly)
close shard
Close(.part), Rename
```

Sequential `Write` instead of `WriteAt` is allowed if you write strictly in increasing `start` order and do not skip pad. Traversal order = increasing `start` order (that is how we pack).

`Encode`:

- `bf16`: convert to BF16 LE row-major (§5.7), return one `data` blob.
- `nf4` / `vq`: convert to `float32` row-major of the logical `shape` (without pad — column padding is done by the codec) and call the codec package. The container **does not** know the LUT or k-means. It receives a `map`/struct of blobs and checks lengths.

The writer **does not** hold `[][]byte` of all blobs. After `WriteAt` the blob is released.

Do not open shards in parallel. Do not prefetch the neighboring tensor.

### 3.5. Error mid pass B

Delete `.part`. Do not leave a half-written `<out>`. Do not try to “rewrite JSON at the start” after partial blobs.

---

## 4. Read algorithm

### 4.1. Open

```
f ← os.Open(path)                    # not mmap, not syscall.Mmap
N ← binary.LittleEndian.Uint64(ReadAt(0, 8))
check N as in §1.2
js ← ReadAt(8, N)
parse and validate JSON (§1.3, §2)
for each blob of each tensor:
    check start/end, formula sizes, start%64==0, end≤size(f)
    overlaps — error
keep the header in memory; keep the file descriptor open
```

Do not use `mmap` even on Linux: one code path for Windows/WSL. Sequential reading of all tensors is still `ReadAt` (or `io.NewSectionReader(f, start, len)`).

Do not read all blobs on `Open`.

### 4.2. Fetch one tensor by name

The name is a **CHR0 name** (without `.weight`). Exact match. There is no `.weight` alias: that is CLI work, not the container’s.

```
info ← tensors[name]   # no key → error not_found
result.kind, codec, shape, layer ← from info
for each blob key that the codec has:
    buf ← make([]byte, end-start)
    ReadAt(buf, start)               # exactly end-start bytes, otherwise truncated
    result.blobs[key] ← buf
return result, without decoding nf4/vq
```

Peak RAM when reading one tensor ≈ the sum of its blobs, not the whole `.chr`.

Re-reading the same name: `ReadAt` again (no cache of all tensors). Header cache — yes.

`Close` closes `*os.File`.

### 4.3. Reading “as an iterator” (compress-inverse verify)

Verify must not load the whole model: iterate names (order — sorted keys or traversal order — does not matter for metrics) and `Get` one at a time. After processing a tensor, release the blobs.

---

## 5. Safetensors parser and shards

### 5.1. There is no magic

A safetensors file **does not** start with ASCII magic. Map:

```
offset 0        uint64le  st_header_nbytes = Ns     # 2 ≤ Ns ≤ 100_000_000
offset 8        Ns bytes  JSON object
offset 8+Ns     …         raw tensors
```

There is **no** pad-to-64 in safetensors. Data starts immediately after the JSON. `data_offsets` in ST are `[start, end)` **from the start of the data section**, i.e. file address = `8 + Ns + start`. Do not confuse with CHR0.

`Ns` checks are the same as §1.2 (except the minimum: `Ns≥2`). First JSON byte = `{`. `__metadata__` is not a tensor, skip it (not copied into CHR0).

The ST reader is also `ReadAt`, one open file.

### 5.2. Tensor object in ST

Required keys: `dtype` (string), `shape` (array of integer ≥0), `data_offsets` (two integers, `start ≤ end`).

`dtype` (this slice):

| `dtype` | bytes/element | action |
|---|---|---|
| `F32` | 4 | accept |
| `F16` | 2 | accept |
| `BF16` | 2 | accept |
| other (`I64`, `U8`, `F8_*`, `BOOL`, `F64`, …) | — | error `unsupported dtype` |

Case is exact: `BF16`, not `bf16`.

`Π shape * elem_size == data_offsets[1]−data_offsets[0]`, otherwise error. Rank 0 (empty `shape`) and any axis `0` — error (there are no empty weights). Rank ≥ 3 — error (not in CHR0).

The tensor’s file range must not extend past EOF.

ST-header key order: parse with `json.Decoder` by tokens (not a `map` on the first pass), so traversal order inside a shard is the order in the file. Duplicate tensor keys in one ST — error.

### 5.3. Single file vs directory

Compressor input is the `--in` path.

| `--in` | Action |
|---|---|
| Ordinary file `*.safetensors` | One shard; all tensors of the file (except `__metadata__`). Do **not** read a sibling `index.json` |
| Directory, `model.safetensors.index.json` present | Shards only from `weight_map`. Ignore other `*.safetensors` in the directory |
| Directory, no index, exactly one `*.safetensors` | That file |
| Directory, no index, `model.safetensors` present (even if there are more files) | That file |
| Otherwise | Error: need an index or an unambiguous `.safetensors` |

Do not recurse into subdirectories. `adapter_model.safetensors` by itself is not selected by the rules above.

### 5.4. `model.safetensors.index.json`

Object with required `weight_map` (object: full **HF name** → relative path to the shard). The `metadata` key (often `total_size`) is **ignored**: it is not a checksum and not a limit.

`weight_map` value:

- only a relative path, no leading `/`, after `path.Clean` does not start with `..`;
- file = `filepath.Join(dir, value)` must exist;
- `/` separators normalized via `filepath`.

Traversal algorithm (LRU=1 file):

1. Unique shards = the set of `weight_map` values after `Clean`.
2. **Sort paths UTF-8** (determinism; HF names are `model-00001-of-00004` and lexicographic order = number).
3. For shard `S`: open; for tensors **in ST-header order** whose `weight_map[hfName]` points at `S` — process; close.
4. A tensor in `weight_map` that is not in the named shard — error (after walking all, a missing list).
5. A tensor in a shard that is not in `weight_map` — **skip** (not an error).
6. One HF name in two shards is impossible through a single map; if somehow one key… JSON object last-wins; the writer parses by tokens and a duplicate key in `weight_map` is an error.

Empty `weight_map` — error.

Do not open the next shard until the previous is closed. Do not keep a mmap of “all shards”.

### 5.5. HF name → CHR0 name

```
if the name ends with ".weight" (last dotted component is exactly "weight"):
    chrName = name without that suffix   # exactly once
else:
    chrName = name as-is             # .bias, inv_freq, …
```

Do not touch `foo.weight_scale`. `foo.weight.weight` → `foo.weight` (one strip).

### 5.6. `config.json` and emitting root fields

If `--in` is a directory (or the file’s directory when `--in` is a file) contains `config.json` — read it (ordinary JSON, not ST). Fields:

| CHR0 | Source, first found |
|---|---|
| `arch` | `model_type` (string). Missing — `"unknown"` |
| `hidden_size` | `hidden_size` / `n_embd` / `d_model` |
| `intermediate_size` | `intermediate_size` / `ffn_dim` / `n_inner` |
| `num_layers` | `num_hidden_layers` / `n_layer` / `num_layers` |
| `vocab_size` | `vocab_size` |

If a field is missing — **derive from tensors** (after classification):

- `hidden_size`: length of `model.norm` (kind=norm, name ends with `.norm` without `layers`); else `shape[1]` of `embed`; else `shape[1]` of the first `q`; else error.
- `intermediate_size`: `shape[1]` of the first `down`; else `shape[0]` of the first `up`; else `0`.
- `num_layers`: `max(layer)+1` among tensors with `layer`; if there are none — `0`.
- `vocab_size`: `shape[0]` of `embed` or `lm_head`; else `0`.
- `arch`: `"unknown"`.

A conflict “config says X, tensors clearly do not fit” is **not** checked in v1 (the 64×64 q and 32×128 down toy is valid). `num_layers` from config, if present, is taken from config; then each `layer` must be `< num_layers`.

### 5.7. What to do with non-weights

**Skip** tensors whose canonical name contains `inv_freq`, `rotary_emb`, or ends with `.sin` / `.cos`. They are not in `.chr`; `verify` counts `skipped`.

Other unknown 1D — `kind=other`, `codec=bf16`. Unknown rank-2 — compress with the chosen `--codec`.

Why skip RoPE: it is not GEMM weights; `inv_freq` is reconstructed from the config. Interface with [integrity-cli.md](integrity-cli.md) §3.1.

### 5.8. Dtype conversion before the CHR0 blob

ST yields **raw** LE bytes + `dtype`. Conversion is in the CHR0 writer, not in the ST parser.

| Target | Source BF16 | F16 | F32 |
|---|---|---|---|
| blob `codec=bf16` | memcpy | F16→F32 (IEEE) → BF16 RNE | F32→BF16 RNE |
| nf4/vq codec input | BF16→F32 (shift+0) | F16→F32 | memcpy as `[]float32` via `math.Float32frombits`, LE |

F32→BF16 round-to-nearest-even: like truncating the low 16 bits of F32 with round-to-even (add `0x7FFF + (bit16)` to the low bits, then `>>16`; NaN: keep exponent=255 and a non-zero mantissa in the high 7 bits; Inf/sign — as in IEEE). For tests of norms from a BF16 source the conversion is **not** called: bytes 1-to-1.

NaN/Inf in a `bf16` copy: bits as they came out. For nf4/vq the container hands F32 to the codec; NaN policy is the codec spec (encode error), not CHR0.

Row-major C-contiguous, like safetensors. Ampere fragment-major is **not** this slice.

### 5.9. Parser memory

`Read(name)` reads only one tensor. After return the shard may stay open (pass B), but the writer does not keep the previous tensor’s buffer. Do not make a `map[string][]byte` of the whole model.

---

## 6. Classification: name → `kind` + `codec`

Input: CHR0 name (already without `.weight`).

### 6.1. `kind` — by the **last** dotted component

Let `base = chrName` without the `.bias` suffix, if present (for kind; **in JSON the name with `.bias` remains**). `last` = substring after the last `.`, or all of `base` if there are no dots.

First match in the table top to bottom. Comparison is exact, case-sensitive.

| `last` | `kind` | HF examples |
|---|---|---|
| `q_proj` | `q` | `model.layers.0.self_attn.q_proj.weight` |
| `k_proj` | `k` | `…k_proj.weight` |
| `v_proj` | `v` | `…v_proj.weight` |
| `o_proj` | `o` | `…o_proj.weight` |
| `wo` | `o` | InternLM2 `attention.wo.weight` |
| `wqkv` | `qkv` | InternLM2 fused QKV `attention.wqkv.weight` |
| `gate_proj` | `gate` | `…mlp.gate_proj.weight` |
| `up_proj` | `up` | `…mlp.up_proj.weight` |
| `down_proj` | `down` | `…mlp.down_proj.weight` |
| `w1` | `gate` | Mixtral expert |
| `w3` | `up` | Mixtral expert |
| `w2` | `down` | Mixtral expert |
| `embed_tokens` | `embed` | `model.embed_tokens.weight` |
| `wte` | `embed` | GPT-2 |
| `lm_head` | `lm_head` | `lm_head.weight` |
| `norm`, `input_layernorm`, `post_attention_layernorm`, `post_feedforward_layernorm`, `pre_feedforward_layernorm`, `final_layernorm`, `final_norm`, `attention_norm`, `ffn_norm`, `rms_norm`, `q_norm`, `k_norm`, `ln_f`, `ln_1`, `ln_2`, `ln_3` | `norm` | RMSNorm / LayerNorm |
| otherwise, if `last` contains substring `layernorm` or `layer_norm` or `rmsnorm` | `norm` | fallback for `model.norm.weight` already covered by `last=norm` |
| everything else | `other` | `inv_freq`, `mlp.gate` (MoE router), fused `c_attn`, … |

`q_norm` is `norm`, not `q`: the table looks at `last`, `q_proj` ≠ `q_norm`.

`model.norm` → `last=norm` → `norm`.

### 6.2. `codec`

Let `file_codec` ∈ {`nf4`,`vq`} be the compressor flag (one for the whole `.chr`).

```
if chrName ends with ".bias":     codec = bf16
else if kind ∈ {norm, other}:         codec = bf16
else if kind ∈ {q,k,v,o,qkv,gate,up,down,embed,lm_head}:
                                         codec = file_codec
else:                                   impossible
```

Unknown Linear (`c_attn`, `dense_h_to_4h`) stay `other`+`bf16`, **not** silently quantized. Why: a foreign axis layout breaks group-wise K. Llama/Qwen `*_proj` are covered by the table; Mixtral `w1/w2/w3` too.

### 6.3. Summary table (Llama/Qwen slice + toy)

| HF name | CHR0 name | kind | codec with `--codec nf4` |
|---|---|---|---|
| `model.layers.0.self_attn.q_proj.weight` | `model.layers.0.self_attn.q_proj` | `q` | `nf4` |
| `model.layers.0.self_attn.q_proj.bias` | `model.layers.0.self_attn.q_proj.bias` | `q` | `bf16` |
| `model.layers.0.self_attn.k_proj.weight` | `…k_proj` | `k` | `nf4` |
| `model.layers.0.self_attn.v_proj.weight` | `…v_proj` | `v` | `nf4` |
| `model.layers.0.self_attn.o_proj.weight` | `…o_proj` | `o` | `nf4` |
| `model.layers.0.mlp.gate_proj.weight` | `…gate_proj` | `gate` | `nf4` |
| `model.layers.0.mlp.up_proj.weight` | `…up_proj` | `up` | `nf4` |
| `model.layers.0.mlp.down_proj.weight` | `…down_proj` | `down` | `nf4` |
| `model.layers.0.input_layernorm.weight` | `…input_layernorm` | `norm` | `bf16` |
| `model.layers.0.post_attention_layernorm.weight` | `…post_attention_layernorm` | `norm` | `bf16` |
| `model.layers.0.self_attn.q_norm.weight` | `…q_norm` | `norm` | `bf16` |
| `model.layers.0.self_attn.rotary_emb.inv_freq` | as in HF | `other` | `bf16` |
| `model.norm.weight` | `model.norm` | `norm` | `bf16` |
| `model.embed_tokens.weight` | `model.embed_tokens` | `embed` | `nf4` |
| `lm_head.weight` | `lm_head` | `lm_head` | `nf4` |
| `model.layers.0.mlp.gate.weight` (router) | `…mlp.gate` | `other` | `bf16` |
| `model.layers.0.block_sparse_moe.experts.3.w1.weight` | `…w1` | `gate` | `nf4` |

With `--codec vq` the codec column for Linear/embed/lm_head is `vq`, the rest unchanged.

### 6.4. `layer` from the name

Look in dotted segments for a pair `layers`, `<n>` (non-negative integer, ordinary `Atoi`). First such pair from the left:

- `model.layers.0.self_attn.q_proj` → `0`
- `model.layers.31.mlp.down_proj` → `31`

No `layers` segment — no `layer` field (`model.embed_tokens`, `lm_head`, `model.norm`). The `h` segment (GPT-2) is **not** recognized: then `kind` is often `other`, `layer` is absent. This slice is Llama/Qwen.

---

## 7. `n_in` padding and logical shape

### 7.1. Where what lives

In JSON there is **one** `shape` — logical, as in HF, without pad.

Padding exists **only** inside nf4/vq blobs:

```
n_in_pad(n_in, g) = ceil(n_in / g) * g = ((n_in + g − 1) / g) * g     # integers ≥1
```

- nf4: `g = 64` = `group_size`
- vq: `g = 8`
- bf16: no pad, blob length from the logical `shape`

A second shape is **not** written in the header. Do not write `shape_padded`. Do not store Ampere fragment-major. For CPU checking the blobs are **row-major**, as in [compressor.md](../compressor.md) §6.2.

Padding is **columns on the right** of each row (`k = n_in … n_in_pad-1`), source-matrix values there = `0` (the codec does this before pack). There are no extra rows: `n_out` is not padded in the v1 container (the 64-tile along M is the kernel’s concern, not the file’s).

Rank 1 (`norm`, bias, `inv_freq`): only `bf16`, the pad formula is not applied.

A quantized tensor must be rank 2, otherwise error (do not “pretend that n_out=1” silently — that could be done explicitly, but v1 rejects it).

### 7.2. Blob lengths (reader check)

Notation: `n_out = shape[0]`, `n_in = shape[1]`.

**nf4** (`n_in_pad = n_in_pad(n_in, 64)`):

| blob | dtype | logical in-memory form | `end-start` |
|---|---|---|---|
| `data` | uint8 | `[n_out, n_in_pad/2]` | `n_out * n_in_pad / 2` |
| `scale` | FP16 | `[n_out, n_in_pad/64]` | `n_out * (n_in_pad/64) * 2` |

`n_in_pad` is a multiple of 64 → `n_in_pad/2` is an integer.

**vq** (`n_in_pad = n_in_pad(n_in, 8)`, `M=2`, `k=256`, `g=8`):

| blob | dtype | form | `end-start` |
|---|---|---|---|
| `codebook` | FP16 | `[2, 256, 8]` | `2 * 256 * 8 * 2 = 8192` (always) |
| `index` | uint8 | `[n_out, n_in_pad/8, 2]` | `n_out * (n_in_pad/8) * 2` |

**bf16**: `2 * product(shape)`.

### 7.3. Row-major inside a blob (CPU)

Let index `0` be the slowest (row).

- nf4 `data`: byte with linear index `r * (n_in_pad/2) + c`. Nibble packing — nf4 spec / compressor §6.2: low 4 bits = weight `W[r, 2c]`, high = `W[r, 2c+1]`.
- nf4 `scale`: FP16 `scale[r, j]` at address `(r * (n_in_pad/64) + j) * 2`.
- vq `codebook`: `codebook[m, i, d]` at address `((m * 256 + i) * 8 + d) * 2`.
- vq `index`: `index[r, g, m]` byte at address `(r * (n_in_pad/8) + g) * 2 + m` (last axis `M`, two consecutive uint8 per group).
- bf16: element with multi-index as in NumPy C-order, 2 bytes per element.

The kernel’s `(row_tile, col_group)` tile is **not** rearranged in this slice. Verify reads the whole matrix row-major and decodes it whole.

### 7.4. Pad example

`shape=[3,70]`, nf4: `n_in_pad=128`, `data` = `3*64=192` bytes, `scale` = `3*2*2=12` bytes. In JSON `"shape":[3,70]`, not `[3,128]`.

vq: `n_in_pad=72`, `index` = `3*9*2=54` bytes, `codebook` = 8192.

---

## 8. Limits and degenerate cases

| Limit | Value | Why |
|---|---|---|
| `header_nbytes` (CHR0 and ST) | 2…100_000_000 | Like safetensors; JSON in RAM |
| Tensor name | 1…1024 bytes | garbage |
| Number of tensors | indirectly by the header; hard cap ≤ 1_000_000 | protection against a cycle |
| Rank | 1 or 2 | |
| Axis | ≤ 2^24−1 (16 777 215) | vocab/hidden 32B << this |
| Product of axes × 4 | ≤ 4 GiB (`1<<32`) | one tensor in RAM; 32B lm_head BF16 ≈ 1.56 GiB |
| Blobs per tensor | ≤ 3 | |
| `version` | 1 | |
| `tile` | 64 and 8 | |

Empty file / `< 8` bytes: error `truncated`.

`tensors: {}` or no tensors after the filter: error `no tensors`. The writer does not create such a `.chr`.

Duplicates:

- two identical keys in JSON `tensors` (token parser) — error;
- two HF names → one CHR0 name — error on write;
- overlapping `[start,end)` — error.

`header_nbytes` lies — §1.2–1.3, do not “trim” and do not read to EOF.

A file with valid JSON but `max_end > size(file)`: error even if a particular `Get` does not touch the truncated blob? **Yes, on `Open`**: all ranges are checked.

An offset not a multiple of 64: error.

`shape` and the actual blob size do not match: error on `Open` (do not wait for `Get`).

Tied `lm_head` missing from ST: not an error, it is simply not in `.chr`.

Scalar / 0-dim / 0-size: error.

---

## 9. Checksum

In v1 there is **no** checksum field, no trailer, no required `*.chr.sha256`.

Container integrity = a repeated `ReadAt` yields the same blob bytes that were written. Weight integrity = `verify` (RMSE/maxabs) per the integrity spec, not a CRC in CHR0.

The writer does not compute SHA-256. The reader does not look for a sidecar.

---

## 10. Interface with codecs

The container hands the codec a logical `W[n_out, n_in]` in `float32` (row-major) and accepts finished blobs. There are no quantization formulas here.

### 10.1. NF4 — required keys

`group_size` = 64 (JSON integer).  
Blobs: `data`, `scale`. No `zero`.

| Key | dtype | shape in terms of logical + pad | bytes |
|---|---|---|---|
| `data` | uint8 | `[n_out, n_in_pad/2]`, `n_in_pad=n_in_pad(n_in,64)` | `n_out * n_in_pad / 2` |
| `scale` | IEEE 754 binary16 (FP16), LE | `[n_out, n_in_pad/64]` | `n_out * (n_in_pad/64) * 2` |

Packing of byte `data[r,c]` (compressor.md §6.2): nibble is a level index 0..15, **not** offset INT4; low 4 bits = `W[r, 2c]`, high = `W[r, 2c+1]`. Scale is FP16, not BF16.

### 10.2. VQ — required keys

`group_size` = 8, `n_codebooks` = 2, `codebook_bits` = 8.

| Key | dtype | shape | bytes |
|---|---|---|---|
| `codebook` | FP16 LE | `[n_codebooks, 2^codebook_bits, group_size]` = `[2, 256, 8]` | 8192 |
| `index` | uint8 | `[n_out, n_in_pad/8, n_codebooks]` = `[n_out, n_in_pad/8, 2]` | `n_out * (n_in_pad/8) * 2` |

The `M` axis of `index` is last: for a group of 8 weights, two consecutive bytes `(i1, i2)`. Codebook for **this** matrix: do not share `codebook` between `q` and `down`.

### 10.3. BF16

One `data` blob: BF16 LE, logical `shape`, no pad.

### 10.4. What the container checks, what it does not do

Checks: key set, lengths, 64-align of starts, one file-codec.

Does not check: that nibbles ∈ 0..15 are meaningful, that the codebook is a k-means result, MSE. That is packages `internal/nf4` and `internal/vq`.

---

## Appendix A. Package contract (no code)

`internal/safetensors`:

- parse `--in` into a list `{HFName, DType, Shape, ShardPath, DataStart, DataEnd}`;
- `OpenShard` / `Close`; `ReadAt` of one tensor → `[]byte` native;
- not know CHR0.

`internal/chr0`:

- `Align64`, blob-length formulas, classifier §6, JSON stabilization, `Write`, `Open`, `Get`;
- not know the NF4 LUT or k-means;
- `encoding/binary`, `encoding/json` (`DisallowUnknownFields`, `SetEscapeHTML(false)`, `UseNumber`), `os.File`.

Tensor names in `.chr` never contain `.weight`. Bias and `inv_freq` contain their own suffixes.

---

## Appendix B. Errors the reader must catch (for tests)

1. File length 0, 3, 7.
2. `N` larger than EOF.
3. `N=516`, but the JSON bytes are not JSON / truncated / two objects.
4. BOM before `{`.
5. `magic: "chr0"` / `version: 2`.
6. `tile.row: 32`.
7. Empty `tensors`.
8. Duplicate name.
9. `q_proj` with `codec: bf16` while another `nf4` Linear is present — mix in `Q`.
10. `norm` with `codec: nf4`.
11. `data` length ≠ formula.
12. `start=575` (not a multiple of 64).
13. Overlapping ranges.
14. ST `dtype: I64`.
15. ST `data_offsets` from the start of the **file** instead of the data section — caught by a `numel*size` mismatch.
16. `index.json` with `../escape.safetensors`.
17. Two `.weight` and without, collapsing into one CHR0 name.
18. `layer` missing on `model.layers.0.mlp.down_proj`.
19. `layer:0` vanished from JSON because of omitempty.
20. `group_size: 32` on nf4.

# Codec A: group-wise NF4, group 64 (CPU)

Stage-A roundtrip integrity. Not the container, not VQ, not INT4-RTN, not GPTQ/AWQ, not double-quant of scales.

Canonical implementation — package `internal/nf4` on Go 1.22 without CGO. From this spec table-driven tests and golden hex are written **without Python and without bitsandbytes**.

Normative sources of the levels: the hardcoded table `get_4bit_type("nf4")` in [bitsandbytes/functional.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py) and the LUT `nf4_dequantization_lut` in `csrc/kernels.cu`. These are the same 16 literals QLoRA uses (Dettmers et al., 2023). Do **not** recompute `scipy.stats.norm.ppf` quantiles: the canon is the §1 literals, not a table from the PDF.

---

## 0. Frozen constants and non-goals

| Symbol | Value | Meaning |
|---|---:|---|
| `group_size` / `G_nf4` | **64** | Group along axis K / `n_in`. Another size in v1 is an error. |
| Number of levels | 16 | Nibble = LUT index, not offset INT4. |
| Scale | IEEE 754 **binary16** | Not BF16, not float32 on disk. |
| Double-quant of scales | none | No nested absmax field. |
| `zero` | none | This is not asymmetric INT4. |

**Matrix.** `W[n_out, n_in]`, row-major, last axis is the Linear input (`n_in` = K). Groups **do not** cross rows: row `r`, group `g` is 64 elements `W[r, 64g .. 64g+63]`.

**Compute.** Input F32 / F16 / BF16 is converted once to float32. Quantization and decode for `verify` are float32. Intermediate BF16 / FP16 of the product LUT × scale is **forbidden**.

**Non-goals (forbidden in this slice):** INT4 + zero-point, GPTQ, AWQ, `desc_act`, double-quant of scales, GPU kernel, Ampere fragment-major, VQ codebook.

**Nibble packing — not like CUDA bitsandbytes.** In bnb’s `kQuantizeBlockwise` the even weight goes into the **high** nibble (`q = idx[2c] << 4 | idx[2c+1]`). Here, as in `docs/compressor.md` §3 and §6.2: the **low** nibble = `W[r, 2c]`, high = `W[r, 2c+1]`. Copying packing from CUDA bnb is not allowed.

---

## 1. Table of 16 levels

Index 0 = −1.0 … index 15 = +1.0, strictly increasing. Index 7 is **plus-zero** (`0x00000000`), not `−0`.

The literals below are the exact strings from `get_4bit_type("nf4")`. In Go they are written as `float32(...)`. IEEE 754 binary32 after round-to-nearest-even from this decimal (coincides with the path “first binary64, then binary32”: for these 16 numbers double rounding does not diverge).

| idx | nibble | literal (canon) | binary32 bits | float32 value |
|---:|:---:|---|---|---|
| 0 | `0x0` | `-1.0` | `0xBF800000` | −1 |
| 1 | `0x1` | `-0.6961928009986877` | `0xBF3239B1` | −0.6961928009986877 |
| 2 | `0x2` | `-0.5250730514526367` | `0xBF066B30` | −0.5250730514526367 |
| 3 | `0x3` | `-0.39491748809814453` | `0xBECA32A0` | −0.39491748809814453 |
| 4 | `0x4` | `-0.28444138169288635` | `0xBE91A24D` | −0.28444138169288635 |
| 5 | `0x5` | `-0.18477343022823334` | `0xBE3D353F` | −0.18477343022823334 |
| 6 | `0x6` | `-0.09105003625154495` | `0xBDBA7871` | −0.09105003625154495 |
| 7 | `0x7` | `0.0` | `0x00000000` | 0 |
| 8 | `0x8` | `0.07958029955625534` | `0x3DA2FAFF` | 0.07958029955625534 |
| 9 | `0x9` | `0.16093020141124725` | `0x3E24CAE3` | 0.16093020141124725 |
| 10 | `0xA` | `0.24611230194568634` | `0x3E7C04DD` | 0.24611230194568634 |
| 11 | `0xB` | `0.33791524171829224` | `0x3EAD033A` | 0.33791524171829224 |
| 12 | `0xC` | `0.44070982933044434` | `0x3EE1A4B8` | 0.44070982933044434 |
| 13 | `0xD` | `0.5626170039176941` | `0x3F1007AB` | 0.5626170039176941 |
| 14 | `0xE` | `0.7229568362236023` | `0x3F3913B3` | 0.7229568362236023 |
| 15 | `0xF` | `1.0` | `0x3F800000` | 1 |

Test `TestNF4TableBits`: `math.Float32bits(NF4[i])` equals the bits column. A 1-ulp mismatch is a fail, even if the decimal writing “looks similar”. Do not use shortened blog values (`-0.6962`) and do not confuse the literal `0.44070982933044434` with the typo `0.44070983934402466` (those two decimals have **the same** binary32 `0x3EE1A4B8`; still copy the table canon into the source).

The asymmetry (7 negatives + zero + 8 positives) is a property of QLoRA `use_extra_value=True`, not a bug.

---

## 2. Encode of one vector of length 64

Group input: 64 values already in float32. Notation: `g[0..63]`, LUT `L[0..15]` from §1.

### 2.1. Dtype conversion (before slicing into groups)

| Input | How to get float32 |
|---|---|
| F32 | as-is |
| F16 | exact expansion IEEE binary16 → binary32 |
| BF16 | high 16 bits of float32, low 16 bits zeros |

From here all arithmetic is binary32, round-to-nearest-even, without `FMA` on the distance step (see §2.4).

### 2.2. Step order (normative)

1. **Reject non-numbers.** If any `g[i]` is not finite (`NaN` or `±Inf`) → encode error, not substitution with 0.
2. **Scale in float32.**  
   `s32 = max_i |g[i]|`.  
   If `s32 == 0` (the whole group is zeros, including `±0`) → `s32 = 1`.  
   `−0` in `abs` yields `+0`; `max` over zeros stays 0, the “→ 1” rule fires.
3. **Scale on disk.** `s16 = float32_to_fp16_RNE(s32)` (IEEE 754-2008 binary16, ties to even; subnormals allowed).  
   If `s16` is not **finite and strictly positive** → encode error (overflow to Inf, underflow to `+0` of a non-zero group). Do not write Inf/NaN into the `scale` blob.
4. **The scale we actually divide by** is the one decode will see:  
   `s = fp16_to_float32(s16)`.  
   Dividing by the original `s32` is **forbidden**: otherwise the first encode and encode(decode(·)) would diverge because of scale rounding.
5. **Normalization.** For each `i`: `u[i] = g[i] / s` (float32 division). Then clip to `[-1, 1]`:  
   `u = min(1, max(-1, u))`.  
   Clip is needed when `s16` rounded down and `|g_max| / s > 1`. After clip the extreme lands in `L[0]` or `L[15]`.
6. **Nearest level by L2 in float32.**  
   Index  
   \[
   \mathrm{idx}[i] = \arg\min_{j=0}^{15}\ (u[i] - L[j])^{2}.
   \]
   Distance is computed this way and only this way:

   ```
   d = u - L[j]          # float32 subtraction
   d2 = d * d           # float32 multiply, not FMA(d,d,0) as a separate requirement
   ```

   `abs(u - L[j])` on `[-1, 1]` yields the same argmin (the square is monotone). Normatively the square is fixed, as in the requirements “nearest by L2”.
7. **Ties.** If `d2` are equal for two (adjacent) levels — **smaller index**. That is the same choice as bitsandbytes `dQuantizeNF4`: comparison with the midpoint via strict `>`, equality goes to the lower branch. In 1D on a strictly increasing grid a tie happens only between neighbors when `u` is the exact midpoint.

Group result: 64 `uint8` indices in `0..15` and one scale `s16`.

### 2.3. Why not copy the CUDA midpoint tree as-is

`dQuantizeNF4` is a binary tree of 15 thresholds, each threshold intended as the midpoint of neighboring levels. That is equivalent to L2 **plus** tie → smaller index, if the thresholds are midpoints of **the same** `float32` LUT.

Four threshold literals in `kernels.cu` diverge by 1 ulp from `float32((L[j]+L[j+1])/2)` (`0.8614784181118011`, `0.5016634166240692`, `0.1202552504837513`, `-0.8480964004993439`). The norm is §2.2 against the §1 table, not the CUDA tree. An implementation **may** search the index by binary search over midpoints `M[j] = float32( (L[j] + L[j+1]) * 0.5 )` and the rule `u > M[j] → up`, else down. That must coincide with §2.2. Golden fixtures §8–§9 sit far from 1 ulp of a threshold, so the bnb tree and L2 coincide on them; unit tests still check indices against §2.2.

Midpoints of neighboring `L[j]` (for reference, not a second canon):

| between idx | midpoint float32 | bits |
|---:|---|---|
| 0–1 | −0.8480963706970215 | `0xBF591CD8` |
| 1–2 | −0.6106328964233398 | `0xBF1C5270` |
| 2–3 | −0.4599952697753906 | `0xBEEB8480` |
| 3–4 | −0.33967941999435425 | `0xBEADEA76` |
| 4–5 | −0.23460739850997925 | `0xBE703CEC` |
| 5–6 | −0.13791173696517944 | `0xBE0D38BC` |
| 6–7 | −0.045525018125772476 | `0xBD3A7871` |
| 7–8 | 0.03979014977812767 | `0x3D22FAFF` |
| 8–9 | 0.120255246758461 | `0x3DF64862` |
| 9–10 | 0.2035212516784668 | `0x3E5067E0` |
| 10–11 | 0.2920137643814087 | `0x3E9582D4` |
| 11–12 | 0.3893125355243683 | `0x3EC753F9` |
| 12–13 | 0.5016634464263916 | `0x3F006D04` |
| 13–14 | 0.6427869200706482 | `0x3F248DAF` |
| 14–15 | 0.8614784479141235 | `0x3F5C89DA` |

### 2.4. Contractions and precision

On `[-1, 1]` 16 subtractions are safe. Forbidden to replace `(u-L[j])^2` with a float64 comparison “for precision” in one place and float32 in another: indices must be as with float32. Compiler FMA on `d*d` does not shift argmin of the golden fixtures here; for portability compute two operations, as in §2.2.

---

## 3. Decode

For each logical weight with index `nib ∈ 0..15` and group scale `s16`:

```
s  = fp16_to_float32(s16)     # exact expansion, not “via BF16”
w  = L[nib] * s               # both factors float32, product float32
```

Order is mandatory: first the scale to float32, then multiply by the level. Do not cast `L[nib]` to FP16. Do not cast the product to FP16/BF16 on the `verify` / `chr decode` path (output is F32, see the CLI interface). The Ampere kernel later narrows to BF16 for MMA itself; that is not this slice.

`L[nib]` is taken from §1, not from a recomputation of quantiles.

Padding columns after `n_in` are **not** returned (§6).

---

## 4. Matrix pseudocode, complexity, peak memory

`n_in_padded = 64 * ceil(n_in / 64)`. `n_groups = n_in_padded / 64`. If `n_out < 1` or `n_in < 1` → encode error (an empty weight tensor is not compressed in v1).

```
EncodeNF4(W[n_out, n_in] → float32):
  data  = uint8[n_out][n_in_padded/2]
  scale = fp16[n_out][n_groups]
  for r = 0 .. n_out-1:                  # can stream: one row in flight
      for g = 0 .. n_groups-1:
          for k = 0 .. 63:
              c = 64*g + k
              grp[k] = (c < n_in) ? W[r, c] : 0
          idx[0..63], s16 = EncodeGroup64(grp)   # §2
          scale[r, g] = s16
          for k = 0 .. 31:
              data[r, 32*g + k] = (idx[2k+1] << 4) | idx[2k]
  return data, scale

DecodeNF4(data, scale, n_out, n_in):
  W_hat = float32[n_out][n_in]
  for r, for g:
      s = fp16_to_float32(scale[r, g])
      for k = 0 .. 63:
          c = 64*g + k
          if c >= n_in: continue
          b = data[r, 32*g + k/2]
          nib = (k % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F)
          W_hat[r, c] = L[nib] * s
  return W_hat
```

**Complexity.** Time `Θ(n_out · n_in_padded)`: one max-abs pass (64 comparisons) + 64×16 distances. That is ~10³ FLOP per group, not GEMM. Packing is bit shifts, `Θ(n_out · n_in_padded / 2)` bytes written.

**Peak memory.** Stream by rows: hold one float32 row (`n_in · 4` B), a packed row `n_in_padded/2`, row scales `n_groups · 2`. For `down_proj` 4096×14336 that is ≈ 56 KiB per row + output blobs can be written straight to the file. Full materialization of a 4096×14336 F32 input ≈ 224 MiB — allowed for CPU-verify of a fat matrix, not for a 32B compressor; the codec must be able to stream by rows (interface: “give a row / a stripe of rows”).

Full-materialization orientation (not required):

| Buffer | Formula | 4096×4096 | 4096×14336 |
|---|---|---:|---:|
| F32 input | `n_out·n_in·4` | 64 MiB | 224 MiB |
| `data` uint8 | `n_out·n_in_padded/2` | 8 MiB | 28 MiB |
| `scale` FP16 | `n_out·n_groups·2` | 0.5 MiB | 1.75 MiB |

Streaming by rows reduces the peak to `O(n_in)` plus what already sits in `.chr` on disk.

---

## 5. Nibble packing and unpacking

Target machines: x86_64, Windows/WSL — little-endian. The `data` blob is a `uint8` array, **endianness does not apply to a byte**. FP16 scales are little-endian `uint16` (§7, §10).

### 5.1. Masks

Byte `data[r, c]`, `c = 0 .. n_in_padded/2 − 1`:

```
lo  =  data[r, c]       & 0x0F      # index of W[r, 2c]
hi  = (data[r, c] >> 4) & 0x0F      # index of W[r, 2c+1]
data[r, c] = (hi << 4) | lo
```

`nib` is a LUT index 0..15, not `code + 8` and not a sign in the high bit of the nibble.

Inside a group the byte index in the row: weight column `j` lives in byte `j >> 1` of that row; even `j` is the low nibble.

### 5.2. Test vector: 2 weights → 1 byte

Packing is a pure function of two indices, without scale. Normalized `u` already in `[-1, 1]` (as after §2.5–2.6).

| `u0`, `u1` | idx0, idx1 | byte | breakdown |
|---|---|---|---|
| `0.0`, `1.0` | 7, 15 | **`0xF7`** | `0xF7 & 0x0F = 7` → `L[7]=0`; `0xF7 >> 4 = 15` → `L[15]=1` |
| `1.0`, `0.0` | 15, 7 | **`0x7F`** | low 15, high 7 |
| `L[3]`, `L[12]` | 3, 12 | **`0xC3`** | `L[3]=-0.39491748809814453`, `L[12]=0.44070982933044434` |

Control: if an implementation yields `0x7F` on `(0, 1)` — it packed the bnb way (even weight in the high nibble). That is a fail.

### 5.3. Kernel tile (layout only, not this package)

As `docs/compressor.md` §6.2: tile `(row_tile=i, col_group=j)` — rows `[64i, 64i+64)`, weight columns `[8j, 8j+8)` → bytes `data[64i : 64i+64, 4j : 4j+4]` (4 bytes = 8 nibbles), scale `scale[64i:64i+64, floor(8j/64)]`. The CPU codec writes row-major, not fragment-major.

---

## 6. Padding, empty rows, NaN/Inf

### 6.1. `n_in` padding

If `n_in % 64 ≠ 0`, on encode the tail of each row is filled with **zeros** up to `n_in_padded`. In CHR0 JSON `shape = [n_out, n_in]` is **logical**, without padding. Blobs have padded width (§10). Decode cuts columns `≥ n_in`.

Padding does not increase `s` if the group has a non-zero logical weight (`max(|w|, 0, 0, …) = |w|`). Tail zeros get index **7**.

Padding zeros are not “another group with scale=1 in metadata separately”: they enter the last group of the row.

Example: `W` of shape `[1, 2] = [0.0, 1.0]`. One group of 64: two logical + 62 zeros. `s16` = FP16(`1.0`) = `0x3C00`. Indices `[7, 15]` + 62×`7`. Packed 32 bytes:

```
F7 77 77 77 77 77 77 77 77 77 77 77 77 77 77 77
77 77 77 77 77 77 77 77 77 77 77 77 77 77 77 77
```

Decode returns only `[0.0, 1.0]`.

### 6.2. Empty row (all weights zeros)

A row of zeros is valid. Each of its groups: `s32=0 → s=1`, all indices 7, `s16 = 0x3C00`. This is not an error and not a “skip row”.

`n_out < 1` or `n_in < 1` is an error (§4), that is a different case.

### 6.3. NaN / Inf

Any `NaN` or `±Inf` in the input group (after conversion to float32, including Inf from F16/BF16) → **encode error**. Do not replace with 0, do not skip the group, do not write a partial blob of this matrix.

If `s32` is finite but `fp16(s32)` is not finite-positive (overflow `> 65504`, underflow to 0 of a non-zero group) → also encode error. Normal LLM weights do not hit this; silently writing Inf into `scale` is forbidden: decode then yields Inf and breaks verify.

### 6.4. Signs of zero

`−0.0` is finite. `abs(-0)=0`. The nearest level to `u=−0` after `s=1` is index 7. The LUT stores `+0`.

---

## 7. FP16 scale: bits and endian

- Storage: binary16 little-endian. `1.0` → bits `0x3C00` → in the file bytes `00 3C`.
- Write: RNE, ties to even, as in `docs/spec/vq.md` §7.2. binary16 subnormals allowed.
- Read: exact expansion to float32.
- `+0` as a stored scale is **forbidden** (except the path we already replaced with `s=1` for a zero group).
- Not BF16: `1.0` in BF16 is also `0x3C00` in the high 16 bits of float32, but among arbitrary `s` BF16 ≠ FP16.

Common values for tests:

| float32 | binary16 bits | LE bytes |
|---|---|---|
| `1.0` | `0x3C00` | `00 3C` |
| `2.0` | `0x4000` | `00 40` |
| `2.25` | `0x4080` | `80 40` |
| `2.5` | `0x4100` | `00 41` |
| `0.5` | `0x3800` | `00 38` |

Layout of `scale[r, g]` in the blob: offset `(r * n_groups + g) * 2` bytes from the start of the `scale` blob.

---

## 8. Golden example: 64 weights

One row, `n_in = 64` (no padding), `n_groups = 1`.

**Input** (count in float32, not in decimal “approximately”):

```
W[i] = float32(i) / float32(63)     for i = 0,1,…,63
```

`63` and `i ≤ 63` are exact in binary32. Division is float32. Equivalent to `float32(float64(i)/63)` for these `i` (both paths yield one binary32).

`max |W| = W[63] = 1.0` → `s32 = 1.0` → `s16 = 0x3C00`. `s = 1.0f32`. `u[i] = W[i]`. Clip does not fire.

**Indices** (nearest L2, tie → smaller; here there are no ties):

```
i:   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
idx: 7  7  7  8  8  8  8  8  9  9  9  9  9 10 10 10

i:  16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
idx: 10 10 10 11 11 11 11 11 11 12 12 12 12 12 12 12

i:  32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
idx: 13 13 13 13 13 13 13 13 13 14 14 14 14 14 14 14

i:  48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
idx: 14 14 14 14 14 14 14 15 15 15 15 15 15 15 15 15
```

Thresholds at which the index changes (for debugging, `u = i/63`):

| idx | `i` | condition |
|---|---|---|
| 7 | 0..2 | `u ≤ 0.03979014977812767` |
| 8 | 3..7 | up to `0.120255246758461` |
| 9 | 8..12 | up to `0.2035212516784668` |
| 10 | 13..18 | up to `0.2920137643814087` |
| 11 | 19..24 | up to `0.3893125355243683` |
| 12 | 25..31 | up to `0.5016634464263916` |
| 13 | 32..40 | up to `0.6427869200706482` |
| 14 | 41..54 | up to `0.8614784479141235` |
| 15 | 55..63 | above, including `1.0` |

The tightest gap in this fixture: `i=13`, `u ≈ 0.206349209` versus midpoint 9–10 `≈ 0.20352125` (gap ~3·10⁻³, thousands of ulp). The index is unambiguous.

**Packed 32 bytes** (low nibble = even `i`):

```
77 87 88 88 99 99 A9 AA AA BA BB BB CB CC CC CC
DD DD DD DD ED EE EE EE EE EE EE FE FF FF FF FF
```

Concatenated, uppercase:

```
778788889999A9AAAABABBBBCBCCCCCCDDDDDDDDEDEEEEEEEEEEEEFEFFFFFFFF
```

**Scale:** `s = 1.0`, bits `0x3C00`, in the blob `00 3C`.

Test `TestNF4Golden64`: encode of this row → 32 bytes as above and `scale` one `uint16` `0x3C00`. Decode → `L[idx[i]] * 1.0` with the LUT float32 bits, not “close to i/63”.

---

## 9. Test invariants without a model

Metrics orig vs reconstruct (after decode to float32, orig also in float32):

```
e_i    = float32( L[idx_i] * s ) − float32(W_i)
rmse   = sqrt( mean_i (float64(e_i)²) )
maxabs = max_i |e_i|
mae    = mean_i |e_i|
```

`mean` over **logical** elements, without padding. Sum of squares is in float64 from already-computed float32 `e_i`.

### 9.1. Zero matrix

`W = 0` of any shape with `n_out, n_in ≥ 1`, including non-multiples of 64. All nibbles of logical (and padded) positions = **7**. All scales `0x3C00`. Decode is exact zeros.

### 9.2. Constant on a group

A group of 64 copies of `c ≠ 0`, finite. `s32 = |c|`, `s16 = fp16(|c|)`. After division by `s = f32(s16)` all `u = sign(c)` (with clip if `s16 < |c|`). All indices **0** if `c < 0`, **15** if `c > 0`.

Check: `c = −2.25` → `s16 = 0x4080`, nibbles all `0x0`, decode = `−1.0 * 2.25 = −2.25` (2.25 is exact in FP16). `c = 0.5` → `s16 = 0x3800`, all indices 15, decode = `1.0 * 0.5`.

### 9.3. Idempotence of encode ∘ decode

Let `P = encode(W)`, `Ŵ = decode(P)`, `P₂ = encode(Ŵ)`, `Ŵ₂ = decode(P₂)`.

Require:

1. `decode(P)` is bit-stable: a repeated decode of the same blobs → the same `float32` bits.
2. `P₂` matches `P` (the same nibbles and the same scale bits) on fixtures where encode followed §2.2.

Why this is so when `s = max|g|`. A non-zero group always contains an element with `|g| = s32`. After steps 3–5 it encodes to index 0 or 15 (`|u|=1` after clip). Then `max|Ŵ| = |L[0 or 15]| · s = s`, the second encode writes the same `s16` and the same `u = L[idx]` — LUT fixed points, argmin returns the same index. Zero group: `Ŵ=0`, again `s=1`, index 7.

FP16 caveat: quantization **must** be against `f32(s16)`, not against raw `s32`. Otherwise `P₂.idx` can drift from `P.idx` by 1 nibble at a threshold. With §2.2 the invariant holds; the “caveat” in the requirements is exactly this scale cast, not “sometimes it is allowed not to match”.

Do not require `Ŵ = W` bit-for-byte: the codec is lossy.

### 9.4. Strict thresholds on the 2×64 fixture

Matrix `W[2, 64]` float32:

```
W[0, i] = float32(i) / float32(63)                 # as §8
W[1, i] = float32(2*i − 63) / float32(63)       # uniform grid [-1, 1]
```

Both scales `0x3C00` (`s=1`). Packed:

Row 0 — §8 (`7787…FFFF`).

Row 1, 32 bytes:

```
00 00 10 11 11 11 21 22 22 33 43 44 54 55 66 76
87 88 99 AA BA BB CC CC DD DD EE EE EE FE FF FF
```

Concatenated:

```
00001011111121222233434454556676878899AABABBCCCCDDDDEEEEEEFEFFFF
```

Row-1 indices (for debugging):

```
0 0 0 0 0  1 1 1 1 1 1 1 1  2 2 2 2 2  3 3 3  4 4 4 4
5 5 5  6 6 6  7 7  8 8 8  9 9  10 10 10  11 11 11
12 12 12 12  13 13 13 13  14 14 14 14 14 14 14  15 15 15 15 15
```

Computed metrics orig vs decode (128 elements, formula in the §9 header):

| metric | value | strict test threshold |
|---|---:|---:|
| rmse | 0.05184147829055911 | **≤ 0.05185** |
| maxabs | 0.14507704973220825 | **≤ 0.14508** |
| mae | 0.03979102736047935 | **≤ 0.03980** |

Where maxabs: row 1, `i=5`, `W = float32(−53/63) ≈ −0.841269850730896`, index 1, reconstruct `L[1] ≈ −0.6961928009986877`, `|e| ≈ 0.14507705`.

The main assert of this fixture is **packed bytes and scale bits**. rmse/maxabs thresholds are a net in case decode multiplies the wrong way. If packed matched, the metrics must match the table with a 10⁻⁷ margin; the thresholds are slightly wider so as not to catch summation order.

Orientation, **not** a unit gate: for random `N(0,1)` per group of 64 reconstruct RMSE is usually ≪ 0.1 (often ~0.03–0.08: NF4 error on normalized × typical `E[max|g|]`). Do not use a random seed in CI.

### 9.5. Required test names `internal/nf4`

| Name | Arrange | Assert |
|---|---|---|
| `TestNF4TableBits` | LUT constant | 16× `Float32bits` = §1 |
| `TestNF4Golden64` | `W[i]=i/63` | packed hex §8, scale `0x3C00` |
| `TestNF4Golden2x64` | fixture §9.4 | both packed, two scales `0x3C00`, rmse/maxabs/mae ≤ thresholds |
| `TestNF4PackNibble` | pairs §5.2 | bytes `F7`, `7F`, `C3` |
| `TestNF4ZeroMatrix` | zeros `[3,64]` and `[1,65]` | nibbles 7, scale `0x3C00`, decode 0 |
| `TestNF4Constant` | `c=−2.25` and `c=0.5` on a group | indices 0 / 15, scale bits §9.2 |
| `TestNF4Idempotent` | §8 and §9.4 | encode(decode(P)) = P |
| `TestNF4Pad65` | `[1,2]=[0,1]` | packed §6.1, decode of shape `[1,2]` |
| `TestNF4RejectNaN` | one NaN | encode error |
| `TestNF4RejectInf` | one Inf | encode error |
| `TestNF4RejectEmpty` | `n_out=0` or `n_in=0` | error |
| `TestNF4GroupSize` | `group_size≠64` at the CHR0 interface | reader/encoder error |
| `TestNF4DecodeExact` | packed §8 | `W_hat[i] = L[idx[i]]` bitwise |
| `TestNF4ScaleLE` | scale `1.0` | blob bytes `00 3C` |

---

## 10. Interface with CHR0

The codec does not describe the JSON file as a whole. Contract of a tensor with `codec: "nf4"`. Offsets `[start, end)` from the start of the file, as in `docs/compressor.md` §6.1. The container aligns blobs to 64.

### 10.1. Required keys

| Key | Type | v1 value | Required |
|---|---|---|---|
| `codec` | string | `"nf4"` | yes |
| `shape` | `[int, int]` | logical `[n_out, n_in]` | yes |
| `group_size` | int | **64** | yes |
| `data` | `[int, int]` | `[start, end)` packed uint8 | yes |
| `scale` | `[int, int]` | `[start, end)` FP16 | yes |
| `kind` | string | per CHR0 rules | yes, not the codec’s concern |
| `layer` | int | if a layered tensor | per CHR0 |

No `zero`, no `codebook` / `index`, no nested absmax. Reader: `group_size ≠ 64` → error of this slice. Field `zero` present → error (that would be another codec).

### 10.2. Blobs

Let `n_in_padded = 64 * ceil(n_in / 64)`, `n_groups = n_in_padded / 64`.

**`data`**

- dtype: `uint8`
- shape: `[n_out, n_in_padded/2]`
- layout: row-major, byte `(r, c)` per §5
- `end - start = n_out * n_in_padded / 2`

**`scale`**

- dtype: FP16 little-endian
- shape: `[n_out, n_groups]`
- `end - start = n_out * n_groups * 2`

Mini-example (offsets made up, lengths mandatory): `q_proj` 64×64:

```json
"model.layers.0.self_attn.q_proj": {
  "layer": 0,
  "kind": "q",
  "codec": "nf4",
  "shape": [64, 64],
  "group_size": 64,
  "data":  [4096, 6144],
  "scale": [6144, 6272]
}
```

`6144 − 4096 = 2048 = 64 × 32`. `6272 − 6144 = 128 = 64 × 1 × 2`.

Length reference for `4096×4096` (as in the compressor, but **with the correct length**, not like the illustration `data: [4096, 8200]` in `docs/compressor.md` §6.1):

- `data`: `4096 × 4096 / 2 = 8_388_608`
- `scale`: `4096 × (4096/64) × 2 = 524_288`

The writer computes `end = start + nbytes`.

### 10.3. Signatures for `chr decode` / `verify`

```
EncodeNF4(W f32[n_out, n_in]) → data u8[n_out, n_in_padded/2], scale f16[n_out, n_groups]

DecodeNF4(data, scale, n_out, n_in) → W_hat f32[n_out, n_in]
```

`DecodeNF4` is a pure function of the blobs and the logical shape. It does not emit padding outward.

Norms and bias are not encoded by this codec (in CHR0 they are `bf16`).

---

## 11. Checklist for `internal/nf4`

- LUT §1 bitwise, without scipy.
- Group encode: max-abs → FP16 scale → divide by `f32(s16)` → clip → L2 → tie = smaller index.
- Decode: `L[nib] * f32(s16)` in float32.
- Packing: low = even column; test `0xF7` / `0x7F`.
- Pad only in encode; `shape` is logical.
- NaN/Inf / empty shape / non-finite scale → error.
- Golden hex §8 and 2×64 §9.4.
- Interface: `group_size: 64`, blob shapes §10, no `zero`.
- Not INT4, not double quant, not the CUDA tree as a second canon.

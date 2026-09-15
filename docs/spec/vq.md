# Codec B: additive VQ 2×8, residual k-means (CPU)

Stage-B format diagnostics. Not chat quality, not AQLM, not calibration on activations X. Codebook **per matrix**, not per layer. Count only weight MSE.

Canonical implementation — package `internal/vq` on Go 1.22 without CGO, without GPU, without faiss, without beam, without sklearn. This spec is enough to write the package and table-driven tests on a tiny matrix.

---

## 0. Frozen constants and non-goals

| Symbol | Value | Meaning |
|---|---:|---|
| `M` | 2 | Number of codebooks. M=1 is **not** done in v1. |
| `k` | 256 | Centroids in each codebook. |
| `B` / `group_size` | 8 | Vector length = group along `n_in`. |
| `codebook_bits` | 8 | Index is one byte, `k = 2^8`. |
| `iters` default | 20 | Full Lloyd cycles on **each** codebook. |
| `--seed` default | 0 | `uint64`. |
| `--chunk` default | 262144 | Vectors per assignment chunk. |
| `--tol` default | off | Early stop only if a threshold was given. |

**Storage.** Codebook `codebook[M][256][8]` is IEEE 754 **binary16** (FP16), little-endian. Indices are `uint8`. k-means arithmetic is **float32**, sum accumulators are **float64**.

**Reconstruction of a group of 8 weights:**

\[
\hat g = C_1[i_1] + C_2[i_2] \quad \text{(componentwise, float32 after FP16→float32)}
\]

For `verify` the decode output is float32. Comparison with original W is lossy. Comparison of decode with “codebook + indices” is **bit-exact**.

**v1 non-goals (forbidden to implement in this slice):** GPU / CUDA / faiss / beam search / GPTVQ / block fine-tune / Hessian / imatrix / weighted k-means / mini-batch Sculley / sklearn / a second codebook of a different dimension / `M≠2` / `k≠256` / `B≠8`.

**Determinism.** The same `(W, seed, iters, chunk, tol)` → the same `uint8` indices and the same FP16 codebook bits on one Go compiler / one architecture. Tie-break rules and RNG consumption order are fixed below. Golden indices in unit tests are taken from Go, not from NumPy.

---

## 1. Residual k-means formulas: shapes, init → iterate → residual → second codebook

### 1.1. Input, padding, slicing into vectors

Input: matrix `W[n_out, n_in]`, row-major. Allowed source dtypes: `F32`, `F16`, `BF16`. Before clustering each element is converted **once** to float32:

- F32 → as-is;
- F16 → IEEE binary16 → float32;
- BF16 → high 16 bits of float32, low 16 bits zeros.

`NaN` or `Inf` on input → **encode error**, not silently. Empty axes `n_out < 1` or `n_in < 1` → encode error.

Padding only along `n_in`, only with zeros, only in the internal encode buffer:

\[
n_{\mathrm{in\_padded}} = 8 \left\lceil \frac{n_{\mathrm{in}}}{8} \right\rceil, \qquad
G = \frac{n_{\mathrm{in\_padded}}}{8}, \qquad
N = n_{\mathrm{out}} \cdot G.
\]

CHR0 metadata stores the **logical** `shape = [n_out, n_in]`, not padded. `G` and the `index` blob size are computed from padded. On decode the extra `n_in_padded - n_in` columns are dropped.

Vector with linear number `n ∈ {0,…,N−1}`:

\[
r = \left\lfloor n / G \right\rfloor, \quad
j = n \bmod G,
\]

\[
V[n, d] = \begin{cases}
W[r,\ 8j+d] & \text{if } 8j+d < n_{\mathrm{in}} \\
0 & \text{otherwise}
\end{cases}
\quad d=0..7.
\]

Shapes:

| Tensor | Shape | dtype when counting | Comment |
|---|---|---|---|
| `W` logical | `[n_out, n_in]` | float32 | As in JSON `shape`. |
| `W_pad` | `[n_out, n_in_padded]` | float32 | Needed only as a view; need not be materialized separately from `V`. |
| `V` | `[N, 8]` | float32 | `N = n_out * G`. Row-major: `V[n*8 + d]`. |
| `R` | `[N, 8]` | float32 | Working residual. Starts as a copy of `V`. |
| `C_m` (compute) | `[256, 8]` | float32 | One codebook. |
| `C` (file) | `[2, 256, 8]` | FP16 | See §7. |
| `π_m` | `[N]` | uint8 | Assignment of codebook `m`. |
| `index` (file) | `[n_out, G, 2]` | uint8 | See §6. |

`V` need not be a separate allocation if `W_pad` is already row-major float32: then `V` is a view. `R` is a **separate** buffer: after the first codebook `V` is no longer equal to the residual. The source `W` can be released after building `R` for compress.

### 1.2. Additive VQ as the objective

We seek codebooks \(C^{(0)}, C^{(1)} \in \mathbb{R}^{256 \times 8}\) and indices \(\pi^{(0)}, \pi^{(1)} \in \{0,\ldots,255\}^N\):

\[
\hat V[n] \approx C^{(0)}[\pi^{(0)}(n)] + C^{(1)}[\pi^{(1)}(n)].
\]

Bits/weight: \(M \cdot 8 / 8 = 2\). Codebook volume per matrix: \(2 \cdot 256 \cdot 8 \cdot 2 = 8192\) bytes.

This is **residual k-means** (AQLM / additive quantization initialization), not a joint beam over a pair of indices. Each codebook greedily approximates the current residual by MSE:

\[
L_m = \frac{1}{N} \sum_{n=0}^{N-1} \left\| R^{(m)}[n] - C^{(m)}[\pi^{(m)}(n)] \right\|_2^2.
\]

Do **not** take sqrt in the loss or in assignment: argmin over \(L_2^2\) coincides with argmin over \(L_2\).

### 1.3. Outer loop over codebooks (exactly M=2)

`Q16(x)` = float32 → FP16 (roundTiesToEven) → float32. Then the residual of the second codebook matches what decode will subtract from the written codebook, not from the “raw” float32 Lloyd centroid.

```
R[N, 8] ← copy(V)          # float32
create PCG(seed, 0)       # one generator per matrix, §2.1
idx_sub ← ReservoirIndices(N, n_sub=min(N, 65536), rng)   # once

for m = 0, 1:
    C_f32[256, 8], π[N] ← KMeans(R, k=256, iters, chunk, tol, rng, idx_sub)
    C_fp16[m] ← float32_to_fp16(C_f32)          # roundTiesToEven, §7
    # residual for the next codebook — through the written codebook
    for n = 0 .. N-1:                           # in chunks, like assignment
        R[n] ← R[n] − fp16_to_float32(C_fp16[m][ π[n] ])
    π_store[m] ← π

pack index from π_store[0], π_store[1]       # §6
return codebook=C_fp16, index
```

After `m=1` the final residual `R` is the reconstruction error relative to the **written** FP16 codebooks. It can be discarded; for logs MSE = `mean(R²)` over `N*8` elements (padding zeros enter the internal MSE; in `verify` against W they do not, there the logical shape is used).

If `float32_to_fp16` produced Inf (centroid outside the binary16 range, `|x| > 65504`) — encode error. Does not happen on LLM weights; on synthetic data with giant numbers — yes.

### 1.4. One KMeans on the current residual `R`

```
C ← KMeansPlusPlus(R[idx_sub], k=256, rng)      # §2
C_old ← zeros
for iter = 1 .. iters:                          # iters=0 → body does not run
    π ← AssignChunks(R, C, chunk)               # §3
    C, count ← UpdateMeans(R, π)                # §4
    δ ← max_{j: count[j]>0} ‖C[j] − C_old[j]‖_2
    C_old ← C
    ResplitEmpty(C, count, rng)                 # §4.2, after δ
    if tol is set and δ < tol: break
π ← AssignChunks(R, C, chunk)                   # final assignment under the final C
return C, π
```

**Why the final assignment.** After the last update the centroids moved, and `π` is still from the previous `C`. We write to the file indices consistent with the final float32 codebook (before the FP16 cast). After the FP16 cast we **do not** recompute indices: test §9.2 requires consistency with the written codebook, not optimality of the FP16 codebook.

**`iters = 0`:** only k-means++ and one assignment, no Lloyd. Allowed. Default 20.

**`iters < 0`:** error.

Each of the two codebooks runs its own Lloyd independently. The `iter` counter is not shared.

### 1.5. Shapes inside one Lloyd iteration

| Buffer | Shape | dtype | Lives |
|---|---|---|---|
| `C` | `[256, 8]` | float32 | whole codebook |
| `cnorm2` | `[256]` | float32 | iteration; `∑_d C[j,d]²` |
| `x_chunk` | `[T, 8]` | view into `R` | chunk |
| `dist` | `[T, 256]` | float32 | chunk; need not store whole, §3.4 |
| `π` | `[N]` | uint8 | whole codebook |
| `sum` | `[256, 8]` | float64 | iteration |
| `count` | `[256]` | int64 | iteration |

`T = min(chunk, N − offset)`.

---

## 2. k-means++ on a subsample

### 2.1. RNG

One generator per **one matrix**:

```
rng = math/rand/v2.New( math/rand/v2.NewPCG(uint64(seed), uint64(0)) )
```

Go 1.22+. Methods only `IntN(n)` (`[0, n)`) and `Float64()` (`[0, 1)`). Not `math/rand` v1 (different algorithm). Do not hash the tensor name into the seed: two matrices with the same `--seed` get the **same** sequence, each from its own `NewPCG(seed, 0)`.

Consumption order on a matrix:

1. Reservoir of subsample indices (0 calls if `N ≤ 65536`).
2. Codebook 0: k-means++ (IntN + Float64), then on each Lloyd iteration — empty-cluster resplit noise (Float64).
3. Codebook 1: the same, the generator is **not** reset.

Do not parallelize streams: an RNG race is forbidden. Assignment inside Lloyd does not depend on RNG.

### 2.2. Subsample size

\[
n_{\mathrm{sub}} = \min(N,\ 65536).
\]

Exactly that, not “about 64k”. `65536 = 2^{16}` is a spec constant, not a flag.

Subsample indices are computed **once** from `N` and reused for both codebooks. Vector values at those indices are taken from the **current** `R` (for codebook 1 that is already the residual).

### 2.3. How to sample without shuffling all of N

A full shuffle of indices `0..N-1` on a fat matrix is `N` `int64` values (for `N = 7_340_032` ≈ 56 MiB) plus a permutation pass. We do not do that.

**Chosen method: Algorithm R (Vitter), reservoir over indices.** Not a stride: `W` has structure along rows, a uniform stride systematically holes rows.

```
n_sub = min(N, 65536)
idx_sub[0 .. n_sub) — uint32
if N ≤ 65536:
    idx_sub[i] = i                          # no RNG
else:
    for i = 0 .. n_sub-1:
        idx_sub[i] = i                      # first n_sub vectors
    for t = n_sub .. N-1:
        j = rng.IntN(t + 1)                 # uniform in [0, t]
        if j < n_sub:
            idx_sub[j] = t
```

This is a uniform set of `n_sub` **distinct** indices. The order in array `idx_sub` is fixed by the algorithm and participates in k-means++ (the first centroid is uniform over **positions** of the reservoir).

Reservoir memory: `65536 * 4 = 262144` bytes for indices. The subsample vectors themselves are gathered before the codebook’s k-means++: `S[s, :] = R[idx_sub[s], :]`, shape `[n_sub, 8]` float32 = `65536 * 8 * 4 = 2_097_152` bytes. This buffer `S` can be reused.

### 2.4. k-means++ on `S[n_sub, 8]`

Arthur, Vassilvitskii 2007, squared L2, no sqrt.

**Start.** `j0 = rng.IntN(n_sub)`. `C[0] = S[j0]`.

**Step t = 1 .. 255.** For each row `s`:

\[
d^2[s] = \min_{0 \le j < t} \ \sum_{d=0}^{7} \bigl(S[s,d] - C[j,d]\bigr)^2
\]

(minimum over already chosen centroids; with equal distances the inner argmin does not matter — the same number lands in \(d^2\)).

\[
\Sigma = \sum_{s=0}^{n_{\mathrm{sub}}-1} d^2[s].
\]

- If `Σ > 0`: let `r = rng.Float64() * Σ`. Cumulative:

  ```
  acc = 0
  pick = n_sub - 1                  # fallback for rounding error
  for s = 0 .. n_sub-1:
      acc ← acc + d²[s]
      if acc >= r:
          pick = s
          break
  C[t] = S[pick]
  ```

  Comparison `acc >= r`, not `>`. If `r = 0` and leading `d²` are zeros, the first `s` with a non-zero accumulation is taken, or `s=0` when `acc>=0`.

- If `Σ = 0`: all subsample points already coincide with one of the chosen centroids (typical: 4 unique vectors, `k=256`). Take `pick = rng.IntN(n_sub)` **with repeats** and copy `C[t] = S[pick]`. Duplicates will be split by empty-cluster resplit on Lloyd (§4.2).

**If `n_sub < k`** (tiny matrix, e.g. `1×8` → `N=1`). After exhausting subsample diversity (`Σ=0` or unique steps ran out) the remaining centroids are copies + the same `Σ=0` / `IntN` mechanism. Lloyd + resplit are mandatory, otherwise 255 empty clusters would stay exact duplicates, and the assignment tie-break would dump everyone into the smaller index — that is fine for MSE, resplit is still needed so the codebook does not contain 255 identical rows without noise (for determinism the noise must be there).

k-means++ is **not** counted as a Lloyd iteration and does not look at `--tol`.

Init complexity: \(n_{\mathrm{sub}} \cdot (1+2+\cdots+255) \cdot O(B) \approx 65536 \cdot 32640 \cdot 8\) mul-add ≈ \(1.7 \cdot 10^{10}\) FLOP in the worst case, once per codebook. Acceptable on CPU; cheap on `n_sub ≪ 65536`.

---

## 3. Assignment in chunks

### 3.1. Formula

For a vector `x ∈ R^8` and centroid `c_j`:

\[
\mathrm{dist}(x, c_j)
 = \|x\|_2^2 + \|c_j\|_2^2 - 2\, x^\top c_j
 = \sum_d x_d^2 + \sum_d c_{j,d}^2 - 2 \sum_d x_d c_{j,d}.
\]

**Do not take sqrt.** Argmin over `dist` = argmin over \(L_2\).

For a fixed `x` the term `‖x‖²` does not depend on `j` and **may be omitted** in argmin (it does not affect the index choice). An implementation is allowed to compute

\[
j^\star(x) = \arg\min_j \bigl( \|c_j\|_2^2 - 2\, x^\top c_j \bigr).
\]

If the SSE itself is needed for an iteration log — then the full `dist`. SSE is not written to the file.

### 3.2. dtypes and operation order

| Quantity | dtype | How to compute |
|---|---|---|
| `x`, `C[j]` | float32 | as they sit in `R` / `C` |
| `dot = x·c_j` | float32 | `acc = 0; for d=0..7: acc += x[d]*c[j][d]` (mul, then add; **no FMA requirement**, `d` in increasing order) |
| `cnorm2[j]` | float32 | once per iteration: `∑_d C[j,d]*C[j,d]`, same `d` order |
| `xnorm2` | float32 | optional, same order |
| `dist` | float32 | `xnorm2 + cnorm2[j] − (dot + dot)`  (`2*dot` as `dot+dot`, not via float64) |
| `π[n]` | uint8 | argmin, §3.3 |

Forbidden to materialize a `(N, 256, 8)` or `(T, 256, 8)` tensor of differences. That for `T=N=7_340_032` is:

\[
7\,340\,032 \times 256 \times 8 \times 4 = 60\,129\,542\,144\ \text{bytes} \approx 56\ \text{GiB}.
\]

Instant OOM. Compute `dot` as GEMM `x_chunk[T,8] @ C[256,8]^T → [T,256]` or fused with per-row argmin (§3.4).

`cnorm2` is 256 float32 = 1024 bytes; recomputing every chunk is not forbidden, cheaper once per iteration.

### 3.3. Argmin and ties

Scan `j = 0, 1, …, 255` in that order:

```
best_j = 0
best_d = dist(x, C[0])
for j = 1 .. 255:
    d = dist(x, C[j])
    if d < best_d:          # strictly less
        best_d = d
        best_j = j
π = uint8(best_j)
```

**Ties → smaller index.** Equality `d == best_d` does not take the branch, the previous (smaller) `j` remains. Not `<=`. Not “random among tied”.

`best_d` starts from `j=0`, even if it is Inf/NaN — on a valid `R` (input without NaN/Inf, finite centroids) `dist` is finite.

### 3.4. Chunk loop

```
offset = 0
while offset < N:
    T = min(chunk, N - offset)
    x = R[offset : offset+T]          # view [T, 8]
    π[offset : offset+T] = Argmin(x, C)   # §3.1–3.3
    offset += T
```

The last chunk is shorter. `chunk < 1` → encode error. `chunk > N` is allowed: one chunk of length `N`.

**Two allowed memory layouts per chunk** (numerically identical argmin if dot is float32 with order `d=0..7`):

1. **Materialize `dist[T, 256]` float32**, then per-row argmin. Peak see §8.
2. **Fused:** for each of the `T` vectors keep 256 float32 (or a running-min immediately). Peak `dist` = `256*4` bytes. On CPU without BLAS this is the natural path. Indices are the same.

The spec **does not** require BLAS. If BLAS/GEMM is enabled later, it must obey the “smaller index” tie-break (after GEMM — the same scan `j=0..255` with strict `<`). A different fold order of the 8 dot terms can shift `dist` bits and, rarely, the index; for unit tests §9.1/§9.2 that does not matter (there distances are 0 vs >0). Cross-architecture golden indices on Gaussian noise are **not** promised.

Chunks are independent. Parallelizing over chunks in v1 is **not allowed** (simplicity + no question about reducing sums). Threads are not this slice.

---

## 4. Update: sums, counters, empty clusters

### 4.1. Mean

After assignment (or **in the same pass** as assignment — preferred on CPU, one walk of `R`):

```
sum[256, 8] = 0                  # float64
count[256] = 0                   # int64
for n = 0 .. N-1:
    j = π[n]
    count[j] += 1
    for d = 0 .. 7:
        sum[j, d] += float64(R[n, d])
for j = 0 .. 255:
    if count[j] > 0:
        for d = 0 .. 7:
            C[j, d] = float32( sum[j, d] / float64(count[j]) )
    # else do not touch C[j] — resplit in §4.2
```

Sums in float64: at `N ≈ 7·10^6` and coordinates ~1, float32-accumulator error is already noticeable. One division per occupied cluster. `count` is never > `N`; `int64` has margin.

A two-pass variant (first `π`, then `index_add` over `π`) is equivalent and also allowed. Mini-batch / exponential moving average — **no**.

An empty cluster (`count[j] = 0`) does not define a mean. Do not write `0/0`, do not silently zero `C[j]`.

### 4.2. Empty clusters — resplit of the fattest

After update, **once per iteration**:

1. `t = argmax_j count[j]`. Ties → **smaller** `j`. If all `count` are zeros (does not happen at `N≥1`) — error.
2. For each `j` in order `0,1,…,255`, if `count[j] == 0`:

   \[
   C[j, d] = C[t, d] + (2 u_{j,d} - 1) \cdot \varepsilon, \quad d=0..7,
   \]

   where `u_{j,d} = rng.Float64() ∈ [0, 1)`, \(\varepsilon = 10^{-5}\) (exactly `1e-5`, not a relative scale).

   Eight `Float64` calls per empty `j`, order `d=0..7`. Empties that are not there do not touch the RNG.

Counters are **not** changed: this is a virtual resplit. All empty clusters of this iteration are noisy copies of **one and the same** fattest (different noise). The next assignment will redistribute points. If the noisy copies are empty again — on the next iteration resplit again from the current fattest.

`ε = 1e-5` is representable in float32. Do not shift `C[t]` itself (requirements: noise on the new centroid, not “spread a pair by ±ε”).

Do not delete empty slots: the codebook is always 256 rows.

On a fixture of 4 exact vectors: k-means++ almost surely (and with orthogonal vectors — necessarily, see §9.1) puts all 4 into init; Lloyd leaves 4 live means equal to those vectors; 252 empties get `fattest + noise`; assignment of points with distance 0 stays on the exact centroid because of strict `<`. Reconstruction MSE ≈ 0.

### 4.3. When resplit is relative to `δ`

`δ` (§5) is computed **before** resplit, only over clusters with `count[j] > 0`. Otherwise every iteration with empties would give `δ ≥ ε` and would break early stopping.

---

## 5. Early stopping

Default: **always exactly `iters` cycles** of assign → update → resplit, plus a final assignment. `--tol` not set → no check.

If `--tol τ` was given with `τ ≥ 0` (float64/float32, one number in the CLI):

After update and **before** resplit:

\[
\delta = \max_{\,j:\ \mathrm{count}[j]>0}
\sqrt{\sum_{d=0}^{7} \bigl(C[j,d] - C^{\mathrm{old}}[j,d]\bigr)^2 }.
\]

Here sqrt is needed: the threshold is in units of L2 centroid displacement, not SSE. If there are no empties and all 256 are live — maximum over all `j`.

Stop if `δ < τ`. This iteration’s resplit **still runs** (the codebook must not go into the file with exact duplicates of empties). Then the final assignment and exit from this codebook’s Lloyd.

`C_old` is a copy of `C` after the **previous** iteration’s resplit (on iteration 1 — the state right after k-means++). That is, we measure the shift that update made from assignment, not resplit noise.

Minimum iterations when `τ` is set: one (there is something to measure). `τ = 0` means stop only when occupied centroids are bitwise unchanged.

Recommended value if it is ever turned on in the CLI by default: do not enable. On the 4-vector synthetic `δ=0` already after 1–2 iterations; on a Gaussian 20 iterations is cheaper than arguing about a threshold.

The `--iters` flag sets an **upper** bound. `--tol` without `--iters` is still capped at 20.

---

## 6. Packing `index[n_out, n_in_padded/8, M]`

### 6.1. In-memory layout

Shape `[n_out, G, M]` with `M=2`, dtype `uint8`, C-order / row-major, last axis fastest:

\[
\mathrm{offset}(r, j, m) = (r \cdot G + j) \cdot 2 + m.
\]

For one group of 8 weights — **two consecutive** bytes: first the index of codebook 0, immediately after it the index of codebook 1.

```
index[r, j, 0] = π^{(0)}[r*G + j]     # i1, codebook C1 = codebook[0]
index[r, j, 1] = π^{(1)}[r*G + j]     # i2, codebook C2 = codebook[1]
```

Bytes of the `index` blob have no extra padding inside the blob. Aligning the blob to 64 bytes is the CHR0 container’s job, not the codec’s: the codec yields exactly

\[
|\mathrm{index}| = n_{\mathrm{out}} \cdot G \cdot 2
\]

bytes. For `4096×14336`: `G = 1792`, `|index| = 4096 * 1792 * 2 = 14\,680\,064` bytes.

The Ampere kernel reads tile `(row_tile=i, col_group=j)` as `index[64i : 64i+64, j, :]`. Inside one row `r` groups `j` are consecutive in memory (`i1,i2` then the next group). A slice of 64 rows at a fixed `j` has stride `G·2` bytes. For CPU-verify the layout is row-major, not Tensor Core fragment-major.

### 6.2. Numeric example: 1×16 weights → 2 groups → 4 index bytes

Logical `W` of shape `[1, 16]`, `n_in % 8 = 0`, `G = 2`, `N = 2`. Two groups of 8 weights — two “vectors”, `M=2` indices each.

Suppose encode (or a hand-made codebook for a decode test) gave:

| group `j` | `W` columns | `i1 = index[0,j,0]` | `i2 = index[0,j,1]` |
|---|---|---:|---:|
| 0 | 0..7 | 0 | 7 |
| 1 | 8..15 | 1 | 0 |

Linear memory of the `index` blob (4 bytes):

```
offset:  0     1     2     3
byte:   i1g0  i2g0  i1g1  i2g1
hex:    00    07    01    00
```

This is “1×16 weights → 2 groups, two uint8 per group”. If the informal phrasing says “2 indices” — that means 2 groups; the file holds 4 bytes.

Decode of these 4 bytes with the codebook from §7.3:

\[
\hat g_0 = C[0][0] + C[1][7], \qquad \hat g_1 = C[0][1] + C[1][0].
\]

### 6.3. Padding in indices

`W` of shape `[1, 10]`: `n_in_padded = 16`, `G = 2`, the `index` blob is still 4 bytes. The second group is weights `[W[0,8], W[0,9], 0,0,0,0,0,0]`. Decode computes 16 values; only 10 logical ones enter `verify` and the output tensor. Padding-group indices **are stored** (otherwise the blob size cannot be derived from `shape` and `group_size`).

---

## 7. FP16 codebook: axes `[M, 256, 8]`, little-endian

### 7.1. Axis and byte order

`codebook[m, j, d]`:

- `m = 0..1` — slowest axis (codebook 0 whole, then codebook 1);
- `j = 0..255` — centroid number = uint8 index value;
- `d = 0..7` — group coordinate, coincides with `W[r, 8j+d]`.

Byte offset of an element from the start of the `codebook` blob:

\[
\mathrm{byte\_offset}(m,j,d) = \bigl((m \cdot 256 + j) \cdot 8 + d\bigr) \cdot 2.
\]

Each element is IEEE 754 binary16, **little-endian** (as x86_64 / Windows / WSL). Blob size is always

\[
2 \cdot 256 \cdot 8 \cdot 2 = 8192
\]

bytes, independent of `n_out, n_in`.

Codebook 0 occupies bytes `[0, 4096)`, codebook 1 — `[4096, 8192)`. Centroid `j` of codebook `m` is 16 consecutive bytes.

CHR0 may align the blob to 64; 8192 already divides by 64, there is no tail pad.

### 7.2. float32 ↔ FP16 conversion

- **Write:** round-to-nearest, ties to even (IEEE 754-2008). binary16 subnormals allowed. Overflow → encode error, not Inf in the file.
- **Read / reconstruct / residual between codebooks:** FP16 → float32 by exact mantissa expansion (like a hardware `cvt`). Intermediate BF16 is **not** used.
- Computing Lloyd in FP16 is **forbidden**.

Reconstruct identity: for each `d`

```
w_hat = fp16_to_f32(codebook[0, i1, d]) + fp16_to_f32(codebook[1, i2, d])
```

addition already in float32. Do not add in FP16 (overflow and tail loss). Do not quantize the sum back to FP16 — verify output is float32.

### 7.3. Golden codebook fragment for a layout test (no k-means)

Hand-made codebook, the rest zeros:

| Address | float32 value | binary16 LE hex |
|---|---|---|
| `C[0,0,:] = [1, 0,0,0,0,0,0,0]` | 1.0 → `0x3C00` | bytes `00 3C` plus 14 zeros |
| `C[0,1,:] = [0, 1,0,0,0,0,0,0]` | 1.0 at `d=1` | 2 zeros, then `00 3C`, then 12 zeros |
| `C[1,7,:] = [0, 0, 0.5, 0,0,0,0,0]` | 0.5 → `0x3800` | offset `4096 + 7*16 + 4 = 4212`: `00 38` |

First 16 bytes of the blob: `00 3C 00 00 00 00 00 00 00 00 00 00 00 00 00 00`.

Together with the indices of §6.2:

\[
\hat g_0 = [1, 0, 0.5, 0, 0, 0, 0, 0], \qquad
\hat g_1 = [0, 1, 0, 0, 0, 0, 0, 0].
\]

Bitwise: `1.0f32 = 0x3f800000`, `0.5f32 = 0x3f000000`. This example checks gather+add and layout, not Lloyd.

---

## 8. Peak memory: matrix 4096×14336, chunk=1e6

### 8.1. Problem sizes

\[
n_{\mathrm{out}}=4096,\quad n_{\mathrm{in}}=14336=8\cdot 1792 \text{ (padding not needed)},
\]

\[
N = 4096 \cdot 14336 / 8 = 7\,340\,032.
\]

float32 image of `W` / `V` / `R`:

\[
4096 \cdot 14336 \cdot 4 = 234\,881\,024\ \text{bytes} = 224\ \text{MiB}.
\]

### 8.2. Assignment chunk at `chunk = 1_000_000`

If `dist[T, 256]` float32 is materialized (`T = 1e6`):

| Chunk buffer | Shape | Bytes |
|---|---|---:|
| `dist` | `[1e6, 256]` float32 | **1 024 000 000** |
| `xnorm2` (if computed) | `[1e6]` float32 | 4 000 000 |
| `cnorm2` | `[256]` float32 | 1 024 |
| `π` of the piece (can write straight into full `π`) | `[1e6]` uint8 | 1 000 000 |
| `x` | view into `R` | 0 |
| **assignment workspace total** | | **1 029 001 024** ≈ 981 MiB |

`1e6 × 256 × 4 = 1_024_000_000` bytes — that is the answer to “how many bytes per assignment chunk at chunk=1e6” in the GEMM-distance-buffer formulation.

Forbidden buffer `(T, 256, 8)` float32 at the same `T`: `1e6 × 256 × 8 × 4 = 8_192_000_000` bytes ≈ 7.63 GiB — **do not allocate**.

`compressor.md` §5 estimated `1M×256×2 ≈ 0.5 GB` (as if 2 bytes per distance). Our count is in float32: **4 bytes**, ≈ **1.024 GB** for `dist`.

Fused per-row argmin: instead of 1.024 GB keep `256 × 4 = 1024` bytes. On CPU v1 **that is the recommended implementation**. The 1.024 GB figures remain an upper bound if someone goes the GEMM route.

### 8.3. Recommended `chunk`

**Default `--chunk 262144`** (`2^{18}`).

- `dist[262144, 256]` float32 = `262144 × 256 × 4 = 268_435_456` bytes = 256 MiB — if materialized.
- Fused: chunk peak is negligible, 262144 vectors × 8 × 4 = 8 MiB of a hot pass over `R` sits worse in cache than a tiny chunk, but has less loop overhead over chunks.
- Compromise with `compressor.md` §4.1 (“256k–1M”).

Alternatives (not default): `65536` (64 MiB GEMM buffer, more loop overhead), `1000000` (peak §8.2, on 16 GB RAM still fits next to 224 MiB `R`).

`chunk` does not affect the assignment result, only peak and speed. Changing it in index-integrity tests is allowed.

### 8.4. Encode peak for the whole 4096×14336 matrix

Minimal set (in-place residual, fused argmin):

| Buffer | Bytes |
|---|---:|
| `R` float32 | 234 881 024 |
| `π` of two codebooks, uint8 (can be two `[N]` or packed immediately) | 14 680 064 |
| `idx_sub` uint32 | 262 144 |
| `S` subsample float32 | 2 097 152 |
| `C` float32 of the current codebook | 8 192 |
| `sum` float64 + `count` | 256×8×8 + 256×8 = 18 432 |
| fused dist | 1 024 |
| **peak ≈** | **≈ 252 MiB** |

If plus a copy of the source `W` float32 (for simultaneous verify in the same process): +224 MiB ≈ 476 MiB.

If plus a materialized `dist` at `chunk=1e6`: **≈ 1.27 GB**. At default `chunk=262144` and materializing `dist`: **≈ 520 MiB**.

On lm_head ~1.6 GB BF16 (~3.2 GB float32) this slice **does not** cut stripes: the algorithm is the same. If the float32 image does not fit in RAM — that is the zone of `docs/spec/integrity-cli.md` (two-pass: reservoir by streaming rows, then assignment by stream). For unit tests stripes are not needed. The codebook is still one per matrix.

---

## 9. Test invariants without a model

Fixtures are synthetic. Do not download a model. Thresholds are strict; do not apply them to a live 8B.

Common rules:

- `seed=0`, `iters=20`, `chunk` any ≥1, `tol` off unless stated otherwise.
- MSE / RMSE over the logical `[n_out, n_in]`, float32, without padding columns:

\[
\mathrm{MSE} = \frac{1}{n_{\mathrm{out}} n_{\mathrm{in}}} \sum_{r,c} \bigl(W[r,c] - \hat W[r,c]\bigr)^2, \quad
\mathrm{RMSE} = \sqrt{\mathrm{MSE}}.
\]

- “Raw mean”: \(\hat W \equiv \bar w = \mathrm{mean}(W)\). Then MSE_mean = \(\mathrm{Var}(W)\) (population, divisor `n_out*n_in`, not `n-1`).

### 9.1. Exactly 4 distinct dim-8 vectors, repeated

Template rows (exact representation in float32 and in FP16):

```
a = [0, 0, 0, 0, 0, 0, 0, 0]
b = [1, 0, 0, 0, 0, 0, 0, 0]
c = [0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 1, 0, 0, 0, 0, 0]
```

`W` of shape `[16, 8]`: row `r` = `{a,b,c,d}[r % 4]`. Then `N = 16`, four unique vectors, each 4 times. `n_in` is already a multiple of 8.

Orthogonality of `{b,c,d}` and zero `a` guarantee: after choosing the first centroid in k-means++, squared distances of non-zero **different** templates are strictly larger than those of copies of the already chosen one; all 4 templates will land in init before duplicates.

After `iters=20` (2 would suffice):

1. Among the 256 rows of `fp16_to_f32(codebook[0])` all four vectors are present: for each template `v ∈ {a,b,c,d}`

   \[
   \min_{j=0..255} \| C^{(0)}[j] - v \|_\infty < 10^{-4}.
   \]

   In fact 0: `0` and `1` are exact in FP16.

2. `RMSE(W, decode(encode(W))) < 10^{-5}`, `maxabs < 10^{-4}`.

3. Codebook-0 indices on identical rows of `W` coincide (if the codebook happens to have two exact centroid duplicates — the tie-break takes the smaller `j`, the index is still the same).

4. Used codebook-1 centroids (those `j` that appear in `index[:,:,1]`) have `‖C^{(1)}[j]‖_∞ < 10^{-3}`: residual after a perfect first codebook ≈ 0.

This is the test “the codebook contains them to FP16 precision, MSE ≈ 0”. We do not check that the **unused** 252 codebook rows are zeros — there is resplit noise there.

Additionally, fully degenerate: a zero matrix `[8, 16]` → RMSE = 0 relative to `W`, all groups go to one codebook-0 index (the smaller among exact zeros).

### 9.2. reconstruct(encode(W)) bit-consistent with the written codebook

For **any** successfully encoded `W` (including §9.1, §9.3, the hand-made codebook §7.3):

Let `(codebook_fp16, index)` be the encode output. Independently of encode, the reference:

```
for r, j, d:
    i1 = index[r, j, 0]
    i2 = index[r, j, 1]
    ref[r, 8j+d] = f32(codebook[0,i1,d]) + f32(codebook[1,i2,d])
trim columns to n_in
```

`Decode(codebook, index, shape)` must yield `ref` **with the same float32 bits** (`math.Float32bits` on each element), not “close”. Comparison with original `W` is **not** done here.

Consequences:

- a repeated decode of the same blobs — the same bits;
- encode has no right to reconstruct from float32 Lloyd centroids, bypassing the FP16 cast;
- a container roundtrip `.chr` → the same 8192 codebook bytes and the same `index` bytes → the same `ref`.

### 9.3. Original vs reconstruct: better than mean / not worse than a random single codebook

Fixture `W` `[32, 128]`, `N = 512` (larger than `k`, one codebook cannot memorize all vectors). Elements are deterministic “noise” without Box–Muller, row-major:

```
x = uint32(seed)              # for this test seed=1, not 0: xorshift32 does not start from 0
if x == 0: x = 1
for each element:
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5                 # all shifts on uint32
    W = float32( (x % 2001) ) / 1000.0 − 1.0     # ≈ Uniform[-1, 1]
```

The population variance of this matrix is a concrete number; the implementation computes it from `W`. Thresholds:

1. **Main:** `MSE(W, decode(encode(W))) < 0.5 * Var(W)`. The raw mean gives exactly `Var(W)`; 2×8 must be noticeably better. (Readiness criterion from the requirements: “MSE after 2×8 is noticeably smaller than that of the raw mean”.)

2. **Random single codebook of 256:** build `C_rand[256, 8]` with the same xorshift, continuing the state after filling `W` (another 256×8 calls of the same formula). Assignment §3 to **one** codebook, reconstruct = `C_rand[π_rand]` without a second codebook. Require

   \[
   \mathrm{MSE}_{\mathrm{VQ2}} \le \mathrm{MSE}_{\mathrm{rand1}}.
   \]

   On this fixture the inequality is strict with a large margin: random 256 vectors in the cube `[-1,1]^8` do not land in the cloud of `W`.

3. Do not require `MSE = 0`. Do not require bitwise matching of indices across machines.

A matrix with fewer than `k` vectors (`N ≤ 256`) is **not** suitable for this test: then one codebook can memorize all points, MSE≈0, comparison with mean is trivially “yes”, and with a “random codebook” it depends on luck.

### 9.4. More checks (required in `internal/vq`, cheap)

| Name | Arrange | Assert |
|---|---|---|
| `PadNin` | `W` `[1, 10]`, finite numbers | `G=2`, `|index|=4`, decode of shape `[1,10]`, padding columns do not leak out |
| `Layout1x16` | hand-made codebook §7.3 + bytes `00 07 01 00` | decode = `[1, 0, 0.5, 0,0,0,0,0,  0,1,0,0,0,0,0,0]`, bits §9.2 |
| `TieSmallerIndex` | `x=[1,0,…,0]`, `C[0]=C[1]=x`, the rest far | `π=0` |
| `DistNoSqrt` | micro-example §9.5 | argmin = 0 |
| `RejectNaN` | one NaN in `W` | encode error |
| `RejectInf` | one Inf | encode error |
| `ItersZero` | §9.1, `iters=0` | does not crash; RMSE may be > 0, but decode is bit-consistent with the codebook |
| `SeedDeterminism` | two encodes of §9.3, `seed=1` | `codebook` and `index` bytes matched |
| `EmptyMatrix` | `n_out=0` or `n_in=0` | error |

### 9.5. Dist micro-example (hand calculation)

`x = [1,0,0,0,0,0,0,0]`, `‖x‖² = 1`.

| `j` | `c_j` | `‖c‖²` | `x·c` | `dist = 1 + ‖c‖² − 2 x·c` |
|---|---|---:|---:|---:|
| 0 | `[1,0,0,0,0,0,0,0]` | 1 | 1 | `1+1−2=0` |
| 1 | `[0,1,0,0,0,0,0,0]` | 1 | 0 | `1+1−0=2` |
| 2 | `[0.5,0,0,0,0,0,0,0]` | 0.25 | 0.5 | `1+0.25−1=0.25` |

Argmin = 0. If `c_3 = c_0` is added, tie → still 0.

---

## 10. Interface with CHR0

The codec does not write the JSON as a whole — only the tensor-field contract with `codec: "vq"`. Offsets `[start, end)` from the start of the file, as in `docs/compressor.md` §6.1. The container pads blobs to 64.

### 10.1. Required JSON keys of a tensor

| Key | Type | v1 value | Required |
|---|---|---|---|
| `codec` | string | `"vq"` | yes |
| `shape` | `[int, int]` | logical `[n_out, n_in]` | yes |
| `group_size` | int | **8** | yes |
| `n_codebooks` | int | **2** | yes |
| `codebook_bits` | int | **8** | yes |
| `codebook` | `[int, int]` | `[start, end)` of the codebook blob | yes |
| `index` | `[int, int]` | `[start, end)` of the index blob | yes |
| `kind` | string | as in CHR0 (`q`/`down`/…) | yes, but not the codec’s concern |
| `layer` | int | if a layered tensor | per CHR0 rules |

There are no `data`, `scale`, `zero` fields on `vq`. Reader: if `group_size ≠ 8` or `n_codebooks ≠ 2` or `codebook_bits ≠ 8` — error (this slice does not know other modes).

### 10.2. Blobs

**`codebook`**

- dtype: FP16 little-endian
- shape: `[n_codebooks, 2^codebook_bits, group_size] = [2, 256, 8]`
- `end - start = 8192`
- axes and offsets — §7

**`index`**

- dtype: uint8
- shape: `[n_out, n_in_padded/8, n_codebooks] = [n_out, G, 2]`
- `n_in_padded = 8 * ceil(n_in / 8)`, `n_in` from `shape[1]`
- `end - start = n_out * G * 2`
- layout — §6

Example (`start=20000` offset is made up; lengths are mandatory):

```json
"model.layers.0.mlp.down_proj": {
  "layer": 0,
  "kind": "down",
  "codec": "vq",
  "shape": [4096, 14336],
  "group_size": 8,
  "n_codebooks": 2,
  "codebook_bits": 8,
  "codebook": [20000, 28192],
  "index": [28192, 14708256]
}
```

Check: `28192 − 20000 = 8192`. `14708256 − 28192 = 14_680_064 = 4096 × 1792 × 2`. The writer computes `end = start + nbytes`, does not paste the example by hand. In `docs/compressor.md` §6.1 the `index` offset is a container illustration, not a length reference.

### 10.3. Decode as the contract for `chr decode` / `verify`

Logical signatures (not code):

```
EncodeVQ(W f32[n_out,n_in], seed u64, iters int, chunk int, tol optional)
  → codebook f16[2,256,8], index u8[n_out,G,2]

DecodeVQ(codebook f16[2,256,8], index u8[n_out,G,2], n_out, n_in)
  → W_hat f32[n_out, n_in]
```

`DecodeVQ` does not read seed/iters: it is a pure function of the codebook and indices. It does not return padding columns.

Norms / bias are not encoded by this codec (always `bf16` in CHR0).

---

## 11. Complexity and expected time (orientation, not a test)

One assignment on 4096×14336: GEMM `(N×8)×(8×256)` ≈ `N * 256 * 8 = 1.50·10^{10}` FMA ≈ 30 GFLOP, as `docs/compressor.md` §4.2.

20 iterations × 2 codebooks + 2 final assignments ≈ 42 assignments ≈ 1.3 TFLOP plus update and k-means++. On CPU pure Go — minutes for one fat matrix, not GPU milliseconds. For v1 integrity that is acceptable; a live 8B/32B on CPU is a separate decision (this spec does not include faiss and GPU). Mini-batch as a compressor “fallback path” is **not** included: it changes the result and breaks determinism §2.

---

## 12. Checklist for `internal/vq`

- [ ] Pad `n_in` to a multiple of 8, logical shape on the outside.
- [ ] `R` float32, two codebooks in sequence, residual through `Q16(C)`.
- [ ] PCG(`seed`, 0); reservoir `min(N,65536)`; k-means++; Lloyd ≤20.
- [ ] Assignment in chunks, `dist` without sqrt, ties → smaller index, no `(N,256,8)` tensor.
- [ ] Update float64-sums; empties → fattest + `Uniform[-1e-5, 1e-5)` from the same RNG.
- [ ] `--tol` optional, `δ` by L2 of occupied centroids before resplit.
- [ ] `index` row-major `[n_out,G,2]`, FP16 codebook `[2,256,8]` LE.
- [ ] Tests §9.1–§9.4 without a model.
- [ ] JSON fields `group_size: 8`, `n_codebooks: 2`, `codebook_bits: 8`.

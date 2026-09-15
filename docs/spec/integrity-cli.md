# CLI `chr` and integrity protocol

User-facing contract of the `chr` utility (Go 1.22, CPU, no CGO) and what counts as **successful integrity** for lossy codecs NF4/VQ. Byte layout of `.chr` — [chr0.md](chr0.md) and [compressor.md](../compressor.md) §6. Quantization formulas — [nf4.md](nf4.md) / [vq.md](vq.md). They are not here.

This is **not** inference and **not** WikiText. Three check levels are **separated**; they must not be mixed:

| Level | Question | Reference | Bit-exact with orig BF16? |
|---|---|---|---|
| 1. Container | re-reading `.chr` yields the same blobs | payload bytes | yes, on `.chr` blobs |
| 2. Codec | `decode(pack(W))` matches the codec reference | golden from `nf4.md` / `vq.md` | for nf4/vq — **no** (this is not about orig) |
| 3. Weight | orig vs reconstruct | RMSE / MAE / maxabs | **no** for Linear/embed/lm_head; **yes** for norm/bias (`codec=bf16`) |

The live-8B criterion (PPL, chat) is [compressor.md](../compressor.md) §7, **not** the `verify` exit code. CLI on synthetics does not replace that.

---

## 0. Frozen decisions

- Language: Go 1.22. Module `chr`. No CGO. Dependencies: standard library. In tests — only `testing` (and `os`/`path/filepath`/`bytes` from stdlib). No `cobra`, cobra-like, `testify`, `safetensors` PyPI, `torch`.
- Binary: `chr`, package `cmd/chr`.
- Commands: `compress`, `decode`, `verify`. No other subcommands in v1 (`help`/`version` are not required; `-h` on the root and on a subcommand — yes).
- `--codec` on compress: only `nf4` | `vq` (lowercase). `int4` in the container JSON is allowed by the neighboring spec — **the v1 CLI does not encode and does not verify `int4`**: if encountered → code 1.
- In one `.chr` — one lossy codec for all compressible tensors; norm/bias always `bf16` ([compressor.md](../compressor.md) §6, chr0 requirements).
- Tensor names in `.chr` — as in HuggingFace, **without** the `.weight` suffix. If in orig safetensors the key is `….q_proj.weight`, in `.chr` and in the verify report it is `….q_proj`.
- `decode` always writes **one** safetensors file, dtype **F32**, logical shape (padding trimmed). Why F32: on CPU it is easier to compare and inspect in numpy; BF16 decode would hide a scale rounding error. For a live 8B a full dump ≈ 32 GB — therefore a live run is checked with `verify`, not `decode`.
- Exit codes: **0** ok, **1** input/contract/I/O error, **2** only for `verify` when metrics were computed and a threshold failed. `compress`/`decode` have no code 2.
- A model is not placed in the repository or in tests and is not downloaded. Tests write fixtures themselves.
- Flags — package `flag` (`flag.ContinueOnError`), not `flag.ExitOnError`: otherwise a parse error would give os.Exit(2) and coincide with “threshold failed”. Parse error → code **1**.
- One thread. Do not open shards in parallel. In RAM — no more than one tensor (or one stripe, §7).

---

## 1. Flags, defaults, thresholds

### 1.1. argv parsing

```
chr <command> [flags]
```

- No command / unknown command / `-h` on the root: usage on stderr, code 1 (for `-h` on the root, 0 is allowed — **we choose 0 for `-h`/`--help`, 1 for an unknown command and empty argv**).
- Flags only after the command name. `chr --in x compress` — error, code 1.
- No positional arguments. Everything is flags.
- stdin as `--in -` is not supported.
- `--out` is overwritten without asking. The parent directory must exist (no `mkdir -p`), otherwise code 1.

General usage shape (wording is free, flag names are not):

```
chr compress --in <safetensors|dir|index.json> --out <file.chr> --codec nf4|vq [flags]
chr decode   --in <file.chr> --out <out.safetensors>
chr verify    --orig <safetensors|dir|index.json> --chr <file.chr> [flags]
```

### 1.2. `chr compress`

| Flag | Type | Default | Required | Meaning |
|---|---|---|---|---|
| `--in` | path | — | yes | one `.safetensors`, **or** a directory with shards, **or** a path to `*.safetensors.index.json` |
| `--out` | path | — | yes | `.chr` path (extension is not checked) |
| `--codec` | `nf4`\|`vq` | — | **yes** | no silent default: nf4 and vq are different files and different quality |
| `--group-size` | int | `64` if `--codec=nf4`, `8` if `vq` | no | v1 accepts **only** the codec’s canonical size |
| `--seed` | int64 | `0` | no | VQ codebook RNG (k-means++ / resplit). For nf4 **ignored** (no warning) |
| `--iters` | int | `20` | no | Lloyd iterations; VQ only. nf4 ignores. `<1` → code 1 |
| `--chunk` | int | `262144` | no | dim-8 vectors per one VQ assignment. nf4 ignores. `<256` → code 1. Matches [vq.md](vq.md) (memory peak, not 1e6). |
| `--stripe-bytes` | int64 | `268435456` (256 MiB) | no | F32 working-buffer threshold after which a tensor is cut into stripes (§7) |
| `--stripe-rows` | int | `4096` | no | desired stripe height. `<1` → code 1 |
| `--arch` | string | `unknown` | no | `arch` field of the CHR0 header |
| `--hidden-size` | int | `0` | no | if 0 — try to derive from tensors `kind=q`/`embed`/`norm`, else leave 0 |
| `--intermediate-size` | int | `0` | no | similarly from `up`/`gate`/`down` |
| `--num-layers` | int | `0` | no | max `layer` + 1 from names, else 0 |
| `--vocab-size` | int | `0` | no | from `embed`/`lm_head` `shape[0]`, else 0 |
| `--quiet` | bool | false | no | do not write progress to stderr |

`--group-size` in v1:

- `--codec=nf4` and value ≠ 64 → code 1, text `chr: nf4 group-size must be 64`.
- `--codec=vq` and value ≠ 8 → code 1, `chr: vq group-size must be 8`.
- The flag exists so the command matches future codecs and the documentation; currently this is an invariant check, not a free parameter.

`--codec` is required: absence / empty string / `NF4` / `int4` / `bf16` → code 1.

Progress (stderr, not stdout), one line per tensor, unless `--quiet`:

```
compress  model.layers.0.self_attn.q_proj  [64, 64]  nf4  3ms
```

Tensor order: as the orig iterator (§3.2). After the last — a line `wrote <path> tensors=<n>`.

Encode errors (NaN/Inf in the input, a holey safetensors, an unreadable shard) → code 1, the tensor in `--out` is not considered valid (delete an unfinished file **on error**, so a half-CHR0 is not left).

### 1.3. `chr decode`

| Flag | Type | Default | Required |
|---|---|---|---|
| `--in` | path | — | yes, `.chr` file |
| `--out` | path | — | yes, **one** `.safetensors` |

No other flags. No `--dtype`, no `--name`, no shards on output.

Each tensor from the CHR0 header is written as F32, logical `shape` from JSON (not padded). Names — CHR0 keys. See §4.

stderr progress: `decode  <name>  [d0, d1, …]  <codec>`.

### 1.4. `chr verify`

| Flag | Type | Default | Required |
|---|---|---|---|
| `--orig` | path | — | yes, same forms as `--in` of compress |
| `--chr` | path | — | yes |
| `--fail-rmse` | float64 | from table A by the file’s lossy codec | no |
| `--fail-maxabs` | float64 | from table A by the file’s lossy codec | no |
| `--json` | bool | false | no | stdout = JSON; do not print human text |
| `--quiet` | bool | false | no | only summary / worst (human mode). Ignored with `--json` |

`--fail-rmse` / `--fail-maxabs` thresholds apply **per each** lossy tensor separately (not on the model average). One `lm_head` above the threshold → code 2, even if all Linear are “green”.

`codec=bf16` (norm, bias): flag thresholds **do not apply**. The requirement is bit-exact in the sense of §3.5. This is not disabled in v1: a broken norm is a container bug, not “lossy”.

If `.chr` has no lossy tensors (bf16 only) — table A defaults as for `nf4`, unused in practice.

Negative `--fail-rmse` or `--fail-maxabs` → code 1. `+Inf` is allowed (effectively disables this threshold for lossy). NaN in the flag → code 1.

### 1.5. Table A — CLI and unit-test defaults (synthetics)

These numbers are the **binary default** and what tests §5 rely on **if the test does not pass flags**. They are sized for fixtures of scale ~N(0,1) / U[-1,1] and small `n`.

| Codec | `--fail-rmse` | `--fail-maxabs` | Why so |
|---|---|---|---|
| `nf4` | `0.08` | `0.50` | group-64 NF4 on N(0,1): RMSE is usually ≪ 0.1; maxabs ≤ half of the widest LUT gap (~0.139) × `s=max\|g\|`. For a group of 64 N(0,1) samples `s` is rarely > 3.5 → maxabs ≲ 0.49. 0.50 is tight, but without false FAIL on honest NF4. |
| `vq` | `0.40` | `2.50` | residual 2×8 without X. On tiny fixtures RMSE ≪ 0.40 (often ~0). On a square ~256×256 N(0,1) RMSE should be < `rms(W)≈1`; 0.40 means “the codebook actually learned, this is not the mean”. maxabs 2.50 — several σ, not a codebook blow-up. |

Zeros, constants, 4 repeated VQ vectors — should pass **an order of magnitude** below these numbers; tests §5 set stricter asserts for them, not relying only on the CLI default.

### 1.6. Table B — live 8B (not for unit tests)

**Not the binary default.** Mark this explicitly in help and in this file. On live weights std is often 0.01–0.03 for Linear and larger for `embed`/`lm_head`; an absolute synthetic threshold either strangles embed, or (if loosened globally) lets a dead codec through on Linear.

After downloading 8B **always pass flags explicitly**:

| Codec | `--fail-rmse` | `--fail-maxabs` | Comment |
|---|---|---|---|
| `nf4` | `0.12` | `2.0` | For a live model, **not for unit tests**. Slack for `embed_tokens` / `lm_head`. A typical Linear RMSE will be 10⁻³…10⁻² — the threshold does not “check chat quality”, it catches a broken pack (zeros, swapped nibbles). |
| `vq` | `0.50` | `8.0` | For a live model, **not for unit tests**. Weight-only 2×8 without imatrix; expectation per [compressor.md](../compressor.md) §7 — the model still talks, PPL worse than NF4. The threshold catches a NaN codebook / swapped index axes, not SOTA 2-bit. |

Live PPL / “chat on 10 questions” the CLI **does not compute**. If `verify` passed table B, and PPL is “in outer space” — that is not a CLI bug, that is the compressor / runtime criterion of §7.

### 1.7. Resolving `--in` / `--orig`

1. Path is an ordinary file and the name ends with `.safetensors` (or it is a file, and the safetensors magic/header reads) → one file.
2. Path is a file `*.safetensors.index.json` → shards by `weight_map`, base = `dirname(path)`.
3. Path is a directory:
   - if `model.safetensors.index.json` exists → as item 2;
   - else if exactly one `*.safetensors` → that one;
   - else code 1 (`chr: ambiguous safetensors directory`).
4. No file / no permissions → code 1.

Shards: key `weight_map[tensor_name]` is a relative path from the index directory. Name iteration — **lexicographic sort of keys** (stable verify output, independent of JSON-object order).

**One** shard is open at any moment. Filename in the map changed — close the previous fd, open a new one. Forbidden to keep a map of all mmaps.

---

## 2. `verify` printing

Per-tensor metrics are computed over **logical** elements (padding is not included), orig and reconstruct in float32 (§3.4):

```
n      = ∏ shape
mae    = (1/n) Σ |ŵ_i − w_i|
rmse   = sqrt( (1/n) Σ (ŵ_i − w_i)² )
maxabs = max |ŵ_i − w_i|
rms    = sqrt( (1/n) Σ w_i² )          # orig; report only
rel    = rmse / rms   if rms > 0, else 0 if rmse==0, else +Inf
```

`n=0` (empty shape) → code 1, not a “skip”.

Extra fields `rms` and `rel` are **printed**, but are **not** v1 thresholds (there are no `--fail-rel` flags). On a live model `rel` shows whether the codec is worse than the constant 0 (`rel ≥ 1`).

### 2.1. Human text (stdout, if no `--json`)

Encoding UTF-8, `\n`. First lines — header, then a table, then a summary block. Column widths are not rigidly fixed (HF names are long); the field separator inside a tensor line is **two or more spaces** or a tab. For machine parsing in tests use `--json`.

Example (synthetic 3 tensors, nf4):

```
chr verify
orig:        /tmp/fake.safetensors
chr:         /tmp/fake.nf4.chr
linear_codec: nf4
thresholds:  rmse<=0.08  maxabs<=0.50  bf16=exact

name                                       shape          codec  n        rmse          mae           maxabs        rms          rel      status
model.layers.0.mlp.down_proj               [32, 128]     nf4    4096     2.134512e-02  1.540011e-02  7.812500e-02  9.912000e-01  2.15e-02  ok
model.layers.0.self_attn.q_proj             [64, 64]      nf4    4096     1.001003e-02  7.100000e-03  4.125000e-02  1.002000e+00  1.00e-02  ok
model.norm                                 [64]           bf16   64      0.000000e+00  0.000000e+00  0.000000e+00  1.000000e+00  0.00e+00  ok

worst: model.layers.0.mlp.down_proj  rmse=2.134512e-02  maxabs=7.812500e-02
summary: tensors=3  lossy=2  bf16=1  skipped=0  fail=0  mean_rmse_lossy=1.567757e-02
PASS
```

Rules:

- `status`: `ok` or `FAIL`. For a bf16 mismatch also `FAIL`.
- Tensor lines — **sort by `name`** (as the iterator of §1.7).
- Metric numbers: `%.6e`. Integers — decimal.
- `worst`: among tensors with `n>0` pick the maximum `rmse`; on a tie — larger `maxabs`; on a tie — lexicographically smaller name. If there are no tensors — line `worst: (none)`.
- `mean_rmse_lossy` is **not** weighted by `n`, but the mean over lossy tensors (each matrix layer is equal). If lossy=0, write `n/a`.
- Last line: `PASS` if the code will be 0; `FAIL` if the code will be 2. On code 1 this template may not finish printing — then an error message on stderr (§2.3).
- On code 2 the table is **complete** (do not stop at the first FAIL): all bad tensors need to be visible.

`--quiet`: do not print the header or the per-tensor table; `worst` + `summary` + `PASS`/`FAIL` remain.

### 2.2. `--json` (stdout, one object)

Keys snake_case. No human text on stdout. On stderr — only code-1 errors.

```json
{
  "orig": "/tmp/fake.safetensors",
  "chr": "/tmp/fake.nf4.chr",
  "linear_codec": "nf4",
  "thresholds": {
    "rmse": 0.08,
    "maxabs": 0.50,
    "bf16_exact": true
  },
  "tensors": [
    {
      "name": "model.layers.0.mlp.down_proj",
      "shape": [32, 128],
      "codec": "nf4",
      "n": 4096,
      "rmse": 0.02134512,
      "mae": 0.01540011,
      "maxabs": 0.078125,
      "rms": 0.9912,
      "rel": 0.02154,
      "ok": true
    }
  ],
  "worst": {
    "name": "model.layers.0.mlp.down_proj",
    "rmse": 0.02134512,
    "maxabs": 0.078125
  },
  "summary": {
    "tensors": 3,
    "lossy": 2,
    "bf16": 1,
    "skipped": 0,
    "fail": 0,
    "mean_rmse_lossy": 0.01567757
  },
  "failed": [],
  "ok": true,
  "exit": 0
}
```

- `tensors` — the same order as in the human table.
- `failed` — names with `ok=false`, lexicographically.
- `worst`: `null` if `tensors` is empty.
- `exit` duplicates the process code (0 or 2 in this object). On code 1 JSON is **not required** to be a valid report: it is allowed to write no JSON at all, only stderr. If an error is discovered **after** some tensors have already been computed (e.g. extra at the end) — still code 1, JSON need not be emitted. Simpler for the implementer: code 1 → stderr only.
- `ok` = (`exit==0`).
- floats as JSON numbers (not strings). Tests compare with relative `1e-5`, not byte-for-byte JSON.

### 2.3. Code-1 messages (stderr)

Stable prefix `chr: ` — tests bind to it.

| Situation | Message (exactly the template) |
|---|---|
| no file | `chr: open <path>: …` (`os.PathError` text is allowed after the prefix) |
| compressible / required bf16 missing from chr | `chr: missing tensor in chr: <name>` |
| name is in chr, not among stored orig | `chr: extra tensor in chr: <name>` |
| unsupported codec in chr | `chr: unsupported codec: <codec> (<name>)` |
| NaN/Inf in orig on encode | `chr: non-finite value in tensor <name>` |
| bad `--codec` / `--group-size` | as in §1.2 |
| orig shape ≠ logical `shape` in chr | `chr: shape mismatch: <name> orig=<a> chr=<b>` |
| orig dtype not F32/F16/BF16 | `chr: unsupported dtype <dtype> (<name>)` |

Several missing/extra: one line per name is allowed, then a shared `chr: tensor set mismatch`. Code 1, even if metrics of some tensors have already been computed.

---

## 3. What to compare with what (name sets)

### 3.1. Classifier (one for compress and verify)

First the canonical name: if the orig key ends with `.weight` and **does not** end with `.bias` — strip the `.weight` suffix. Then match by the **last path component** and by substrings (order — first matching branch top to bottom):

| Condition on the canonical name | `kind` | Class | In `.chr`? |
|---|---|---|---|
| contains `inv_freq` **or** `rotary_emb` **or** suffix `.sin` / `.cos` | skip | **skip** | no |
| suffix `.bias` | `other` (or the parent’s kind, if convenient for the kernel; for the CLI `other` is enough) | **store_bf16** | yes, `codec=bf16` |
| last component `embed_tokens` **or** name `tok_embeddings` / `wte` | `embed` | **compress** | yes, `--codec` |
| last component `lm_head` **or** (`output` and the name **does not** contain `norm`) | `lm_head` | **compress** | yes |
| last component `q_proj` | `q` | **compress** | yes |
| `k_proj` | `k` | **compress** | yes |
| `v_proj` | `v` | **compress** | yes |
| `o_proj` | `o` | **compress** | yes |
| `gate_proj` | `gate` | **compress** | yes |
| `up_proj` | `up` | **compress** | yes |
| `down_proj` | `down` | **compress** | yes |
| contains `norm` or `layernorm` or `ln_f` | `norm` | **store_bf16** | yes, `codec=bf16` |
| rank-2 and none of the branches above | `other` | **compress** | yes |
| otherwise (unknown 1D/3D+) | `other` | **store_bf16** | yes |

Why skip on `inv_freq`/rotary, not a copy: these are not weights the schema GEMM eats; the cos/sin cache is reconstructed from `theta`. Extra blobs in `.chr` only confuse verify. Why unknown rank-2 goes to compress: better to wrongly compress a rare Linear than to drop a matrix.

Class **skip**:

- not in `.chr` → **not an error**, counter `skipped++`, no row in the table;
- is in `.chr` → **extra**, code 1.

### 3.2. Set contract (a choice, not an “or”)

Let `S_store(orig)` be the canonical names of class `compress` or `store_bf16`.
Let `S_chr` be the `tensors` keys in the CHR0 JSON.

- `S_store(orig) ⊆ S_chr` otherwise **missing**, code 1.
- `S_chr ⊆ S_store(orig)` otherwise **extra**, code 1.
- In total **equality**. All compressible orig **must** be in chr. Extra in chr is an error. Skip from orig does not require presence and has no right to appear.

Why not “skip missing Linear”: then `verify` would be green on a file that forgot `lm_head` (1.6 GB), and the kernel would crash or take garbage at inference. This is a file contract, not a quality threshold → code **1**, not 2.

verify work order (pseudo):

```
header ← ReadCHR0Header(chr)          # JSON in RAM, do not load blobs
visited ← ∅
for name in sorted(S_orig_keys):
    canon, class ← Classify(name)
    if class == skip: skipped++; continue
    if canon ∉ header.tensors: missing → err1
    W ← LoadOneOrigTensor(name)       # F32, release shard buffers
    Ŵ ← DecodeFromCHR0(header, canon) # F32, logical shape; one tensor
    compare shape; accumulate a report row
    visited += canon
    W, Ŵ = nil
if header.tensors − visited ≠ ∅: extra → err1
print report
if there was a bf16 mismatch or lossy > threshold: exit 2
else exit 0
```

Code 1 has priority over code 2: an incomplete name set is not masked by “also RMSE is large”.

### 3.3. Orig dtype → float32

Only `F32`, `F16` (IEEE binary16), `BF16` are allowed. Everything else — code 1.

Conversion to F32 is per-element, little-endian, as in safetensors. BF16: high 16 bits of float32, low zeros (standard expansion).

### 3.4. Reconstruct

- `nf4`: decode per [nf4.md](nf4.md) to float32, scale FP16→F32, trim padded `n_in` to logical.
- `vq`: `C1[i1]+C2[i2]` float32, codebook FP16→F32, trim pad.
- `bf16`: read the raw BF16 blob, each element → float32.

Padding does not enter the metrics.

### 3.5. Bit-exact for norm/bias

We compare `Ŵ` with the **BF16 projection of orig**, not with “raw F32 orig, if it happened to be stored that way”:

```
ref_i = float32( round_to_bf16( orig_f32_i ) )   # RNE
require Ŵ_i == ref_i   (bitwise as float32)
```

If orig is already BF16, `round_to_bf16` is identity, this is true bit-exact orig↔chr.

Why not compare with raw F32: the container stores lossless tensors as BF16 ([compressor.md](../compressor.md) §6.2). F32→BF16 loses the low mantissa bits; that is not a codec bug. **Synthetic tests write norm/bias as BF16**, then `ref` = orig.

Any inequality of `Ŵ` and `ref` → `ok=false`, code 2 (threshold “zero”), not code 1.

Lossy tensors **need not** land at the NF4 LUT / codebook level relative to orig: we compare only the §2 metrics with table A/B.

### 3.6. What we do not compare

- `.chr` bytes with orig safetensors.
- F32 decode with orig BF16 bit-exact for `nf4`/`vq`.
- Shard order, 64 alignment, JSON whitespace — that is level 1, container tests, not `verify`.
- There is no checksum in the file ([compressor.md](../compressor.md) §6.2). `verify` computes metrics itself.

---

## 4. `decode` writes F32

Frozen: **always F32**, no dtype-choice flag.

Rules:

- One output safetensors file (magic, JSON header, `data_offsets`, dtype `"F32"`).
- Even if orig was sharded — one file on output.
- Names = CHR0 keys (without `.weight`).
- `shape` = logical from CHR0 (e.g. `[32, 100]`, even though nf4 pack went with `n_in_padded=128`).
- Tensor order in JSON: lexicographic (Go `encoding/json` serializes a map that way).
- Write as a stream, do not gather all F32 in RAM: like CHR0, sizes are known first → `data_offsets` can be computed before writing the payload. One tensor in RAM (or a stripe, if someone calls decode on 8B — for 8B F32 lm_head ~2.1 GB; decode **may** cut write stripes, see §7; for v1 unit tests stripes in decode are not required, tensors are tiny).
- After decode: `chr verify` **does not** read this F32 file; verify is always orig safetensors vs `.chr`. Decode is for eyes / external scripts.

Why not BF16 on output: eyeball checks and numpy `float32` match what `internal/verify` computes. NF4 scales and the VQ codebook already live in FP16 inside `.chr`; F32 output does not make them more precise, it only does not hide an error in a second BF16 rounding.

---

## 5. Unit / golden tests without a model

Wave 2 slices packages and **these** functions 1:1. Fixtures: the test itself writes safetensors (helper `internal/safetensors` or a test-only writer). No internet, no `*.bin` in git.

Package layout (so CLI tests do not exec the binary):

- `internal/chr0`, `internal/safetensors`, `internal/nf4`, `internal/vq` — codecs and container.
- `internal/verify` — metrics, classifier, `Run(opts) Result` with `ExitCode int`.
- `cmd/chr` — flag parsing + calling `verify`/`compress`/`decode`. CLI tests: file `run.go` with `func run(args []string) int` in package `main`, `run_test.go` next to it. **Not** `os.Exit` inside `run`; `os.Exit` only in `main`.

Fixture helper (need not be a separate package): write a map `name → {dtype, shape, []float32}`. For bit-exact norm — dtype `BF16`.

Deterministic values without a global RNG: `w[i,j] = float32((17*i + 13*j) % 100)/50 - 1` (range about [-1, 0.98]).

Below is the required list. Name = name of `func TestXxx(t *testing.T)`. Subcases — `t.Run`.

### 5.1. `TestContainerRoundtripBlobs`  
package: `internal/chr0` (may duplicate via a `cmd/chr` call in a subcase)

**Arrange.** Build CHR0 with three tensors as in the chr0 example (q 64×64 nf4, down 32×128 nf4 or vq — does not matter for the container, all three can be `codec=bf16` so the codec is not pulled in): known payload bytes (e.g. 128 bytes of `0xA5`). Write the file. Open again via `ReadAt`.

**Assert.**

- `header.magic=="CHR0"`, `version==1`.
- For each tensor `ReadAt(start, end-start)` is **byte-for-byte** equal to the written slice (`bytes.Equal`).
- Re-reading the same offsets — the same hash/bytes (level-1 idempotence).
- Do not compare with orig BF16.

### 5.2. `TestNF4Golden`  
package: `internal/nf4`

**Arrange.** Vector/matrix and expected packed hex + FP16 scale **from `docs/spec/nf4.md`** (golden 1×64 or 2×64 — as they freeze there). Encode in memory, without `.chr`.

**Assert.**

- packed bytes = golden hex.
- scale bits = golden.
- `Decode(Encode(W))` matches a hand `lut[nib]*float32(s)` bitwise in F32.
- **Do not** require `Decode(Encode(W)) == W`.
- Zero matrix 2×64 (subcase `zero`): all nibbles = LUT zero index, scale=1 (as nf4.md).
- A CLI subcase is optional; the `nf4` package is enough. Level-2 integrity.

### 5.3. `TestVQToy`  
package: `internal/vq`

**Arrange.** Matrix `n_out=4`, `n_in=16` (8×2 groups per row): exactly **4 distinct** dim-8 vectors, each repeated (e.g. rows alternate v0..v3). `seed=0`, `iters=20`, `chunk=256`. Encode → codebook FP16 + index.

**Assert.**

- reconstruct `C1[i1]+C2[i2]` from the **written** codebook matches `Decode` bitwise (level 2; this is not orig).
- MSE(orig, reconstruct) ≤ `1e-4` (FP16 codebook; on 4 clusters k=256 must learn).
- `n_codebooks=2`, `codebook_bits=8`, `group_size=8` if looking through a CHR0 wrapper (subcase `via_chr` can be deferred to 5.6).

### 5.4. `TestVerifyFailCode`  
package: `internal/verify` + a mirror in `cmd/chr` (`TestRunVerifyExitCodes`)

**Arrange.** One compressible tensor, 8×64, values `w[i,j]` as the formula above (not a constant, not zeros). Compress `--codec nf4` in a temp dir. Two `verify.Run` calls:

1. `--fail-rmse 1e-12 --fail-maxabs 1e-12` (impassable for honest NF4 on this fixture).
2. the same files, table A defaults (or explicit 0.08 / 0.50).
3. CLI subcase: `run([]string{"verify", "--orig", …, "--chr", …, "--fail-rmse", "1e-12", "--fail-maxabs", "1e-12"})`.

**Assert.**

1. `ExitCode==2`, in the report `fail>=1`, `ok=false`, tensor name in `failed`. Not 1.
2. `ExitCode==0`, `fail==0`.
3. CLI returns 2, human stdout contains `FAIL`, stderr without `chr: missing`.

Additionally in the same function `t.Run("input_missing_orig")`: non-existent `--orig` → code **1**, not 2.

### 5.5. `TestPad`  
package: `internal/nf4` + `internal/vq` + one end-to-end in `internal/verify`

**Arrange.**

- NF4: `n_out=4`, `n_in=100` (100 % 64 ≠ 0). Orig F32/BF16. Encode with pad zeros to 128.
- VQ: `n_out=2`, `n_in=12` (pad to 16).

**Assert.**

- In metadata/CHR0 `shape` = logical `[4,100]` / `[2,12]`, not padded.
- `Decode` last-axis length = 100 / 12.
- verify metrics: `n=400` and `n=24`, not 512 / 32.
- Pad tail does not affect: if orig in columns 0..99 matched decode, maxabs is counted only there.
- A second encode of decode(W) on the logical piece need not match the first pack (lossy); for NF4 — second-encode idempotence as in nf4.md, if that test already covers it.

### 5.6. `TestFakeModelThreeTensors`  
package: `cmd/chr` (end-to-end) and/or `internal/verify`

**Arrange.** One safetensors, three tensors (names **without** `.weight`), as the tiny container example:

| Name | shape | dtype | class |
|---|---|---|---|
| `model.layers.0.self_attn.q_proj` | `[64, 64]` | BF16 | compress |
| `model.layers.0.mlp.down_proj` | `[32, 128]` | BF16 | compress |
| `model.norm` | `[64]` | BF16 | store_bf16 |

Fill q/down with the §5 formula; norm = 1.0. Two compress runs: `--codec nf4` and `--codec vq --seed 0`. Then `decode` to `out.safetensors` and `verify --orig` vs each `.chr`.

**Assert.**

- `S_chr` = three names, `kind` q / down / norm, codecs nf4|vq / nf4|vq / bf16.
- verify code 0 on table A defaults for the corresponding codec.
- `model.norm`: rmse=mae=maxabs=0, `codec=bf16`.
- `decode`: dtype F32, three tensors, logical shapes, `n` of elements matches; q/down **not** bit-exact with orig; norm — bit-exact with orig BF16→F32.
- `--json` parses, `summary.tensors==3`, `summary.bf16==1`, `summary.lossy==2`.
- Peak: in verify there is no moment when all three orig buffers are live at once (in the test it is enough to check the API contract: `Load` one at a time; no goroutines). A hard RSS assert in unit is not required.

### 5.7. More tests without a model (wave 2 — also required, otherwise the CLI is holey)

Names are frozen so they are not lost.

| Function | Package | Arrange | Assert |
|---|---|---|---|
| `TestVerifyMissingCompressable` | `internal/verify` | fake from 5.6, in chr delete `down_proj` (or compress and swap JSON — simpler: build chr from two tensors) | code **1**, message `missing tensor in chr: model.layers.0.mlp.down_proj`, not code 2 |
| `TestVerifyExtraTensor` | `internal/verify` | orig only `model.norm`; chr with q_proj + norm | code **1**, `extra tensor in chr: …` |
| `TestVerifySkipInvFreq` | `internal/verify` | orig: `q_proj` 8×64 + `model.layers.0.self_attn.rotary_emb.inv_freq` 1D; compress | inv_freq not in chr; verify code 0, `skipped==1` |
| `TestRunCompressRequiresCodec` | `cmd/chr` | `run({"compress","--in",p,"--out",q})` | code 1 |
| `TestRunHelpExitZero` | `cmd/chr` | `run({"-h"})` and `run({"verify","-h"})` | code 0 |
| `TestDecodeWritesF32` | `internal/safetensors` or `cmd/chr` | decode after 5.6 | header dtype of each tensor `"F32"` |

Do not duplicate NF4 golden hex here as numbers — source of truth is `nf4.md`. If at wave 2 `nf4.md` still has no hex, `TestNF4Golden` uses the zero/constant invariants from the nf4 requirements and must not `t.Skip` hex: then assert a zero matrix and a constant `c=0.5` (one level, `scale=|c|`).

---

## 6. After download (live run, no weights in the repo)

Weights are **not** committed. A local HF snapshot is assumed, variable `$MODEL` is a directory with `model.safetensors.index.json` (or one file).

Commands are an illustration; thresholds are **table B**, not the default:

```
# NF4, one file on disk
chr compress --in "$MODEL" --out /data/llama8b.nf4.chr --codec nf4 --quiet
chr verify  --orig "$MODEL" --chr /data/llama8b.nf4.chr \
    --fail-rmse 0.12 --fail-maxabs 2.0 --json > /data/llama8b.nf4.verify.json

# VQ 2×8 weight-only, same orig, different file
chr compress --in "$MODEL" --out /data/llama8b.vq2.chr --codec vq \
    --seed 0 --iters 20 --chunk 262144 \
    --stripe-bytes 268435456 --stripe-rows 4096
chr verify  --orig "$MODEL" --chr /data/llama8b.vq2.chr \
    --fail-rmse 0.50 --fail-maxabs 8.0 --json > /data/llama8b.vq2.verify.json
```

A full F32 dump is **not** needed for acceptance and would eat ~32 GB:

```
# only if there is space; not part of unit
chr decode --in /data/llama8b.nf4.chr --out /tmp/llama8b.nf4.f32.safetensors
```

Expectation per [compressor.md](../compressor.md) §7 (this is **not** the `chr` exit code):

1. 8B NF4: WikiText no worse than +1.0 to BF16; chat is coherent. `verify` table B must be PASS even before PPL — otherwise the pack is broken and PPL is about nothing.
2. 8B VQ: take PPL and record it; chat “alive/dead”. Dead chat with PASS `verify` means “the codec is honestly lossy”, not “the CLI needs a fix”.
3. Kernel comparison with llama.cpp Q4 — not this binary.

Compressor RSS/VRAM peak on a live 8B: one matrix + a stripe (§7). If `embed_tokens` / `lm_head` (~1.05 GB BF16) are processed without stripes at default `--stripe-bytes` — that is an implementation bug of §7.

---

## 7. Memory bounds

### 7.1. Hard rules (all commands)

1. **One open shard.** LRU=1 file. No `errgroup` over shards.
2. **One logical tensor** (or one of its stripes) in working buffers. After encode/decode/metrics the slices are released; do not accumulate `[][]float32` of all layers.
3. Do not hold orig BF16 and a full F32 copy of the **whole** matrix at once: convert a **stripe**.
4. CHR0 header (JSON) — entirely in RAM. That is megabytes, not gigabytes.
5. Output blobs: append to disk, not `[]byte` of the whole compressed model.
6. Goroutines over tensors are forbidden in v1 (simplicity of RSS accounting).

Peak we aim for on a CPU host with a live 32B lm_head (~1.56 GB BF16, `n_out≈128256`, `n_in=5120`):

```
F32 stripe 4096 × 5120 × 4 ≈ 80 MB
+ VQ index stripe + codebook 8 KB
+ chunk assignment 1e6 × 256 × 4 ≈ 1 GB  ← cut by --chunk
≈ 1–1.2 GB , not 1.56×4 GB F32 of the whole matrix
```

### 7.2. When to enable stripes

Let `n_out`, `n_in` be the logical matrix shape (2D). Working size of “the whole matrix in F32”:

```
full_f32 = n_out * n_in * 4
```

If `full_f32 > --stripe-bytes` **and** the tensor is 2D → stripe mode. Otherwise load the tensor whole (unit tests always whole: 64×64 F32 = 16 KB).

Stripe height:

```
rows = min(--stripe-rows, n_out)
shrink rows while rows * n_in * 4 > --stripe-bytes and rows > 1
if even rows=1 does not fit in --stripe-bytes — still one row (progress matters more than the limit)
```

For NF4 the group is along `n_in`: a stripe along rows **does not break** scales. Encode/decode of rows are independent. One pass over stripes, blobs `data`/`scale` are written sequentially row-major as in [compressor.md](../compressor.md) §6.2.

1D (norm) is never striped.

v1 unit tests **need not** exercise stripes (fixtures < threshold). A test of the threshold itself can be omitted without a 300 MB fixture. The implementation for a live 1.6 GB `lm_head` is **required** to be in compress/verify/decode.

### 7.3. VQ on stripes — two-pass codebook

k-means on a subset, assignment on all. Codebook **per matrix**, not per stripe.

**Pass I — init (subsample → Lloyd).**

Do not take only the first rows of `embed`/`lm_head`: those are special tokens, the codebook will be garbage.

Fixed subsample algorithm:

- `vecs_per_row = n_in_padded / 8`
- `target = min(N, 65536)`, `N = n_out * vecs_per_row`
- `rows_needed = ceil(target / vecs_per_row)`
- `stride = max(1, n_out / rows_needed)`
- take rows `0, stride, 2*stride, …` until `target` vectors are gathered (the last incomplete row — trim).
- On this set: k-means++ (`--seed`) and Lloyd `--iters` per [vq.md](vq.md). Obtain `C[0:M]`.
- Empty clusters — resplit as in vq.md, the same seed-stream.

This is **one** sequential orig-read pass (only selected rows; unselected ones are not converted to F32).

**Pass II — assign.**

- Codebook is frozen. All rows, in stripes.
- Assignment **in `--chunk` vector chunks**, do not materialize `(N, 256, 8)`.
- Write `index` in stripes at already-known offsets (`n_out` is known from the orig header).
- Residual of the second codebook: as in vq.md (after C1 on the vector), without a second training on full N in v1 stripe mode. (If the tensor is **not** striped — full residual k-means per vq.md, both trainings on all vectors.)

In total for a huge tensor: 2 orig reads (init subsample + assign). For a small one: 1 read, full vq.md algorithm.

`--chunk` only cuts the assignment/update GEMM-like loop, it is **not** stripe height. On one stripe 4096×14336 vectors = 7.3e6, the inner loop is still in packs of 1e6.

### 7.4. NF4 and stripes

One pass. Per stripe: F32 rows × `n_in`, pack, append `data` and `scale`, forget the stripe. No double-quant.

### 7.5. `verify` and stripes

The same thresholds `full_f32 > stripe-bytes`. Compute metrics **accumulatively** over stripes:

```
sum_sq_err, sum_abs, maxabs, n, sum_sq_orig
```

Do not gather a full `ŵ−w`. RMSE at the end from the sums. Otherwise 8B lm_head is again 2×2 GB.

Orig stripe and chr-decode stripe of the same height; row indices match.

### 7.6. What not to do

- Do not read two shards as prefetch “just in case”.
- Do not mmap 32B whole as a fallback path in v1 (on Windows/WSL it is `ReadAt` anyway).
- Do not hold F32 of the whole model “for JSON convenience”.
- Do not enable stripes in synthetic unit tests (the 256 MiB threshold will not hit them). Do not write a test that allocates 2 GB.

---

## 8. Interface with neighboring specs (fields, not bytes)

The CLI **does not** redefine the layout. After compress the following must result:

**nf4** (per tensor): `codec=nf4`, `group_size=64`, logical `shape`, blobs `data` uint8 `[n_out, n_in_padded/2]`, `scale` FP16 `[n_out, n_in_padded/64]`. No `zero`.

**vq:** `codec=vq`, `group_size=8`, `n_codebooks=2`, `codebook_bits=8`, `codebook` FP16 `[2,256,8]`, `index` uint8 `[n_out, n_in_padded/8, 2]`.

**bf16:** `codec=bf16`, `data` raw little-endian BF16, `shape` as orig.

`kind`, `layer` — by the classifier of §3.1. `tile` in the header root: `{ "row": 64, "col_group": 8 }` as in [compressor.md](../compressor.md) §6.1 (for the kernel; CPU-verify does not rearrange tiles, blobs are row-major).

If by wave 2 `chr0.md` clarifies header writing (two passes vs sidecar) — compress follows **it**. Here it is enough: after a successful compress the file is read by `internal/chr0` and passes level 1.

There is no checksum in `.chr`. The CLI does not write and does not check a neighboring `*.sha256`.

---

## 9. Exit codes and `run()`

| Code | When |
|---|---|
| 0 | the command did what it promised; for verify all required tensors were compared and thresholds/bit-exact are ok |
| 1 | flags, paths, dtype, NaN, name set, I/O, unsupported codec, group-size, holey CHR0 JSON, `header_nbytes` lies |
| 2 | verify only: name set matched, metrics computed, at least one tensor `ok=false` |

`ctrl-c` / `ctx.Done`: a separate code is not specified; the process will die by signal.

`internal/verify.Result`:

```
ExitCode int
Report   # the same graph as JSON §2.2
Err      error  # for code 1; for 0/2 may be nil
```

`cmd/chr` prints Report and returns `ExitCode`.

---

## 10. What wave 2 must not ask

- Default `--codec` — none, the flag is required.
- Decode dtype — F32.
- Missing Linear — error 1, not skip.
- Extra in chr — error 1.
- inv_freq — skip.
- Norms — bit-exact of the BF16 projection.
- nf4/vq — not bit-exact to orig; table A thresholds in the default, table B only in §6.
- Stripes — at `full_f32 > 256MiB`; VQ then init on a strided 64k subsample, assign over all.
- Tests — names §5.1–5.7, without an LLM.

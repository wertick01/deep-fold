# Floor 1 protocol: GPU-NF4 oracle, VRAM, and safety

Wave 2.0 acceptance from agent 4. If the kernel, loader, or host argue with this page about **thresholds and what FAIL means** — this page wins; about bytes and stitches, [stitch-gpu.md](stitch-gpu.md), [nf4.md](nf4.md), [chr0.md](chr0.md) win.

Floor 1 from [token-loop.md](../token-loop.md) §7.3 is **the kernel against CPU-decode of our own codec**. Floor 2 (KL against a BF16 dump, WikiText, greedy-match) is not this wave and not this file.

---

## 1. One command

```text
conda activate torch-gpu
python gpu/tests/oracle_gate.py --chr C:\dev\models\qwen25-3b.nf4.chr
```

Prints a check table, then `maxabs`, `rmse`, `smi_before`, `smi_after`, `display_reserved_mib`, and a verdict.

| Exit code | Verdict | What it means |
|---:|---|---|
| 0 | `PASS` | every required check passed; this *is* acceptance |
| 2 | `FAIL` | a numeric or safety invariant was broken (same as `chr verify`) |
| 3 | `BLOCKED` | a required check **could not** run (no `gpu/nf4`, no `.chr`, no CUDA) |

`3` is **not** green. It distinguishes “the kernel lies” from “there is no kernel yet”: those two states must not share one code, or the absence of a kernel looks like success.

Useful flags: `--name` (a different tensor), `--tol` (maxabs threshold), `--vram-extra-mib`, `--json <path>` (full machine-readable report), `--crosscheck-f32 <path>` (see §7).

Without `--chr`, self-checks and toy checks still run in a meaningful way; the verdict will be `BLOCKED`.

---

## 2. What the oracle actually computes

```
W_hat[r, c] = float32(LUT[nib(r, c)]) * float32(scale[r, c / 64])     # nf4.md §1, §3
Y_cpu       = W_hat @ x                                               # float32 matmul
```

- `LUT` — 16 literals from [nf4.md](nf4.md) §1, checked **bit-exact** (`struct.unpack` == the bits column).
- The low nibble of a byte = **even** `K` (not CUDA bitsandbytes packing).
- `scale` FP16 → float32 by exact expand, then multiply. Not “via BF16”.
- `x` is the same BF16 the kernel received, lifted to float32: input rounding is **not** charged to the kernel as error.
- `y_gpu` (BF16) is lifted to float32; comparison is in float64.

The oracle reads packed **from the file** with its own read-only parser ([chr0_min.py](../../gpu/tests/chr0_min.py)), not from the loader’s tensors. That lets us separately check that the loader’s VRAM holds exactly the disk bytes (X1), without tying the oracle to the code under test.

**Forbidden** to compare `y_gpu` with `F.linear` on the original BF16 weights: that is the quantization cost, a different floor. Original safetensors are not opened by this gate at all.

### Thresholds

| Quantity | Threshold | Why |
|---|---|---|
| `maxabs(Y_gpu − Y_cpu)` | **≤ max(0.05, ½ ULP BF16(\|Y_cpu\|))** at `rms(x) ≈ 1` | [stitch-gpu.md](stitch-gpu.md), token-loop §7.3 floor 1 |
| `rmse` | report, no gate | catches “bad everywhere” vs “bad in one element” |

An element above 0.05 when `|Y| ~ O(1)` is a **kernel bug**, not quantization. A peak of 0.058 on `y≈17` with the same answer as 2×n16 is BF16 rounding, not FAIL. Look first at: the `k >= K` mask on the tail, group scale `k/64`, nibble order, `cp.async` stage race at `K >= 512`.

Measured reference (RTX 3080, sm_86, torch 2.5.1, Qwen2.5-3B `model.layers.0.mlp.gate_proj`, `M=11008`, `K=2048`, `N=1`):

| Check | maxabs | rmse |
|---|---:|---:|
| F1 toy 128×256 | 0.003191 | 0.000620 |
| F2 toy 130×65 | 0.001399 | 0.000305 |
| F3 3B `gate_proj` | **0.015759** | 0.002654 |
| F4 3B, different seed `x` | 0.018568 | 0.002683 |

The oracle’s own noise (the same product, float64 accumulation vs float32) is `7.4e-06`, three orders of magnitude below the threshold: 0.0158 is the BF16 output and the kernel’s summation order, not numpy arithmetic.

---

## 3. Functional checks

| ID | What | PASS means |
|---|---|---|
| L0–L5 | LUT/packing/decode/encode against golden fixtures [nf4.md](nf4.md) §1, §5.2, §6.1, §8, §9.4 | the oracle is allowed to claim anything; FAIL here voids every number below |
| C0 | toy CHR0 writer against the byte gold [chr0.md](chr0.md) §2.4 | `N=516`, offsets and file size matched; S4 fixtures are legal |
| K0 | the kernel imports and launches | `gpu.nf4` found, one launch succeeded |
| K1 | the kernel binary is newer than its sources | the gate does not attest a stale `.pyd` (§6) |
| F1 | toy 128×256, `N=1` | `maxabs ≤ 0.05` |
| F2 | toy 130×65 (`K_pad=128`) | `maxabs ≤ 0.05`; and `y` is **bit-identical** for packed with garbage in the padding and with nibble-7 there |
| F3 | 3B `gate_proj`, `N=1` | `maxabs ≤ 0.05` |
| F4 | same `W`, different seed `x` | `maxabs ≤ 0.05` again, and the blake2b digest of `packed`/`scale` did not change |
| F5 | two calls in a row with the same `x` | `torch.equal(y1, y2)` — bit-stable |
| X1 | loader bytes == `.chr` bytes | `packed`/`scale` matched bytewise, `M/K/K_pad` matched |

F2 deliberately puts **random garbage** in columns `K … K_pad−1`, even though a real `.chr` writes nibble 7 there. That is stricter: if the kernel treats pad as live `K`, `y` will diverge. The nibble-7 variant is computed second and compared bit-identical to the first.

---

## 4. Safety checks

| ID | What | PASS means | FAIL means |
|---|---|---|---|
| S1 | `nvidia-smi memory.used` before/after `materialize` + one `gate_proj` | `Δ < sizeof(W_bf16)` **and** `Δ ≤ packed+scale+x+y + 20 MiB`, and the torch allocator peak is in the same budget | a scratch `W` or an extra arena appeared in HBM |
| S2 | no `[M, K]` fp16/bf16/fp32 tensor on device | no aten op produced a cuda float tensor of size ≥ `M*K`; no `cudaMalloc`/`malloc(` in `gpu/nf4/*.cu,cpp` | dequant went through HBM |
| S3 | `.chr` read-only | every `open` of this path is `"rb"`, `mtime_ns` and size did not change | the test mutates the artifact |
| S4 | corrupt header | 5 corruptions (truncation, `start` not a multiple of 64, overlapping blobs, `group_size=32`, `data` length ≠ formula) rejected **by both** the loader and the gate parser; **0** kernel launches | the loader trusts the header |
| S5 | NaN | behavior is pinned, no “silent” result (§5) | NaN was swallowed or leaked across rows |
| S6 | `packed` immutable after HtoD | digest and `data_ptr` did not change, `dtype=uint8`, no `.mul_` | the kernel mutates the input in place |
| S7 | display | `display_reserved_mib` in the log as a number (§8) | the figure was ignored |
| S8 | a full `chr decode` is not part of the test | an **AST scan** of the `gpu/tests` tree finds no process launches except `nvidia-smi` in `gpu_probe.py`; cross-check capped at 256 MiB | the test could dump 12 GiB |
| S9 | lying about `M`/`K`/`K_pad` | the Python boundary rejects undersized `packed`/`scale`, a wrong `K_pad`, a short/CPU/fp16 `x`; the context is alive after a series of refusals | OOB read past the end of the blob (§6) |

Measured reference, 3B `gate_proj`, `N=1`:

```
packed 10.75 MiB + scale 0.67 + x 0.004 + y 0.021 = legit 11.45 MiB
smi_before 1673 -> smi_after 1687, delta 14 MiB  (limit 31.45, W_bf16 = 43.0, W_f32 = 86.0)
torch peak allocated delta 12.07 MiB
load_s 0.024, gemm_ms 0.638
```

`Δ = 14` vs `legit = 11.45`: the gap is driver allocation granularity and the kernel module, not a second weight buffer. The spec threshold is “< 20 MiB above legit”, i.e. 31.45.

### smi resolution (why S1 is not for every matrix)

`memory.used` reports **reserved**, and torch reserves large blocks in ~20 MiB segments. Measured on `q_proj` `[2048, 2048]`: `Δ smi = 22 MiB` with `torch peak = 2.14 MiB` and `legit = 2.13`. There is no scratch `W` there — it is one fresh allocator segment.

Hence the gate rule:

- if `sizeof(W_bf16) ≥ 24 MiB` — **smi decides** (plus the allocator counter). The acceptance matrix `gate_proj` (43 MiB) falls here;
- if smaller — smi physically cannot tell `W` from segment granularity, `torch peak allocated` decides, and `Δ smi` stays in the log with a note. That way S1 does not become a false alarm on `q_proj` and does not lose its teeth on `gate_proj`.

Weakening the threshold for large matrices is forbidden: there `Δ smi` resolves `W` confidently.

---

## 5. NaN: pinned behavior (S5)

The kernel is **not required** to catch NaN. The gate records what happens and forbids two specific scenarios: silent swallowing and leakage across rows.

| NaN source | Observed | Norm |
|---|---|---|
| `x[k] = NaN`, `N=1` | **all** `M` elements of `y` are non-finite | that is how it should be: every row sums over all of `K`. FAIL if `y` is entirely finite — that means NaN was swallowed |
| `scale[r, g] = NaN` (synthetic) | only row `r` is non-finite; the other rows are finite | FAIL if other rows are infected: row `y[i]` reads only `W[i, :]`, anything else is a race or OOB |

A NaN scale **cannot** come from disk: [nf4.md](nf4.md) §6.3 makes a non-finite scale an encode error, and `+0` is forbidden. So the second table row is a synthetic tensor, not a file. If such a scale ever appears in a real `.chr`, the compressor is at fault, not the kernel.

The kernel does not raise a separate “there was a NaN in `y`” flag and must not in this wave: the cost is a branch on every MMA. The weaker guarantee is stated as: **a non-finite input yields a non-finite output, not a silent zero**.

---

## 6. Found and fixed: OOB at the Python boundary (S9)

The `gpu/nf4/bindings.cpp` boundary checked dtype, device, and `x.size(0) == K`, but **did not** check that `packed`/`scale` even cover the claimed `M` and `K_pad`. The kernel indexes `packed` with stride `K_pad/2` up to row `M−1`, so an undersized blob read past the end of the allocation.

Repro before the patch: `packed`/`scale` for 64 rows, claimed `M=128` → **no exception**, the kernel read ~8 KiB past the end of the buffer and returned plausible numbers. That is more dangerous than a crash: the test would have “passed”.

The patch (agent 4, host-side validation only, `.cu` logic untouched) — three `TORCH_CHECK`s in `nf4_gemm`: `K_pad == 64*ceil(K/64)`, `packed.numel() >= M*K_pad/2`, `scale.numel() >= M*K_pad/64`. The test is S9. This is the only reason `gpu/tests` is allowed to patch `gpu/nf4` at all: OOB/safety, with a test, no “speedups” and no number-fudging.

### K1: why the gate checks the binary’s date

`torch.utils.cpp_extension` keeps the build version **in-process**. If a rebuild fails (no `cl.exe`/`ninja` on PATH), a later import can silently load the previous `.pyd` — and every gate number will describe code that is not in the repository. K1 compares the loaded binary’s mtime with `bindings.cpp`, `nf4_gemm.cu`, `chr_gpu.h` and fails the gate if the binary is older.

Rebuild on Pavel’s machine (both MSVC and ninja must be on PATH):

```text
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
conda activate torch-gpu
python gpu/nf4/setup.py build_ext --inplace
```

`ninja.exe` lives in `<env>\Scripts`, so without `conda activate` the JIT build cannot find it; the gate adds `<prefix>`, `<prefix>\Scripts`, `<prefix>\Library\bin` to `PATH` just in case.

---

## 7. Coverage bounds (what this gate does **not** prove)

- **`nvidia-smi` is the source of truth for VRAM**, and it is shared across the card: another process, the window manager, or a second agent moves `memory.used`. The gate takes the median of three samples and measures the delta around a narrow window, but under concurrent work on the same card S1 can wobble. Re-run alone.
- **The S2 watcher sees only aten ops.** A direct `cudaMalloc` inside the C++ extension is invisible to it; only smi (S1) and the source scan catch it. Hence the rule: `Δ smi` is the gate, the watcher is a refinement.
- **S3 cannot forbid a write mmap** from foreign code: it patches `builtins.open`/`os.open` and checks `mtime`/size. That will catch an actual write; it will not catch a “right to write”.
- **Two threads/two streams on one matrix are not checked** (S6 in the spec does not require this either): `packed` is treated as immutable after HtoD, which the digest confirms.
- **`N > 1` (prefill) is not checked by this gate**: `oracle_gate.py` runs decode only. Prefill is a separate file `gpu/tests/oracle_prefill.py`, see §11.
- **SASS is not read** (no `cuobjdump` in this environment): “there is `HMMA.16816`” is agent 2’s check, not this one.
- **Floor 2 is absent**: no KL, no PPL, no greedy-match, no `generate`.

### Optional: X2, check against an already existing F32 dump

`--crosscheck-f32 <file>` reads **one** tensor from an already-on-disk `chr decode` dump (hard cap 256 MiB per slice) and checks it against numpy-decode of the same `packed`. The point: prove that the Python oracle matches the canonical Go implementation, not only itself.

Measured on `qwen25-3b.nf4.f32.safetensors`, `gate_proj`: **bit-identical**, `maxabs = 0`, slice 86 MiB.

The check is **optional** and does not affect the verdict: the dump is not a required artifact. Creating it from the gate is forbidden (§S8): 12 GiB on this disk is not part of the test.

---

## 8. Negative controls: the gate has teeth

A gate that never failed proves nothing. Verified by swapping the kernel via `DEEPFOLD_NF4_GEMM="module:attr"` (a wrapper around the real `nf4_gemm`, file outside the repo):

| Swap | Result |
|---|---|
| `y[0,0] += 0.30` | F1, F2, F3, F4 → FAIL (`maxabs ≈ 0.300`), exit 2. F5 stayed PASS — the shift is deterministic, and that is correct |
| `torch.empty(M, K, bf16, device)` before the honest call | S1 → FAIL (`Δ smi = 100 MiB`, `torch peak = 100.07`), S2 → FAIL (caught `aten.empty.memory_format (11008, 2048) torch.bfloat16`). **F3 still PASS**: the numbers are right, but a scratch `W` in HBM is still a fail |
| undersized `packed`/`scale` (before the §6 patch) | the kernel silently read past the end of the buffer, no exception → this became S9 |

The second case is the main one: it shows that the safety half catches a hidden dequant **even when the math agrees**. Do not set `DEEPFOLD_NF4_GEMM` in an acceptance run.

---

## 9. How to read `display_reserved_mib`

Per the spec this is `smi_used − torch_allocated` after an empty context, so the number is large and **includes more than the display**:

```
display_reserved_mib : 1675.0   (smi_boot=1415 MiB — display + foreign processes, cuda_ctx=260 MiB)
```

- `smi_boot` — `memory.used` before the gate created a context: display, browser, foreign sessions.
- `cuda_ctx` — `smi_after_ctx − smi_boot`, the cost of one empty torch CUDA context (~260 MiB on this machine).
- The 12 GB planner must budget **both**: on a 3080 with a monitor attached, “free” ≈ `12288 − smi_boot − cuda_ctx`.

The number is always printed; ignoring it is a separate FAIL (S7).

---

## 10. Ownership

Agent 4 owns `gpu/tests/` and this file. Edits in `gpu/nf4`, `gpu/chr0`, `gpu/host`, Go packages, notebooks — read and import only, with the single exception of §6 (OOB + test). “Fudging” the LUT, the scale, or a threshold so something matches is not allowed: then the gate stops being a proof and becomes decoration.

Section §11 and the file `gpu/tests/oracle_prefill.py` — agent 9 (wave 3). The same ban on “fudging” applies there.

---

## 11. Floor 1 for prefill: `N > 1` (wave 3)

The same floor, the same threshold, a different dimension. The oracle does not change: `W_hat` is CPU-decode of our own bytes, `Y_cpu = W_hat @ X` in float32, `X` is now `[K, N]`. Everything §2 says about the LUT, nibbles, scales, and float32 still holds; what is new is only masking on `N` and a second tail.

```text
conda activate torch-gpu
python gpu/tests/oracle_prefill.py --chr C:\dev\models\qwen25-3b.nf4.chr
```

Exit codes are the same: `0` PASS, `2` FAIL, `3` BLOCKED. Useful flags: `--n` (prefill width, default 16), `--n-tail` (tail, 3), `--n-over` (one above the kernel cap, 17), `--allow-skip`, `--no-l1`, `--json`.

### 11.1 Checks

| ID | What | PASS means |
|---|---|---|
| P0 | `oracle_gate.py` in full, **by import** | `main()` returned 0; all of wave 2 (N=1) is in place |
| K2 | kernel `N` range | `N=16` is computed; `N` above the cap is **refused or correct** (see §11.3) |
| P1 | toy 128×256, `N=16` | `maxabs ≤ 0.05` |
| P2 | toy 130×65 (`K_pad=128`), `N=3` | `maxabs ≤ 0.05`; pad along `K` did not enter `y`; pad along `N` did not enter live columns |
| P3 | 3B `gate_proj`, `N=16` | `maxabs ≤ 0.05` |
| S | `Δ smi` around one prefill | `Δ < sizeof(W_bf16)` and `Δ ≤ legit(N) + 20 MiB`, no cuda-float `≥ M*K` |
| P4 | `N=1` after prefill edits | `maxabs ≤ 0.05` **and the number matched F1/F2/F3 from P0 exactly** |
| L1 | `gpu/loop/smoke.py`, if present | “Paris” obtained, tok/s recorded; otherwise SKIP (§11.4) |

Be careful with the ID `L1`: in `oracle_gate.py` it is a LUT self-check (§3), in `oracle_prefill.py` it is the token-loop smoke. The name came from the wave 3 spec; both show up in one log because P0 prints the whole gate table. Look at the check text, not the letter.

**P0 is launched by import, not as a process.** `subprocess` from `gpu/tests` is forbidden (S8), and `oracle_prefill.py` itself passes the same AST scan. So `oracle_gate.main(argv)` is called in this same process, and its `--json` is read back — that is where P4 takes the N=1 reference numbers.

**P4 compares against the number, not the threshold.** The same seed gives the same `x`, the same `W`, and the same kernel, so `maxabs` must match *bit-exact*, not “also pass 0.05”. That is the regression sensor: a prefill edit that quietly shifted the `N=1` path shows up here even if 0.05 still holds.

**P2 checks two independent tails.** Tail along `K`: columns `K … K_pad−1` are filled with garbage and, on a second run, with nibble 7 — `y` must be bit-identical (like F2, §3). Tail along `N`: the same `x` is expanded by one “wild” column (`137.0`) to `N+1`, and the first `N` columns of the result must not move. The kernel pads `N` to 16 internally, so the epilogue mask is exactly where `N=3` breaks unnoticed.

### 11.2 Measured

RTX 3080, sm_86, torch 2.5.1 / cu124, Qwen2.5-3B `model.layers.0.mlp.gate_proj` (`M=11008`, `K=2048`), kernel with prefill tile `BM=64, BN=16, BK=128`:

| Check | `maxabs` | `rmse` |
|---|---:|---:|
| P1 toy 128×256, `N=16` | 0.002756 | 0.000662 |
| P2 toy 130×65, `N=3` | 0.001829 | 0.000363 |
| P3 3B `gate_proj`, `N=16` | **0.018726** | 0.002676 |
| P4 `N=1` (F1 / F2 / F3) | 0.003191 / 0.001399 / 0.015759 | — |

`0.0187` at `N=16` vs `0.0158` at `N=1` (F3) is the same nature: BF16 output and summation order, just the max is taken over 16 columns instead of one. The threshold is untouched.

Prefill was bit-identical to per-column decode: on P1 `maxabs(y_prefill − y_column-by-column) = 0.000000`. That is not a requirement (two tilings are allowed to round differently), but useful news: `N=16` is not “almost the same”, it is exactly the same.

VRAM and time:

```
S: legit(packed 10.75 + scale 0.67 + x 0.062 + y 0.336) = 11.82 MiB
   smi_before 2148 -> smi_after 2160, delta 12 MiB  (limit 31.82, W_bf16 = 43.0)
   torch peak allocated delta 12.67 MiB, cuda-float tensors >= M*K: 0
   prefill N=16: 0.765 ms => 0.048 ms/column  (decode N=1: ~0.70 ms/column, ~15x)
```

`x` and `y` grow exactly by `N` (0.004 → 0.062 and 0.021 → 0.336 MiB), `packed`/`scale` do not grow at all — there is no scratch `W` in HBM, and that is the main thing S was supposed to show. The smi-resolution rule from §4 applies unchanged: `W_bf16 = 43 MiB ≥ 24`, so smi decides here.

### 11.3 Cap on `N` and a “silently wrong” answer

`nf4_gemm.cu` launches at most `N=16` per call and expects the host to slice a wider prefill; `N=17` is refused (`-2`, and at the Python boundary — `RuntimeError: N=17 not in 1..16`). K2 accepts **either of two honest answers**: an exception (“slice it yourself”) or a correct `y` (“the cap was raised”, checked against the oracle). The third is forbidden: a plausible `y` computed by a tile that only covers 16 columns. That is the one that would poison prefill silently — like the undersized `packed` in §6.

### 11.4 L1 is a witness, not a gate

If `gpu/loop/smoke.py` exists, it is **imported** (again: no child processes) and called with a substituted `sys.argv`; its `sys.exit()` is caught. Without that, a foreign `argparse` would kill the gate before the verdict is printed.

Measured (graph=linears, 64 tokens after 16 warmup, two runs): `decode_tok_s = 12.4` both times, `prefill_5_ms = 96.7 … 97.9` at `prefill_chunk = 16`, `N>1 = True`.

L1 is PASS on `greedy_en_paris`, not on smoke’s overall exit code: its `rc` includes gates this oracle does not own (the RU chat template, for example), and a foreign tokenizer failure must not redden prefill. Smoke’s overall verdict and its gate line are printed in full — nothing is hidden. `--no-l1` turns L1 off: it loads the whole model (~20–40 s).

**L1 VRAM numbers inside this gate are not canonical.** Smoke lives in the same process as the oracle, so its `vram_decode_mb` includes the gate’s context and blobs (measured 3849 and 5703 MiB in two runs), and its own leak gate `smi_flat_16_to_64` wobbles from foreign work on the same card (§7). Prefill VRAM is S’s job, which measures a narrow window before the model is loaded. The real smoke figure must be taken from a separate run, not from here.

### 11.5 If the kernel still only does `N=1`

Then P1–P3 have nothing to launch. The script prints **who exactly** refused — the `gpu/nf4/__init__.py` wrapper or the `.cu` itself — and exits with code `3`. That distinction is not cosmetic: the wrapper can forbid `N>1` after the kernel has already learned it, and then “prefill does not work” is untrue.

With `--allow-skip`, P1–P3 become SKIP, and the remaining gates are P0, P4, and S: `N=1` regression and the absence of a scratch `W` are still checked. Without the flag such a run is **not green**, exactly by the logic of §1: “there is no kernel yet” must not be shown as success.

While the kernel was decode-only, the script still printed numbers for `N=16`/`N=3` assembled from an `N`-fold `N=1` call (rows `R1`–`R3`). That is **not** prefill, it does not affect P1–P3, and it is marked as reference: the point is to check the oracle itself for `N>1` and to know in advance which number must come out. The then-measured `0.002756 / 0.001829 / 0.018726` matched what the real prefill tile later produced.

### 11.6 What this file does not prove

- **Writes past the end of `y` are not checked.** The ABI returns a `y` allocated by the kernel itself, so there is nothing to put as guard bytes past the last column. It is caught only indirectly: the `N` tail (P2) checks reads, not writes.
- **Slicing `N > 16` is not checked end-to-end.** K2 pins that the boundary honestly refuses; that the host slices correctly is a `gpu/loop` question, not this file’s.
- **Everything in §7 still holds**: smi is shared across the card, the watcher sees only aten ops, SASS is not read, floor 2 (KL/PPL/greedy-match against BF16) is still absent here.
- **`N` between 4 and 15 is sampled** (`--n-tail`): 1, 3, 4, 16, 17 were checked. There is no full sweep over `N`.

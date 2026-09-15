# Plan: VQ residency (H3) up to 32B on an RTX 3080 12 GB

**Status 2026-09-14: H3 is closed on quality at 3B.** Occupancy/prefill VQ do not fix chat. `compress --codec auto` **no longer** picks VQ.

H2-6 (live 32B overflow) — PASS: 2.31 tok/s, smoke 3/3. Live plan and numbers —
[`docs/plan-h2-ring.md`](plan-h2-ring.md), plate
[`docs/img/h2-qwen25-32b.png`](img/h2-qwen25-32b.png). This file remains a
VQ tombstone, not an instruction “download 32B under 2 bits”.

Live canary `C:\dev\models\qwen25-3b.vq2.chr` (738 MiB, `codec=vq`): kernel = CPU reconstruct (`verify_vq` V7 PASS), greedy collapses to “ll”. Not a load bug.

| What was VQ-quantized, rest BF16 | Smoke Paris/Berlin/323 |
|---|---|
| nothing (BF16) | 3/3 |
| only `gate_proj` (36 matrices) | **0/3** |
| only `up_proj` | 2/3 |
| only `down_proj` | 2/3 |
| all attention qkvo | 2/3 |
| mlp gate+up+down | 0/3 |
| all Linear, embed BF16 | 0/3 |

Python residual k-means on one `gate_proj`: rel_mse ≈ 0.12 (row/column scale does not save it). NF4 on the same slices cosine ≈ 0.996. Format 2×8 without scales does not hold SwiGLU.

Pinned H2D on this WDDM machine: **24.3 GB/s** on 256 MiB (pageable 8.0 GB/s). H2 with pin makes sense; without pin — no.

`--codec vq` remains for the kernel oracle. 32B that must speak: NF4 overflow (H2), not 2-bit.

Hand out to agents **one wave at a time**. Do not start the next until the previous gate is closed. Do **not** start waves 3–6 for H3.

Goal that is still alive: 27B/32B on 12 GB that **answer in chat** faster than the 1–2 tok/s floor. That is H2, not H3. Not a goal: Gemma, 70B, “faster than Marlin”.

---

## Paste into the agent’s first message

```
Repo C:\dev\deep-fold. Read docs/plan-vq-resident.md in full, then
only your wave (named below). Do not start neighboring waves. Do not call Task /
Cloud / *-pro. Python: C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe
(conda env torch-gpu). Card: RTX 3080 12 GB, sm_86 only.

Wave: <N — name>
Do only its Definition of Done. If the gate is red — stop and write
why; do not “also fix” the next wave.
```

Fill in the wave number. One chat = one wave.

---

## Frozen facts (do not dispute)

Stage A (NF4) is **done and measured** on this card. Do not rewrite the NF4 kernel “so VQ is easier”.

| Fact | Number / place |
|---|---|
| 3B NF4 decode | 28.4 tok/s paired, TTFT 139 ms. `C:\dev\models\qwen25-3b.nf4.chr` |
| 14B NF4 | 6.56 tok/s, weights 7 483 MiB, on card. `docs/runs/qwen25-14b/` |
| 20B NF4 | 5.01 tok/s, weights 10 062 MiB, peak smi 11 828 / 12 288, headroom 460 MiB. `docs/runs/internlm20b/` |
| Runtime overhead for `codec auto` | 1 800 MiB (`gpu/cli/codec.py`) |
| 32B NF4 | does not fit (−6.1 GiB). 32B VQ 2×8 + embed INT4 | fits, leftover ~3.1 GiB. `docs/vram-3080.md` |
| CLI `compress`/`run --codec auto` | NF4 if it fits, else refuse. VQ only `--codec vq` (oracle). `gpu/cli/test_codec.py` |
| VQ GPU | `CompressedVqLinear` + `gpu/vq/vq_gemm.cu`. **N=1 only**. `BM=128`, `split_k=1` |
| Live `.vq2.chr` on disk | `C:\dev\models\qwen25-3b.vq2.chr` (738 MiB). Smoke FAIL (gate_proj) |
| Gemma / Phi-3 / MoE / vision / GGUF | refuse until walk. No 27B in the catalog: size target is **Qwen2.5-32B** |
| RAM→VRAM ring (H2) | **unlocked**. Pinned H2D 24.3 GB/s. See end of file |

Invariant of the whole plan: **HBM has no dense `[M,K]` BF16 layer**. Dequant only in registers / smem. If after load the VQ working set ≈ BF16 — a bug, not a win.

Honesty as in the README: tok/s VQ vs NF4 — different stacks only if the loop changes; vs BF16 `generate` do not rank as a kernel benchmark; do not cite `nvidia-smi` on 14B/20B/32B as “it fit”.

---

## Environment

```
conda activate torch-gpu
cd C:\dev\deep-fold
python -m gpu.cli doctor
```

| | |
|---|---|
| Python | `C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe` |
| Models | `C:\dev\models\Qwen2.5-3B-Instruct`, `Qwen2.5-14B-Instruct`, `internlm2_5-20b-chat` |
| NF4 chr | `C:\dev\models\qwen25-3b.nf4.chr` and neighbors |
| Compressor | `chr.exe` in the repo root or `DEEPFOLD_CHR_BIN` |
| Commits | only if Pavel asks explicitly. Conventional commits. Do not `--amend` others’ history |
| Runs | live CSV in `C:\dev\models\runs\...` (outside git). Copy into `docs/runs/` only on request, do not overwrite old plates with a fixture |

Unit tests without GPU (must stay green after every code wave):

```
python gpu/cli/test_codec.py
python gpu/cli/test_cli.py
python gpu/loop/test_attach.py
python gpu/host/test_attach.py
python gpu/chr0/test_chr0.py
```

With GPU, after waves 2–3:

```
python gpu/host/verify_vq.py
python gpu/vq/verify.py
```

---

## Do not (refuse)

- `cudaMallocManaged`, oversubscribe, “let Windows page it”. 14B BF16 already showed 0.92 tok/s. H2 — pinned + two slots only.
- Write occupancy/prefill VQ “so 32B starts talking”. 3B VQ smoke is red; the kernel has nothing to do with it.
- Dequant a layer into HBM, Huffman, 2¹⁶ codebook, `wmma`, non-Ampere arch.
- Download Qwen2.5-32B (~65 GB) without an explicit “yes” from Pavel. `docs/models.md` deliberately does not take it.
- Change Go packages `internal/*` unless the wave is about the compressor. CHR0 VQ format already exists.
- Raise NF4 `LIVE_MAX_N` “while we’re at it”. NF4 live is already 32 (`gpu/nf4/plan.py`). VQ has its own `vq_max_n`.
- Edit README numbers without a live CSV.
- Launch `Task`, Cloud Agents, `*-pro`, best-of-n. Work in this chat, `model inherit`.
- “Also fix” a neighboring wave if your own gate is red.

---

## Waves

### Wave 0 — orientation (15 min, no code)

Read: this file, `docs/spec/vq.md` §0–7, `docs/spec/stitch-gpu.md`, `gpu/cli/codec.py`, `gpu/host/vq_linear.py`, `gpu/vq/vq_gemm.cu` (header + `chr_vq_gemm`), `gpu/loop/graph.py` (`vq_max_n`).

**DoD:** in the agent’s reply: 8–12 lines “where VQ is now / what is missing / what is my wave’s gate”. No patch.

---

### Wave 1 — first live `.vq2.chr` on 3B

**Why.** Without a file there is no e2e. The kernel already does N=1: that is enough for the model to talk. TTFT will be bad — expected, not a wave-1 bug.

**Files.** Not the kernel. Allowed: `docs/models.md` (a line about VQ 3B), nothing in `gpu/vq/`.

**Commands.**

```
conda activate torch-gpu
cd C:\dev\deep-fold
python -m gpu.cli compress --in C:\dev\models\Qwen2.5-3B-Instruct --codec vq --out C:\dev\models\qwen25-3b.vq2.chr
python gpu/host/verify_vq.py --chr C:\dev\models\qwen25-3b.vq2.chr
python -m gpu.cli run --model C:\dev\models\Qwen2.5-3B-Instruct --chr C:\dev\models\qwen25-3b.vq2.chr
```

Smoke — three prompts from the README (Paris / Berlin / 323). Record: packed MiB, smi after load, TTFT, decode tok/s, smoke pass/fail. Comparing to NF4 of the same session is **not required** (isolate processes, 12 GB).

**DoD.**

- File `C:\dev\models\qwen25-3b.vq2.chr` exists, header `codec=vq`.
- `verify_vq.py` V7 (real matrix) PASS.
- `run` prints a decode-only warning and still answers; Paris/Berlin/323 smoke passes.
- In the reply: a table of numbers. Do not invent tok/s.

**Stop.** Compressor crashes / host OOM / smoke fails → do not go to 14B. Do not touch kernel occupancy in this wave.

**Expect.** Compress 3B VQ (residual k-means, CPU) — minutes–tens of minutes, not seconds. Do not kill it if tensors are progressing.

---

### Wave 2 — VQ decode occupancy (like NF4 split-K)

**Why.** `vq_gemm.cu` currently `BM=128`, `split_k=1`: on 3B `q_proj` too few CTAs for 70 SMs. NF4 already fixed this (`gpu/nf4/plan.py` SMALL + split-K, target ~140 CTA). Without this, 32B VQ will stay “sort of talks, 3–5 tok/s”.

**Files (only these, plus tests).**

- `gpu/vq/vq_gemm.cu`
- `gpu/vq/__init__.py` / `bindings.cpp` if the launch ABI changes
- plan mirror if needed: new `gpu/vq/plan.py` **modeled on** `gpu/nf4/plan.py`, not a copy of NF4 group=64
- `gpu/vq/verify.py`, `gpu/host/verify_vq.py`
- `gpu/include/chr_gpu.h` only if there is no other way; do not break the NF4 ABI

VQ group = 8, codebook 8 KiB in smem. Do not drag in the NF4 LUT.

**Do.** Decode `N=1`: `BM=64` (or whichever tile gives ≥70 CTA on 3B `q_proj` 2048×2048 and on GQA `k_proj`), split-K like NF4. Prefill **not** in this wave: `N!=1` still -2 / exception.

**Checks.**

```
python gpu/host/verify_vq.py
python gpu/vq/verify.py
```

Reference: `Y_cpu = reconstruct_vq @ x` (float32), `maxabs(Y_gpu − Y_cpu) ≤ 0.05` at rms(x)≈1. This is a kernel bug, not quantization. Do not compare to BF16 safetensors.

**DoD.**

- Synthetic + 3B `q_proj` from `.vq2.chr` (if wave 1 is not there yet — synthetic is required, 3B is a bonus).
- V5 lives: no `weight` parameter `[M,K]`.
- V4 lives: `N!=1` still refuses (prefill is wave 3).
- If there is a GPU: ncu on 3B `q_proj` N=1 is not required; if there is — occupancy above “wave-2 NF4 starvation”. Do not publish tok/s from ncu.

**Stop.** maxabs > 0.05 on a real matrix → do not mask with a threshold. Do not enable prefill “to check occupancy”.

**Parallel.** Can run at the same time as wave 1: different files. Conflict only if both touch `verify_vq.py` — then wave 2 owns verify.

---

### Wave 3 — VQ prefill `N>1`

**Waits:** wave 2 green (decode occupancy + oracle).

**Why.** TokenLoop currently walks the prompt one token at a time (`vq_max_n` catches N=2 and returns 1). On 32B the first token will be unbearable.

**Do.** The same N belts as NF4 live, without blindly copy-pasting BK/group:

| N | meaning |
|---|---|
| 1 | decode, wave 2 |
| 2..8 | BN=8 |
| 9..16 | BN=16 |
| 17..32 | only if decode+ n16 oracle is green; otherwise leave host chunk ≤16 |

`gpu/vq/__init__.py` must not `raise` on N>1 once the kernel can. `vq_max_n` — the same trick as `nf4_max_n`: probe launch N=2, return `probe` (currently probe comes from NF4 `LIVE_MAX_N`; give VQ its own ceiling, without breaking NF4 32).

**Attention.** `verify_vq.py` **V4** currently requires N≠1 to fail. After prefill: V4 = “N=16 matches the oracle”, plus a separate refuse on N>cap. Update the test, do not delete it.

**DoD.**

- Oracle N=1 and N=16 (and N=cap) ≤ 0.05 maxabs on toy and on 3B `q_proj`.
- `TokenLoop` on VQ 3B: `prefill_chunk > 1`. `run` no longer prints “walked one token at a time”.
- CLI/loop unit tests green. `linear_max_n("vq")` does not import the NF4 kernel for nothing and vice versa.

**Stop.** n32, if maxabs > 0.05 — leave cap=16, as NF4 did with n32 numerics. Do not raise TokenLoop above a green oracle.

---

### Wave 4 — remeasure 3B VQ after the kernel

**Waits:** waves 1 and 3.

**Why.** Separate “VQ format is slow” from “starved kernel”.

**Do.** Isolated process, same three prompts, `max_new_tokens=64`, greedy. Not at the same time as NF4 on 12 GB.

Record in `C:\dev\models\runs\vq-qwen25-3b-<date>/`: `summary.csv`, `messages.csv`, notes (codec=vq, prefill_chunk, smi, torch reserved).

Compare to live NF4 3B (28.4 tok/s / 139 ms paired) **in the report text**, do not mix CSVs into one folder.

**DoD.**

- Smoke PASS.
- Decode tok/s and TTFT named. Goal is not “beat NF4”. Goal: decode is not a catastrophe (guide: not worse than ~0.5× NF4 3B without explanation), TTFT is not “a minute for a 20-token prompt”.
- Packed weights ~ 3B × 2 bit ≈ half of NF4 1 563 MiB (~0.8 GiB), not 5.9 GiB BF16.

**Stop.** Smoke fail or working set ≈ BF16 → residency bug, fix the host, not 14B.

---

### Wave 5 — 14B VQ quality vs NF4 (hard eval)

**Waits:** wave 4, 3B smoke green.

**Why.** 32B will be VQ only. If 2 bits kill chat on 14B — 32B is pointless, need the H2 hatch (four bits + tail).

**Do.**

```
python -m gpu.cli compress --in C:\dev\models\Qwen2.5-14B-Instruct --codec vq --out C:\dev\models\qwen25-14b.vq2.chr
```

Then the same 12-item hard eval as `docs/eval-hard-qwen25.md` / `gpu.lab.hard`, greedy, `max_new_tokens=256`, isolated worker. Codec **vq only** (14B BF16 not required; 14B NF4 is already 10/12).

**DoD.**

- Table of 12 items: hit/miss vs NF4 14B (10/12).
- Mean TTFT, decode tok/s, peak smi, torch reserved.
- Gate verdict (below).

**Gate (hard).**

- H3 plan PASS: no worse than **8/12** on this sheet **and** Paris/Berlin/323 smoke. 12 items are a regression, not MMLU.
- H3 FAIL: ≤6/12 or the model does not speak coherently → **do not download 32B**. Write a report. Do not start H2 yourselves: wait for Pavel.
- Grey zone 7/12: stop, ask Pavel.

Do not claim “lossless quantization”, even if VQ ≥ NF4 on 12 items.

---

### Wave 6 — 32B resident VQ (only after wave 5 PASS + “yes” to download)

**Why.** This is `docs/schema.md` stage B and the answer to “27B on 12 GB”. There is no Gemma-27B in the stack. Qwen2.5-32B is the named size.

**Disk.** ~65 GB safetensors. `docs/models.md` currently forbids it. The agent does **not** download itself. Pavel downloads or writes “download”. Compress **tensor by tensor from disk**, not `from_pretrained` whole on GPU. Peak RAM — one float32 tensor (+ k-means working buffers), see `docs/spec/vq.md` / `docs/cpu-roundtrip.md`.

```
python -m gpu.cli compress --in <32B-dir> --codec auto
# auto on 12 GB must pick vq (gpu/cli/test_codec.py already checks this on shapes)
python -m gpu.cli run --model <32B-dir>
```

**DoD.**

- `decide()` = vq. File `.vq2.chr`.
- After load: packed ~8 GiB, smi < 12288 with headroom for KV, **torch reserved ≈ packed, not 62 GiB**.
- Paris/Berlin/323 smoke PASS.
- Decode tok/s recorded. Guide “like 20B NF4” ~5 tok/s; the schema wanted ≥10 — that is **after** the kernel, not a wave-6 promise.
- Hard eval 12 items — desirable, not a smoke blocker. If done: isolated, conscious max_seq (KV).

**Stop.** OOM, working set > 12 GiB dedicated + huge shared, smoke fail, auto picked nf4 (bug in `codec.py` / shapes).

Do not compare to BF16 generate on 32B (will not fit). No “BF16 tok/s” column.

---

## What counts as “taken to the end”

1. 3B VQ canary **recorded as FAIL** (this happened). VQ occupancy — optional for the oracle, not a chat blocker.
2. `auto` does not pack VQ. `--codec vq` warns.
3. H2: pinned host + two slots for one packed NF4 matrix, overlap copy/GEMM, Paris/Berlin/323 smoke on a model that does not fit whole (or on 3B with an artificial budget).
4. Download 32B only after Pavel’s explicit “yes”.

---

## Agent report (end of wave)

```
Wave: N
Status: PASS | FAIL | BLOCKED
Done: …
Numbers (if e2e): weight_mib / smi / torch_reserved / ttft_ms / decode_tok_s / smoke
Files: …
Tests: command → PASS/FAIL
Next-wave gate: can | cannot, because …
Did not touch: …
```

---

## H2 (unlocked: H3 canary red)

NF4 32B does not fit (~17.6 GiB weights). The card holds ~10 GiB packed + overhead. Copy the tail **pinned** H2D, not pageable.

Measured 2026-09-14 on this 3080:

| | 256 MiB | GB/s |
|---|---:|---:|
| `pin_memory=True` non_blocking | 10.31 ms | **24.3** |
| pageable | 31.18 ms | 8.0 |

The “H2 makes sense” threshold (~15 GB/s) is passed. Two slots for **one packed matrix**, overlap copy/GEMM, no `cudaMallocManaged`. Turn off QKV-fork on streams for the overflow group (one slot cannot be written from three streams).

The ceiling is still ≈ the current kernel (~3 tok/s on 32B NF4), not 10. That beats the 1–2 tok/s floor “haul the whole model over PCIe every token” and **talks** (NF4 quality).

Done 2026-09-14: product smoke **2.31 tok/s**, see `docs/plan-h2-ring.md`. Do not cite pageable ~0.8.

Do not start H2 via managed/oversubscribe. 14B BF16 on this card already gave 0.92 tok/s.

# H2: NF4 overflow ring (pinned host, two slots)

**Status 2026-09-14:** C0 closed. **H2-1…H2-6 PASS.**

Product smoke Qwen2.5-32B-Instruct on RTX 3080 12 GB:
`python -m gpu.lab.h2_trace --no-timing` →
`C:\dev\models\runs\h2-qwen25-32b-20260914-234048`
(copy of `data_path.md` / `messages.json` / `gate.txt` in
[`docs/runs/h2-qwen25-32b/`](runs/h2-qwen25-32b/)).

| | |
|---|---|
| Decode | **2.31 tok/s** mean (2.30–2.32; ~432 ms/tok) |
| llama.cpp Q4_K_M, same smoke | **1.5 tok/s** decode (`docs/eval-32b.md`, 2026-09-15) |
| TTFT | **1006 ms** mean (2 chunks, `LIVE_MAX_N=32`) |
| Smoke | Paris / Berlin / 323 **3/3**, `gate.txt` PASS |
| Packed NF4 | **16599 MiB** — does not fit on the card |
| Resident HBM | **9716 MiB** (`report.device_mib`, not 16601) |
| Host tail | **96** matrices, **6885 MiB**, pin 6885/6885, slot **71.72 MiB** |
| smi decode | **11926–11933 MiB**, flat |
| Floor “haul the tail over PCIe” | beaten (not 0.8; not 10) |
| ~3 tok/s ceiling | not reached (wall 432 ms > copy-floor 277 ms) |
| Hard-12 | **not run** |
| BF16 32B | **not run** (will not fit) |
| VQ | **no** (`--codec auto` does not pick VQ) |

Wave goal: 32B **talks** faster than the 1–2 tok/s floor. Ceiling ≈ current kernel
(~3 tok/s if overflow weights were already in HBM), not 10. Quality is NF4.

Plate: [`docs/img/h2-qwen25-32b.png`](img/h2-qwen25-32b.png),
`python -m gpu.lab.h2_plate --redraw`.

## Frozen (do not dispute)

Hardware: sm_86, WDDM, display. Pinned H2D **24.3 GB/s** (256 MiB); pageable 8.0 —
dead path. One H2D copy engine; two H2Ds do not add up. Compute ∥ one copy — yes.

Live NF4 = **4.25 bit**, group 64. Packed 32B ≈ **16599–16601 MiB**, not the paper
17577 (4.5 bit in `vram-3080.md`). Hole ≈ 6.1 GiB + overhead 1800 MiB.

| Rule | Decision |
|---|---|
| Slots | **2** device arenas, size = packed+scale of the worst overflow matrix (**71.72 MiB** on 32B gate/up/down). Addresses are static. Not a layer, not lm_head. |
| Third slot | not v1 (no second H2D; 72 MiB is better given to resident FFN) |
| Pin | **256 MiB** slabs at load; do not read `.chr` on a token. Check `cpu_is_pinned()`, not `bool(tensor.is_pinned)` — that is a method, always true. |
| Copy | one `copy_stream` (not default); `copy_(non_blocking)` from pinned. On WDDM: **timing `e_copy` + CPU join** before prefetch. `elapsed_time` only if `timing=True`. CLI: `ring_timing=False` (join exists, no copy counter). |
| H2D unit | whole matrix (data+scale), not a tile |
| Kernel | do not touch `chr_nf4_gemm`. `LIVE_MAX_N=32`. No `[M,K]` BF16 in HBM. No managed. |
| Prefetch | depth = 1 (next overflow matrix; N=1 also prefetches lm_head) |
| Graphs | plan A on DEVICE. HOST GEMM after bind — CUDA graph from a static slot (`HostSlotGemm`); no copy in graph on WDDM. Graphs are not the speed headline. |
| QKV/gate-up fork | only if **all** members are DEVICE. Mixed: serial, do not copy resident into a slot |
| Prefill | one H2D per overflow matrix **per chunk** `N≤32`, not per column |
| Embed / lm_head / norms / bias / all qkvo | **always resident** |
| Overflow | all `down_proj`, then tail `gate+up` pairs (policy **D**) |
| Chat `max_seq` | **2048** (KV 512 MiB on 32B). 4096 is a separate mode |
| `auto` | NF4 if it fits whole; else **NF4 + overflow**. Never VQ. 70B-class — refuse |
| 32B | tree and `.chr` on disk. Smoke A recorded. Hard-12 — only after an explicit “run it”. |

Canary without 32B: `qwen25-3b.nf4.chr` + fake cap (cut resident packed, not a ballast tensor).
Paris/Berlin/323 smoke. Bus caliber — **256 MiB / 10.31 ms**.
`t_ms = size_MiB × 10.31 / 256`.

**978 MiB** (17577 − 16601) is a 4.5-bit table error in `vram-3080.md`, not
“unknown weights”. Do **not** reserve it in the budget.

Ceiling: if overflow is **smeared** across layers (every `down` on every layer ring),
the copy engine is busy during resident qkv/attn/gateup → `wall ≈ max(copy, gemm)`.
DoD is not 2.7; the goal is to beat the 1–2 floor, not promise 10. Live wall ~432 ms/tok —
copy-floor 277 ms plus WDDM join and incomplete overlap.

### Bug not to cite as design

`Tensor.is_pinned` is a **method**. `bool(arena.is_pinned)` is always True, overflow
ran pageable (~0.76–0.82 tok/s, ~7.3 GiB/s). Fix: `cpu_is_pinned()` in
`gpu/host/host_image.py`. Product number is **2.31**, not 0.8.

### Third-party breakdown (accept / no)

| Move | Verdict |
|---|---|
| Beat the resident **prefix** (A): two slots do not pump the tail | **Yes.** A is dead. D: all `down` travel every layer. |
| Permutation scheduler of MLP / fork as v1 | **No.** D is the seed. |
| packed+scale = **one** H2D, `ready` = both | **Yes.** |
| `record ready` → wait/GEMM/`record done` → wait/overwrite | **Yes.** In CopyRing. |
| Overflow GEMM in a CUDA graph from a static slot | **Done** (`HostSlotGemm`). Do not claim as a tok/s win. |
| Prefetch start of the next token during lm_head | **Yes, cheap**; does not change bytes/token. |
| Subtract 978 MiB “just in case” | **No.** |
| Third slot / half-M / embedding by rows / `cudaHostRegister` of the whole `.chr` | **Not v1.** |
| 2.7 tok/s target as a 32B gate without a file | **No.** Chat gate is 3B canary + live 32B smoke. |
| 10 tok/s / Marlin / 8B llama.cpp Q4 caliber | **No.** |
| Beat llama.cpp Q4_K_M 32B overflow on this 3080 (~1.5 tok/s) | **Measured**, not a 10 tok/s claim. |

## Waves

One chat = one wave. Units without GPU after a code wave:

```
python gpu/cli/test_codec.py
python gpu/cli/test_cli.py
python gpu/loop/test_attach.py
python gpu/host/test_attach.py
python gpu/chr0/test_chr0.py
python gpu/loop/test_ring.py
python gpu/loop/test_host_graph.py
python gpu/lab/test_h2_trace.py
python gpu/lab/test_h2_plate.py
```

### H2-1 — `decide()` overflow, no CUDA — PASS

`Decision.overflow`. 32B/12 GB → nf4 + overflow, not raise, not VQ.

### H2-2 — HostImage + residency plan, CPU — PASS

Pin slabs 256 MiB. `plan_residency` = **D**. `slot_nbytes = max(streamed)`.

### H2-3 — load: resident HBM + host tail + SlotPair — PASS

### H2-4 — CopyRing + decode eager — PASS

### H2-5 — plan A resident only; CLI run on overflow — PASS

### H2-6 — live 32B — PASS (smoke A)

Do not start Hard 12 without an explicit “run it” (`docs/eval-32b.md`).

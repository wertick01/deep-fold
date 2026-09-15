# H2-accel: scorecard of bets on 32B overflow

**Branch:** `exp/h2-32b-accel` from `c129ffb`. Scorecard recorded:
[`docs/runs/h2-accel-32b/`](runs/h2-accel-32b/). Do not start Hard-12 32B without
an explicit “run it”. Product default is still D / `chunk` / `draft="none"`.

**Wave goal:** turn on *switchable* variants on the same TokenLoop / CopyRing
and run them with one lab runner. Do not promise 10 tok/s. Do not touch
`chr_nf4_gemm`. No third slot. Not `DEEPFOLD_COPY_JOIN=0` on WDDM.

Measured basis (`docs/runs/h2-qwen25-32b/data_path.md`):

| | |
|---|---|
| Decode | 2.31 tok/s, ~432 ms/tok |
| Copy floor | 6885 MiB → **277 ms** (256 MiB / 10.31 ms) |
| Prefill | tape **on every chunk** `LIVE_MAX_N=32` |
| Policy D | all 64 `down` HOST; `gate+up` of layers 48–63 HOST |
| Embed / lm_head | DEVICE, 394.5 MiB each, untied |
| CopyRing | 2 slots, ahead=1, join ON, `ring_timing` in CLI off |
| smi decode | ~11930 MiB |

## Frozen (inherit H2)

- 2 slots, one `copy_stream`, H2D unit = whole matrix.
- Join **before** prefetch on Windows. Do not queue `copy_{i+1}` before join.
- `LIVE_MAX_N=32`. Graphs N=1 only. Prefill N>1 eager.
- `auto` never takes VQ. No dense `[M,K]` in HBM.
- Pin-set **qkvo + lm_head** always DEVICE. Embed DEVICE in baseline.
- Do **not** load a draft model on the same GPU (no ~350 MiB headroom).
- Python→C++, FlexGen batch, QuIP#, 3rd slot, extra H2D streams — **out of wave**.

## Scorecard matrix

Each variant is a runner flag, not a separate binary. Baseline is always in the table.

| id | What it changes | Decode | TTFT |
|---|---|---|---|
| `baseline` | as now: D, chunk-prefill, `step()` | basis | basis |
| `profile` | `ring_timing=True` on one decode | diagnostics 155 ms | — |
| `verify-k` | teacher-force greedy in blocks k∈{2,4,8} | measures `T_verify` | no |
| `spec-lookup` | greedy generate + n-gram draft from prompt, k=4 | yes, if lookup is alive | no |
| `prefill-hold` | CopyRing hold: one H2D of the matrix for all layer chunks | **no** | yes, long prompt |
| `pairs-stride` | same 16 `gate+up` pairs, not the tail, but every 4th layer | floor 277 ms the same | no |
| `host-embed` | packed embed on CPU, refill 5 MLP into resident | ~15 ms/tok if refill | weakly |

Not in this wave’s matrix: CPU NF4 GEMV, a new codec, row-split of a matrix,
same-GPU 3B draft.

Scorecard DoD: JSON + `SUMMARY.txt` with the fields below. Paris/Berlin/323 smoke on
variants that change generate (not on `plan-only`). Greedy `verify-k`
must match `baseline` on tokens.

## A. CopyRing hold (needed for `prefill-hold`)

File: `gpu/loop/ring.py`. Do **not** change the product ping-pong for decode.

Today `bind_for_gemm` = wait + join + prefetch next + one GEMM + `record_gemm`.
Prefill needs a path “copied the matrix once, ran N≤32 several
times on the slot, then record and prefetch”.

API (names can be refined, semantics cannot):

```
CopyRing.bind_hold(gemm) -> (packed, scale)
    wait e_copy, CPU join (WDDM), return slot views.
    do NOT call prefetch of the NEXT matrix.
    _active_slot stays occupied.

CopyRing.gemm_hold()  # no-op / assert active
    slot still must not be overwritten.

CopyRing.release_hold(gemm)
    record e_gemm, clear active, prefetch() as after a normal record.
```

Rules:

- While hold is active, a second `bind_*` is RuntimeError (as today without record).
- Decode N=1 stays on `bind_for_gemm` / `record_gemm`. Not hold.
- Counters: `total_copies` grows on issue, not on every hold GEMM.
- CPU test: one issue, three fake GEMMs, one record; `ahead` and overwrite-forbid.
- GPU canary 3B fake-overflow: N=8 twice on one HOST down without a second H2D.

WDDM: join once per matrix, **before** any hold GEMM and **before** prefetch next.

## B. Prefill weight-stationary MLP (`prefill-hold`)

Files: `gpu/loop/generate.py` (`prefill` / `_prefill_layers`), not `step()`.

Attention **stays left-to-right by chunks inside a layer**: layer i KV for
token t+1 needs layer i KV of tokens ≤ t. Must not compute layer 5 down before
layer 4 residual.

Invert only **HOST MLP** inside an already-computed layer:

```
for layer:
    for chunk in prompt_chunks:          # N≤32
        DEVICE qkv / rope / kv / attn / o
        if gate+up DEVICE: compute the chunk’s down-input (can into a list)
    for each HOST matrix of this layer in consume order:
        bind_hold
        for chunk: nf4_gemm N≤32 on the slot
        release_hold
    residual += down outputs
```

Superchunk: if storing `[T, intermediate]` does not fit (108 MiB × 2 on 32B
T=2048), cut the prompt into superchunks **S=256** (gate `[256,27648]` BF16 ≈
13.5 MiB). Number of tape passes = `ceil(T/S)`, not `ceil(T/32)`.

TokenLoop flag: `prefill_mode: Literal["chunk", "hold"] = "chunk"`.
Default `"chunk"` = current behavior (tape on every chunk).

Checks:

- CPU: `CopyRing.total_copies` count on prefill T=64, chunk=32, one HOST
  down: `chunk` → 2 copies; `hold` → 1 copy (or 1 per superchunk).
- GPU 3B fake-overflow: greedy prefill `hold` vs `chunk` — same argmax
  of the last token (like smoke prefill N vs N=1).
- Decode tok/s on `hold` need not grow; TTFT on a long prompt — yes.

## C. Speculative / verify-k

New file `gpu/loop/speculate.py`. `TokenLoop.generate` only calls it.

**C1. `verify_block(loop, ids[k], start_pos) -> logits [k, vocab]`**

One `forward(..., all_positions=True)`. k∈1..32. k>32 — ValueError (two
forwards = two tapes; do not hide that).

**C2. Lab `measure_verify(loop, greedy_tokens, k)`**

After the same prefill as baseline:

1. Record the greedy trajectory via `step()` (control).
2. `reset` + the same prefill.
3. Feed the same tokens in blocks of k. Wall time per block, `h2d_bytes`,
   greedy match.

`E_break_even = T_verify(k) / T_step`. If at k=4/8 this is > 1.5 — write in SUMMARY
“spec does not pay off on this card”, do not turn `spec-lookup` into the
product default.

**C3. `spec-lookup` (only if C2 is not failed)**

Draft without a second model: n-gram repeat from the already-known prompt+prefix
(length 2–3). Verify greedy: accept the matching prefix, on divergence
emit the target argmax (like Leviathan greedy). KV: `seq_len = start + n_accept`;
do not read the cache tail.

`generate(..., speculate: int = 1, draft: str = "none")`.
`speculate==1` or `draft=="none"` = current `step()` loop.

Greedy `spec-lookup` on Paris/Berlin/323 may not speed up (no repeats) —
that is not a quality FAIL if tokens = baseline.

No sampling. No GPU draft.

## D. `pairs-stride`

File: `gpu/host/residency.py`.

New policy `"pairs_stride"` in `POLICIES`:

- like D, all `down` HOST (on a 32B cap this is inevitable);
- 16 `gate+up` pairs not tail 48..63, but layers `3,7,11,...,63` (every 4th,
  16 of them). If fewer than 16 pairs — take as many as needed up to cap.

Tape is still consume order (`_streamed_tape`). Tape bytes on 32B
**match** D (± one matrix if cap cuts differently — document it).

`load_chr_nf4` / `load_model(..., residency_policy=)` pass the string through.
Default `"D"`. CLI: `--residency pairs_stride`.

CPU: `test_residency.py` — WHO layer ids, pin-set untouched, n_host pairs = D.

Do not split a matrix by rows.

## E. `host-embed`

Packed embed is **not** on the CopyRing tape (that would add 394 MiB/token).

Path:

1. `plan_residency(..., pin_embed=False)` or policy `"D_host_embed"`:
   embed not in PIN_KINDS; does **not** land on the `host` tape; materializes on
   **CPU** (pinned optional).
2. `Nf4Embedding.attach` CPU `ChrMatrix`. `dequant_nf4_rows` already follows
   `packed.device`. Forward: CPU rows → `.to(device, non_blocking)` into
   a small staging `[n, hidden]` **not** on the ring `copy_stream`.
3. Cap refill: +nbytes(embed) to resident MLP. Whole matrices: 2
   `gate+up` pairs (4×71.72) + 1 `down`. Do not leave half a pair.
4. 32B untied: `lm_head` stays DEVICE. 3B tied: host-embed **refuse**
   (shared packed; do not break the tie).

Control run: host-embed **without** refill — smi −~394, tok/s ≈ baseline.
Second: with refill — `h2d_bytes` per decode-forward smaller by 5 arenas.

Do not mix tiny H2D embed with `CopyRing.copy_stream` without a separate stream
or the default stream after join.

## F. Scorecard runner

New `gpu/lab/h2_accel.py` (do not break `h2_trace` default).

```
python -m gpu.lab.h2_accel --variant baseline,profile,verify-k,prefill-hold,pairs-stride,host-embed
python -m gpu.lab.h2_accel --variant verify-k --k 2,4,8 --force-overflow --max-seq 512
python -m gpu.lab.h2_accel --plan-only --variant pairs-stride,host-embed
```

`--model` / `--chr` as in `h2_trace`. `--force-overflow` = 3B canary cap.
Default 32B paths as in `h2_metrics`.

Output `$DEEPFOLD_RUNS/h2-accel-<stamp>/`:

- `plate.json` — schema `deepfold.h2_accel.v1`
- `SUMMARY.txt` — table variant × {prefill_ms, tok/s, h2d_MiB/forward,
  h2d_copies, smi, smoke, notes}
- for `profile`: per-copy ms if timing on; else honestly `copy_ms=null`

JSON fields per variant: `id`, `prompt_len`, `prefill_ms`, `decode_tok_s`,
`decode_ms_per_tok`, `h2d_bytes`, `h2d_copies`, `h2d_forwards`,
`copy_floor_ms`, `smi_mib`, `residency_policy`, `prefill_mode`,
`speculate`, `n_host`, `resident_mib`, `smoke`, `greedy_match_baseline`.

`--plan-only` does not load GPU: only `plan_residency` + expected bytes.

CPU tests `gpu/lab/test_h2_accel.py`: parser, schema, plan-only on toy
descs / skip without `.chr`.

## Code waves (file owners — do not overlap)

### Accel-1 — ring hold + prefill-hold

`gpu/loop/ring.py`, `gpu/loop/generate.py` (`prefill` / `_prefill_layers` /
`prefill_mode` flag), `gpu/loop/test_ring.py` (new tests, old PASS).

Do not change decode `step()` semantics.

### Accel-2 — speculate / verify-k

`gpu/loop/speculate.py`, `gpu/loop/test_speculate.py`,
`TokenLoop.generate` **only** new kwargs with defaults that preserve
the current loop.

Do not import HuggingFace generate. Do not load a second model.

### Accel-3 — residency + embed + runner

`gpu/host/residency.py`, `gpu/host/test_residency.py`,
`gpu/host/model.py` (host embed seat), `gpu/host/embedding.py` if needed,
`gpu/lab/h2_accel.py`, `gpu/lab/test_h2_accel.py`, pass-through of
`residency_policy` in `load_model` / `h2_trace` / `sessions` **with default D**.

## Units without GPU (after each sub-wave)

```
python gpu/host/test_residency.py
python gpu/loop/test_ring.py
python gpu/loop/test_speculate.py
python gpu/lab/test_h2_accel.py
python gpu/cli/test_cli.py
```

GPU canary (if 3B `.chr` is on disk):

```
python -m gpu.lab.h2_accel --force-overflow --variant baseline,verify-k,prefill-hold --k 4 --max-seq 512 --max-new-tokens 16
```

32B scorecard — a separate command by the card owner, not CI:

```
python -m gpu.lab.h2_accel --variant baseline,profile,verify-k,prefill-hold,pairs-stride,host-embed
```

## What counts as success of a variant

| id | Success | Fail (leave the flag, default off) |
|---|---|---|
| `profile` | copy_ms is not None; sum ≈ or < wall | — |
| `verify-k` | greedy match; T_verify/k recorded | T_verify(8) > 2×T_step → spec default off |
| `spec-lookup` | tokens = baseline | need not be faster on smoke |
| `prefill-hold` | prefill copies ↓; greedy match | smoke TTFT (2 chunks) may not move |
| `pairs-stride` | h2d_bytes = D; smoke | tok/s did not grow — OK, a row in the table |
| `host-embed` | without refill smi↓ tok/s≈; with refill bytes↓ | OOM / tied 3B |

Honest ceiling: even perfect overlap is not above ~3.6 tok/s on the same tape.
`verify-k` can beat this floor **per accepted token**, not per forward.

## Scorecard 2026-09-15 (3080)

Live plates: `C:\dev\models\runs\h2-accel-32b-20260915-190303` and warm D
`h2-accel-32b-warm-baseline-20260915-201046`. Git copy:
[`docs/runs/h2-accel-32b/`](runs/h2-accel-32b/).

`--max-seq 512` → 90 HOST, 6455 MiB/fwd, floor 260 ms (not 96 / 6885 / 277:
less KV, higher cap). Warm D: **2.35 tok/s**, prefill 990 ms. The first
`baseline` in the matrix is a cold start, not T_1.

| id | Result |
|---|---|
| `profile` | `copy_ms` 366 ms/fwd vs floor 260. Lab. |
| `verify-k` | greedy match. T_8=3098 ms/block; T_8/T_step≈7.3 → spec default off |
| `spec-lookup` | tokens = D, **0.008 tok/s**. Leave the flag, default `"none"` |
| `prefill-hold` | copies −270; short-smoke TTFT worse. Opt-in for a long prompt |
| `pairs-stride` | bytes = D, tok/s did not grow |
| `host-embed` | −359 MiB/fwd, smi≈D because of refill, tok/s did not grow |

Code merging to `main` does not change generate until the caller passes new
kwargs / policy. Hold and verify primitives are needed for the next wave; n-gram
lookup must not go into the product.

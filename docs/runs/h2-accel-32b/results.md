# H2-accel 32B comparison (2026-09-15)

Switchable overflow bets on Qwen2.5-32B-Instruct NF4, RTX 3080 12 GB, WDDM.
TZ: [`docs/plan-h2-accel.md`](../../plan-h2-accel.md). Live dumps stay under
`C:\dev\models\runs\h2-accel-*`. This folder is the git copy.

Smoke is the three lab prompts (Paris / Berlin / 323), greedy,
`max_new_tokens=64`, `max_seq=512`. Every generate variant scored **3/3** and
**greedy_match_baseline=true**.

`--max-seq 512` raises the overflow cap vs the recorded H2 tape
(`docs/runs/h2-qwen25-32b/`, max_seq=2048): **90 HOST / 6455 MiB/forward**,
not 96 / 6885. Copy floor **260 ms**, not 277. Decode is still copy-bound.

## Warm T_1 (use this)

| | |
|---|---|
| Policy | D, `prefill_mode=chunk`, `draft=none` |
| Decode | **2.35 tok/s** (~425 ms/tok) |
| Prefill | 990 ms (short chat prompts, two N=32 chunks) |
| Copies | 2070 across 23 forwards |
| smi | 12013 MiB |
| Graph | captured (N=1) |

Matches the older H2 smoke (2.31 tok/s at max_seq=2048). Ceiling with this
tape is still ~3.6 tok/s if copy were fully hidden. 10 tok/s is not in reach
without fewer bytes per forward or more than one accepted token per tape.

The matrix's first `baseline` row is a **cold** load (0.08 tok/s, prefill
26 s). Ignore it for tok/s / TTFT.

## Matrix (same process, GPU warming)

From `SUMMARY.matrix.txt` / `plate.json`. Decode tok/s on early rows is
below warm T_1 because the 16 GB `.chr` and clocks were still coming up.

| id | tok/s | prefill | copies | n_host | smi | Decision |
|---|---|---|---|---|---|---|
| baseline (cold) | 0.08 | 26 s | 2070 | 90 | 11963 | ignore |
| profile | 1.05 | 2.0 s | 2070 | 90 | 11894 | `copy_ms` 8415 / 23 fwd ≈ **366 ms** vs floor 260. Timing join extra. Keep as lab. |
| verify-k | 1.73 (`step()`) | 1.17 s | 2070 | 90 | 11939 | greedy match. k=8 **3098 ms/block** (387 ms/tok). T_verify(8)/T_step ≈ **7.3** |
| spec-lookup | **0.008** | 57 s | 1980 | 90 | 11951 | tokens = baseline; n-gram pays verify. **Product default stays `draft="none"`** |
| prefill-hold | 2.33 | **3.8 s** | **1800** | 90 | 11926 | copies −270 (= three extra chunk tapes). Decode unchanged. Short-prompt TTFT **worse** |
| pairs-stride | 2.14 | 4.0 s | 2070 | 90 | 11859 | same tape bytes as D. No tok/s win |
| host-embed | 2.19 | 3.8 s | 1955 | 85 | 11886 | 6096 MiB/fwd (−359). smi ≈ D (refill 5 MLP). No tok/s win |

3B `--force-overflow` canary (`docs/runs/h2-accel-3b-canary/`): hold copies
2300→2000, prefill 235→181 ms, ~10 tok/s, verify k=4 = 121 ms/block match.

## Merge / is this code needed?

**Merging `exp/h2-32b-accel` into `main` does not change the product path**
if defaults stay what they are today:

- `load_model(..., residency_policy="D")`
- `TokenLoop(..., prefill_mode="chunk")`
- `generate(..., speculate=1, draft="none")` → the same `step()` loop
- CLI `--residency` defaults to `D`

Failed *defaults* are not failed *primitives*:

| Keep | Why |
|---|---|
| `CopyRing.bind_hold` / `release_hold` | Only TTFT lever that actually dropped copies. Opt-in `prefill_mode="hold"` for long prompts (S≥256). Short smoke must stay `chunk`. |
| `verify_block` / `measure_verify` | Lab measurement of T_k. Next draft (not n-gram) can reuse this. |
| `pairs_stride` / `D_host_embed` | Load-only WHO knobs, default D. Negative result is documented; they do not run unless asked. |
| `gpu.lab.h2_accel` | Comparison runner. Not on the `deepfold` freeze. |

**Do not product-default** `draft="lookup"`: on this card it is a footgun
(0.008 tok/s) even though greedy tokens match. Leave the kwarg; leave the
default `"none"`. Neighbour CLI should not grow a `--draft` flag.

Deleting the branch would throw away the only hold/verify APIs and the
recorded WHO experiments. They do not sit on the token path unless a caller
opts in. Next speed bets (fewer bytes / a draft that accepts k≥8) build on
this, they do not replace the CopyRing 2-slot join contract.

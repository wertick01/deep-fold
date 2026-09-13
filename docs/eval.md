# Hard eval: reasoning quality and speed

Smoke lab (notebooks 03/04/05, Paris / Berlin / 323) is **not** a quality
score. It only checks that the model still talks in English and can multiply
17×19. This page is a separate plate: tasks hard enough that 3B, 14B, and 20B
can fail, run with the **same prompts** on BF16 and NF4, reporting accuracy
**and** speed.

Harness: `python -m gpu.lab.hard` and
[`notebooks/06_hard_eval.ipynb`](../notebooks/06_hard_eval.ipynb). Isolated
workers, same rule as [`lab.md`](lab.md): one codec per process, then exit.
Do not load both copies on a 12 GB card.

## What we claim, and what we do not

- **Smoke stays smoke.** A pass on Paris / Berlin / 323 does not mean the
  compressed model reasons as well as BF16. Do not put those needles in a
  README quality table.
- **Compression is lossy NF4.** A quality drop is in scope. The point of this
  eval is to measure it, not to hide it.
- **Speed is a different stack** on each side (HF `generate` + dense GEMM vs
  fused NF4 `TokenLoop`). Report both. Do not rank them as a kernel benchmark.
- **3B vs 14B (from smoke, not from this plate):** when both fit, 3B NF4 is
  slower than BF16 (~17 vs ~23 tok/s). 14B NF4 wins decode because BF16 spills
  (~28 GiB CUDA working set vs ~7.8 GiB NF4). Hard eval must still report both
  quality and those same speed/VRAM columns. Do not infer 20B numbers.
- **Live 3B/14B plate is in git.** Appendix with full questions, replies,
  misses, and per-message prefill/decode times:
  [`eval-hard-qwen25.md`](eval-hard-qwen25.md). InternLM 20B is not on that
  plate.

Not in this wave: WikiText, KL vs a BF16 dump, extra codecs, NF4 kernel
changes.

## Tasks

Committed fixture: [`gpu/lab/data/hard_items.json`](../gpu/lab/data/hard_items.json)
(12 independent items + a 4-turn history script). No HuggingFace download.
Tests and CI use that file only.

| id | Kind | Why it can fail |
|---|---|---|
| `gsm8k-lamps` … `gsm8k-stickers` | multi-step arithmetic | missed intermediate, unit mix-up |
| `trap-sheep` | “all but 9” | 3B often answers 8 or 17 |
| `logic-yesno` | syllogism | “yes” on a valid “no” |
| `code-sum`, `code-loop` | mental execution | off-by-one on `range` |
| `trap-batball` | CRT bat-and-ball | many small models answer `0.10` |
| `prefill-warehouse` | long prompt, buried number | distractors 1800 / 312 / 4501 |

Greedy. Same `max_new_tokens = 256` on both codecs (smoke uses 64; that is
too short for these items). `max_seq` is 2048 on 3B/14B and 1024 on internlm
20B so the KV window is not the 20B leftover.

### Two protocols

1. **Independent (default).** Each item is a fresh single-user turn. KV is
   reset. TTFT is a clean prefill. This is the number to quote first.
2. **History (`--history`).** Four turns about one crate of screws. Each next
   prompt is the chat so far (prior user **and** assistant text).
   `TokenLoop.generate` still resets KV, so this is **growing prefill**, not
   incremental KV reuse. Both codecs get the same packed string. Use it to
   stress context length, not to claim a KV-cache win.

### Larger sets later

`DEEPFOLD_HARD` may point at a local JSON with the same schema
(`independent` / `history` arrays of `{id, kind, prompt, gold, needles?, note}`).
Do not download GSM8K in tests. If you add a live GSM8K split, keep a tiny
committed fixture so `python -m gpu.lab.test_hard` stays offline.

## Metrics

Same session recorder as smoke (`timeline.csv`, `events.csv`, `messages.csv`,
`summary.csv`) plus **`hard_scores.csv`**:

| Metric | Where | Notes |
|---|---|---|
| Accuracy | `hard_scores.csv` `correct` | GSM8K: `#### N`, else “the answer is”, else last number. yes/no: last Yes/No. Fail closed. |
| Open / human | `pending_human` | First cut has no open items. Raw text stays in `messages.csv` for a later dual rating (NF4 better / worse / parity). |
| TTFT | `messages.prefill_ms`, `summary.mean_ttft_ms` | Prefill. NF4 is `N≤16` chunks. |
| Decode tok/s | `messages.decode_tok_s` | Different stacks. |
| nvidia-smi | `summary.vram_after_load_smi_mib`, peak | Dedicated VRAM, capped at 12288. |
| CUDA working set | `summary.vram_after_load_torch_mib` | 14B BF16 ~28 GiB (shared GPU memory). Do not read a full smi line as “it fit.” |
| Cross-run index | `hard_matrix.csv` | One row per (model, codec): `weight_mib` next to `expected_weight_mib` (bytes on disk), working set, smi peak, TTFT, tok/s, accuracy, notes. |

`messages.quality_ok` on a hard run is the hard scorer (not Paris). The smoke
plate’s panel C still **labels** needles as Paris / Berlin / 323, so notebook
06 does **not** use that plate as the quality figure. Quote `hard_scores.csv`.

## How to run

```powershell
conda activate torch-gpu
cd <this-repo>
python -m gpu.lab.test_hard
python -m gpu.lab.hard --list
python -m gpu.lab.hard --lab qwen25-3b
python -m gpu.lab.hard --lab qwen25-14b
python -m gpu.lab.hard --lab internlm20b
python -m gpu.lab.hard --lab qwen25-3b --history
python -m gpu.lab.hard --matrix --plate docs\img\hard-eval-qwen25.png
python -m gpu.lab.hard --redraw <run-root> --plate docs\img\hard-eval-qwen25.png
```

`--redraw` rebuilds the plate from a finished (or still-filling) run root using
`hard_matrix.csv`, each `hard_scores.csv` and the frozen `hard_script.json`.
No GPU, no model, no re-run — so a matrix that was launched without `--plate`,
or one whose last worker is still going, can be drawn at any time.

`--matrix` is the notebook-06 pass: BF16 3B, BF16 14B, NF4 3B, NF4 14B, in
that order — uncompressed both models, then compressed both models. One
isolated worker per pair, and the card is checked back down in between, so two
models are never resident. It writes one directory per run
(`<slug>-<codec>/`, four frozen CSVs + `hard_scores.csv` +
`<codec>/worker.log`) plus `hard_matrix.csv`, the cross-run index the summary
plate is drawn from.

`--lab internlm20b` skips NF4 when `internlm2_5-20b.nf4.chr` is missing; BF16
is still a recorded session (likely spill / OOM on 12 GB). That is the
measurement, not a crashed cell.

Or Run All on [`notebooks/06_hard_eval.ipynb`](../notebooks/06_hard_eval.ipynb):
it runs the same four-run matrix (3B and 14B, BF16 then NF4) and ends with the
summary infographic — CUDA working set, nvidia-smi peak, TTFT, decode tok/s
and accuracy per run, with the fixture's questions, golds and each run's
answer in a table underneath. `RUN_20B = True` appends InternLM; it is off by
default so 20B cannot block 3B + 14B. The notebook's pre-flight cell prints
the weight bytes found on disk per row (3B ~5.9k / ~1.6k MiB, 14B ~28k /
~7.5k MiB), which is how a reader checks that a 14B row really loaded 14B.

14B BF16 at ~0.9 tok/s with `max_new_tokens=256` × 12 items is a long run.
Close other fat GPU processes first. Do not commit weights or `.chr` files.

## Honesty checklist

- Same prompts, both codecs, greedy, same cap.
- Isolated processes. `empty_cache` in Jupyter is not an unload.
- Quality from the scorer + raw text, never from smoke needles.
- Speed and VRAM from the same session as the replies.
- No 20B row until someone measures it.
- NF4 quality drop is a result, not a bug, unless the driver materialized a
  layer (VRAM after load sitting on the BF16 line — that *is* a bug; see
  [`lab.md`](lab.md)).

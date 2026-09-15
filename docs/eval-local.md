# Local eval: GSM8K parquet → JSON, WikiText, InternLM 20B

Instructions for an engineer **without** the chat where Pavel downloaded the
corpora. The corpora are already on disk. **Do not** download from HuggingFace
Hub. `python -m gpu.lab.eval` has **no** `--download`. `DEEPFOLD_EVAL` must
point at **already local** JSON/JSONL. Child workers set
`HF_HUB_OFFLINE=1`; do not call `datasets.load_dataset` on eval.

Python on this machine:

```powershell
$py = "C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe"
Set-Location C:\dev\deep-fold
```

**Do not** `pip install` into `torch-gpu`. **Do not** commit `C:\dev\models\eval`.
**Do not** overwrite `docs/runs/qwen25-3b` and `docs/runs/internlm20b`.
`LIVE_MAX_N` stays 32; do not raise n64 in this task.

Two different plates:

| Command | What it is |
|---|---|
| `python -m gpu.lab.eval` | this document: JSON with `kind=gsm8k` / `ppl` / … |
| `python -m gpu.lab.hard` | 12 independent items + optional history. **Not** WikiText |

## 1. What Pavel already downloaded, and where

Everything lives outside the git repo `C:\dev\deep-fold`:

```
C:\dev\models\eval\
  gsm8k\                 Hub-dataset clone openai/gsm8k (git + LFS)
    main\test-00000-of-00001.parquet     (~419 KB, 1319 rows)
    main\train-00000-of-00001.parquet    (do not use on the first run)
    socratic\...                        ignore until needed
  wikitext-2\            Hub-dataset clone wikitext-2 (git + LFS)
    data\test-00000-of-00001.parquet      (2183 `text` rows)
    data\validation-00000-of-00001.parquet
    data\train-00000-of-00001.parquet
  gsm8k-200.json          harness conversion (see §2); ~105 KB, 200 items
```

`eval_source()` only looks at `*.json` / `*.jsonl` **in the root** of
`DEEPFOLD_EVAL`, not inside `gsm8k\main\*.parquet`. Empty directory →
the committed 8-item fixture `gpu/lab/data/eval_items.json` and a
log note, with no download.

## 2. Convert GSM8K **main test** → JSON

Harness schema (`gpu/lab/eval.py`): objects with `id`, `kind`, `prompt`,
`gold`, optional `task`. For GSM8K: `kind=gsm8k`, `task=gsm8k`,
`gold` is an integer (as a string), the prompt asks for the final answer as
`#### N`, matching the fixture.

First slice: **200–500** **test** rows (not train). Disk already has the
**200** first rows of the official test (1319 total):

`C:\dev\models\eval\gsm8k-200.json`

`DEEPFOLD_EVAL` accepts a **file or a directory**. A file is unambiguous
(if other `*.json` files appear in the root of `C:\dev\models\eval`, a
directory takes the first by name). Prefer:

```powershell
$env:DEEPFOLD_EVAL = "C:\dev\models\eval\gsm8k-200.json"
```

Check without GPU:

```powershell
$py = "C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe"
$env:DEEPFOLD_EVAL = "C:\dev\models\eval\gsm8k-200.json"
Set-Location C:\dev\deep-fold
& $py -m gpu.lab.eval --list
```

Expected: `200 items  source=local  C:\dev\models\eval\gsm8k-200.json`.

### Repeat the conversion (pyarrow is already in torch-gpu)

Do not install `datasets`. Do not download from Hub. `pyarrow` 23 is already
in the env.

```powershell
$py = "C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe"
Set-Location C:\dev\deep-fold
& $py -c @'
from pathlib import Path
import json
import pyarrow.parquet as pq
from gpu.lab.hard import extract_number

parquet = Path(r"C:\dev\models\eval\gsm8k\main\test-00000-of-00001.parquet")
out = Path(r"C:\dev\models\eval\gsm8k-200.json")
n = 200
suffix = "Show the steps, then put the final integer after ####."
table = pq.read_table(parquet)
items = []
for i in range(n):
    q = str(table.column("question")[i].as_py() or "").strip()
    gold = extract_number(str(table.column("answer")[i].as_py() or ""))
    if not q or not gold:
        raise SystemExit(f"bad row {i}")
    items.append({
        "id": f"gsm8k-main-test-{i:04d}",
        "kind": "gsm8k",
        "task": "gsm8k",
        "gold": gold,
        "prompt": f"{q}\n\n{suffix}",
        "note": "openai/gsm8k main test parquet, local convert; gold from ####",
    })
payload = {
    "plate": "eval",
    "max_new_tokens": 256,
    "source_parquet": str(parquet),
    "split": "main/test",
    "n_taken": n,
    "n_parquet": table.num_rows,
    "note": "First 200 GSM8K main test rows. Keep outside git.",
    "items": items,
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out} items={len(items)}")
'@
```

To take 500 instead of 200, change `n = 200` and the file name
(`gsm8k-500.json`). Do not use train on this slice.

## 3. Run Qwen2.5-3B: BF16, then NF4

12 GB card. **Do not** load both codecs in one process. The harness already
isolates: `run_eval` → one `gpu.lab.worker` at a time, the process exits,
VRAM is returned, then the next codec.

Before starting, the card should be almost empty (~2 GiB display, **no**
fat `python.exe` in `nvidia-smi`):

```powershell
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

If a foreign lab / competitor / ncu is hanging around — **do not** start.

```powershell
$py = "C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe"
$env:DEEPFOLD_EVAL = "C:\dev\models\eval\gsm8k-200.json"
$out = "C:\dev\models\runs\eval-qwen25-3b-gsm8k-200-20260913"
Set-Location C:\dev\deep-fold
& $py -m gpu.lab.eval --lab qwen25-3b --codecs bf16,nf4 --out $out
```

`--out` must be a **new** directory under `C:\dev\models\runs\`, not
`docs/runs/qwen25-3b`. If `eval-qwen25-3b-gsm8k-200-20260913` already
exists (on this machine the first run started 2026-09-13), pick
another date in the name. Do not start a second eval while `nvidia-smi` shows
`torch-gpu\python.exe` on the card.

The worker writes the CSV **after all 200 items**, not one by one. While
`gpu.lab.worker` is alive, `$out\qwen25-3b-bf16\` may be empty — that is not a
hung run if GPU ~8 GiB and utilization is not zero.

Time expectation: 200 items × greedy × `max_new_tokens=256`
(worst case). On 3B that is tens of minutes per codec, both codecs —
about an hour+. This is a deliberate 200-item slice, not the full test 1319.

Output:

```
$out\eval_script.json          frozen prompts + provenance
$out\eval_scores.csv          merged BF16+NF4 after each worker
$out\qwen25-3b-bf16\         isolated BF16 (messages.csv, eval_scores.csv, …)
$out\qwen25-3b-nf4\
```

## 4. WikiText-2: this is not GSM8K, the NLL adapter exists, PPL numbers do not yet

On disk: `C:\dev\models\eval\wikitext-2\data\`.

| Split | File | Why |
|---|---|---|
| **test** | `data\test-00000-of-00001.parquet` | the only split from which PPL may *eventually* be published |
| validation | `data\validation-00000-of-00001.parquet` | adapter debugging, not an “official” number |
| train | `data\train-00000-of-00001.parquet` | not for the report |

Column `text` (articles/paragraphs), not `question`/`answer`. PPL is
**prefix loglikelihood**, not extract `#### N`.

Adapter: `gpu/lab/nll.py`. Both codecs compute teacher-forced NLL
(`logits[t]` → token `t+1`), **without** a chat template. `kind=ppl` no longer
goes through `generate`. The result lives in `loglikelihood.csv` next to
`messages.csv` (the messages schema is frozen, you cannot add an `nll` column).
`score_messages` reads the sidecar and writes `eval_scores.csv`.

Still **missing**:

1. JSON with `kind=ppl` from WikiText-2 test (parquet itself is not read by the harness).
2. Rolling windows for articles longer than `max_seq` (currently: empty cell,
   not silent truncation).
3. A published number in the README.

**Do not invent WikiText PPL.** An empty cell is more honest than a zero. CPU
check of the adapter: `python -m gpu.lab.test_nll` (no 3B).

When JSON exists: a separate file, `DEEPFOLD_EVAL` pointing at it, card free
(not in parallel with GSM8K). Do not mix GSM8K accuracy and PPL into one figure.

### WikiText-2 test → `kind=ppl` JSON (local slice, not official PPL)

Parquet is already on disk. The harness **does not** read parquet itself.
Conversion is `C:\dev\models\eval\_convert_wikitext_ppl.py` (also outside git):
Qwen2.5-3B tokenizer, `max_seq=2048`, prefixes shorter than 2 tokens and longer
than `max_seq` do not enter the slice (no rolling windows). On this machine
the entire test fits in 2048 (max 562 tokens; 7 rows `<2`).

Slice used for the 2026-09-13 plate:

`C:\dev\models\eval\wikitext2-test-ppl-fit50.json` — first 50 test rows
whose length is in `[2, 2048]`. This is **not** official WikiText-2 test PPL
(no corpus concatenation, no rolling windows). PPL only as
`exp(nll / n_tokens)` on rows with a real NLL; do not fill an empty cell with
zero. Do not write the number into the README.

Live run (teacher-forced, isolated workers):
`C:\dev\models\runs\eval-qwen25-3b-wikitext-ppl-20260913-fit50-nll`
plus `ppl_summary.json` in the same directory. Isolated `gpu.lab.worker`
must pass `--plate eval` and `--items-json` into `run_bf16` / `run_nf4`,
otherwise `kind=ppl` goes through `generate` (that is what happened with
`eval-qwen25-3b-wikitext-ppl-20260913-fit50` — not PPL).

## 5. InternLM 20B hard (NF4 only)

This is **`python -m gpu.lab.hard`**, 12 items, not 200 GSM8K and not WikiText.
BF16 20B on 12 GB — spill/OOM plus stale
`prepare_inputs_for_generation` on transformers 5; **do not patch**
(see `gpu/lab/sessions.py`: the patch yields fluent repetition, a fake
baseline). NF4 `.chr` only.

New `--out`, **not** `docs/runs/internlm20b`:

```powershell
$py = "C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe"
# card free (see nvidia-smi above)
Set-Location C:\dev\deep-fold
& $py -m gpu.lab.hard --lab internlm20b --codec nf4 --out C:\dev\models\runs\hard-internlm20b-nf4-YYYYMMDD
```

Hard uses the **`--codec`** flag (a single value), not `--codecs`.
Do not run `--codec both` on 20B in this env.

Run 2026-09-14 (card was free, no OOM, ~17.6 min):

`C:\dev\models\runs\hard-internlm20b-nf4-20260914`

**8/12**, all 12 items. Misses: train `30` vs `240`, machines `6` vs
`108`, sheep `8` vs `9`, bat-and-ball `0` vs `0.05`. Mean TTFT **1318 ms**,
**4.4 tok/s**, peak `nvidia-smi` **12067 MiB**, `max_seq=1024`. This is **not**
the smoke 605 ms / 5.01 tok/s at `max_seq=512` and **not** `docs/runs/internlm20b`.
Not a quality headline. BF16 was not patched.

## 6. What never goes into git

- All of `C:\dev\models\eval\` — Hub clones, `.git`, LFS, parquet.
- Converted JSON (even 200 items; even more so 1319 / train).
- `C:\dev\models\runs\` — live runs, logs, CSV.
- `*.chr`, model weights.
- Do not commit or push `C:\dev\models\eval`.

The repo keeps only the tiny fixtures
`gpu/lab/data/eval_items.json` (8) and `hard_items.json` (12).

## 7. How to read scores vs README honesty

**GSM8K / eval plate** — `eval_scores.csv`:

| Column | Meaning |
|---|---|
| `correct` | `true`/`false` for `gsm8k`; empty for `ppl` |
| `extracted` vs `gold` | number after `####` / “the answer is” / last number |
| `nll`, `n_tokens` | teacher-forced NLL from the sidecar; empty if the adapter did not compute (no round-trip, prefix > max_seq, no `kind=ppl`) |
| accuracy | share of `correct=true` among rows with non-empty `correct` |

Root `$out\eval_scores.csv` is both codecs. Do not confuse with
`hard_scores.csv`.

**12-item hard** (`python -m gpu.lab.hard`) — `hard_scores.csv`. This is
**not** WikiText, not lm-eval, not the full GSM8K test. Do not put it in the
README as “WikiText quality” and do not replace a 200-item local slice
with the headline “GSM8K”.

**8-item eval fixture** (if `DEEPFOLD_EVAL` is unset) — smoke
harness, not quality.

**Do not** write an invented accuracy into the README. When the 200-item run
finishes, the figure lives in `$out\eval_scores.csv` and in the harness log
(`codec  items  accuracy …`). Touch the README only as a separate decision,
with an explicit caption “200 / 1319 main test, greedy, max_new=256”.

Smoke Paris / Berlin / 323 is still not quality
(see [`eval.md`](eval.md)).

## Next — work order after this eval

1. **WikiText PPL** — local slice already ran; the number is **not** in the README.
2. **InternLM 20B hard** — done, §5 above. Do not overwrite `docs/runs/internlm20b`.
3. **Competitor isolated venvs** — bitsandbytes live smoke exists outside git;
   the rest SKIP. Do not `pip` into `torch-gpu`.
4. **ncu: n32 vs 2×n16** — done:
   `C:\dev\models\runs\ncu-n32-vs-2xn16-20260913`. True n32 on 3B `q_proj`
   **69 µs** vs two n16 **113 µs** (**1.63×**). Occupancy did not drop.
   **Numerics `--plan-n` on live 3B (2026-09-14).** Layer-0 `q_proj` N=32:
   maxabs **0.05847** on one element (`y[383,1]` −17.5 vs −17.44153).
   n32 is **bit-identical** to 2×n16 on the same `x`; n16 on `x[:,:16]` gives the
   same peak. This is half-ULP BF16, not a tile bug. Floor: max(0.05, ½ ULP).
   e2e 3B 2026-09-14 pair: `C:\dev\models\runs\qwen25-3b-paired-20260914`,
   `prefill_chunk=32`, TTFT **48 vs 92 ms**, decode **24.8 vs 28.7 tok/s**.
   NF4-only n32 same day: 91 ms / 28.6. `LIVE_MAX_N=32`. n64 was not measured.
5. **Do not claim vs Marlin.** Occupancy / DRAM from `docs/runs/ncu/` —
   our kernel, not tok/s against Marlin / AWQ / bitsandbytes / llama.cpp.

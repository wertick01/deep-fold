# Локальный eval: GSM8K parquet → JSON, WikiText, InternLM 20B

Инструкция для инженера **без** чата, в котором Pavel качал корпуса.
Корпуса уже лежат на диске. **Не** качать с HuggingFace Hub. У
`python -m gpu.lab.eval` **нет** `--download`. `DEEPFOLD_EVAL` должен
указывать на **уже локальный** JSON/JSONL. Дочерние воркеры ставят
`HF_HUB_OFFLINE=1`; `datasets.load_dataset` на eval не вызывать.

Python этой машины:

```powershell
$py = "C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe"
Set-Location C:\dev\deep-fold
```

**Не** `pip install` в `torch-gpu`. **Не** коммитить `C:\dev\models\eval`.
**Не** перезаписывать `docs/runs/qwen25-3b` и `docs/runs/internlm20b`.
`LIVE_MAX_N` по-прежнему 16; TokenLoop в этой задаче не поднимать.

Две разные тарелки:

| Команда | Что это |
|---|---|
| `python -m gpu.lab.eval` | этот документ: JSON с `kind=gsm8k` / `ppl` / … |
| `python -m gpu.lab.hard` | 12 независимых пунктов + optional history. **Не** WikiText |

## 1. Что Pavel уже скачал и куда

Всё вне git-репозитория `C:\dev\deep-fold`:

```
C:\dev\models\eval\
  gsm8k\                 клон Hub-датасета openai/gsm8k (git + LFS)
    main\test-00000-of-00001.parquet     (~419 KB, 1319 строк)
    main\train-00000-of-00001.parquet    (не брать на первый прогон)
    socratic\...                        игнорировать, пока не понадобится
  wikitext-2\            клон Hub-датасета wikitext-2 (git + LFS)
    data\test-00000-of-00001.parquet      (2183 строки `text`)
    data\validation-00000-of-00001.parquet
    data\train-00000-of-00001.parquet
  gsm8k-200.json          конвертация harness (см. §2); ~105 KB, 200 items
```

`eval_source()` смотрит только `*.json` / `*.jsonl` **в корне** каталога
`DEEPFOLD_EVAL`, не внутри `gsm8k\main\*.parquet`. Пустой каталог →
закоммиченная 8-пунктная фикстура `gpu/lab/data/eval_items.json` и
пометка в логе, без скачивания.

## 2. Конвертация GSM8K **main test** → JSON

Схема harness (`gpu/lab/eval.py`): объекты с `id`, `kind`, `prompt`,
`gold`, опционально `task`. Для GSM8K: `kind=gsm8k`, `task=gsm8k`,
`gold` — целое (как строка), промпт просит финальный ответ в форме
`#### N`, как фикстура.

Первый срез: **200–500** строк **test** (не train). На диске уже лежат
**200** первых строк official test (1319 всего):

`C:\dev\models\eval\gsm8k-200.json`

`DEEPFOLD_EVAL` принимает **файл или каталог**. Файл однозначен
(если в корне `C:\dev\models\eval` появятся другие `*.json`, каталог
возьмёт первый по имени). Предпочтительно:

```powershell
$env:DEEPFOLD_EVAL = "C:\dev\models\eval\gsm8k-200.json"
```

Проверка без GPU:

```powershell
$py = "C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe"
$env:DEEPFOLD_EVAL = "C:\dev\models\eval\gsm8k-200.json"
Set-Location C:\dev\deep-fold
& $py -m gpu.lab.eval --list
```

Ожидание: `200 items  source=local  C:\dev\models\eval\gsm8k-200.json`.

### Повторить конвертацию (pyarrow уже в torch-gpu)

Не ставить `datasets`. Не качать Hub. `pyarrow` 23 уже есть в env.

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

Чтобы взять 500 вместо 200, смените `n = 200` и имя файла
(`gsm8k-500.json`). Train не использовать на этом срезе.

## 3. Прогон Qwen2.5-3B: BF16, затем NF4

Карта 12 GB. Оба кодека **не** грузить в одном процессе. Harness уже
изолирует: `run_eval` → по одному `gpu.lab.worker`, процесс выходит,
VRAM возвращается, затем следующий кодек.

Перед стартом карта должна быть почти пустой (~2 GiB display, **нет**
жирного `python.exe` в `nvidia-smi`):

```powershell
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Если висит чужой lab / competitor / ncu — **не** стартовать.

```powershell
$py = "C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe"
$env:DEEPFOLD_EVAL = "C:\dev\models\eval\gsm8k-200.json"
$out = "C:\dev\models\runs\eval-qwen25-3b-gsm8k-200-20260913"
Set-Location C:\dev\deep-fold
& $py -m gpu.lab.eval --lab qwen25-3b --codecs bf16,nf4 --out $out
```

`--out` обязан быть **новым** каталогом под `C:\dev\models\runs\`, не
`docs/runs/qwen25-3b`. Если `eval-qwen25-3b-gsm8k-200-20260913` уже
существует (на этой машине первый прогон стартовал 2026-09-13), возьмите
другую дату в имени. Не запускайте второй eval, пока `nvidia-smi` показывает
`torch-gpu\python.exe` на карте.

Воркер пишет CSV **после всех 200 пунктов**, не по одному. Пока жив
`gpu.lab.worker`, в `$out\qwen25-3b-bf16\` может быть пусто — это не зависший
прогон, если GPU ~8 GiB и utilization не ноль.

Ожидание по времени: 200 пунктов × greedy × `max_new_tokens=256`
(худший случай). На 3B это десятки минут на кодек, оба кодека —
примерно час+. Это сознательный срез 200, не полный test 1319.

Выход:

```
$out\eval_script.json          замороженные промпты + provenance
$out\eval_scores.csv          слитый BF16+NF4 после каждого воркера
$out\qwen25-3b-bf16\         isolated BF16 (messages.csv, eval_scores.csv, …)
$out\qwen25-3b-nf4\
```

## 4. WikiText-2: это не GSM8K, адаптер NLL есть, числа PPL ещё нет

На диске: `C:\dev\models\eval\wikitext-2\data\`.

| Split | Файл | Зачем |
|---|---|---|
| **test** | `data\test-00000-of-00001.parquet` | единственный split, с которого *когда-нибудь* можно публиковать PPL |
| validation | `data\validation-00000-of-00001.parquet` | отладка адаптера, не «официальное» число |
| train | `data\train-00000-of-00001.parquet` | не для отчёта |

Колонка `text` (статьи/абзацы), не `question`/`answer`. PPL — это
**loglikelihood префикса**, не extract `#### N`.

Адаптер: `gpu/lab/nll.py`. Оба кодека считают teacher-forced NLL
(`logits[t]` → токен `t+1`), **без** chat template. `kind=ppl` больше не
ходит в `generate`. Результат живёт в `loglikelihood.csv` рядом с
`messages.csv` (схема messages заморожена, колонку `nll` туда нельзя).
`score_messages` читает sidecar и пишет `eval_scores.csv`.

Пока **нет**:

1. JSON с `kind=ppl` из WikiText-2 test (parquet сам по себе harness не читает).
2. Rolling windows для статей длиннее `max_seq` (сейчас: пустая ячейка,
   не молчаливая обрезка).
3. Опубликованного числа в README.

**Не выдумывать WikiText PPL.** Пустая ячейка честнее нуля. CPU-проверка
адаптера: `python -m gpu.lab.test_nll` (без 3B).

Когда будет JSON: отдельный файл, `DEEPFOLD_EVAL` на него, карта свободна
(не параллельно с GSM8K). Не смешивать accuracy GSM8K и PPL в одной цифре.

### WikiText-2 test → `kind=ppl` JSON (локальный срез, не official PPL)

Parquet уже на диске. Harness **не** читает parquet сам. Конвертация —
`C:\dev\models\eval\_convert_wikitext_ppl.py` (тоже вне git): Qwen2.5-3B
tokenizer, `max_seq=2048`, префиксы короче 2 токенов и длиннее `max_seq`
не попадают в срез (rolling windows нет). На этой машине весь test
укладывается в 2048 (max 562 токена; 7 строк `<2`).

Срез, с которым гоняли тарелку 2026-09-13:

`C:\dev\models\eval\wikitext2-test-ppl-fit50.json` — первые 50 строк test,
у которых длина в `[2, 2048]`. Это **не** official WikiText-2 test PPL
(нет склейки корпуса, нет rolling windows). PPL только как
`exp(nll / n_tokens)` по строкам с реальным NLL; пустую ячейку нулём
не заполнять. Число не писать в README.

Живой прогон (teacher-forced, isolated workers):
`C:\dev\models\runs\eval-qwen25-3b-wikitext-ppl-20260913-fit50-nll`
плюс `ppl_summary.json` в том же каталоге. Isolated `gpu.lab.worker`
обязан прокинуть `--plate eval` и `--items-json` в `run_bf16` / `run_nf4`,
иначе `kind=ppl` уходит в `generate` (так вышло у
`eval-qwen25-3b-wikitext-ppl-20260913-fit50` — не PPL).

## 5. InternLM 20B hard (только NF4)

Это **`python -m gpu.lab.hard`**, 12 пунктов, не 200 GSM8K и не WikiText.
BF16 20B на 12 GB — spill/OOM плюс stale
`prepare_inputs_for_generation` на transformers 5; **не патчить**
(см. `gpu/lab/sessions.py`: патч даёт fluent repetition, фейковый
baseline). Только NF4 `.chr`.

Новый `--out`, **не** `docs/runs/internlm20b`:

```powershell
$py = "C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe"
# карта свободна (см. nvidia-smi выше)
Set-Location C:\dev\deep-fold
& $py -m gpu.lab.hard --lab internlm20b --codec nf4 --out C:\dev\models\runs\hard-internlm20b-nf4-YYYYMMDD
```

У hard флаг **`--codec`** (единственное число), не `--codecs`.
`--codec both` не запускать на 20B в этом env.

## 6. Что никогда не попадает в git

- Весь `C:\dev\models\eval\` — Hub-клоны, `.git`, LFS, parquet.
- Сконвертированный JSON (даже 200 пунктов; тем более 1319 / train).
- `C:\dev\models\runs\` — живые прогоны, логи, CSV.
- `*.chr`, веса моделей.
- Не коммитить и не пушить `C:\dev\models\eval`.

В репозитории остаются только крошечные фикстуры
`gpu/lab/data/eval_items.json` (8) и `hard_items.json` (12).

## 7. Как читать скоры vs честность README

**GSM8K / eval plate** — `eval_scores.csv`:

| Колонка | Смысл |
|---|---|
| `correct` | `true`/`false` для `gsm8k`; пусто для `ppl` |
| `extracted` vs `gold` | число после `####` / «the answer is» / последнее число |
| `nll`, `n_tokens` | teacher-forced NLL из sidecar; пусто, если адаптер не посчитал (нет round-trip, prefix > max_seq, нет `kind=ppl`) |
| accuracy | доля `correct=true` среди строк с непустым `correct` |

Корень `$out\eval_scores.csv` — оба кодека. Не путать с
`hard_scores.csv`.

**12-item hard** (`python -m gpu.lab.hard`) — `hard_scores.csv`. Это
**не** WikiText, не lm-eval, не полный GSM8K test. Не ставить в README
как «качество на WikiText» и не подменять 200-пунктный локальный срез
заголовком «GSM8K».

**8-item eval fixture** (если `DEEPFOLD_EVAL` не задан) — smoke
harness, не качество.

**Не** вписывать выдуманный accuracy в README. Когда 200-пунктный прогон
закончится, цифра живёт в `$out\eval_scores.csv` и в логе harness
(`codec  items  accuracy …`). README трогать только отдельным решением,
с явной подписью «200 / 1319 main test, greedy, max_new=256».

Smoke Paris / Berlin / 323 по-прежнему не качество
(см. [`eval.md`](eval.md)).

## Next — порядок работ после этого eval

1. **WikiText JSON + прогон.** Адаптер NLL уже в `gpu/lab/nll.py`. Дальше —
   конвертировать **test** split в `kind=ppl` JSON (вне git), прогнать на
   свободной карте, **потом** считать `exp(nll / n_tokens)`. До прогона
   числа в README нет. Rolling windows, если статья длиннее `max_seq`.
2. **InternLM 20B hard**, если GPU свободна: команда §5, новый out dir.
   Не патчить BF16 remote code. Не затирать `docs/runs/internlm20b`.
3. **Competitor isolated venvs** — рецепт [`competitor-venvs.md`](competitor-venvs.md).
   Не `pip` в `torch-gpu`. Не воровать GPU, если там уже жирный python.
4. **ncu: n32 vs 2×n16.** Prefill-оракул `--plan-n` уже есть;
   `LIVE_MAX_N` всё ещё 16. Сначала замерить широкий тайл, **потом**
   решать, размораживать ли TokenLoop. Pavel в этой задаче не размораживал.
5. **Не утверждать vs Marlin.** Occupancy / DRAM из `docs/runs/ncu/` —
   наше ядро, не tok/s против Marlin / AWQ / bitsandbytes / llama.cpp.
)

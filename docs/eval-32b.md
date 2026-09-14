# Qwen2.5-32B: лист вопросов

Модель: `C:\dev\models\Qwen2.5-32B-Instruct`  
CHR: `C:\dev\models\qwen25-32b.nf4.chr`  
`max_seq=2048`, greedy, isolated process, 12 GB, H2 overflow (NF4, не VQ).

## A — дым (3) — PASS 2026-09-14

Те же `gpu.lab.script.MESSAGES` / needles Paris / Berlin / 323.
`max_new_tokens=64`. PASS = 3/3. Это «говорит», не качество 32B.

Команда:

```
python -m gpu.lab.h2_trace --no-timing
```

Живой дамп: `C:\dev\models\runs\h2-qwen25-32b-20260914-234048`  
Копия в git: [`docs/runs/h2-qwen25-32b/`](runs/h2-qwen25-32b/)  
Пластина: [`docs/img/h2-qwen25-32b.png`](img/h2-qwen25-32b.png)

| # | Промпт | Ответ | tok/s | TTFT |
|---|---|---|---:|---:|
| 1 | What is the capital of France? | The capital of France is Paris. | 2.31 | 1004 ms |
| 2 | And the capital of Germany? | The capital of Germany is Berlin. | 2.32 | 993 ms |
| 3 | What is 17 times 19? | 323 | 2.30 | 1021 ms |

Mean decode **2.31 tok/s**, mean TTFT **1006 ms**. `gate.txt` = PASS.

H2 sanity с этого прогона:

- `overflow=True`, `codec=nf4`, не VQ
- `report.device_mib` = **9716** (не ~16601)
- два слота, все `down_proj` HOST, gate+up HOST на L48–63
- smi decode **11926–11933**, не растёт по шагам
- tok/s **2.31** > пола 1–2; потолок ~3 не достигнут
- pin 6885/6885 MiB

Не цитировать pageable ~0.8 ток/с: это баг `Tensor.is_pinned` как bool.

## B — hard 12 — не запускали

Как 14B NF4 (10/12). Сверять с 14B NF4, не с BF16 32B (не влезет).
`gpu/lab/data/hard_items.json` → `python -m gpu.lab.hard`. `max_new_tokens=256`.

Не стартовать без явного «гоняй» у Павла: 12 пунктов × ~256 токенов на 2.3 ток/с —
это уже минуты на пункт, и это регрессия, не WikiText / GSM8K / MMLU.

| id | gold |
|---|---|
| gsm8k-lamps | 164 |
| gsm8k-money | 44 |
| gsm8k-tank | 90 |
| gsm8k-train | 240 |
| gsm8k-machines | 108 |
| gsm8k-stickers | 48 |
| trap-sheep | 9 |
| logic-yesno | no |
| code-sum | 15 |
| code-loop | 33 |
| trap-batball | 0.05 |
| prefill-warehouse | 7429 |

Опционально history crate (4 хода) из того же JSON — KV, не качество кодека.

```
python -m gpu.lab.hard --model C:\dev\models\Qwen2.5-32B-Instruct --chr C:\dev\models\qwen25-32b.nf4.chr
```

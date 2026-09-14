# Модели для проверки `chr`

Скачиваем **BF16 safetensors**, не GGUF и не GPTQ: компрессору нужны исходные веса. Каталог `models/` в git не входит.

| Роль на 3080 12 ГБ | Репозиторий | Скачать | BF16 в 12 ГБ | После нашего NF4 |
|---|---|---|---|---|
| С запасом | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | **6,2 ГБ** | да (~6,2 ГБ весов, leftover ~5 ГБ) | да, воздух |
| Впритык | [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | **29,5 ГБ** | на карте нет: CUDA working set ~28 ГиБ, spill в shared GPU memory; `nvidia-smi` упирается в 12288 МиБ | да, leftover ~3,5 ГБ (на карте, без spill) |
| Не влезает | [internlm/internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat) | **~40 ГБ** | нет (~40 ГБ) | NF4 ~11,3 ГБ весов: сначала стандартный `.chr`; TokenLoop режет fused `wqkv` |
| Overflow | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | **~65 ГБ** | нет | packed NF4 ~16,6 ГиБ тоже не влезает; H2: resident ~9,7 ГиБ + pinned host tail ~6,9 ГиБ. Дым 2,31 ток/с. Не VQ |

32B лежит в `C:\dev\models\Qwen2.5-32B-Instruct`, CHR `C:\dev\models\qwen25-32b.nf4.chr`.
Дым Paris/Berlin/323 записан 2026-09-14 (`docs/eval-32b.md`, пластина
`docs/img/h2-qwen25-32b.png`). Hard-12 на 32B не гонять без явного «гоняй».
20B остаётся resident-NF4 калибром (все веса на карте).

Опционально позже: `Qwen/Qwen2.5-7B-Instruct` (~15 ГБ) — в BF16 уже нет, в NF4 свободно. Не обязателен для корзин выше.

Порядок качания: 3B → 14B → 20B → 32B (32B уже на диске).

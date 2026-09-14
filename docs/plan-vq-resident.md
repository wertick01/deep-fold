# План: VQ-резидентность (H3) до 32B на RTX 3080 12 GB

**Статус 2026-09-14: H3 по качеству закрыт на 3B.** Occupancy/prefill VQ не чинят чат. `compress --codec auto` больше **не** выбирает VQ.

H2-6 (живой 32B overflow) — PASS: 2,31 ток/с, дым 3/3. Живой план и цифры —
[`docs/plan-h2-ring.md`](plan-h2-ring.md), пластина
[`docs/img/h2-qwen25-32b.png`](img/h2-qwen25-32b.png). Этот файл остаётся
надгробием VQ, не инструкцией «качай 32B под 2 бита».

Живой canary `C:\dev\models\qwen25-3b.vq2.chr` (738 MiB, `codec=vq`): ядро = CPU reconstruct (`verify_vq` V7 PASS), greedy схлопывается в «ll». Не баг загрузки.

| Что квантовали VQ, остальное BF16 | Дым Paris/Berlin/323 |
|---|---|
| ничего (BF16) | 3/3 |
| только `gate_proj` (36 матриц) | **0/3** |
| только `up_proj` | 2/3 |
| только `down_proj` | 2/3 |
| весь attention qkvo | 2/3 |
| mlp gate+up+down | 0/3 |
| все Linear, embed BF16 | 0/3 |

Python residual k-means на одном `gate_proj`: rel_mse ≈ 0.12 (ряд/столбец scale не спасает). NF4 на тех же срезах cosine ≈ 0.996. Формат 2×8 без шкал не держит SwiGLU.

Pinned H2D на этой WDDM-машине: **24.3 GB/s** на 256 MiB (pageable 8.0 GB/s). H2 с pin имеет смысл; без pin — нет.

`--codec vq` остаётся для оракула ядра. 32B, который должен говорить: NF4 overflow (H2), не 2-bit.

Раздавать агентам **по одной волне**. Следующую не начинать, пока не закрыт gate предыдущей. Волны 3–6 по H3 **не** стартовать.

Цель, которая ещё жива: 27B/32B на 12 GB, которые **отвечают в чате** быстрее пола 1–2 ток/с. Это H2, не H3. Не цель: Gemma, 70B, «быстрее Marlin».

---

## Вставить агенту в первый месседж

```
Репозиторий C:\dev\deep-fold. Читай docs/plan-vq-resident.md целиком, потом
только свою волну (указана ниже). Не начинай соседние волны. Не вызывай Task /
Cloud / *-pro. Python: C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe
(conda env torch-gpu). Карта: RTX 3080 12 GB, sm_86 only.

Волна: <N — название>
Сделай только её Definition of Done. Если gate красный — остановись и напиши
почему, не чини «заодно» следующую волну.
```

Подставь номер волны. Один чат = одна волна.

---

## Замороженные факты (не оспаривать)

Этап A (NF4) **сделан и измерен** на этой карте. Не переписывать NF4-ядро «чтобы VQ было проще».

| Факт | Число / место |
|---|---|
| 3B NF4 decode | 28.4 tok/s paired, TTFT 139 ms. `C:\dev\models\qwen25-3b.nf4.chr` |
| 14B NF4 | 6.56 tok/s, веса 7 483 MiB, на карте. `docs/runs/qwen25-14b/` |
| 20B NF4 | 5.01 tok/s, веса 10 062 MiB, peak smi 11 828 / 12 288, запас 460 MiB. `docs/runs/internlm20b/` |
| Overhead рантайма для `codec auto` | 1 800 MiB (`gpu/cli/codec.py`) |
| 32B NF4 | не влезает (−6.1 GiB). 32B VQ 2×8 + embed INT4 | влезает, leftover ~3.1 GiB. `docs/vram-3080.md` |
| CLI `compress`/`run --codec auto` | NF4 если влезает, иначе refuse. VQ только `--codec vq` (оракул). `gpu/cli/test_codec.py` |
| VQ GPU | `CompressedVqLinear` + `gpu/vq/vq_gemm.cu`. **Только N=1**. `BM=128`, `split_k=1` |
| Живой `.vq2.chr` на диске | `C:\dev\models\qwen25-3b.vq2.chr` (738 MiB). Дым FAIL (gate_proj) |
| Gemma / Phi-3 / MoE / vision / GGUF | refuse до walk. 27B в каталоге нет: цель размера — **Qwen2.5-32B** |
| Кольцо RAM→VRAM (H2) | **разблокирован**. Pinned H2D 24.3 GB/s. См. конец файла |

Инвариант всего плана: **в HBM нет плотного `[M,K]` BF16 слоя**. Деквант только в регистрах / smem. Если после load VQ working set ≈ BF16 — баг, не победа.

Честность как в README: tok/s VQ vs NF4 — разные стеки только если меняется цикл; vs BF16 `generate` не ранжировать как kernel benchmark; `nvidia-smi` на 14B/20B/32B не цитировать как «влезло».

---

## Окружение

```
conda activate torch-gpu
cd C:\dev\deep-fold
python -m gpu.cli doctor
```

| | |
|---|---|
| Python | `C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe` |
| Модели | `C:\dev\models\Qwen2.5-3B-Instruct`, `Qwen2.5-14B-Instruct`, `internlm2_5-20b-chat` |
| NF4 chr | `C:\dev\models\qwen25-3b.nf4.chr` и соседние |
| Компрессор | `chr.exe` в корне репо или `DEEPFOLD_CHR_BIN` |
| Коммиты | только если Павел явно просит. Conventional commits. Не `--amend` чужой истории |
| Прогоны | живые CSV в `C:\dev\models\runs\...` (вне git). В `docs/runs/` копировать только по просьбе, не затирать старые пластины фикстурой |

Юнит-тесты без GPU (должны оставаться зелёными после каждой волны кода):

```
python gpu/cli/test_codec.py
python gpu/cli/test_cli.py
python gpu/loop/test_attach.py
python gpu/host/test_attach.py
python gpu/chr0/test_chr0.py
```

С GPU, после волн 2–3:

```
python gpu/host/verify_vq.py
python gpu/vq/verify.py
```

---

## Не делать (refuse)

- `cudaMallocManaged`, oversubscribe, «пусть Windows подкачает». 14B BF16 уже показал 0.92 ток/с. H2 — только pinned + два слота.
- Писать occupancy/prefill VQ «чтобы 32B заговорил». 3B VQ дым красный; ядро тут ни при чём.
- Разжимать слой в HBM, Huffman, книга 2¹⁶, `wmma`, не-Ampere arch.
- Качать Qwen2.5-32B (~65 ГБ) без явного «да» от Павла. `docs/models.md` это сознательно не берёт.
- Менять Go-пакеты `internal/*`, если волна не про компрессор. Формат CHR0 VQ уже есть.
- Поднимать `LIVE_MAX_N` у NF4 «за компанию». NF4 live уже 32 (`gpu/nf4/plan.py`). VQ имеет свой `vq_max_n`.
- Редактировать README цифры без живого CSV.
- Запускать `Task`, Cloud Agents, `*-pro`, best-of-n. Работа в этом чате, `model inherit`.
- «Починить заодно» соседнюю волну, если свой gate красный.

---

## Волны

### Волна 0 — ориентация (15 мин, без кода)

Прочитать: этот файл, `docs/spec/vq.md` §0–7, `docs/spec/stitch-gpu.md`, `gpu/cli/codec.py`, `gpu/host/vq_linear.py`, `gpu/vq/vq_gemm.cu` (заголовок + `chr_vq_gemm`), `gpu/loop/graph.py` (`vq_max_n`).

**DoD:** в ответе агента: 8–12 строк «где VQ сейчас / чего нет / какой gate моей волны». Без патча.

---

### Волна 1 — первый живой `.vq2.chr` на 3B

**Зачем.** Без файла нет e2e. Ядро уже умеет N=1: этого хватит, чтобы модель заговорила. TTFT будет плохим — это ожидаемо, не баг волны 1.

**Файлы.** Не ядро. Можно: `docs/models.md` (строка про VQ 3B), ничего в `gpu/vq/`.

**Команды.**

```
conda activate torch-gpu
cd C:\dev\deep-fold
python -m gpu.cli compress --in C:\dev\models\Qwen2.5-3B-Instruct --codec vq --out C:\dev\models\qwen25-3b.vq2.chr
python gpu/host/verify_vq.py --chr C:\dev\models\qwen25-3b.vq2.chr
python -m gpu.cli run --model C:\dev\models\Qwen2.5-3B-Instruct --chr C:\dev\models\qwen25-3b.vq2.chr
```

Smoke — три промпта из README (Paris / Berlin / 323). Записать: packed MiB, smi after load, TTFT, decode tok/s, pass/fail дыма. Сравнить с NF4 той же сессии **не обязательно** (процессы изолировать, 12 GB).

**DoD.**

- Файл `C:\dev\models\qwen25-3b.vq2.chr` существует, header `codec=vq`.
- `verify_vq.py` V7 (реальная матрица) PASS.
- `run` печатает предупреждение decode-only и всё же отвечает; дым Paris/Berlin/323 проходит.
- В ответе: таблица чисел. Не выдумывать tok/s.

**Stop.** Компрессор падает / OOM хоста / дым не проходит → не идти в 14B. Ядро occupancy не трогать в этой волне.

**Ожидание.** Compress 3B VQ (residual k-means, CPU) — минуты–десятки минут, не секунды. Не убивать, если идёт прогресс по тензорам.

---

### Волна 2 — occupancy VQ decode (как NF4 split-K)

**Зачем.** `vq_gemm.cu` сейчас `BM=128`, `split_k=1`: на 3B `q_proj` мало CTA на 70 SM. NF4 это уже чинил (`gpu/nf4/plan.py` SMALL + split-K, цель ~140 CTA). Без этого 32B VQ останется «вроде говорит, 3–5 ток/с».

**Файлы (только эти, плюс тесты).**

- `gpu/vq/vq_gemm.cu`
- `gpu/vq/__init__.py` / `bindings.cpp` если меняется ABI launch
- зеркало плана, если нужно: новый `gpu/vq/plan.py` **по образцу** `gpu/nf4/plan.py`, не копия NF4 group=64
- `gpu/vq/verify.py`, `gpu/host/verify_vq.py`
- `gpu/include/chr_gpu.h` только если без этого нельзя; не ломать NF4 ABI

VQ group = 8, книга 8 KiB в smem. Не тащить NF4 LUT.

**Сделать.** Decode `N=1`: `BM=64` (или тот тайл, что даёт ≥70 CTA на 3B `q_proj` 2048×2048 и на GQA `k_proj`), split-K как NF4. Prefill **не** в этой волне: `N!=1` по-прежнему -2 / исключение.

**Проверки.**

```
python gpu/host/verify_vq.py
python gpu/vq/verify.py
```

Эталон: `Y_cpu = reconstruct_vq @ x` (float32), `maxabs(Y_gpu − Y_cpu) ≤ 0.05` при rms(x)≈1. Это баг ядра, не квантования. Не сравнивать с BF16 safetensors.

**DoD.**

- Синтетика + 3B `q_proj` из `.vq2.chr` (если волны 1 ещё нет — синтетика обязательна, 3B бонус).
- V5 жив: нет параметра `weight` `[M,K]`.
- V4 жив: `N!=1` по-прежнему отказывается (префилл — волна 3).
- Если есть GPU: ncu на 3B `q_proj` N=1 не обязателен; если есть — occupancy выше «волны-2 NF4 голода». Не публиковать tok/s из ncu.

**Stop.** maxabs > 0.05 на реальной матрице → не маскировать порогом. Не включать prefill «чтобы проверить занятость».

**Параллель.** Можно одновременно с волной 1: разные файлы. Конфликт только если оба правят `verify_vq.py` — тогда волна 2 владеет verify.

---

### Волна 3 — VQ prefill `N>1`

**Ждёт:** волна 2 зелёная (decode occupancy + оракул).

**Зачем.** TokenLoop сейчас ходит по промпту по одному токену (`vq_max_n` ловит N=2 и возвращает 1). На 32B первый токен будет невыносим.

**Сделать.** Те же пояса N, что у NF4 live, без слепого копипаста BK/group:

| N | смысл |
|---|---|
| 1 | decode, волна 2 |
| 2..8 | BN=8 |
| 9..16 | BN=16 |
| 17..32 | только если decode+ n16 оракул зелёный; иначе оставить host chunk ≤16 |

`gpu/vq/__init__.py` не должен `raise` на N>1, когда ядро умеет. `vq_max_n` — тот же трюк, что `nf4_max_n`: пробный launch N=2, вернуть `probe` (сейчас probe приходит из `LIVE_MAX_N` NF4; для VQ завести свой потолок, не ломая NF4 32).

**Внимание.** `verify_vq.py` **V4** сейчас требует, чтобы N≠1 падал. После prefill: V4 = «N=16 совпадает с оракулом», плюс отдельный отказ на N>cap. Обновить тест, не удалять.

**DoD.**

- Оракул N=1 и N=16 (и N=cap) ≤ 0.05 maxabs на toy и на 3B `q_proj`.
- `TokenLoop` на VQ 3B: `prefill_chunk > 1`. `run` больше не печатает «walked one token at a time».
- Юнит-тесты CLI/loop зелёные. `linear_max_n("vq")` не импортирует NF4 ядро зря и наоборот.

**Stop.** n32, если maxabs > 0.05 — оставить cap=16, как NF4 делал с n32 numerics. Не поднимать TokenLoop выше зелёного оракула.

---

### Волна 4 — перемер 3B VQ после ядра

**Ждёт:** волны 1 и 3.

**Зачем.** Отделить «VQ формат медленный» от «голодное ядро».

**Сделать.** Изолированный процесс, те же три промпта, `max_new_tokens=64`, greedy. Не одновременно с NF4 на 12 GB.

Записать в `C:\dev\models\runs\vq-qwen25-3b-<дата>\`: `summary.csv`, `messages.csv`, notes (codec=vq, prefill_chunk, smi, torch reserved).

Сравнить с живым NF4 3B (28.4 tok/s / 139 ms paired) **в тексте отчёта**, не смешивая CSV в одну папку.

**DoD.**

- Дым PASS.
- Decode tok/s и TTFT названы. Цель не «обогнать NF4». Цель: decode не катастрофа (ориентир: не хуже ~0.5× NF4 3B без объяснения), TTFT не «минута на 20 токенов промпта».
- Packed веса ~ 3B × 2 bit ≈ половина NF4 1 563 MiB (~0.8 GiB), не 5.9 GiB BF16.

**Stop.** Дым fail или working set ≈ BF16 → баг residency, чинить хост, не 14B.

---

### Волна 5 — качество 14B VQ vs NF4 (hard eval)

**Ждёт:** волна 4, дым 3B зелёный.

**Зачем.** 32B будет только VQ. Если 2 бита убивают чат на 14B — 32B бессмысленен, нужен люк H2 (четыре бита + хвост).

**Сделать.**

```
python -m gpu.cli compress --in C:\dev\models\Qwen2.5-14B-Instruct --codec vq --out C:\dev\models\qwen25-14b.vq2.chr
```

Затем тот же 12-item hard eval, что `docs/eval-hard-qwen25.md` / `gpu.lab.hard`, greedy, `max_new_tokens=256`, изолированный worker. Кодек **vq only** (14B BF16 не обязателен; 14B NF4 уже 10/12).

**DoD.**

- Таблица 12 пунктов: hit/miss vs NF4 14B (10/12).
- Mean TTFT, decode tok/s, peak smi, torch reserved.
- Вердикт gate (ниже).

**Gate (жёсткий).**

- PASS плана H3: не хуже **8/12** на этом листе **и** дым Paris/Berlin/323. 12 пунктов — регрессия, не MMLU.
- FAIL H3: ≤6/12 или модель не говорит связно → **не качать 32B**. Писать отчёт. H2 не начинать сами: ждать Павла.
- Серая зона 7/12: остановиться, спросить Павла.

Не утверждать «квантование без потерь», даже если VQ ≥ NF4 на 12 пунктах.

---

### Волна 6 — 32B резидентный VQ (только после PASS волны 5 + «да» на качку)

**Зачем.** Это схема `docs/schema.md` этап B и ответ на «27B на 12 GB». В стеке нет Gemma-27B. Qwen2.5-32B — названный размер.

**Диск.** ~65 ГБ safetensors. `docs/models.md` сейчас запрещает. Агент **не качает** сам. Павел качает или пишет «качай». Compress **по тензорам с диска**, не `from_pretrained` целиком на GPU. Пик RAM — один тензор float32 (+ рабочие буферы k-means), см. `docs/spec/vq.md` / `docs/cpu-roundtrip.md`.

```
python -m gpu.cli compress --in <32B-dir> --codec auto
# auto на 12 GB обязан выбрать vq (gpu/cli/test_codec.py уже так проверяет на шейпах)
python -m gpu.cli run --model <32B-dir>
```

**DoD.**

- `decide()` = vq. Файл `.vq2.chr`.
- После load: packed ~8 GiB, smi < 12288 с запасом под KV, **torch reserved ≈ packed, не 62 GiB**.
- Дым Paris/Berlin/323 PASS.
- Decode tok/s записан. Ориентир «как 20B NF4» ~5 ток/с; схема хотела ≥10 — это **после** ядра, не обещание волны 6.
- Hard eval 12 items — желательно, не блокер дыма. Если делают: isolated, max_seq сознательный (KV).

**Stop.** OOM, working set > 12 GiB dedicated + огромный shared, дым fail, auto выбрал nf4 (баг `codec.py` / шейпы).

Не сравнивать с BF16 generate на 32B (не влезет). Нет колонки «BF16 tok/s».

---

## Что считается «довели до конца»

1. 3B VQ canary **записан как FAIL** (это случилось). Occupancy VQ — по желанию для оракула, не блокер чата.
2. `auto` не пакует VQ. `--codec vq` предупреждает.
3. H2: pinned host + два слота под одну packed NF4-матрицу, overlap copy/GEMM, дым Paris/Berlin/323 на модели, которая не влезает целиком (или на 3B с искусственным бюджетом).
4. 32B качать только после явного «да» Павла.

---

## Отчёт агента (конец волны)

```
Волна: N
Статус: PASS | FAIL | BLOCKED
Сделано: …
Числа (если e2e): weight_mib / smi / torch_reserved / ttft_ms / decode_tok_s / smoke
Файлы: …
Тесты: команда → PASS/FAIL
Gate следующей волны: можно | нельзя, потому что …
Не трогал: …
```

---

## H2 (разблокирован: H3 canary красный)

NF4 32B не влезает (~17.6 GiB весов). Карта держит ~10 GiB packed + overhead. Хвост копировать **pinned** H2D, не pageable.

Замерено 2026-09-14 на этой 3080:

| | 256 MiB | GB/s |
|---|---:|---:|
| `pin_memory=True` non_blocking | 10.31 ms | **24.3** |
| pageable | 31.18 ms | 8.0 |

Порог «H2 имеет смысл» (~15 GB/s) пройден. Два слота под **одну packed-матрицу**, overlap copy/GEMM, без `cudaMallocManaged`. QKV-fork на стримах выключить на overflow-группе (один слот нельзя писать с трёх стримов).

Потолок всё равно ≈ текущее ядро (~3 ток/с на 32B NF4), не 10. Это бьёт пол 1–2 ток/с «везём всю модель по PCIe каждый токен» и **говорит** (качество NF4).

Сделано 2026-09-14: продуктовый дым **2.31 ток/с**, см. `docs/plan-h2-ring.md`. Не цитировать pageable ~0.8.

Не начинать H2 через managed/oversubscribe. 14B BF16 на этой карте уже дал 0.92 ток/с.

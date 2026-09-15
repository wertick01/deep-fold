# H2-accel: сверка ставок на 32B overflow

**Ветка:** `exp/h2-32b-accel` от `c129ffb`. Сверка записана:
[`docs/runs/h2-accel-32b/`](runs/h2-accel-32b/). Hard-12 32B не стартовать без
явного «гоняй». Product default по-прежнему D / `chunk` / `draft="none"`.

**Цель волны:** включить *переключаемые* варианты на том же TokenLoop / CopyRing
и прогнать их одним лаб-раннером. Не обещать 10 ток/с. Не трогать
`chr_nf4_gemm`. Не третий слот. Не `DEEPFOLD_COPY_JOIN=0` на WDDM.

Измеренный базис (`docs/runs/h2-qwen25-32b/data_path.md`):

| | |
|---|---|
| Decode | 2.31 ток/с, ~432 мс/ток |
| Copy floor | 6885 MiB → **277 мс** (256 MiB / 10.31 мс) |
| Prefill | лента **на каждый чанк** `LIVE_MAX_N=32` |
| Policy D | все 64 `down` HOST; `gate+up` слоёв 48–63 HOST |
| Embed / lm_head | DEVICE, по 394.5 MiB, untied |
| CopyRing | 2 слота, ahead=1, join ON, `ring_timing` в CLI off |
| smi decode | ~11930 MiB |

## Заморожено (наследовать H2)

- 2 слота, один `copy_stream`, единица H2D = целая матрица.
- Join **до** prefetch на Windows. Не ставить `copy_{i+1}` в очередь до join.
- `LIVE_MAX_N=32`. Graphs N=1 only. Prefill N>1 eager.
- `auto` никогда не берёт VQ. Нет dense `[M,K]` в HBM.
- Pin-set **qkvo + lm_head** всегда DEVICE. Embed DEVICE в baseline.
- Draft-модель на той же GPU **не** грузить (нет ~350 MiB headroom).
- Python→C++, FlexGen batch, QuIP#, 3-й слот, extra H2D streams — **вне волны**.

## Матрица сверки

Каждый вариант — флаг раннера, не отдельный бинарник. Baseline всегда в таблице.

| id | Что меняет | Decode | TTFT |
|---|---|---|---|
| `baseline` | как сейчас: D, chunk-prefill, `step()` | базис | базис |
| `profile` | `ring_timing=True` на одном decode | диагностика 155 мс | — |
| `verify-k` | teacher-force greedy блоками k∈{2,4,8} | измеряет `T_verify` | нет |
| `spec-lookup` | greedy generate + n-gram draft из prompt, k=4 | да, если lookup живой | нет |
| `prefill-hold` | CopyRing hold: одна H2D матрицы на все чанки слоя | **нет** | да, длинный prompt |
| `pairs-stride` | те же 16 пар `gate+up`, не хвост, а каждый 4-й слой | пол 277 мс тот же | нет |
| `host-embed` | packed embed на CPU, refill 5 MLP в resident | ~15 мс/ток если refill | слабо |

Не в матрице этой волны: CPU NF4 GEMV, новый кодек, row-split матрицы,
same-GPU 3B draft.

DoD сверки: JSON + `SUMMARY.txt` с полями ниже. Smoke Paris/Berlin/323 на
вариантах, которые меняют generate (не на `plan-only`). Greedy `verify-k`
обязан совпасть с `baseline` по токенам.

## A. CopyRing hold (нужно для `prefill-hold`)

Файл: `gpu/loop/ring.py`. Продуктовый ping-pong **не** менять для decode.

Сегодня `bind_for_gemm` = wait + join + prefetch next + один GEMM + `record_gemm`.
Для prefill нужен путь «скопировали матрицу один раз, прогнали N≤32 несколько
раз по слоту, потом record и prefetch».

API (имена можно уточнить, семантика нет):

```
CopyRing.bind_hold(gemm) -> (packed, scale)
    wait e_copy, CPU join (WDDM), вернуть slot views.
    prefetch СЛЕДУЮЩЕЙ матрицы НЕ вызывать.
    _active_slot остаётся занятым.

CopyRing.gemm_hold()  # no-op / assert active
    слот всё ещё нельзя overwrite.

CopyRing.release_hold(gemm)
    record e_gemm, очистить active, prefetch() как после обычного record.
```

Правила:

- Пока hold активен, второй `bind_*` — RuntimeError (как сейчас без record).
- Decode N=1 остаётся на `bind_for_gemm` / `record_gemm`. Не hold.
- Счётчики: `total_copies` растёт на issue, не на каждый GEMM hold.
- CPU тест: одна issue, три fake GEMM, один record; `ahead` и overwrite-запрет.
- GPU canary 3B fake-overflow: N=8 дважды на одном HOST down без второй H2D.

WDDM: join один раз на матрицу, **до** любых GEMM hold и **до** prefetch next.

## B. Prefill weight-stationary MLP (`prefill-hold`)

Файлы: `gpu/loop/generate.py` (`prefill` / `_prefill_layers`), не `step()`.

Внимание **остаётся слева направо по чанкам внутри слоя**: KV слоя i для
токена t+1 нужен KV слоя i токенов ≤ t. Нельзя считать down слоя 5 до
residual слоя 4.

Инверсия только **HOST MLP** внутри уже посчитанного слоя:

```
for layer:
    for chunk in prompt_chunks:          # N≤32
        DEVICE qkv / rope / kv / attn / o
        if gate+up DEVICE: посчитать down-input чанка (можно в список)
    for each HOST matrix of this layer in consume order:
        bind_hold
        for chunk: nf4_gemm N≤32 на слоте
        release_hold
    residual += down outputs
```

Суперчанк: если хранить `[T, intermediate]` не влезает (108 MiB × 2 на 32B
T=2048), резать prompt на суперчанки **S=256** (гейт `[256,27648]` BF16 ≈
13.5 MiB). Число проходов ленты = `ceil(T/S)`, не `ceil(T/32)`.

Флаг TokenLoop: `prefill_mode: Literal["chunk", "hold"] = "chunk"`.
Default `"chunk"` = текущее поведение (лента на каждый чанк).

Проверки:

- CPU: число `CopyRing.total_copies` на prefill T=64, chunk=32, одном HOST
  down: `chunk` → 2 copies; `hold` → 1 copy (или 1 на суперчанк).
- GPU 3B fake-overflow: greedy prefill `hold` vs `chunk` — тот же argmax
  последнего токена (как smoke prefill N vs N=1).
- Decode tok/s на `hold` не обязан расти; TTFT на длинном prompt — да.

## C. Speculative / verify-k

Новый файл `gpu/loop/speculate.py`. `TokenLoop.generate` только вызывает его.

**C1. `verify_block(loop, ids[k], start_pos) -> logits [k, vocab]`**

Один `forward(..., all_positions=True)`. k∈1..32. k>32 — ValueError (два
форварда = две ленты; не прятать это).

**C2. Lab `measure_verify(loop, greedy_tokens, k)`**

После того же prefill, что у baseline:

1. Записать greedy траекторию через `step()` (контроль).
2. `reset` + тот же prefill.
3. Скормить те же токены блоками k. Стенка на блок, `h2d_bytes`,
   совпадение greedy.

`E_break_even = T_verify(k) / T_step`. Если на k=4/8 это > 1.5 — в SUMMARY
написать «spec не окупается на этой карте», `spec-lookup` не включать в
продуктовый default.

**C3. `spec-lookup` (только если C2 не провален)**

Draft без второй модели: повтор n-gram из уже известного prompt+префикса
(длина 2–3). Verify greedy: принять совпавший префикс, на расхождении
выдать argmax target (как Leviathan greedy). KV: `seq_len = start + n_accept`;
хвост кэша не читать.

`generate(..., speculate: int = 1, draft: str = "none")`.
`speculate==1` или `draft=="none"` = текущий цикл `step()`.

Greedy `spec-lookup` на Paris/Berlin/323 может не ускорить (нет повторов) —
это не FAIL качества, если токены = baseline.

Нет sampling. Нет GPU draft.

## D. `pairs-stride`

Файл: `gpu/host/residency.py`.

Новая политика `"pairs_stride"` в `POLICIES`:

- как D, все `down` HOST (на 32B cap это неизбежно);
- 16 пар `gate+up` не хвост 48..63, а слои `3,7,11,...,63` (каждый 4-й,
  16 штук). Если пар меньше 16 — взять столько, сколько нужно до cap.

Tape по-прежнему consume order (`_streamed_tape`). Байты ленты на 32B
**совпадают** с D (± одна матрица, если cap режет иначе — задокументировать).

`load_chr_nf4` / `load_model(..., residency_policy=)` прокинуть строку.
Default `"D"`. CLI: `--residency pairs_stride`.

CPU: `test_residency.py` — WHO layer ids, pin-set не тронут, n_host pairs = D.

Не делить матрицу по строкам.

## E. `host-embed`

Packed embed **не** на CopyRing tape (это добавило бы 394 MiB/токен).

Путь:

1. `plan_residency(..., pin_embed=False)` или policy `"D_host_embed"`:
   embed не в PIN_KINDS; **не** попадает в `host` tape; материализуется на
   **CPU** (pinned optional).
2. `Nf4Embedding.attach` CPU `ChrMatrix`. `dequant_nf4_rows` уже следует
   `packed.device`. Forward: CPU rows → `.to(device, non_blocking)` в
   маленький staging `[n, hidden]` **не** на `copy_stream` кольца.
3. Cap refill: +nbytes(embed) к resident MLP. Целые матрицы: 2 пары
   `gate+up` (4×71.72) + 1 `down`. Не оставлять половину пары.
4. 32B untied: `lm_head` остаётся DEVICE. 3B tied: host-embed **refuse**
   (общий packed; не разрывать tie).

Контрольный прогон: host-embed **без** refill — smi −~394, tok/s ≈ baseline.
Второй: с refill — `h2d_bytes` на decode-forward меньше на 5 арен.

Не смешивать tiny H2D embed с `CopyRing.copy_stream` без отдельного stream
или default stream после join.

## F. Раннер сверки

Новый `gpu/lab/h2_accel.py` (не ломать `h2_trace` default).

```
python -m gpu.lab.h2_accel --variant baseline,profile,verify-k,prefill-hold,pairs-stride,host-embed
python -m gpu.lab.h2_accel --variant verify-k --k 2,4,8 --force-overflow --max-seq 512
python -m gpu.lab.h2_accel --plan-only --variant pairs-stride,host-embed
```

`--model` / `--chr` как у `h2_trace`. `--force-overflow` = 3B canary cap.
Default 32B пути как в `h2_metrics`.

Выход `$DEEPFOLD_RUNS/h2-accel-<stamp>/`:

- `plate.json` — schema `deepfold.h2_accel.v1`
- `SUMMARY.txt` — таблица variant × {prefill_ms, tok/s, h2d_MiB/forward,
  h2d_copies, smi, smoke, notes}
- для `profile`: per-copy ms, если timing on; иначе честно `copy_ms=null`

Поля JSON на вариант: `id`, `prompt_len`, `prefill_ms`, `decode_tok_s`,
`decode_ms_per_tok`, `h2d_bytes`, `h2d_copies`, `h2d_forwards`,
`copy_floor_ms`, `smi_mib`, `residency_policy`, `prefill_mode`,
`speculate`, `n_host`, `resident_mib`, `smoke`, `greedy_match_baseline`.

`--plan-only` не грузит GPU: только `plan_residency` + ожидаемые bytes.

CPU тесты `gpu/lab/test_h2_accel.py`: parser, schema, plan-only на toy
descs / без `.chr` skip.

## Волны кода (владельцы файлов — не пересекаться)

### Accel-1 — ring hold + prefill-hold

`gpu/loop/ring.py`, `gpu/loop/generate.py` (`prefill` / `_prefill_layers` /
флаг `prefill_mode`), `gpu/loop/test_ring.py` (новые тесты, старые PASS).

Не менять семантику decode `step()`.

### Accel-2 — speculate / verify-k

`gpu/loop/speculate.py`, `gpu/loop/test_speculate.py`,
`TokenLoop.generate` **только** новые kwargs с default, сохраняющими
текущий цикл.

Не импортировать HuggingFace generate. Не грузить вторую модель.

### Accel-3 — residency + embed + раннер

`gpu/host/residency.py`, `gpu/host/test_residency.py`,
`gpu/host/model.py` (host embed seat), `gpu/host/embedding.py` если нужно,
`gpu/lab/h2_accel.py`, `gpu/lab/test_h2_accel.py`, прокидка
`residency_policy` в `load_model` / `h2_trace` / `sessions` **с default D**.

## Юниты без GPU (после каждой подволны)

```
python gpu/host/test_residency.py
python gpu/loop/test_ring.py
python gpu/loop/test_speculate.py
python gpu/lab/test_h2_accel.py
python gpu/cli/test_cli.py
```

GPU canary (если 3B `.chr` на диске):

```
python -m gpu.lab.h2_accel --force-overflow --variant baseline,verify-k,prefill-hold --k 4 --max-seq 512 --max-new-tokens 16
```

32B сверка — отдельная команда владельца карты, не CI:

```
python -m gpu.lab.h2_accel --variant baseline,profile,verify-k,prefill-hold,pairs-stride,host-embed
```

## Что считать успехом варианта

| id | Успех | Провал (оставить флаг, default off) |
|---|---|---|
| `profile` | copy_ms не None; сумма ≈ или < стены | — |
| `verify-k` | greedy match; T_verify/k записан | T_verify(8) > 2×T_step → spec default off |
| `spec-lookup` | токены = baseline | быстрее не обязан на смоуке |
| `prefill-hold` | copies prefill ↓; greedy match | TTFT смоука (2 чанка) может не сдвинуться |
| `pairs-stride` | h2d_bytes = D; smoke | tok/s не вырос — ок, строка в таблице |
| `host-embed` | без refill smi↓ tok/s≈; с refill bytes↓ | OOM / tied 3B |

Потолок честный: даже идеальный overlap не выше ~3.6 ток/с при той же ленте.
`verify-k` может обойти этот пол **на принятый токен**, не на forward.

## Сверка 2026-09-15 (3080)

Живые plate: `C:\dev\models\runs\h2-accel-32b-20260915-190303` и тёплый D
`h2-accel-32b-warm-baseline-20260915-201046`. Git-копия:
[`docs/runs/h2-accel-32b/`](runs/h2-accel-32b/).

`--max-seq 512` → 90 HOST, 6455 MiB/fwd, пол 260 мс (не 96 / 6885 / 277:
KV меньше, cap выше). Тёплый D: **2.35 ток/с**, prefill 990 мс. Первый
`baseline` в матрице — холодный старт, не T_1.

| id | Итог |
|---|---|
| `profile` | `copy_ms` 366 мс/fwd vs пол 260. Лаб. |
| `verify-k` | greedy match. T_8=3098 мс/блок; T_8/T_step≈7.3 → spec default off |
| `spec-lookup` | токены = D, **0.008 ток/с**. Флаг оставить, default `"none"` |
| `prefill-hold` | копии −270; TTFT короткого смоука хуже. Opt-in на длинный prompt |
| `pairs-stride` | байты = D, ток/с не вырос |
| `host-embed` | −359 MiB/fwd, smi≈D из-за refill, ток/с не вырос |

Код на слиянии с `main` не меняет generate, пока caller не передаст новые
kwargs / policy. Примитивы hold и verify нужны следующей волне; n-gram
lookup в продукт не включать.

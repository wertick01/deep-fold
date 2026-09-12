# CLI `chr` и протокол целостности

Пользовательский контракт утилиты `chr` (Go 1.22, CPU, без CGO) и то, что считается **успешной целостностью** для lossy-кодеков NF4/VQ. Байт-расклад `.chr` — [chr0.md](chr0.md) и [compressor.md](../compressor.md) §6. Формулы квантования — [nf4.md](nf4.md) / [vq.md](vq.md). Здесь их нет.

Это **не** инференс и **не** WikiText. Три уровня проверки **разделены**; путать их нельзя:

| Уровень | Вопрос | Эталон | Bit-exact с orig BF16? |
|---|---|---|---|
| 1. Контейнер | повторное чтение `.chr` даёт те же блобы | байты payload | да, на блобах `.chr` |
| 2. Кодек | `decode(pack(W))` совпадает с эталоном кодека | golden из `nf4.md` / `vq.md` | для nf4/vq — **нет** (это не про orig) |
| 3. Вес | orig vs reconstruct | RMSE / MAE / maxabs | **нет** для Linear/embed/lm_head; **да** для norm/bias (`codec=bf16`) |

Критерий живой 8B (PPL, чат) — [compressor.md](../compressor.md) §7, **не** код выхода `verify`. CLI на синтетике это не заменяет.

---

## 0. Зафиксированные решения

- Язык: Go 1.22. Модуль `chr`. Без CGO. Зависимости: стандартная библиотека. В тестах — только `testing` (и `os`/`path/filepath`/`bytes` из stdlib). Никакого `cobra`, `cobra`‑подобных, `testify`, `safetensors` PyPI, `torch`.
- Бинарь: `chr`, пакет `cmd/chr`.
- Команды: `compress`, `decode`, `verify`. Других подкоманд в v1 нет (`help`/`version` не требуются; `-h` на корне и на подкоманде — да).
- `--codec` на compress: только `nf4` | `vq` (нижний регистр). `int4` в JSON контейнера разрешён соседней спекой — **CLI в v1 не кодирует и не verify'ит `int4`**: встретил → код 1.
- В одном `.chr` — один lossy-кодек на все compressable-тензоры; norm/bias всегда `bf16` ([compressor.md](../compressor.md) §6, ТЗ chr0).
- Имена тензоров в `.chr` — как в HuggingFace, **без** суффикса `.weight`. Если в orig safetensors ключ `….q_proj.weight`, в `.chr` и в отчёте verify — `….q_proj`.
- `decode` всегда пишет **один** файл safetensors, dtype **F32**, логический shape (паддинг обрезан). Почему F32: на CPU проще сверять и смотреть в numpy; BF16 decode оставил бы скрытую ошибку округления шкалы. Для живой 8B полный dump ≈ 32 ГБ — поэтому живой прогон проверяют `verify`, а не `decode`.
- Коды выхода: **0** ок, **1** ошибка ввода/контракта/I/O, **2** только у `verify` при посчитанных метриках и провале порога. У `compress`/`decode` кода 2 нет.
- Модель в репозиторий и в тесты не кладётся и не скачивается. Фикстуры пишет сам тест.
- Флаги — пакет `flag` (`flag.ContinueOnError`), не `flag.ExitOnError`: иначе parse-ошибка даст os.Exit(2) и совпадёт с «провал порога». Parse-ошибка → код **1**.
- Один поток. Шарды не открывать параллельно. В RAM — не больше одного тензора (или одной полосы, §7).

---

## 1. Флаги, дефолты, пороги

### 1.1. Разбор argv

```
chr <command> [flags]
```

- Нет команды / неизвестная команда / `-h` на корне: usage на stderr, код 1 (для `-h` на корне допустимо 0 — **выбираем 0 для `-h`/`--help`, 1 для неизвестной команды и пустого argv**).
- Флаги только после имени команды. `chr --in x compress` — ошибка, код 1.
- Позиционных аргументов нет. Всё — флаги.
- stdin как `--in -` не поддерживается.
- `--out` перезаписывается без вопроса. Родительская директория должна существовать (без `mkdir -p`), иначе код 1.

Общий вид usage (текст вольный, имена флагов — нет):

```
chr compress --in <safetensors|dir|index.json> --out <file.chr> --codec nf4|vq [flags]
chr decode   --in <file.chr> --out <out.safetensors>
chr verify    --orig <safetensors|dir|index.json> --chr <file.chr> [flags]
```

### 1.2. `chr compress`

| Флаг | Тип | Default | Обязателен | Смысл |
|---|---|---|---|---|
| `--in` | path | — | да | один `.safetensors`, **или** директория с шардами, **или** путь к `*.safetensors.index.json` |
| `--out` | path | — | да | путь `.chr` (расширение не проверяем) |
| `--codec` | `nf4`\|`vq` | — | **да** | нет тихого дефолта: nf4 и vq — разные файлы и разное качество |
| `--group-size` | int | `64` если `--codec=nf4`, `8` если `vq` | нет | v1 принимает **только** канонический размер кодека |
| `--seed` | int64 | `0` | нет | RNG книжки VQ (k-means++ / респлит). Для nf4 **игнорируется** (без warning) |
| `--iters` | int | `20` | нет | итерации Lloyd; только VQ. nf4 игнорирует. `<1` → код 1 |
| `--chunk` | int | `262144` | нет | векторов dim-8 на один assignment VQ. nf4 игнорирует. `<256` → код 1. Совпадает с [vq.md](vq.md) (пик памяти, не 1e6). |
| `--stripe-bytes` | int64 | `268435456` (256 MiB) | нет | порог F32-рабочего буфера, после которого тензор режется на полосы (§7) |
| `--stripe-rows` | int | `4096` | нет | желаемая высота полосы. `<1` → код 1 |
| `--arch` | string | `unknown` | нет | поле `arch` заголовка CHR0 |
| `--hidden-size` | int | `0` | нет | если 0 — попытка вывести из тензоров `kind=q`/`embed`/`norm`, иначе оставить 0 |
| `--intermediate-size` | int | `0` | нет | аналогично из `up`/`gate`/`down` |
| `--num-layers` | int | `0` | нет | max `layer` + 1 по именам, иначе 0 |
| `--vocab-size` | int | `0` | нет | из `embed`/`lm_head` `shape[0]`, иначе 0 |
| `--quiet` | bool | false | нет | не писать прогресс на stderr |

`--group-size` в v1:

- `--codec=nf4` и значение ≠ 64 → код 1, текст `chr: nf4 group-size must be 64`.
- `--codec=vq` и значение ≠ 8 → код 1, `chr: vq group-size must be 8`.
- Флаг существует, чтобы команда совпадала с будущими кодеками и с документацией; сейчас это проверка инварианта, не свободный параметр.

`--codec` обязателен: отсутствие / пустая строка / `NF4` / `int4` / `bf16` → код 1.

Прогресс (stderr, не stdout), одна строка на тензор, если не `--quiet`:

```
compress  model.layers.0.self_attn.q_proj  [64, 64]  nf4  3ms
```

Порядок тензоров: как итератор orig (§3.2). После последнего — строка `wrote <path> tensors=<n>`.

Ошибки encode (NaN/Inf во входе, дырявый safetensors, нечитаемый шард) → код 1, тензор в `--out` не считается валидным (недописанный файл **удалить** при ошибке, чтобы не оставить полу-CHR0).

### 1.3. `chr decode`

| Флаг | Тип | Default | Обязателен |
|---|---|---|---|
| `--in` | path | — | да, файл `.chr` |
| `--out` | path | — | да, **один** `.safetensors` |

Других флагов нет. Нет `--dtype`, нет `--name`, нет шардов на выходе.

Каждый тензор из заголовка CHR0 пишется в F32, логический `shape` из JSON (не padded). Имена — ключи CHR0. См. §4.

Прогресс stderr: `decode  <name>  [d0, d1, …]  <codec>`.

### 1.4. `chr verify`

| Флаг | Тип | Default | Обязателен |
|---|---|---|---|
| `--orig` | path | — | да, те же формы, что `--in` у compress |
| `--chr` | path | — | да |
| `--fail-rmse` | float64 | из таблицы A по lossy-кодеку файла | нет |
| `--fail-maxabs` | float64 | из таблицы A по lossy-кодеку файла | нет |
| `--json` | bool | false | нет | stdout = JSON; человеческий текст не печатать |
| `--quiet` | bool | false | нет | только summary / worst (человеческий режим). С `--json` игнорируется |

Пороги `--fail-rmse` / `--fail-maxabs` действуют **по каждому** lossy-тензору отдельно (не на среднее по модели). Один `lm_head` выше порога → код 2, даже если все Linear «зелёные».

`codec=bf16` (norm, bias): пороги из флагов **не применяются**. Требование — bit-exact в смысле §3.5. Это не отключается в v1: сломанная норма — баг контейнера, не «lossy».

Если в `.chr` нет ни одного lossy-тензора (только bf16) — дефолты таблицы A как для `nf4`, на практике не используются.

Отрицательный `--fail-rmse` или `--fail-maxabs` → код 1. `+Inf` допустим (фактически отключает этот порог для lossy). NaN во флаге → код 1.

### 1.5. Таблица A — дефолты CLI и unit-тестов (синтетика)

Эти числа — **дефолт бинаря** и то, на что опираются тесты §5, **если тест не передаёт флаги**. Они рассчитаны на фикстуры с масштабом ~N(0,1) / U[-1,1] и маленьким `n`.

| Кодек | `--fail-rmse` | `--fail-maxabs` | Почему так |
|---|---|---|---|
| `nf4` | `0.08` | `0.50` | group-64 NF4 на N(0,1): RMSE обычно ≪ 0.1; maxabs ≤ половина самой широкой щели LUT (~0.139) × `s=max\|g\|`. Для группы из 64 сэмплов N(0,1) `s` редко > 3.5 → maxabs ≲ 0.49. 0.50 — жёстко, но без ложных FAIL на честном NF4. |
| `vq` | `0.40` | `2.50` | residual 2×8 без X. На крошечных фикстурах RMSE ≪ 0.40 (часто ~0). На квадрате ~256×256 N(0,1) RMSE должен быть < `rms(W)≈1`; 0.40 — «книжка реально выучилась, это не mean». maxabs 2.50 — несколько σ, не взрыв книжки. |

Нули, константы, 4 повторённых вектора VQ — должны проходить **на порядок** ниже этих чисел; тесты §5 для них ставят более жёсткие assert, не полагаясь только на дефолт CLI.

### 1.6. Таблица B — живая 8B (не для unit-тестов)

**Не дефолт бинаря.** Помечать в help и в этом файле явно. На живых весах std часто 0.01–0.03 у Linear и больше у `embed`/`lm_head`; абсолютный порог синтетики либо душит embed, либо (если его ослабить глобально) пропускает мёртвый кодек на Linear.

После скачивания 8B **всегда передавать флаги явно**:

| Кодек | `--fail-rmse` | `--fail-maxabs` | Комментарий |
|---|---|---|---|
| `nf4` | `0.12` | `2.0` | Для живой модели, **не для unit-тестов**. Slack под `embed_tokens` / `lm_head`. У типичного Linear RMSE будет 10⁻³…10⁻² — порог их не «проверяет качество чата», а ловит сорванный pack (нули, перепутанные нибблы). |
| `vq` | `0.50` | `8.0` | Для живой модели, **не для unit-тестов**. Weight-only 2×8 без imatrix; ожидание по [compressor.md](../compressor.md) §7 — модель ещё говорит, PPL хуже NF4. Порог ловит NaN-книжку / перепутанные оси index, не SOTA 2-bit. |

Живой PPL / «чат на 10 вопросов» CLI **не считает**. Если `verify` прошёл по таблице B, а PPL «в космосе» — это не баг CLI, это критерий §7 компрессора / рантайма.

### 1.7. Разрешение `--in` / `--orig`

1. Путь — обычный файл и имя оканчивается на `.safetensors` (или это файл, и магия/заголовок safetensors читается) → один файл.
2. Путь — файл `*.safetensors.index.json` → шарды по `weight_map`, база = `dirname(path)`.
3. Путь — директория:
   - если есть `model.safetensors.index.json` → как п.2;
   - иначе если ровно один `*.safetensors` → он;
   - иначе код 1 (`chr: ambiguous safetensors directory`).
4. Нет файла / нет прав → код 1.

Шарды: ключ `weight_map[tensor_name]` — относительный путь от директории индекса. Итерация имён — **лексикографическая сортировка ключей** (стабильный вывод verify, независимый от порядка JSON-объекта).

Открыт **один** шард в любой момент. Сменился filename в карте — закрыть предыдущий fd, открыть новый. Запрещено держать map всех mmap.

---

## 2. Печать `verify`

Метрики на тензор считаются по **логическим** элементам (паддинг не входит), orig и reconstruct в float32 (§3.4):

```
n      = ∏ shape
mae    = (1/n) Σ |ŵ_i − w_i|
rmse   = sqrt( (1/n) Σ (ŵ_i − w_i)² )
maxabs = max |ŵ_i − w_i|
rms    = sqrt( (1/n) Σ w_i² )          # orig; только для отчёта
rel    = rmse / rms   если rms > 0, иначе 0 если rmse==0, иначе +Inf
```

`n=0` (пустой shape) → код 1, не «пропуск».

Дополнительные поля `rms` и `rel` **печатаем**, но **не** являются порогами v1 (флагов `--fail-rel` нет). На живой модели по `rel` видно, не хуже ли кодек константы 0 (`rel ≥ 1`).

### 2.1. Человеческий текст (stdout, если нет `--json`)

Кодировка UTF-8, `\n`. Первые строки — шапка, затем таблица, затем блок итога. Ширины колонок не фиксируем жёстко (имена HF длинные); разделитель полей внутри строки тензора — **два и более пробела** или таб. Для машинного разбора в тестах используют `--json`.

Пример (синтетика 3 тензора, nf4):

```
chr verify
orig:        /tmp/fake.safetensors
chr:         /tmp/fake.nf4.chr
linear_codec: nf4
thresholds:  rmse<=0.08  maxabs<=0.50  bf16=exact

name                                       shape          codec  n        rmse          mae           maxabs        rms          rel      status
model.layers.0.mlp.down_proj               [32, 128]     nf4    4096     2.134512e-02  1.540011e-02  7.812500e-02  9.912000e-01  2.15e-02  ok
model.layers.0.self_attn.q_proj             [64, 64]      nf4    4096     1.001003e-02  7.100000e-03  4.125000e-02  1.002000e+00  1.00e-02  ok
model.norm                                 [64]           bf16   64      0.000000e+00  0.000000e+00  0.000000e+00  1.000000e+00  0.00e+00  ok

worst: model.layers.0.mlp.down_proj  rmse=2.134512e-02  maxabs=7.812500e-02
summary: tensors=3  lossy=2  bf16=1  skipped=0  fail=0  mean_rmse_lossy=1.567757e-02
PASS
```

Правила:

- `status`: `ok` или `FAIL`. Для bf16 mismatch тоже `FAIL`.
- Строки тензоров — **сортировка по `name`** (как итератор §1.7).
- Числа метрик: `%.6e`. Целые — десятичные.
- `worst`: среди тензоров с `n>0` выбирается максимальный `rmse`; при равенстве — больший `maxabs`; при равенстве — лексикографически меньшее имя. Если тензоров нет — строка `worst: (none)`.
- `mean_rmse_lossy` — среднее **не взвешенное по n**, а среднее по lossy-тензорам (каждый матричный слой равен). Если lossy=0, писать `n/a`.
- Последняя строка: `PASS` если код будет 0; `FAIL` если код будет 2. При коде 1 этот шаблон может не допечататься — тогда на stderr сообщение об ошибке (§2.3).
- При коде 2 таблица **полная** (не останавливаться на первом FAIL): нужно видеть все плохие тензоры.

`--quiet`: шапку, таблицу по тензорам не печатать; остаётся `worst` + `summary` + `PASS`/`FAIL`.

### 2.2. `--json` (stdout, один объект)

Ключи snake_case. Никакого человеческого текста на stdout. На stderr — только ошибки кода 1.

```json
{
  "orig": "/tmp/fake.safetensors",
  "chr": "/tmp/fake.nf4.chr",
  "linear_codec": "nf4",
  "thresholds": {
    "rmse": 0.08,
    "maxabs": 0.50,
    "bf16_exact": true
  },
  "tensors": [
    {
      "name": "model.layers.0.mlp.down_proj",
      "shape": [32, 128],
      "codec": "nf4",
      "n": 4096,
      "rmse": 0.02134512,
      "mae": 0.01540011,
      "maxabs": 0.078125,
      "rms": 0.9912,
      "rel": 0.02154,
      "ok": true
    }
  ],
  "worst": {
    "name": "model.layers.0.mlp.down_proj",
    "rmse": 0.02134512,
    "maxabs": 0.078125
  },
  "summary": {
    "tensors": 3,
    "lossy": 2,
    "bf16": 1,
    "skipped": 0,
    "fail": 0,
    "mean_rmse_lossy": 0.01567757
  },
  "failed": [],
  "ok": true,
  "exit": 0
}
```

- `tensors` — тот же порядок, что в человеческой таблице.
- `failed` — имена с `ok=false`, лексикографически.
- `worst`: `null`, если `tensors` пуст.
- `exit` дублирует код процесса (0 или 2 в этом объекте). При коде 1 JSON **не обязан** быть валидным отчётом: можно не писать JSON вообще, только stderr. Если ошибка обнаружена **после** того, как часть тензоров уже посчитана (например extra в конце) — всё равно код 1, JSON можно не выдавать. Проще для реализатора: код 1 → только stderr.
- `ok` = (`exit==0`).
- float как JSON numbers (не строки). Тесты сравнивают с `1e-5` относительно, не байт-в-байт JSON.

### 2.3. Сообщения кода 1 (stderr)

Стабильный префикс `chr: ` — на него завязывают тесты.

| Ситуация | Сообщение (ровно шаблон) |
|---|---|
| нет файла | `chr: open <path>: …` (текст `os.PathError` допустим после префикса) |
| compressable / обязательный bf16 отсутствует в chr | `chr: missing tensor in chr: <name>` |
| имя есть в chr, нет среди хранимых orig | `chr: extra tensor in chr: <name>` |
| неподдерживаемый codec в chr | `chr: unsupported codec: <codec> (<name>)` |
| NaN/Inf в orig на encode | `chr: non-finite value in tensor <name>` |
| плохой `--codec` / `--group-size` | как в §1.2 |
| shape orig ≠ логический `shape` в chr | `chr: shape mismatch: <name> orig=<a> chr=<b>` |
| dtype orig не F32/F16/BF16 | `chr: unsupported dtype <dtype> (<name>)` |

Несколько missing/extra: можно одну строку на имя, затем общий `chr: tensor set mismatch`. Код 1, даже если метрики части тензоров уже посчитаны.

---

## 3. Что с чем сравнивать (множества имён)

### 3.1. Классификатор (один на compress и verify)

Сначала каноническое имя: если ключ orig оканчивается на `.weight` и **не** оканчивается на `.bias` — отрезать суффикс `.weight`. Дальше матч по **последнему path-компоненту** и по подстрокам (порядок — первая совпавшая ветка сверху вниз):

| Условие на каноническое имя | `kind` | Класс | В `.chr`? |
|---|---|---|---|
| содержит `inv_freq` **или** `rotary_emb` **или** суффикс `.sin` / `.cos` | skip | **skip** | нет |
| суффикс `.bias` | `other` (или kind родителя, если удобно ядру; для CLI достаточно `other`) | **store_bf16** | да, `codec=bf16` |
| последний компонент `embed_tokens` **или** имя `tok_embeddings` / `wte` | `embed` | **compress** | да, `--codec` |
| последний компонент `lm_head` **или** (`output` и имя **не** содержит `norm`) | `lm_head` | **compress** | да |
| последний компонент `q_proj` | `q` | **compress** | да |
| `k_proj` | `k` | **compress** | да |
| `v_proj` | `v` | **compress** | да |
| `o_proj` | `o` | **compress** | да |
| `gate_proj` | `gate` | **compress** | да |
| `up_proj` | `up` | **compress** | да |
| `down_proj` | `down` | **compress** | да |
| содержит `norm` или `layernorm` или `ln_f` | `norm` | **store_bf16** | да, `codec=bf16` |
| rank-2 и ни одна ветка выше | `other` | **compress** | да |
| иначе (1D/3D+ неизвестное) | `other` | **store_bf16** | да |

Почему skip на `inv_freq`/rotary, а не копия: это не веса, которые ест GEMM схемы; cos/sin кэш восстанавливается из `theta`. Лишние блобы в `.chr` только путают verify. Почему неизвестный rank-2 идёт в compress: лучше ошибочно сжать редкий Linear, чем выкинуть матрицу.

Класс **skip**:

- нет в `.chr` → **не ошибка**, счётчик `skipped++`, строки в таблице нет;
- есть в `.chr` → **extra**, код 1.

### 3.2. Контракт множеств (выбор, не «или»)

Пусть `S_store(orig)` — канонические имена с классом `compress` или `store_bf16`.
Пусть `S_chr` — ключи `tensors` в JSON CHR0.

- `S_store(orig) ⊆ S_chr` иначе **missing**, код 1.
- `S_chr ⊆ S_store(orig)` иначе **extra**, код 1.
- Итого **равенство**. Все compressable orig **обязаны** быть в chr. Extra в chr — ошибка. Skip из orig не требует присутствия и не имеет права появиться.

Почему не «пропуск missing Linear»: тогда `verify` зелёный на файле, в котором забыли `lm_head` (1.6 ГБ), а ядро на инференсе падает или берёт мусор. Это контракт файла, не порог качества → код **1**, не 2.

Порядок работы verify (псевдо):

```
header ← ReadCHR0Header(chr)          # JSON в RAM, блобы не грузить
visited ← ∅
for name in sorted(S_orig_keys):
    canon, class ← Classify(name)
    if class == skip: skipped++; continue
    if canon ∉ header.tensors: missing → err1
    W ← LoadOneOrigTensor(name)       # F32, отпустить шардовые буферы
    Ŵ ← DecodeFromCHR0(header, canon) # F32, logical shape; один тензор
    сравнить shape; накопить строку отчёта
    visited += canon
    W, Ŵ = nil
if header.tensors − visited ≠ ∅: extra → err1
напечатать отчёт
если был bf16 mismatch или lossy > порога: exit 2
иначе exit 0
```

Код 1 имеет приоритет над кодом 2: неполное множество имён не маскируем «ещё и RMSE большой».

### 3.3. Orig dtype → float32

Допускаются только `F32`, `F16` (IEEE binary16), `BF16`. Остальное — код 1.

Конвертация в F32 поэлементно, little-endian, как в safetensors. BF16: старшие 16 бит float32, младшие нули (стандартное расширение).

### 3.4. Reconstruct

- `nf4`: decode по [nf4.md](nf4.md) в float32, шкала FP16→F32, обрезать padded `n_in` до логического.
- `vq`: `C1[i1]+C2[i2]` float32, книга FP16→F32, обрезать pad.
- `bf16`: прочитать сырой BF16 blob, каждый элемент → float32.

Паддинг в метрики не входит.

### 3.5. Bit-exact для norm/bias

Сравниваем `Ŵ` с **BF16-проекцией orig**, не с «сырым F32 orig, если вдруг так записали»:

```
ref_i = float32( round_to_bf16( orig_f32_i ) )   # RNE
требуем Ŵ_i == ref_i   (побитово как float32)
```

Если orig уже BF16, `round_to_bf16` — тождество, это настоящий bit-exact orig↔chr.

Почему не сравнивать с сырым F32: контейнер хранит lossless-тензоры как BF16 ([compressor.md](../compressor.md) §6.2). F32→BF16 теряет младшие биты мантиссы; это не баг кодека. **Синтетические тесты пишут norm/bias в BF16**, тогда `ref` = orig.

Любое неравенство `Ŵ` и `ref` → `ok=false`, код 2 (порог «ноль»), не код 1.

Lossy-тензоры **не** обязаны попадать в уровень NF4 LUT / книгу относительно orig: сравниваем только метрики §2 с таблицей A/B.

### 3.6. Что не сравниваем

- Байты `.chr` с orig safetensors.
- F32 decode с orig BF16 bit-exact для `nf4`/`vq`.
- Порядок шардов, alignment 64, JSON whitespace — это уровень 1, тесты контейнера, не `verify`.
- Контрольной суммы в файле нет ([compressor.md](../compressor.md) §6.2). `verify` сам считает метрики.

---

## 4. `decode` пишет F32

Зафиксировано: **всегда F32**, флага выбора dtype нет.

Правила:

- Один выходной файл safetensors (магия, JSON header, `data_offsets`, dtype `"F32"`).
- Даже если orig был шардирован — на выходе один файл.
- Имена = ключи CHR0 (без `.weight`).
- `shape` = логический из CHR0 (например `[32, 100]`, хотя nf4 pack шёл с `n_in_padded=128`).
- Порядок тензоров в JSON: лексикографический (Go `encoding/json` так сериализует map).
- Писать стримом, не собирать все F32 в RAM: как CHR0, сначала известны размеры → можно посчитать `data_offsets` до записи payload. Один тензор в RAM (или полоса, если кто-то позовёт decode на 8B — для 8B F32 lm_head ~2.1 ГБ; decode **может** резать полосами записи, см. §7; для v1 unit-тестов полосы в decode необязательны, тензоры крошечные).
- После decode: `chr verify` **не** читает этот F32-файл; verify всегда orig safetensors vs `.chr`. Decode — для глаз / внешних скриптов.

Почему не BF16 на выходе: проверка «глазами» и numpy `float32` совпадают с тем, что считает `internal/verify`. Шкалы NF4 и книга VQ и так живут в FP16 внутри `.chr`; выход F32 не делает их точнее, он только не прячет ошибку в повторном округлении BF16.

---

## 5. Unit / golden тесты без модели

Волна 2 нарезает пакеты и **эти** функции 1:1. Фикстуры: тест сам пишет safetensors (хелпер `internal/safetensors` или test-only writer). Никакого интернета, никаких `*.bin` в git.

Пакетная раскладка (чтобы CLI-тесты не exec'али бинарь):

- `internal/chr0`, `internal/safetensors`, `internal/nf4`, `internal/vq` — кодеки и контейнер.
- `internal/verify` — метрики, классификатор, `Run(opts) Result` с `ExitCode int`.
- `cmd/chr` — разбор флагов + вызов `verify`/`compress`/`decode`. Тесты CLI: файл `run.go` с `func run(args []string) int` в package `main`, `run_test.go` рядом. **Не** `os.Exit` внутри `run`; `os.Exit` только в `main`.

Хелпер фикстуры (не отдельный пакет обязательно): записать map `name → {dtype, shape, []float32}`. Для bit-exact norm — dtype `BF16`.

Детерминированные значения без RNG глобального: `w[i,j] = float32((17*i + 13*j) % 100)/50 - 1` (диапазон примерно [-1, 0.98]).

Ниже — обязательный список. Имя = имя `func TestXxx(t *testing.T)`. Подкейсы — `t.Run`.

### 5.1. `TestContainerRoundtripBlobs`  
пакет: `internal/chr0` (можно дублировать вызовом через `cmd/chr` в подкейсе)

**Arrange.** Собрать CHR0 с тремя тензорами как в примере chr0 (q 64×64 nf4, down 32×128 nf4 или vq — не важно для контейнера, можно `codec=bf16` на всех трёх, чтобы не тянуть кодек): известные байты payload (например 128 байт `0xA5`). Записать файл. Открыть снова через `ReadAt`.

**Assert.**

- `header.magic=="CHR0"`, `version==1`.
- Для каждого тензора `ReadAt(start, end-start)` **байт-в-байт** равен записанному слайсу (`bytes.Equal`).
- Повторное чтение тех же офсетов — тот же хеш/байты (идемпотентность уровня 1).
- Не сравнивать с orig BF16.

### 5.2. `TestNF4Golden`  
пакет: `internal/nf4`

**Arrange.** Вектор/матрица и ожидаемые packed hex + FP16 scale **из `docs/spec/nf4.md`** (золотой 1×64 или 2×64 — как там зафиксируют). Encode в памяти, без `.chr`.

**Assert.**

- packed bytes = golden hex.
- scale bits = golden.
- `Decode(Encode(W))` совпадает с ручным `lut[nib]*float32(s)` побитово в F32.
- **Не** требовать `Decode(Encode(W)) == W`.
- Нулевая матрица 2×64 (подкейс `zero`): все нибблы = индекс нуля LUT, scale=1 (как nf4.md).
- CLI-подкейс необязателен; достаточно пакета `nf4`. Целостность уровня 2.

### 5.3. `TestVQToy`  
пакет: `internal/vq`

**Arrange.** Матрица `n_out=4`, `n_in=16` (8×2 группы на строку): ровно **4 различных** вектора dim 8, каждый повторён (например строки чередуют v0..v3). `seed=0`, `iters=20`, `chunk=256`. Encode → codebook FP16 + index.

**Assert.**

- reconstruct `C1[i1]+C2[i2]` из **записанной** книги совпадает с `Decode` побитово (уровень 2; это не orig).
- MSE(orig, reconstruct) ≤ `1e-4` (книга FP16; на 4 кластерах k=256 обязан выучить).
- `n_codebooks=2`, `codebook_bits=8`, `group_size=8` если смотреть через CHR0-обёртку (подкейс `via_chr` можно отложить в 5.6).

### 5.4. `TestVerifyFailCode`  
пакет: `internal/verify` + зеркало в `cmd/chr` (`TestRunVerifyExitCodes`)

**Arrange.** Один тензор compressable, 8×64, значения `w[i,j]` как формула выше (не константа, не нули). Сжать `--codec nf4` в temp dir. Два вызова `verify.Run`:

1. `--fail-rmse 1e-12 --fail-maxabs 1e-12` (непроходимо для честного NF4 на этой фикстуре).
2. те же файлы, дефолты таблицы A (или явные 0.08 / 0.50).
3. подкейс CLI: `run([]string{"verify", "--orig", …, "--chr", …, "--fail-rmse", "1e-12", "--fail-maxabs", "1e-12"})`.

**Assert.**

1. `ExitCode==2`, в отчёте `fail>=1`, `ok=false`, имя тензора в `failed`. Не 1.
2. `ExitCode==0`, `fail==0`.
3. CLI возвращает 2, stdout человеческий содержит `FAIL`, stderr без `chr: missing`.

Дополнительно в этой же функции `t.Run("input_missing_orig")`: несуществующий `--orig` → код **1**, не 2.

### 5.5. `TestPad`  
пакет: `internal/nf4` + `internal/vq` + один сквозной в `internal/verify`

**Arrange.**

- NF4: `n_out=4`, `n_in=100` (100 % 64 ≠ 0). Orig F32/BF16. Encode с pad нулями до 128.
- VQ: `n_out=2`, `n_in=12` (pad до 16).

**Assert.**

- В метаданных/CHR0 `shape` = логический `[4,100]` / `[2,12]`, не padded.
- `Decode` длина последней оси = 100 / 12.
- Метрики verify: `n=400` и `n=24`, не 512 / 32.
- Хвост pad не влияет: если orig в колонках 0..99 совпал с decode, maxabs считается только там.
- Повторный encode decode(W) на логическом куске не обязан совпадать с первым pack (lossy); для NF4 — идемпотентность второго encode как в nf4.md, если тот тест уже покрыл.

### 5.6. `TestFakeModelThreeTensors`  
пакет: `cmd/chr` (сквозной) и/или `internal/verify`

**Arrange.** Один safetensors, три тензора (имена **без** `.weight`), как крошечный пример контейнера:

| Имя | shape | dtype | класс |
|---|---|---|---|
| `model.layers.0.self_attn.q_proj` | `[64, 64]` | BF16 | compress |
| `model.layers.0.mlp.down_proj` | `[32, 128]` | BF16 | compress |
| `model.norm` | `[64]` | BF16 | store_bf16 |

Заполнить q/down формулой §5; norm = 1.0. Два прогона compress: `--codec nf4` и `--codec vq --seed 0`. Затем `decode` в `out.safetensors` и `verify --orig` vs каждый `.chr`.

**Assert.**

- `S_chr` = три имени, `kind` q / down / norm, кодеки nf4|vq / nf4|vq / bf16.
- verify код 0 на дефолтах таблицы A для соответствующего кодека.
- `model.norm`: rmse=mae=maxabs=0, `codec=bf16`.
- `decode`: dtype F32, три тензора, shape логические, `n` элементов совпадает; q/down **не** bit-exact с orig; norm — bit-exact с orig BF16→F32.
- `--json` парсится, `summary.tensors==3`, `summary.bf16==1`, `summary.lossy==2`.
- Пик: в verify нет момента, где одновременно живы все три orig-буфера (в тесте достаточно проверить контракт API: `Load` по одному; горутин нет). Жёсткий RSS-assert в unit не требуется.

### 5.7. Ещё тесты без модели (волна 2 — тоже обязательны, иначе CLI дырявый)

Имена зафиксированы, чтобы не потерялись.

| Функция | Пакет | Arrange | Assert |
|---|---|---|---|
| `TestVerifyMissingCompressable` | `internal/verify` | fake из 5.6, в chr удалить `down_proj` (или сжать и подменить JSON — проще: собрать chr из двух тензоров) | код **1**, сообщение `missing tensor in chr: model.layers.0.mlp.down_proj`, не код 2 |
| `TestVerifyExtraTensor` | `internal/verify` | orig только `model.norm`; chr с q_proj + norm | код **1**, `extra tensor in chr: …` |
| `TestVerifySkipInvFreq` | `internal/verify` | orig: `q_proj` 8×64 + `model.layers.0.self_attn.rotary_emb.inv_freq` 1D; compress | inv_freq нет в chr; verify код 0, `skipped==1` |
| `TestRunCompressRequiresCodec` | `cmd/chr` | `run({"compress","--in",p,"--out",q})` | код 1 |
| `TestRunHelpExitZero` | `cmd/chr` | `run({"-h"})` и `run({"verify","-h"})` | код 0 |
| `TestDecodeWritesF32` | `internal/safetensors` или `cmd/chr` | decode после 5.6 | header dtype каждого тензора `"F32"` |

Golden hex NF4 не дублировать здесь числами — источник истины `nf4.md`. Если на момент волны 2 `nf4.md` ещё без hex, `TestNF4Golden` использует инварианты zero/constant из ТЗ nf4 и помечает hex `t.Skip` нельзя: тогда assert zero-матрицы и константы `c=0.5` (один уровень, `scale=|c|`).

---

## 6. После скачивания (живой прогон, весов в репо нет)

Веса **не** коммитить. Предполагается локальный снимок HF, переменная `$MODEL` — директория с `model.safetensors.index.json` (или одним файлом).

Команды — иллюстрация; пороги **таблица B**, не дефолт:

```
# NF4, один файл на диск
chr compress --in "$MODEL" --out /data/llama8b.nf4.chr --codec nf4 --quiet
chr verify  --orig "$MODEL" --chr /data/llama8b.nf4.chr \
    --fail-rmse 0.12 --fail-maxabs 2.0 --json > /data/llama8b.nf4.verify.json

# VQ 2×8 weight-only, тот же orig, другой файл
chr compress --in "$MODEL" --out /data/llama8b.vq2.chr --codec vq \
    --seed 0 --iters 20 --chunk 262144 \
    --stripe-bytes 268435456 --stripe-rows 4096
chr verify  --orig "$MODEL" --chr /data/llama8b.vq2.chr \
    --fail-rmse 0.50 --fail-maxabs 8.0 --json > /data/llama8b.vq2.verify.json
```

Полный F32 dump **не** нужен для приёмки и съест ~32 ГБ:

```
# только если есть место; не часть unit
chr decode --in /data/llama8b.nf4.chr --out /tmp/llama8b.nf4.f32.safetensors
```

Ожидание по [compressor.md](../compressor.md) §7 (это **не** код выхода `chr`):

1. 8B NF4: WikiText не хуже +1.0 к BF16; чат связный. `verify` таблица B должен быть PASS ещё до PPL — иначе pack сломан и PPL не о чём.
2. 8B VQ: снять PPL и записать; чат «жив/мёртв». Мёртвый чат при PASS `verify` означает «кодек честно lossy», не «надо чинить CLI».
3. Сравнение ядра с llama.cpp Q4 — не этот бинарь.

Пик RSS/VRAM компрессора на живой 8B: одна матрица + полоса (§7). Если `embed_tokens` / `lm_head` (~1.05 ГБ BF16) обрабатываются без полос при `--stripe-bytes` по умолчанию — это баг реализации §7.

---

## 7. Границы памяти

### 7.1. Жёсткие правила (все команды)

1. **Один открытый шард.** LRU=1 файл. Не `errgroup` по шардам.
2. **Один логический тензор** (или одна его полоса) в рабочих буферах. После encode/decode/метрики слайсы отпускаются; не копить `[][]float32` всех слоёв.
3. Не держать одновременно orig BF16 и полную F32-копию **всей** матрицы: конвертировать **полосу**.
4. Заголовок CHR0 (JSON) — в RAM целиком. Это мегабайты, не гигабайты.
5. Выходные блобы: append на диск, не `[]byte` всей сжатой модели.
6. Горутины поверх тензоров в v1 запрещены (простота учёта RSS).

Пик, к которому стремимся на CPU-хосте при живом lm_head 32B (~1.56 ГБ BF16, `n_out≈128256`, `n_in=5120`):

```
F32-полоса 4096 × 5120 × 4 ≈ 80 МБ
+ index-полоса VQ + книга 8 КБ
+ chunk assignment 1e6 × 256 × 4 ≈ 1 ГБ  ← режет --chunk
≈ 1–1.2 ГБ , не 1.56×4 ГБ F32 всей матрицы
```

### 7.2. Когда включать полосы

Пусть `n_out`, `n_in` — логический shape матрицы (2D). Рабочий размер «всей матрицы в F32»:

```
full_f32 = n_out * n_in * 4
```

Если `full_f32 > --stripe-bytes` **и** тензор 2D → режим полос. Иначе грузить тензор целиком (unit-тесты всегда целиком: 64×64 F32 = 16 КБ).

Высота полосы:

```
rows = min(--stripe-rows, n_out)
уменьшать rows, пока rows * n_in * 4 > --stripe-bytes и rows > 1
если даже rows=1 не влезает в --stripe-bytes — всё равно одна строка (прогресс важнее лимита)
```

Для NF4 группа вдоль `n_in`: полоса по строкам **не ломает** шкалы. Encode/decode строки независимы. Один проход по полосам, блобы `data`/`scale` пишутся последовательно row-major как в [compressor.md](../compressor.md) §6.2.

1D (norm) никогда не полосуется.

v1 unit-тесты **не** обязаны прогонять полосы (фикстуры < порога). Тест на сам порог можно не писать без 300 МБ фикстуры. Реализация для живого 1.6 ГБ `lm_head` — **обязана** быть в compress/verify/decode.

### 7.3. VQ на полосах — двухпроходная книга

k-means на подмножестве, assignment на всех. Книга **на матрицу**, не на полосу.

**Проход I — init (subsample → Lloyd).**

Не брать только первые строки `embed`/`lm_head`: это спецтокены, книга будет мусором.

Фиксированный алгоритм subsample:

- `vecs_per_row = n_in_padded / 8`
- `target = min(N, 65536)`, `N = n_out * vecs_per_row`
- `rows_needed = ceil(target / vecs_per_row)`
- `stride = max(1, n_out / rows_needed)`
- взять строки `0, stride, 2*stride, …` пока не наберётся `target` векторов (последняя неполная строка — обрезать).
- На этом множестве: k-means++ (`--seed`) и Lloyd `--iters` по [vq.md](vq.md). Получить `C[0:M]`.
- Пустые кластеры — респлит как в vq.md, тот же seed-stream.

Это **один** последовательный проход чтения orig (только выбранные строки; невыбранные не конвертировать в F32).

**Проход II — assign.**

- Книга заморожена. Все строки, полосами.
- Assignment **чанками `--chunk` векторов**, не материализовать `(N, 256, 8)`.
- Писать `index` полосами в заранее известные офсеты (`n_out` известен с заголовка orig).
- Residual второй книги: как в vq.md (после C1 на векторе), без второго обучения на полном N в v1 полосного режима. (Если тензор **не** полосатый — полный residual k-means по vq.md, оба обучения на всех векторах.)

Итого для огромного тензора: 2 чтения orig (init subsample + assign). Для маленького: 1 чтение, полный алгоритм vq.md.

`--chunk` режет только assignment/update GEMM-подобный цикл, это **не** высота полосы. На одной полосе 4096×14336 векторов = 7.3e6, внутренний цикл всё равно пачками по 1e6.

### 7.4. NF4 и полосы

Один проход. На полосу: F32 rows × `n_in`, pack, append `data` и `scale`, забыть полосу. Double-quant нет.

### 7.5. `verify` и полосы

Те же пороги `full_f32 > stripe-bytes`. Считать метрики **накопительно** по полосам:

```
sum_sq_err, sum_abs, maxabs, n, sum_sq_orig
```

Не собирать полный `ŵ−w`. RMSE в конце из сумм. Иначе 8B lm_head снова 2×2 ГБ.

Orig полоса и chr-decode полоса одной высоты; индексы строк совпадают.

### 7.6. Чего не делать

- Не читать два шарда prefetch «на всякий».
- Не mmap 32B целиком как запасной путь в v1 (на Windows/WSL и так `ReadAt`).
- Не держать F32 всей модели «для удобства JSON».
- Не включать полосы в unit-тестах синтетики (порог 256 MiB их не заденет). Не писать тест, который выделяет 2 ГБ.

---

## 8. Стык с соседними спеками (поля, не байты)

CLI **не** переопределяет расклад. Для compress обязано получиться:

**nf4** (на тензор): `codec=nf4`, `group_size=64`, `shape` логический, блобы `data` uint8 `[n_out, n_in_padded/2]`, `scale` FP16 `[n_out, n_in_padded/64]`. Нет `zero`.

**vq:** `codec=vq`, `group_size=8`, `n_codebooks=2`, `codebook_bits=8`, `codebook` FP16 `[2,256,8]`, `index` uint8 `[n_out, n_in_padded/8, 2]`.

**bf16:** `codec=bf16`, `data` сырой little-endian BF16, `shape` как orig.

`kind`, `layer` — по классификатору §3.1. `tile` в корне заголовка: `{ "row": 64, "col_group": 8 }` как в [compressor.md](../compressor.md) §6.1 (для ядра; CPU-verify тайлы не перекладывает, блобы row-major).

Если `chr0.md` к моменту волны 2 уточнит запись header (два прохода vs sidecar) — compress следует **ему**. Здесь достаточно: после успешного compress файл читается `internal/chr0` и проходит уровень 1.

Контрольной суммы в `.chr` нет. Соседний `*.sha256` CLI не пишет и не проверяет.

---

## 9. Коды выхода и `run()`

| Код | Когда |
|---|---|
| 0 | команда сделала то, что обещала; у verify все обязательные тензоры сверены и пороги/bit-exact ок |
| 1 | флаги, пути, dtype, NaN, множество имён, I/O, неподдерживаемый codec, group-size, дырявый JSON CHR0, `header_nbytes` врёт |
| 2 | только verify: множество имён сошлось, метрики посчитаны, хотя бы один тензор `ok=false` |

`ctrl-c` / `ctx.Done`: не специфицируем отдельный код; процесс умрёт сигналом.

`internal/verify.Result`:

```
ExitCode int
Report   # тот же граф, что JSON §2.2
Err      error  # для кода 1; для 0/2 может быть nil
```

`cmd/chr` печатает Report и возвращает `ExitCode`.

---

## 10. Что волна 2 не должна спрашивать

- Дефолт `--codec` — нет, флаг обязателен.
- Decode dtype — F32.
- Missing Linear — ошибка 1, не skip.
- Extra в chr — ошибка 1.
- inv_freq — skip.
- Нормы — bit-exact BF16-проекции.
- nf4/vq — не bit-exact к orig; пороги таблица A в дефолте, таблица B только в §6.
- Полосы — с `full_f32 > 256MiB`; VQ тогда init по strided subsample 64k, assign по всем.
- Тесты — имена §5.1–5.7, без LLM.
)

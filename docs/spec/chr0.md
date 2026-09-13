# CHR0 v1: контейнер `.chr` и стрим safetensors (CPU, Go без CGO)

Стык: [compressor.md](../compressor.md) §6 (расклад файла) и §9 шаг 2 (писалка). Кодеки NF4/VQ **не** определяются здесь — только слоты под их блобы. Инференс и GPU вне среза.

Пакеты второй волны: `internal/chr0` (этот формат) и `internal/safetensors` (чтение HF). Реализация: стандартная библиотека, `encoding/binary`, `os.File`, **без CGO и без mmap**.

---

## Развилки (сводка — дальше только эти выборы)

| Вопрос | Выбор | Почему |
|---|---|---|
| Как писать JSON vs блобы | Два прохода по **метаданным**, JSON пишется **сразу в финальном размере**, затем блобы. Не sidecar блобов, не «дыра» под JSON | Размеры блобов зависят только от `shape`+кодека, не от значений весов. JSON известен до чтения весов; слот не может «не влезть» |
| mmap vs `ReadAt` | Только `os.File.ReadAt` / последовательный `Write` | Windows/WSL; mmap не нужен для CPU-verify |
| Не-веса (`inv_freq`, rotary, …) | **Пропускать** (нет в `.chr`) | RoPE восстанавливается из `theta`; CLI-тест `TestVerifySkipInvFreq`. Не квантовать. |
| Несколько кодеков в одном файле | Запрещено для Linear/embed/lm_head | ТЗ: один кодек на файл; пример со смешанным nf4+vq в compressor.md §6.1 **не** является нормой v1 |
| Порядок ключей `tensors` | Лексикографическая сортировка UTF-8 | `encoding/json` так сериализует `map[string]T`; не писать свой marshaler |
| Порядок **блобов** на диске | Порядок обхода шардов/тензоров (не JSON-ключи) | Один открытый шард, один тензор; офсеты в JSON связывают имя с байтами |
| `int4` | В схеме чтения есть, писалка v1 **не** создаёт | ТЗ |
| Контрольная сумма в файле | Нет | `verify` считает метрики сам |

---

## 1. Байт за байтом

### 1.1. Карта файла

Все многобайтовые целые в бинарной части — **little-endian** (`binary.LittleEndian`). JSON — текст UTF-8 без BOM.

```
offset 0                uint64le   header_nbytes = N     # длина JSON в байтах, не включая эти 8 байт и не включая pad
offset 8                N байт      header_json           # ровно один JSON-объект
offset 8+N              P байт      pad                   # нули, P = Align64(8+N) − (8+N), 0 ≤ P ≤ 63
offset Align64(8+N)     …           payloads              # каждый блоб начинается на кратном 64
```

Конца-файла-магии нет. Длина файла после успешной записи = `Align64(max_end)`, где `max_end` — максимум всех `end` в заголовке.

`Align64(x)` для неотрицательного целого `x`:

```
Align64(x) = x,                 если x % 64 == 0
           = x + (64 − x % 64), иначе
```

На двухместном дополнении / Go `uint64`: `(x + 63) &^ 63`. Считать **от начала файла**, не «от начала секции».

### 1.2. Offset 0: `header_nbytes`

Прочитать ровно 8 байт. Интерпретация: `uint64` LE.

| Условие | Действие ридера |
|---|---|
| Файл короче 8 байт (в т.ч. пустой) | Ошибка `truncated` |
| `N == 0` или `N == 1` | Ошибка `header_nbytes` (нет валидного объекта `{…}`) |
| `N > 100_000_000` | Ошибка `header_too_large` (тот же потолок, что у safetensors) |
| `8+N > size(file)` | Ошибка `header_nbytes` («врёт»: JSON вылезает за EOF) |

Писалка: `N = len(compact_json)`, без `\n` в конце.

### 1.3. Offset 8: JSON

Прочитать ровно `N` байт. Это **не** C-строка: нулевой байт внутри — часть JSON, и он **запрещён** (см. ниже).

Ридер:

1. Если `N≥1` и байт по offset 8 ≠ `0x7B` (`{`) — ошибка `json_invalid` (в т.ч. BOM `EF BB BF`).
2. `json.Decoder` / `Unmarshal` в объект. Использовать `UseNumber()` (или разбор офсетов через `json.Number`): офсеты не должны пройти как `float64` с дробной частью.
3. После одного top-level value внутри этих `N` байт допускаются только ASCII whitespace: `0x20`, `0x09`, `0x0A`, `0x0D`. Любой другой хвост (второй объект, `NUL`, мусор) — ошибка `json_invalid`.
4. Невалидный UTF-8, оборванный суррогат, не-JSON — ошибка `json_invalid`. Не «чинить» и не искать `{` дальше по файлу.
5. Если JSON валиден, но это массив / число / `null` — ошибка `json_invalid` (нужен объект).

Если `header_nbytes` **больше** реального JSON и в хвост `N` байт попал pad или начало блоба: либо parse error, либо не-whitespace хвост → всё равно ошибка. Если `N` **меньше** реального объекта — parse error. Ридер **не** сканирует файл в поисках `}`.

`DisallowUnknownFields` на корне и на каждом тензоре: неизвестный ключ — ошибка. В v1 нет «игнорировать для forward-compat».

### 1.4. Pad после JSON

`P = Align64(8+N) − (8+N)`.

Писалка пишет `P` байт `0x00`.

Ридер **не** проверяет, что pad — нули, и не проверяет байты «дыр» между блобами. Он только пропускает их по формуле. (Проверка содержимого pad не даёт целостности весов.)

Если `8+N` уже кратно 64, `P=0`, первый блоб начинается сразу после JSON.

Численный пример (этот же файл — §2.4): `N=516` = `0x204`, байты `0..7`:

```
04 02 00 00 00 00 00 00
```

`8+516=524`, `Align64(524)=576`, `P=52` нуля. Первый блоб — offset **576**.

Ещё пример pad блоба (не из игрушечной модели): блоб `[start,end)=[576,582)` (6 байт BF16 × 3), следующий старт = `Align64(582)=640`. Писалка пишет 6 байт данных и 58 нулей.

### 1.5. Блобы

Каждый payload:

- `start % 64 == 0`
- `end > start`
- `end − start` = точный размер содержимого **без** pad
- байты `[start, end)` — содержимое; `[end, Align64(end))` — pad (нули у писалки)

Офсеты `[start, end)` считаются **от начала файла** (не от конца заголовка). Это главное отличие от safetensors.

Перекрытие любых двух `[start,end)` — ошибка (ридер и писалка).

Писалка: после последнего блоба, если `max_end % 64 ≠ 0`, дописать нули до `Align64(max_end)`. Ридер: достаточно `size(file) ≥ max_end`; лишние байты в хвосте **игнорирует** (не ошибка), недостача — ошибка `truncated`.

### 1.6. Endian содержимого блобов

| Содержимое | Байтовый порядок |
|---|---|
| `uint8` (нибблы NF4, индексы VQ) | байт как есть |
| FP16 / BF16 | 16-битный code unit, little-endian |
| JSON-числа | текст, не LE/BE |

Платформа реализации — x86_64 / Windows WSL: native = LE, но писать всё равно через `binary.LittleEndian`, не через host-cast структуры с padding.

---

## 2. JSON-схема заголовка

### 2.1. Корень (все поля обязательны, других ключей нет)

| Ключ | Тип JSON | Ограничения |
|---|---|---|
| `magic` | string | Ровно `"CHR0"` (регистр фиксирован) |
| `version` | integer | Ровно `1`. Иное — ошибка `unsupported version` |
| `arch` | string | Непустая, UTF-8. Источник — §5.6. Не enum |
| `hidden_size` | integer | `≥ 1` |
| `intermediate_size` | integer | `≥ 0` (`0` допустим, если в файле нет MLP) |
| `num_layers` | integer | `≥ 0` |
| `vocab_size` | integer | `≥ 0` (`0` допустим без embed/lm_head) |
| `tile` | object | Ровно `{"row":64,"col_group":8}`. Другие числа в v1 — ошибка |
| `tensors` | object | Не пустой. Ключ — имя тензора CHR0 |

Порядок ключей корня на диске (писалка, struct tags):

`magic`, `version`, `arch`, `hidden_size`, `intermediate_size`, `num_layers`, `vocab_size`, `tile`, `tensors`.

`tile`: только `row` и `col_group`, оба integer, оба обязательны. Это константа среза (тайл Ampere 64×8 из схемы), CPU-verify её не интерпретирует, но ридер **проверяет** значения.

Числа корня — JSON integer без `.` и `e`. Знак `+` не писать. Ридер отвергает `"4096"` (строка) и `4096.0`.

### 2.2. Имя тензора (ключ в `tensors`)

- Как в HuggingFace, **без** суффикса `.weight`.
- Bias: полное имя **с** `.bias`, например `model.layers.0.self_attn.q_proj.bias`.
- Регистр и точки как в источнике. Сравнение имён — байтовое, case-sensitive.
- Длина имени 1…1024 байт UTF-8. Запрещены `U+0000` и ASCII control `0x00–0x1F`.
- Ключи объекта `tensors` на диске: **сортировка по возрастанию байт UTF-8** (как `encoding/json` для map).

Два разных HF-имени, после правила §5.5 дающие одно CHR0-имя — ошибка на записи.

### 2.3. Объект тензора

Общих ключей «на все кодеки» четыре; остальные — по кодеку. Неизвестные ключи запрещены. Лишние ключи чужого кодека запрещены (у `bf16` нет `group_size`).

**Всегда:**

| Ключ | Тип | Правило |
|---|---|---|
| `kind` | string | Ровно одно из: `q`, `k`, `v`, `o`, `qkv`, `gate`, `up`, `down`, `embed`, `lm_head`, `norm`, `other` |
| `codec` | string | `bf16` \| `nf4` \| `int4` \| `vq` |
| `shape` | array of integer | Длина 1 или 2; каждый элемент `≥ 1`; rank 3+ запрещён |
| `layer` | integer | См. ниже |

`layer`:

- Если имя содержит сегмент `layers.<n>` как целую dotted-компоненту (первая такая слева направо, `n` десятичное без знака) — `layer` **обязателен** и равен этому `n` (`0 ≤ layer < num_layers`).
- Иначе (`model.norm`, `model.embed_tokens`, `lm_head`, …) ключа `layer` **нет**.
- `layer: 0` **пишут явно**. Писалка **не** использует `json:",omitempty"` на `int` (иначе нулевой слой исчезнет).

`shape` — **логический** (без паддинга `n_in`). Для матрицы `[n_out, n_in]` ось 0 = строки = `n_out`, ось 1 = `n_in` (как `nn.Linear.weight` в PyTorch). Для вектора `[n]`.

Пара `[start, end)` — JSON-массив ровно из двух integer, `0 ≤ start < end ≤ 2^63-1`, `start % 64 == 0`, `end − start` равен формуле размера блоба (§7, §10).

Порядок ключей внутри тензора (присутствующие):  
`layer`, `kind`, `codec`, `shape`, `group_size`, `n_codebooks`, `codebook_bits`, `data`, `scale`, `zero`, `codebook`, `index`.

#### codec `bf16`

Обязателен `data`. Запрещены: `group_size`, `scale`, `zero`, `codebook`, `index`, `n_codebooks`, `codebook_bits`.

`end − start = 2 * Π shape[i]` (каждый элемент — 2 байта BF16).

Допустим для любого `kind`. Для `kind=norm` и имён с суффиксом `.bias` и для `kind=other` — **единственный** допустимый codec.

#### codec `nf4`

Обязательны: `group_size` (ровно `64`), `data`, `scale`.  
Запрещены: `zero`, `codebook`, `index`, `n_codebooks`, `codebook_bits`.  
`shape` только rank 2.

#### codec `vq`

Обязательны: `group_size` (ровно `8`), `n_codebooks` (ровно `2`), `codebook_bits` (ровно `8`), `codebook`, `index`.  
Запрещены: `data`, `scale`, `zero`.  
`shape` только rank 2.

#### codec `int4` (только чтение чужих файлов)

Как `nf4`, плюс опциональный `zero` той же длины, что `scale`. Писалка v1 **никогда** не ставит `codec=int4`. Ридер контейнера принимает объект и отдаёт сырые блобы; декодера int4 в этом срезе нет.

#### Инвариант «один кодек на файл»

Пусть `Q` — множество `codec` у тензоров с `kind ∈ {q,k,v,o,qkv,gate,up,down,embed,lm_head}` и имя **не** оканчивается на `.bias`.

- Если `Q` непусто, то `Q` — синглтон `{nf4}` или `{vq}` или `{int4}`. Смесь или `bf16` в `Q` — ошибка ридера.
- Нормы, bias, `other` в `Q` не входят и всегда `bf16`.

### 2.4. Пример: игрушечная «модель» из 3 тензоров, `--codec nf4`

Исходник safetensors (порядок тензоров в ST-заголовке = порядок блобов):

| HF-имя | shape | dtype |
|---|---|---|
| `model.layers.0.self_attn.q_proj.weight` | `[64,64]` | BF16 |
| `model.layers.0.mlp.down_proj.weight` | `[32,128]` | BF16 |
| `model.norm.weight` | `[64]` | BF16 |

Рядом `config.json`: `model_type=toy`, `hidden_size=64`, `intermediate_size=128`, `num_hidden_layers=1`, `vocab_size=0`.

Размеры блобов (паддинг `n_in` не нужен: 64 и 128 уже кратны 64):

| CHR0-имя | блоб | байт |
|---|---|---|
| `model.layers.0.self_attn.q_proj` | `data` | `64 * (64/2) = 2048` |
| то же | `scale` | `64 * (64/64) * 2 = 128` |
| `model.layers.0.mlp.down_proj` | `data` | `32 * (128/2) = 2048` |
| то же | `scale` | `32 * (128/64) * 2 = 128` |
| `model.norm` | `data` | `64 * 2 = 128` |

Компактный JSON (это **байтовый эталон** заголовка, `N=516`):

```
{"magic":"CHR0","version":1,"arch":"toy","hidden_size":64,"intermediate_size":128,"num_layers":1,"vocab_size":0,"tile":{"row":64,"col_group":8},"tensors":{"model.layers.0.mlp.down_proj":{"layer":0,"kind":"down","codec":"nf4","shape":[32,128],"group_size":64,"data":[2752,4800],"scale":[4800,4928]},"model.layers.0.self_attn.q_proj":{"layer":0,"kind":"q","codec":"nf4","shape":[64,64],"group_size":64,"data":[576,2624],"scale":[2624,2752]},"model.norm":{"kind":"norm","codec":"bf16","shape":[64],"data":[4928,5056]}}}
```

Тот же объект читаемо (офсеты те же; **не** класть на диск с пробелами — `N` и все `[start,end)` изменятся):

```json
{
  "magic": "CHR0",
  "version": 1,
  "arch": "toy",
  "hidden_size": 64,
  "intermediate_size": 128,
  "num_layers": 1,
  "vocab_size": 0,
  "tile": { "row": 64, "col_group": 8 },
  "tensors": {
    "model.layers.0.mlp.down_proj": {
      "layer": 0,
      "kind": "down",
      "codec": "nf4",
      "shape": [32, 128],
      "group_size": 64,
      "data": [2752, 4800],
      "scale": [4800, 4928]
    },
    "model.layers.0.self_attn.q_proj": {
      "layer": 0,
      "kind": "q",
      "codec": "nf4",
      "shape": [64, 64],
      "group_size": 64,
      "data": [576, 2624],
      "scale": [2624, 2752]
    },
    "model.norm": {
      "kind": "norm",
      "codec": "bf16",
      "shape": [64],
      "data": [4928, 5056]
    }
  }
}
```

Карта файла:

| Регион | `[start, end)` | содержимое |
|---|---|---|
| `header_nbytes` | `[0, 8)` | `04 02 00 00 00 00 00 00` |
| JSON | `[8, 524)` | 516 байт UTF-8 |
| pad JSON | `[524, 576)` | 52 × `00` |
| q `data` | `[576, 2624)` | 2048 × uint8 |
| q `scale` | `[2624, 2752)` | 128 × FP16 |
| down `data` | `[2752, 4800)` | 2048 × uint8 |
| down `scale` | `[4800, 4928)` | 128 × FP16 |
| norm `data` | `[4928, 5056)` | 128 × BF16 |
| EOF | `5056` | `5056 % 64 == 0` |

Ключи в JSON: `down_proj` раньше `q_proj` (лексикографически). Блобы: сначала `q_proj`, потому что так лежит в safetensors. Это нормально.

### 2.5. Те же три тензора, `--codec vq` (поля, не второй эталон файла)

Блобы в порядке обхода:

| имя | блоб | байт | `[start,end)` при `N=593` |
|---|---|---|---|
| q_proj | `codebook` | `2*256*8*2=8192` | `[640, 8832)` |
| q_proj | `index` | `64*(64/8)*2=1024` | `[8832, 9856)` |
| down_proj | `codebook` | 8192 | `[9856, 18048)` |
| down_proj | `index` | `32*(128/8)*2=1024` | `[18048, 19072)` |
| norm | `data` | 128 | `[19072, 19200)` |

`N=593`, `8+593=601`, `Align64(601)=640`. Фрагмент q_proj:

```json
"model.layers.0.self_attn.q_proj": {
  "layer": 0,
  "kind": "q",
  "codec": "vq",
  "shape": [64, 64],
  "group_size": 8,
  "n_codebooks": 2,
  "codebook_bits": 8,
  "codebook": [640, 8832],
  "index": [8832, 9856]
}
```

Книга **на матрицу**, не на слой: у `q` и `down` **разные** диапазоны `codebook`.

---

## 3. Алгоритм записи

### 3.1. Почему не sidecar и не «дыра»

compressor.md §9: «заголовок в память, блобы append, в конце переписать JSON». Это работает только если заранее оставить дыру ≥ финального JSON. Если после блобов JSON вырос (другие цифры офсетов) и не влез — дыра бесполезна.

**Выбор:** размеры всех блобов известны из `shape`+кодека **до** чтения весов. Писалка:

1. Проход A: только заголовки safetensors + `index.json` + `config.json` (килобайты).
2. Стабилизация `N` и офсетов (цикл ниже).
3. Запись `uint64`+JSON+pad.
4. Проход B: один шард, один тензор, encode, записать блобы по уже известным офсетам, исходный буфер отпустить.

Временного файла с сырыми блобами нет. Атомарность: писать в `<out>.part` в том же каталоге, `Close`, затем `os.Rename` на `<out>`. При ошибке удалить `.part`. (Это не sidecar блобов: `.part` и есть выход.)

Если JSON «вырос» — это шаг 2 **до** любого веса; цикл пересчитывает офсеты. Ситуации «блобы уже записаны, JSON не влезает» нет. Если цикл не сошёлся за 8 итераций — ошибка `header_not_stable` (баг стабилизации, не «увеличь дыру»).

### 3.2. Проход A — список тензоров

1. Построить план шардов (§5.4).
2. Для каждого шарда: открыть, разобрать **только** ST-заголовок, закрыть.
3. Для каждого тензора шарда, который входит в план:
   - dtype ∈ {`F32`,`F16`,`BF16`}, иначе ошибка;
   - CHR0-имя = §5.5;
   - `kind`+`codec` = §6;
   - `layer` = §2.3;
   - запомнить `shape`, файл шарда, ST-`data_offsets`, dtype.
4. Проверить уникальность CHR0-имён. Пустой список — ошибка `no tensors`.
5. Посчитать `arch`, размеры модели (§5.6).
6. Для каждого тензора список блобов в **фиксированном порядке полей**:
   - `bf16`: `data`;
   - `nf4`: `data`, `scale`;
   - `vq`: `codebook`, `index`.
   Длины — §7 / §10.

### 3.3. Стабилизация заголовка

Офсеты входят в JSON, `N` зависит от числа цифр, старты блобов зависят от `N`. Цикл:

```
N ← 0
повторить не более 8 раз:
    pos ← Align64(8+N)
    для тензоров в порядке обхода (проход B):
        для блоба тензора в порядке полей:
            start ← pos
            end ← start + nbytes
            запомнить [start,end)
            pos ← Align64(end)
    json ← CompactJSON(корень + tensors с этими офсетами)
    если len(json) == N: готово
    N ← len(json)
    если N > 100_000_000: ошибка header_too_large
иначе: ошибка header_not_stable
```

`CompactJSON`: без пробелов, без HTML-escape (`SetEscapeHTML(false)`), UTF-8, ключи `tensors` отсортированы, корень — struct (не map). `layer:0` присутствует.

На практике сходится за 2 итерации (как в примере: 515→516).

`file_size = Align64(max_end)` после стабилизации. Можно `Truncate(file_size)` на `.part` до записи блобов.

### 3.4. Проход B — веса

Инвариант: **открыт ≤ 1 шард, в RAM ≤ 1 исходный тензор + его закодированные блобы + JSON**.

```
открыть .part, Truncate(file_size)
WriteAt(0, uint64le(N))
WriteAt(8, json)
WriteAt(8+N, P нулей)

current_shard ← нет
для каждого тензора в порядке обхода:
    если шард ≠ current_shard:
        закрыть предыдущий шард
        открыть новый (os.Open)
        current_shard ← этот
    raw ← ReadAt шарда по ST data_offsets        # native F32/F16/BF16
    blobs ← Encode(codec, kind, raw, dtype, shape)  # см. ниже
    raw отпустить (не класть в слайс «все тензоры»)
    для каждого блоба:
        если len(bytes) ≠ end-start: ошибка
        WriteAt(start, bytes)
        дописать нули до Align64(end), если это не дырка перед следующим (при плотной укладке WriteAt следующего затрёт pad — всё равно записать pad явно)
закрыть шард
Close(.part), Rename
```

Последовательный `Write` вместо `WriteAt` допустим, если писать строго в порядке возрастания `start` и не пропускать pad. Порядок обхода = порядок возрастания `start` (так укладываем).

`Encode`:

- `bf16`: привести к BF16 LE row-major (§5.7), вернуть один блоб `data`.
- `nf4` / `vq`: привести к `float32` row-major логического `shape` (без pad — паддинг колонок делает кодек) и вызвать пакет кодека. Контейнер **не** знает LUT и k-means. Получает `map`/структуру блобов и проверяет длины.

Писалка **не** держит `[][]byte` всех блобов. После `WriteAt` блоб отпускается.

Не открывать шарды параллельно. Не prefetch соседний тензор.

### 3.5. Ошибка посередине прохода B

Удалить `.part`. Не оставлять полузаписанный `<out>`. Не пытаться «дописать JSON в начало» после частичных блобов.

---

## 4. Алгоритм чтения

### 4.1. Открытие

```
f ← os.Open(path)                    # не mmap, не syscall.Mmap
N ← binary.LittleEndian.Uint64(ReadAt(0, 8))
проверить N как в §1.2
js ← ReadAt(8, N)
разобрать и валидировать JSON (§1.3, §2)
для каждого блоба каждого тензора:
    проверить start/end, размеры формул, start%64==0, end≤size(f)
    пересечения — ошибка
заголовок держать в памяти; файловый дескриптор держать открытым
```

`mmap` не использовать даже на Linux: один код path для Windows/WSL. Последовательное чтение всех тензоров — всё равно `ReadAt` (или `io.NewSectionReader(f, start, len)`).

Не читать все блобы при `Open`.

### 4.2. Достать один тензор по имени

Имя — **CHR0-имя** (без `.weight`). Точное совпадение. Алиаса с `.weight` нет: это работа CLI, не контейнера.

```
info ← tensors[name]   # нет ключа → ошибка not_found
result.kind, codec, shape, layer ← из info
для каждого ключа блоба, который есть у codec:
    buf ← make([]byte, end-start)
    ReadAt(buf, start)               # ровно end-start байт, иначе truncated
    result.blobs[key] ← buf
вернуть result, не декодируя nf4/vq
```

Пик RAM на чтении одного тензора ≈ сумма его блобов, не весь `.chr`.

Повторное чтение того же имени: снова `ReadAt` (без кэша всех тензоров). Кэш заголовка — да.

`Close` закрывает `*os.File`.

### 4.3. Чтение «как итератор» (compress-обратный verify)

Verify не должен грузить модель целиком: итерация имён (порядок — sorted keys или порядок обхода — неважно для метрик) и `Get` по одному. После обработки тензора блобы отпускаются.

---

## 5. Парсер safetensors и шарды

### 5.1. Магии нет

Файл safetensors **не** начинается с ASCII-магии. Карта:

```
offset 0        uint64le  st_header_nbytes = Ns     # 2 ≤ Ns ≤ 100_000_000
offset 8        Ns байт   JSON-объект
offset 8+Ns     …         сырые тензоры
```

**Нет** pad-до-64 в safetensors. Данные начинаются сразу после JSON. `data_offsets` в ST — это `[start, end)` **от начала секции данных**, т.е. файловый адрес = `8 + Ns + start`. Не путать с CHR0.

Проверки `Ns` — те же, что §1.2 (кроме минимума: `Ns≥2`). Первый байт JSON = `{`. `__metadata__` — не тензор, пропускается (в CHR0 не копируется).

Ридер ST тоже `ReadAt`, один открытый файл.

### 5.2. Объект тензора в ST

Обязательные ключи: `dtype` (string), `shape` (array of integer ≥0), `data_offsets` (два integer, `start ≤ end`).

`dtype` (этот срез):

| `dtype` | байт/элемент | действие |
|---|---|---|
| `F32` | 4 | принять |
| `F16` | 2 | принять |
| `BF16` | 2 | принять |
| иное (`I64`, `U8`, `F8_*`, `BOOL`, `F64`, …) | — | ошибка `unsupported dtype` |

Регистр точный: `BF16`, не `bf16`.

`Π shape * elem_size == data_offsets[1]−data_offsets[0]`, иначе ошибка. Rank 0 (пустой `shape`) и любая ось `0` — ошибка (пустых весов нет). Rank ≥ 3 — ошибка (в CHR0 нет).

Файловый диапазон тензора не должен выходить за EOF.

Порядок ключей ST-заголовка: парсить `json.Decoder` по токенам (не `map` в первый проход), чтобы порядок обхода внутри шарда был порядком в файле. Дубликаты ключей тензоров в одном ST — ошибка.

### 5.3. Один файл vs директория

Вход компрессора — путь `--in`.

| `--in` | Действие |
|---|---|
| Обычный файл `*.safetensors` | Один шард; все тензоры файла (кроме `__metadata__`). `index.json` рядом **не** читать |
| Директория, есть `model.safetensors.index.json` | Шарды только из `weight_map`. Другие `*.safetensors` в каталоге игнорировать |
| Директория, нет index, ровно один `*.safetensors` | Этот файл |
| Директория, нет index, есть `model.safetensors` (даже если есть ещё файлы) | Этот файл |
| Иначе | Ошибка: нужен index или однозначный `.safetensors` |

Подкаталоги не рекурсировать. `adapter_model.safetensors` сам по себе не выбирается правилами выше.

### 5.4. `model.safetensors.index.json`

Объект с обязательным `weight_map` (object: полное **HF-имя** → относительный путь к шарду). Ключ `metadata` (часто `total_size`) **игнорировать**: это не checksum и не предел.

Значение `weight_map`:

- только относительный путь, без ведущего `/`, после `path.Clean` не начинается с `..`;
- файл = `filepath.Join(dir, value)` должен существовать;
- разделители `/` нормализовать через `filepath`.

Алгоритм обхода (LRU=1 файл):

1. Уникальные шарды = множество значений `weight_map` после `Clean`.
2. **Сортировка путей UTF-8** (детерминизм; у HF имена `model-00001-of-00004` и лексикографический порядок = номер).
3. Для шарда `S`: открыть; для тензоров **в порядке ST-заголовка**, чей `weight_map[hfName]` указывает на `S` — обработать; закрыть.
4. Тензор в `weight_map`, которого нет в указанном шарде — ошибка (после прохода по всем, список missing).
5. Тензор в шарде, которого нет в `weight_map` — **пропустить** (не ошибка).
6. Одно HF-имя в двух шардах — невозможно через одну карту; если вдруг один ключ… JSON object last-wins; писалка парсит токенами и дубликат ключа в `weight_map` — ошибка.

Пустой `weight_map` — ошибка.

Не открывать следующий шард, пока не закрыт предыдущий. Не держать mmap «всех шардов».

### 5.5. HF-имя → CHR0-имя

```
если имя оканчивается на ".weight" (последняя dotted-компонента ровно "weight"):
    chrName = имя без этого суффикса   # ровно один раз
иначе:
    chrName = имя как есть             # .bias, inv_freq, …
```

`foo.weight_scale` не трогать. `foo.weight.weight` → `foo.weight` (один strip).

### 5.6. `config.json` и вывод корневых полей

Если `--in` — директория (или каталог файла, когда `--in` файл) содержит `config.json` — прочитать его (обычный JSON, не ST). Поля:

| CHR0 | Источник, первый найденный |
|---|---|
| `arch` | `model_type` (string). Нет — `"unknown"` |
| `hidden_size` | `hidden_size` / `n_embd` / `d_model` |
| `intermediate_size` | `intermediate_size` / `ffn_dim` / `n_inner` |
| `num_layers` | `num_hidden_layers` / `n_layer` / `num_layers` |
| `vocab_size` | `vocab_size` |

Если поля нет — **вывести из тензоров** (после классификации):

- `hidden_size`: длина `model.norm` (kind=norm, имя оканчивается на `.norm` без `layers`); иначе `shape[1]` у `embed`; иначе `shape[1]` у первого `q`; иначе ошибка.
- `intermediate_size`: `shape[1]` у первого `down`; иначе `shape[0]` у первого `up`; иначе `0`.
- `num_layers`: `max(layer)+1` среди тензоров с `layer`; если таких нет — `0`.
- `vocab_size`: `shape[0]` у `embed` или `lm_head`; иначе `0`.
- `arch`: `"unknown"`.

Конфликт «config говорит X, тензоры явно не лезут» в v1 **не** проверяем (игрушка 64×64 q и 32×128 down валидна). `num_layers` из config, если есть, берётся из config; тогда каждый `layer` должен быть `< num_layers`.

### 5.7. Что делать с не-весами

**Пропускать** тензоры, чьё каноническое имя содержит `inv_freq`, `rotary_emb`, или оканчивается на `.sin` / `.cos`. Их нет в `.chr`, `verify` считает `skipped`.

Прочие неизвестные 1D — `kind=other`, `codec=bf16`. Rank-2 неизвестные — compress выбранным `--codec`.

Почему skip RoPE: это не веса GEMM; `inv_freq` восстанавливается из конфига. Стык с [integrity-cli.md](integrity-cli.md) §3.1.

### 5.8. Приведение dtype перед блобом CHR0

ST отдаёт **сырые** LE-байты + `dtype`. Конвертация — в писалке CHR0, не в парсере ST.

| Цель | Источник BF16 | F16 | F32 |
|---|---|---|---|
| блоб `codec=bf16` | memcpy | F16→F32 (IEEE) → BF16 RNE | F32→BF16 RNE |
| вход кодека nf4/vq | BF16→F32 (сдвиг+0) | F16→F32 | memcpy как `[]float32` через `math.Float32frombits`, LE |

F32→BF16 round-to-nearest-even: как усечение младших 16 бит F32 с округлением к чётному (добавить `0x7FFF + (bit16)` к младшим, затем `>>16`; NaN: сохранить экспоненту=255 и ненулевую мантиссу в старших 7 битах; Inf/sign — как в IEEE). Для тестов норм из BF16-источника конверсия **не** вызывается: байты 1-в-1.

NaN/Inf в `bf16`-копии: биты как получились. Для nf4/vq контейнер отдаёт F32 кодеку; политика NaN — спека кодека (ошибка encode), не CHR0.

Row-major C-contiguous, как safetensors. Ampere fragment-major **не** этот срез.

### 5.9. Память парсера

`Read(name)` читает только один тензор. После возврата шард может остаться открытым (проход B), но буфер предыдущего тензора писалка не сохраняет. Не делать `map[string][]byte` всей модели.

---

## 6. Классификация: имя → `kind` + `codec`

Вход: CHR0-имя (уже без `.weight`).

### 6.1. `kind` — по **последней** dotted-компоненте

Пусть `base = chrName` без суффикса `.bias`, если он есть (для kind; **в JSON имя с `.bias` остаётся**). `last` = подстрока после последней `.`, либо всё `base`, если точек нет.

Первое совпадение в таблице сверху вниз. Сравнение точное, case-sensitive.

| `last` | `kind` | Примеры HF |
|---|---|---|
| `q_proj` | `q` | `model.layers.0.self_attn.q_proj.weight` |
| `k_proj` | `k` | `…k_proj.weight` |
| `v_proj` | `v` | `…v_proj.weight` |
| `o_proj` | `o` | `…o_proj.weight` |
| `wo` | `o` | InternLM2 `attention.wo.weight` |
| `wqkv` | `qkv` | InternLM2 fused QKV `attention.wqkv.weight` |
| `gate_proj` | `gate` | `…mlp.gate_proj.weight` |
| `up_proj` | `up` | `…mlp.up_proj.weight` |
| `down_proj` | `down` | `…mlp.down_proj.weight` |
| `w1` | `gate` | Mixtral expert |
| `w3` | `up` | Mixtral expert |
| `w2` | `down` | Mixtral expert |
| `embed_tokens` | `embed` | `model.embed_tokens.weight` |
| `wte` | `embed` | GPT-2 |
| `lm_head` | `lm_head` | `lm_head.weight` |
| `norm`, `input_layernorm`, `post_attention_layernorm`, `post_feedforward_layernorm`, `pre_feedforward_layernorm`, `final_layernorm`, `final_norm`, `attention_norm`, `ffn_norm`, `rms_norm`, `q_norm`, `k_norm`, `ln_f`, `ln_1`, `ln_2`, `ln_3` | `norm` | RMSNorm / LayerNorm |
| иначе, если `last` содержит подстроку `layernorm` или `layer_norm` или `rmsnorm` | `norm` | запас для `model.norm.weight` уже покрыт `last=norm` |
| всё прочее | `other` | `inv_freq`, `mlp.gate` (роутер MoE), fused `c_attn`, … |

`q_norm` — `norm`, не `q`: таблица смотрит на `last`, `q_proj` ≠ `q_norm`.

`model.norm` → `last=norm` → `norm`.

### 6.2. `codec`

Пусть `file_codec` ∈ {`nf4`,`vq`} — флаг компрессора (один на весь `.chr`).

```
если chrName оканчивается на ".bias":     codec = bf16
иначе если kind ∈ {norm, other}:         codec = bf16
иначе если kind ∈ {q,k,v,o,qkv,gate,up,down,embed,lm_head}:
                                         codec = file_codec
иначе:                                   невозможно
```

Неизвестные Linear (`c_attn`, `dense_h_to_4h`) остаются `other`+`bf16`, **не** квантуются молча. Почему: чужой расклад осей ломает group-wise K. Llama/Qwen `*_proj` покрыты таблицей; Mixtral `w1/w2/w3` — тоже.

### 6.3. Сводная таблица (срез Llama/Qwen + игрушка)

| HF-имя | CHR0-имя | kind | codec при `--codec nf4` |
|---|---|---|---|
| `model.layers.0.self_attn.q_proj.weight` | `model.layers.0.self_attn.q_proj` | `q` | `nf4` |
| `model.layers.0.self_attn.q_proj.bias` | `model.layers.0.self_attn.q_proj.bias` | `q` | `bf16` |
| `model.layers.0.self_attn.k_proj.weight` | `…k_proj` | `k` | `nf4` |
| `model.layers.0.self_attn.v_proj.weight` | `…v_proj` | `v` | `nf4` |
| `model.layers.0.self_attn.o_proj.weight` | `…o_proj` | `o` | `nf4` |
| `model.layers.0.mlp.gate_proj.weight` | `…gate_proj` | `gate` | `nf4` |
| `model.layers.0.mlp.up_proj.weight` | `…up_proj` | `up` | `nf4` |
| `model.layers.0.mlp.down_proj.weight` | `…down_proj` | `down` | `nf4` |
| `model.layers.0.input_layernorm.weight` | `…input_layernorm` | `norm` | `bf16` |
| `model.layers.0.post_attention_layernorm.weight` | `…post_attention_layernorm` | `norm` | `bf16` |
| `model.layers.0.self_attn.q_norm.weight` | `…q_norm` | `norm` | `bf16` |
| `model.layers.0.self_attn.rotary_emb.inv_freq` | как в HF | `other` | `bf16` |
| `model.norm.weight` | `model.norm` | `norm` | `bf16` |
| `model.embed_tokens.weight` | `model.embed_tokens` | `embed` | `nf4` |
| `lm_head.weight` | `lm_head` | `lm_head` | `nf4` |
| `model.layers.0.mlp.gate.weight` (роутер) | `…mlp.gate` | `other` | `bf16` |
| `model.layers.0.block_sparse_moe.experts.3.w1.weight` | `…w1` | `gate` | `nf4` |

При `--codec vq` столбец codec для Linear/embed/lm_head — `vq`, остальное без изменений.

### 6.4. `layer` из имени

Искать в dotted-сегментах пару `layers`, `<n>` (неотрицательное целое, обычный `Atoi`). Первая такая пара слева:

- `model.layers.0.self_attn.q_proj` → `0`
- `model.layers.31.mlp.down_proj` → `31`

Нет сегмента `layers` — поля `layer` нет (`model.embed_tokens`, `lm_head`, `model.norm`). Сегмент `h` (GPT-2) **не** распознаём: тогда `kind` часто `other`, `layer` отсутствует. Срез — Llama/Qwen.

---

## 7. Паддинг `n_in` и логический shape

### 7.1. Где что живёт

В JSON **один** `shape` — логический, как в HF, без pad.

Паддинг существует **только** внутри блобов nf4/vq:

```
n_in_pad(n_in, g) = ceil(n_in / g) * g = ((n_in + g − 1) / g) * g     # целые ≥1
```

- nf4: `g = 64` = `group_size`
- vq: `g = 8`
- bf16: pad нет, длина блоба от логического `shape`

Второй shape в заголовке **не** пишем. Не пишем `shape_padded`. Не храним Ampere fragment-major. Для CPU-проверки блобы **row-major**, как [compressor.md](../compressor.md) §6.2.

Паддинг — **колонки справа** каждой строки (`k = n_in … n_in_pad-1`), значения исходной матрицы там = `0` (делает кодек перед pack). Лишних строк нет: `n_out` не паддим в v1 контейнере (тайл 64 по M — забота ядра, не файла).

Rank 1 (`norm`, bias, `inv_freq`): только `bf16`, формула pad не применяется.

Quantized тензор обязан быть rank 2, иначе ошибка (не «сделать вид, что n_out=1» молча — это можно сделать явно, но v1 отвергает).

### 7.2. Длины блобов (проверка ридера)

Обозначения: `n_out = shape[0]`, `n_in = shape[1]`.

**nf4** (`n_in_pad = n_in_pad(n_in, 64)`):

| блоб | dtype | логическая форма в памяти | `end-start` |
|---|---|---|---|
| `data` | uint8 | `[n_out, n_in_pad/2]` | `n_out * n_in_pad / 2` |
| `scale` | FP16 | `[n_out, n_in_pad/64]` | `n_out * (n_in_pad/64) * 2` |

`n_in_pad` кратен 64 → `n_in_pad/2` целое.

**vq** (`n_in_pad = n_in_pad(n_in, 8)`, `M=2`, `k=256`, `g=8`):

| блоб | dtype | форма | `end-start` |
|---|---|---|---|
| `codebook` | FP16 | `[2, 256, 8]` | `2 * 256 * 8 * 2 = 8192` (всегда) |
| `index` | uint8 | `[n_out, n_in_pad/8, 2]` | `n_out * (n_in_pad/8) * 2` |

**bf16**: `2 * product(shape)`.

### 7.3. Row-major внутри блоба (CPU)

Пусть индекс `0` — самый медленный (строка).

- nf4 `data`: байт с линейным индексом `r * (n_in_pad/2) + c`. Упаковка нибблов — спека nf4 / compressor §6.2: младшие 4 бита = вес `W[r, 2c]`, старшие = `W[r, 2c+1]`.
- nf4 `scale`: FP16 `scale[r, j]` по адресу `(r * (n_in_pad/64) + j) * 2`.
- vq `codebook`: `codebook[m, i, d]` по адресу `((m * 256 + i) * 8 + d) * 2`.
- vq `index`: `index[r, g, m]` байт по адресу `(r * (n_in_pad/8) + g) * 2 + m` (последняя ось `M`, два uint8 подряд на группу).
- bf16: элемент с multi-index как в NumPy C-order, 2 байта на элемент.

Тайл `(row_tile, col_group)` ядра в этом срезе **не** перекладывается. Verify читает целую матрицу row-major и декодирует целиком.

### 7.4. Пример pad

`shape=[3,70]`, nf4: `n_in_pad=128`, `data` = `3*64=192` байт, `scale` = `3*2*2=12` байт. В JSON `"shape":[3,70]`, не `[3,128]`.

vq: `n_in_pad=72`, `index` = `3*9*2=54` байт, `codebook` = 8192.

---

## 8. Пределы и вырожденные случаи

| Предел | Значение | Зачем |
|---|---|---|
| `header_nbytes` (CHR0 и ST) | 2…100_000_000 | Как safetensors; JSON в RAM |
| Имя тензора | 1…1024 байт | мусор |
| Число тензоров | косвенно шапкой; жёстко ≤ 1_000_000 | защита от цикла |
| Rank | 1 или 2 | |
| Ось | ≤ 2^24−1 (16 777 215) | vocab/hidden 32B << это |
| Произведение осей × 4 | ≤ 4 GiB (`1<<32`) | один тензор в RAM; 32B lm_head BF16 ≈ 1.56 ГиБ |
| Число блобов на тензор | ≤ 3 | |
| `version` | 1 | |
| `tile` | 64 и 8 | |

Пустой файл / `< 8` байт: ошибка `truncated`.

`tensors: {}` или отсутствие тензоров после фильтра: ошибка `no tensors`. Писалка не создаёт такой `.chr`.

Дубликаты:

- два одинаковых ключа в JSON `tensors` (токен-парсер) — ошибка;
- два HF-имени → одно CHR0-имя — ошибка на записи;
- перекрывающиеся `[start,end)` — ошибка.

`header_nbytes` врёт — §1.2–1.3, не «подрезать» и не читать до EOF.

Файл с валидным JSON, но `max_end > size(file)`: ошибка даже если конкретный `Get` не трогает обрезанный блоб? **Да, на `Open`**: все диапазоны проверяются.

Офсет, не кратный 64: ошибка.

`shape` и фактический размер блоба не сходятся: ошибка на `Open` (не ждать `Get`).

Tied `lm_head` отсутствует в ST: это не ошибка, в `.chr` его просто нет.

Скаляр / 0-dim / 0-size: ошибка.

---

## 9. Контрольная сумма

В v1 **нет** поля checksum, нет трейлера, нет обязательного `*.chr.sha256`.

Целостность контейнера = повторный `ReadAt` даёт те же байты блобов, что записали. Целостность весов = `verify` (RMSE/maxabs) по спеке integrity, не CRC в CHR0.

Писалка не считает SHA-256. Ридер не ищет sidecar.

---

## 10. Стык с кодеками

Контейнер передаёт кодеку логический `W[n_out, n_in]` в `float32` (row-major) и принимает готовые блобы. Формул квантования здесь нет.

### 10.1. NF4 — обязательные ключи

`group_size` = 64 (JSON integer).  
Блобы: `data`, `scale`. `zero` нет.

| Ключ | dtype | shape в терминах логического + pad | байт |
|---|---|---|---|
| `data` | uint8 | `[n_out, n_in_pad/2]`, `n_in_pad=n_in_pad(n_in,64)` | `n_out * n_in_pad / 2` |
| `scale` | IEEE 754 binary16 (FP16), LE | `[n_out, n_in_pad/64]` | `n_out * (n_in_pad/64) * 2` |

Packing байта `data[r,c]` (compressor.md §6.2): ниббл-индекс уровня 0..15, **не** смещённый INT4; младшие 4 бита = `W[r, 2c]`, старшие = `W[r, 2c+1]`. Шкала — FP16, не BF16.

### 10.2. VQ — обязательные ключи

`group_size` = 8, `n_codebooks` = 2, `codebook_bits` = 8.

| Ключ | dtype | shape | байт |
|---|---|---|---|
| `codebook` | FP16 LE | `[n_codebooks, 2^codebook_bits, group_size]` = `[2, 256, 8]` | 8192 |
| `index` | uint8 | `[n_out, n_in_pad/8, n_codebooks]` = `[n_out, n_in_pad/8, 2]` | `n_out * (n_in_pad/8) * 2` |

Ось `M` у `index` — последняя: на группу 8 весов два подряд байта `(i1, i2)`. Книга на **эту** матрицу: не шарить `codebook` между `q` и `down`.

### 10.3. BF16

Один блоб `data`: BF16 LE, логический `shape`, без pad.

### 10.4. Что контейнер проверяет, чего не делает

Проверяет: набор ключей, длины, 64-align стартов, один file-codec.

Не проверяет: что нибблы ∈ 0..15 осмысленно, что книга — результат k-means, MSE. Это пакеты `internal/nf4` и `internal/vq`.

---

## Приложение A. Контракт пакетов (без кода)

`internal/safetensors`:

- разобрать `--in` в список `{HFName, DType, Shape, ShardPath, DataStart, DataEnd}`;
- `OpenShard` / `Close`; `ReadAt` одного тензора → `[]byte` native;
- не знать CHR0.

`internal/chr0`:

- `Align64`, формулы длин блобов, классификатор §6, стабилизация JSON, `Write`, `Open`, `Get`;
- не знать LUT NF4 и k-means;
- `encoding/binary`, `encoding/json` (`DisallowUnknownFields`, `SetEscapeHTML(false)`, `UseNumber`), `os.File`.

Имена тензоров в `.chr` никогда не содержат `.weight`. Bias и `inv_freq` — содержат свои суффиксы.

---

## Приложение B. Ошибки, которые ридер обязан поймать (для тестов)

1. Файл длины 0, 3, 7.
2. `N` больше EOF.
3. `N=516`, но байты JSON — не JSON / обрезаны / два объекта.
4. BOM перед `{`.
5. `magic: "chr0"` / `version: 2`.
6. `tile.row: 32`.
7. Пустой `tensors`.
8. Дубликат имени.
9. `q_proj` с `codec: bf16` при наличии другого `nf4` Linear — смесь в `Q`.
10. `norm` с `codec: nf4`.
11. `data` длина ≠ формуле.
12. `start=575` (не кратно 64).
13. Перекрытие диапазонов.
14. ST `dtype: I64`.
15. ST `data_offsets` от начала **файла** вместо секции данных — поймается несовпадением `numel*size`.
16. `index.json` с `../escape.safetensors`.
17. Два `.weight` и без, схлопывающиеся в одно CHR0-имя.
18. `layer` отсутствует у `model.layers.0.mlp.down_proj`.
19. `layer:0` пропал из JSON из-за omitempty.
20. `group_size: 32` у nf4.

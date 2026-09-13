# GPU ABI: `.chr` → device-тензоры (волна 2)

Что лоадер `gpu/chr0/` отдаёт ядру и на каких условиях. Спорные места решает [stitch-gpu.md](stitch-gpu.md); контейнер — [chr0.md](chr0.md); кодек — [nf4.md](nf4.md). Здесь нет LUT, нет деквантования, нет `.cu`.

---

## 1. `ChrMatrix` — заморожено

```python
@dataclass(frozen=True)
class ChrMatrix:
    name: str              # CHR0-имя, без ".weight"
    M: int                 # логический n_out = shape[0]
    K: int                 # логический n_in  = shape[1]
    K_pad: int             # 64 * ceil(K / 64)
    packed: Tensor         # uint8,   [M, K_pad // 2],  row-major, contiguous
    scale: Tensor          # float16, [M, K_pad // 64], row-major, contiguous
```

Имена полей и порядок не менять: агент 2 читает их через `chr_nf4_dev_t`, агент 3 — напрямую.

`packed` и `scale` лежат на одном `device`. Оба — вьюхи в **одну** device-аллокацию на матрицу (§4), поэтому `packed.data_ptr()` и `scale.data_ptr()` валидны, пока жив сам `ChrMatrix`.

Соответствие C-заголовку `gpu/include/chr_gpu.h` (расклад заморожен, лоадер его не меняет):

| C-поле | Python |
|---|---|
| `int32_t M` | `m.M` |
| `int32_t K` | `m.K` |
| `int32_t K_pad` | `m.K_pad` |
| `const uint8_t *packed` | `m.packed.data_ptr()` |
| `const uint16_t *scale` | `m.scale.data_ptr()` — **биты** binary16, не «fp16-объект» |

`scale` — `torch.float16`, т.е. те же 16 бит, что в файле. Ядро читает как `uint16` / `__half` и расширяет в float32 **до** умножения на `LUT[nib]` ([nf4.md](nf4.md) §3).

---

## 2. API

```python
load_header(path)                                   -> Header
iter_linears(header)                                -> Iterator[str]   # codec == "nf4"
materialize_nf4(path, name, device="cuda", *, header=None) -> ChrMatrix
```

- `load_header` открывает файл **read-only** (`open(path, "rb")`), парсит и **полностью валидирует** заголовок (§5), закрывает файл. Блобы не читает.
- `iter_linears` не трогает диск: имена с `codec == "nf4"` в порядке ключей заголовка (лексикографический UTF-8 от писалки). Сюда попадают `embed_tokens` и `lm_head`, если они nf4 в файле — фильтр по `kind` делает хост, не лоадер.
- `materialize_nf4` — одна матрица. `header=` позволяет переиспользовать уже разобранный заголовок при загрузке многих матриц (парсинг 65 КБ JSON × 300 — единственная причина этого параметра).
- Оригинальный `safetensors` лоадер **не** открывает никогда.

`Header` несёт `magic/version/arch/hidden_size/intermediate_size/num_layers/vocab_size/tile`, `file_size`, и `tensors: Mapping[str, TensorInfo]`. `TensorInfo` — `kind`, `codec`, `shape`, `layer`, `group_size`, блобы `{key: (start, end)}` и производные `M / K / K_pad / n_groups` для rank-2 nf4.

---

## 3. Формулы байт (лоадер отвергает mismatch)

`K_pad = 64 * ceil(K / 64)`, `n_groups = K_pad / 64`. Из [stitch-gpu.md](stitch-gpu.md):

| Блоб | dtype | форма | `end − start` |
|---|---|---|---|
| nf4 `data` | uint8 | `[M, K_pad/2]` | `M * K_pad / 2` |
| nf4 `scale` | FP16 LE | `[M, n_groups]` | `M * n_groups * 2` |
| bf16 `data` | BF16 LE | `shape` | `2 * Π shape` |
| vq `codebook` | FP16 LE | `[2, 256, 8]` | `8192` |
| vq `index` | uint8 | `[M, K_pad_vq/8, 2]` | `M * (K_pad_vq/8) * 2`, `K_pad_vq = 8*ceil(K/8)` |

Офсеты `[start, end)` — **от начала файла**, `start % 64 == 0`.

Проверка размера идёт на `load_header` для **всех** тензоров файла, включая vq/int4-слоты: их размеры валидируются, а `materialize_nf4` на них падает (§5, `CodecError`). Волна 2.0 материализует только `nf4`.

Нибблы (младший = `W[r, 2c]`), LUT, книги — **не забота лоадера**. Он не читает и не переставляет ни одного бита: `packed[r, c]` — это ровно байт файла по офсету `data.start + r*(K_pad/2) + c`.

---

## 4. Как байты попадают в VRAM

**Выбранная стратегия — одна: per-matrix arena, один HtoD на матрицу.**

```
seek(data.start)
readinto(bytearray(data.nbytes + scale.nbytes))   # один pread: data ‖ scale
arena = torch.empty(total, uint8, device)          # одна device-аллокация
arena.copy_(torch.frombuffer(host))                # один cudaMemcpy HtoD
packed = arena[:len_data].view(M, K_pad//2)
scale  = arena[len_data:].view(torch.float16).view(M, n_groups)
host-буфер отпускается
```

Писалка CHR0 всегда кладёт `data`, затем `scale` подряд (порядок полей §2.3 chr0.md), поэтому слитный путь — обычный: `scale.start == data.end` для любой матрицы, у которой `M * K_pad / 2` кратно 64, т.е. для всех реальных. Если между блобами есть pad или чужой блоб, лоадер тихо берёт запасной путь: два `pread` + два HtoD **в тот же** один device-буфер. Форма результата одинакова.

Диск читается **до** первого обращения к device: короткий файл падает, не выделив VRAM.

Следствия, которые это фиксирует:

- **Пик host-RAM = блобы одной матрицы** (для 3B `gate_proj` — 11.42 МиБ), а не файл и не модель. Нет `mmap` всего `.chr`, нет `read()` файла целиком, нет `dict[str, bytes]`.
- **Одна device-аллокация на матрицу**, не по одной на блоб: 434 тензора 3B → ≤ 434 аллокаций из кэширующего аллокатора torch, не «200 мелких `cudaMalloc`» на каждый блоб.
- При загрузке всей модели host- и device-копии **не** сосуществуют: host-буфер матрицы `i` мёртв до чтения матрицы `i+1`.
- `torch.empty(M, K, dtype=bfloat16)` не вызывается никогда — ни «для проверки», ни как промежуток. Черновика `W` в HBM нет ([stitch-gpu.md](stitch-gpu.md)).

Замер на 3080 (`model.layers.0.mlp.gate_proj`, `M=11008, K=2048`): блобы `11 272 192 + 704 512` Б = **11.42 МиБ**, `torch.cuda.memory_allocated` +**12.00 МиБ** (кэширующий аллокатор округляет крупный блок до 2 МиБ), `nvidia-smi memory.used` +**12 МиБ** сверх уже созданного CUDA-контекста. BF16-матрица была бы 43 МиБ, F32 — 86 МиБ. Дельта выше ~14 МиБ — баг.

Запись в `.chr` невозможна по построению: единственный вызов — `open(path, "rb")`.

---

## 5. Ошибки

Все — подклассы `Chr0Error`. Все проверки заголовка выполняются на `load_header`, т.е. **до** любого `to(device)` и любого `cudaMemcpy`.

| Класс | Когда |
|---|---|
| `TruncatedError` | файл < 8 байт; `8+N > size`; `end > size` у любого блоба; `readinto` вернул меньше байт |
| `HeaderError` | `N ∈ {0,1}`; `N > 100_000_000`; первый байт JSON ≠ `{`; невалидный UTF-8/JSON; не объект; хвост не-whitespace; JSON-число с дробной частью; лишний/отсутствующий ключ; `magic ≠ "CHR0"`; `version ≠ 1`; `tile ≠ {row:64,col_group:8}`; пустой `tensors`; дубликат имени; имя с control-байтом; `layer` не сходится с `layers.<n>`; смесь кодеков в Q-множестве |
| `AlignmentError` | `start % 64 ≠ 0` |
| `SizeMismatchError` | `end ≤ start`; `end − start` ≠ формуле §3 |
| `OverlapError` | пересечение любых двух `[start,end)` в файле |
| `CodecError` | `codec` вне `{bf16,nf4,int4,vq}`; `materialize_nf4` на не-`nf4` тензоре; набор ключей не по кодеку |
| `GroupSizeError` | `group_size ≠ 64` у nf4 (и `≠ 8` у vq) |
| `TensorNotFoundError` | имени нет в `tensors` |

Ридер **не** чинит заголовок: не ищет `{` дальше по файлу, не подрезает `N`, не игнорирует неизвестные ключи. Лишний хвост файла за `Align64(max_end)` — не ошибка ([chr0.md](chr0.md) §1.5).

---

## 6. Кто что транспонирует

**Транспонирует хост, не лоадер и не ядро.**

- В файле `W` — row-major `[M, K]`, ось 1 = вход Linear. Лоадер отдаёт эти байты как есть.
- Ядро ждёт `x` в BF16 row-major **`[K, N]`** и пишет `y` BF16 `[M, N]` ([stitch-gpu.md](stitch-gpu.md)).
- HuggingFace даёт активации `[..., N, K]`. Привести их к `[K, N]` — работа `CompressedLinear` (агент 3): `x.reshape(-1, K).t().contiguous()` перед вызовом и обратный reshape `y` после. Лоадер в этом не участвует и `x` не видит.
- Никакого fragment-major и никакой перестановки `packed` на диске или при загрузке: permute — в регистрах ядра.

---

## 7. Проверка

```
python gpu/chr0/test_chr0.py
```

Фикстура happy-path строится на месте: `write_safetensors` (64×128 и 2×65 F32 + норма) → `chr.exe compress --codec nf4` → `materialize_nf4(device="cpu")` → сравнение `packed`/`scale` с `file[start:end]` байт в байт. Malformed-случаи (`end > filesize`, размер ≠ формуле, overlap, `start % 64 ≠ 0`, `group_size=32`, `codec=vq`, битый JSON) собираются вручную: писалка их не производит.

Проверка «до `cuda`»: в тестах truncated-файлов `torch.empty` подменяется на бросающую заглушку; `materialize_nf4(..., "cuda")` обязан упасть `TruncatedError`, а не на заглушке.

Дисциплина памяти (отдельный прогон, 60 матриц 3B подряд, 346.9 МиБ блобов): `device_delta = 360.5 МиБ`, `host RSS delta = 0.0 МиБ`, максимальный host-буфер за раз — 11.42 МиБ.

---

## 8. Что лоадер не делает

Не деквантует, не считает `W_hat`, не открывает safetensors, не пишет в `.chr`, не делает `mmap` на запись, не кэширует блобы между вызовами, не умеет `int4`/`vq` материализовать (парсит слот и отклоняет), не выбирает `stream` и не аллоцирует ничего на токене.

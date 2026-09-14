# H2: NF4 overflow ring (pinned host, two slots)

**Статус 2026-09-14:** C0 закрыта. **H2-1…H2-6 PASS.**

Продуктовый дым Qwen2.5-32B-Instruct на RTX 3080 12 GB:
`python -m gpu.lab.h2_trace --no-timing` →
`C:\dev\models\runs\h2-qwen25-32b-20260914-234048`
(копия `data_path.md` / `messages.json` / `gate.txt` в
[`docs/runs/h2-qwen25-32b/`](runs/h2-qwen25-32b/)).

| | |
|---|---|
| Decode | **2.31 tok/s** mean (2.30–2.32; ~432 ms/tok) |
| TTFT | **1006 ms** mean (2 чанка, `LIVE_MAX_N=32`) |
| Smoke | Paris / Berlin / 323 **3/3**, `gate.txt` PASS |
| Packed NF4 | **16599 MiB** — на карту не влезает |
| Resident HBM | **9716 MiB** (`report.device_mib`, не 16601) |
| Host tail | **96** матриц, **6885 MiB**, pin 6885/6885, слот **71.72 MiB** |
| smi decode | **11926–11933 MiB**, плоский |
| Пол «везём хвост по PCIe» | бит (не 0.8; не 10) |
| Потолок ~3 ток/с | не достигнут (стена 432 мс > copy-floor 277 мс) |
| Hard-12 | **не запускали** |
| BF16 32B | **не запускали** (не влезет) |
| VQ | **нет** (`--codec auto` не выбирает VQ) |

Цель волны: 32B **говорит** быстрее пола 1–2 ток/с. Потолок ≈ текущее ядро
(~3 ток/с, если бы overflow-веса уже были в HBM), не 10. Качество — NF4.

Пластина: [`docs/img/h2-qwen25-32b.png`](img/h2-qwen25-32b.png),
`python -m gpu.lab.h2_plate --redraw`.

## Заморожено (не оспаривать)

Железо: sm_86, WDDM, дисплей. Pinned H2D **24.3 GB/s** (256 MiB); pageable 8.0 —
мёртвый путь. Один H2D copy engine; два H2D не складываются. Compute ∥ один copy — да.

Живой NF4 = **4.25 бит**, group 64. Packed 32B ≈ **16599–16601 MiB**, не бумажные
17577 (4.5 бит в `vram-3080.md`). Дыра ≈ 6.1 GiB + overhead 1800 MiB.

| Правило | Решение |
|---|---|
| Слоты | **2** device-арены, размер = packed+scale худшей overflow-матрицы (**71.72 MiB** на 32B gate/up/down). Адреса статичны. Не слой, не lm_head. |
| Третий слот | не v1 (нет второго H2D; 72 MiB лучше отдать resident FFN) |
| Pin | плиты **256 MiB** при load; `.chr` на токене не читать. Проверять `cpu_is_pinned()`, не `bool(tensor.is_pinned)` — это метод, всегда true. |
| Copy | один `copy_stream` (не default); `copy_(non_blocking)` с pinned. На WDDM: **timing `e_copy` + CPU join** перед prefetch. `elapsed_time` только если `timing=True`. CLI: `ring_timing=False` (join есть, счётчика copy нет). |
| Единица H2D | целая матрица (data+scale), не тайл |
| Ядро | `chr_nf4_gemm` не трогать. `LIVE_MAX_N=32`. Нет `[M,K]` BF16 в HBM. Нет managed. |
| Prefetch | depth = 1 (следующая overflow-матрица; N=1 ещё prefetch lm_head) |
| Graphs | plan A на DEVICE. HOST GEMM после bind — CUDA graph со статического слота (`HostSlotGemm`); copy в graph на WDDM нет. Graphs не headline скорости. |
| QKV/gate-up fork | только если **все** члены DEVICE. Mixed: serial, resident не копировать в слот |
| Prefill | один H2D на overflow-матрицу **на чанк** `N≤32`, не на колонку |
| Embed / lm_head / norms / bias / все qkvo | **всегда resident** |
| Overflow | все `down_proj`, затем хвостовые пары `gate+up` (политика **D**) |
| `max_seq` чата | **2048** (KV 512 MiB на 32B). 4096 — отдельный режим |
| `auto` | NF4 если влезает целиком; иначе **NF4 + overflow**. Никогда VQ. 70B-класс — refuse |
| 32B | дерево и `.chr` на диске. Дым A записан. Hard-12 — только после явного «гоняй». |

Canary без 32B: `qwen25-3b.nf4.chr` + фейковый cap (резать resident packed, не ballast-тензор).
Дым Paris/Berlin/323. Калибр шины — **256 MiB / 10.31 ms**.
`t_ms = size_MiB × 10.31 / 256`.

**978 MiB** (17577 − 16601) — ошибка таблицы 4.5 бит в `vram-3080.md`, не
«неизвестные веса». В бюджет **не** резервировать.

Потолок: если overflow **размазан** по слоям (все `down` каждое кольцо слоя),
copy engine занят во время resident qkv/attn/gateup → `wall ≈ max(copy, gemm)`.
DoD не 2.7; цель — бить пол 1–2, не обещать 10. Живая стена ~432 мс/ток —
copy-floor 277 мс плюс WDDM join и неполный overlap.

### Баг, который не цитировать как дизайн

`Tensor.is_pinned` — **метод**. `bool(arena.is_pinned)` всегда True, overflow
ехал pageable (~0.76–0.82 ток/с, ~7.3 GiB/s). Чинить: `cpu_is_pinned()` в
`gpu/host/host_image.py`. Продуктовая цифра — **2.31**, не 0.8.

### Сторонний разбор (принять / нет)

| Ход | Вердикт |
|---|---|
| Бить резидентный **префикс** (A): два слота не прокачивают хвост | **Да.** A мертва. D: все `down` едут каждый слой. |
| Планировщик перестановок MLP / fork как v1 | **Нет.** D — seed. |
| packed+scale = **один** H2D, `ready` = оба | **Да.** |
| `record ready` → wait/GEMM/`record done` → wait/overwrite | **Да.** В CopyRing. |
| Overflow GEMM в CUDA graph со статического слота | **Сделано** (`HostSlotGemm`). Не заявлять как win tok/s. |
| Prefetch начала следующего токена во время lm_head | **Да, дёшево**; bytes/token не меняет. |
| Вычитать 978 МиБ «на всякий» | **Нет.** |
| Третий слот / half-M / embedding по строкам / `cudaHostRegister` всего `.chr` | **Не v1.** |
| Цель 2.7 ток/с как gate 32B без файла | **Нет.** Gate чата — 3B canary + живой 32B дым. |
| 10 ток/с / Marlin / llama.cpp | **Нет.** |

## Волны

Один чат = одна волна. Юниты без GPU после волны кода:

```
python gpu/cli/test_codec.py
python gpu/cli/test_cli.py
python gpu/loop/test_attach.py
python gpu/host/test_attach.py
python gpu/chr0/test_chr0.py
python gpu/loop/test_ring.py
python gpu/loop/test_host_graph.py
python gpu/lab/test_h2_trace.py
python gpu/lab/test_h2_plate.py
```

### H2-1 — `decide()` overflow, без CUDA — PASS

`Decision.overflow`. 32B/12 GB → nf4 + overflow, не raise, не VQ.

### H2-2 — HostImage + residency plan, CPU — PASS

Pin-плиты 256 MiB. `plan_residency` = **D**. `slot_nbytes = max(streamed)`.

### H2-3 — load: resident HBM + host tail + SlotPair — PASS

### H2-4 — CopyRing + decode eager — PASS

### H2-5 — plan A только resident; CLI run на overflow — PASS

### H2-6 — живой 32B — PASS (дым A)

Hard 12 не стартовать без явного «гоняй» (`docs/eval-32b.md`).

# Стыки спек (волна 1, CPU)

Расхождения первой волны, **зафиксированные** перед кодом. Если спека модуля противоречит этой странице — побеждает эта страница.

GPU-волна: [stitch-gpu.md](stitch-gpu.md).

| Тема | Решение |
|---|---|
| `inv_freq` / `rotary_emb` / `.sin` / `.cos` | **skip**, нет в `.chr`. Классификатор — [integrity-cli.md](integrity-cli.md) §3.1. |
| `--chunk` дефолт | **262144** ([vq.md](vq.md) §8.3). Не 1048576. |
| `--group-size` | nf4 только 64, vq только 8. |
| Имена в `.chr` | без суффикса `.weight`; bias с `.bias`. |
| Decode safetensors | всегда **F32**, один файл. |
| Нормы / bias | `codec=bf16`, bit-exact к BF16-проекции orig. |
| Linear / embed / lm_head | один `--codec` на файл (`nf4` или `vq`). |
| Блобы | row-major, не Ampere fragment-major. |
| Чтение файлов | `os.File.ReadAt`, без mmap, без CGO. |
| JSON CHR0 | compact, `DisallowUnknownFields`, офсеты integer. |
| `hidden_size` | ≥1; compress выводит из config или тензоров, не оставляет 0. |
| RNG VQ | `math/rand/v2` PCG(seed, 0) на **каждую** матрицу заново. |
| Полосы | `full_f32 > 256MiB`; unit-тесты не включают. |
| Модуль Go | `chr` (не github-путь). Пакеты как в integrity-cli §5. |

Пакеты:

```
chr/
  cmd/chr/          # flags + run()
  internal/f16/     # binary16 ↔ float32
  internal/safetensors/
  internal/chr0/
  internal/nf4/
  internal/vq/
  internal/verify/  # classify + metrics + compress/decode orchestration
```

Классификатор живёт в `internal/verify` (или `internal/tensor`), один на compress и verify.

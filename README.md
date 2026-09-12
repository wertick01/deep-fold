# Сжатие больших моделей «как ДНК»

Держать веса сжатыми в видеопамяти и распутывать только тот кусок, который сейчас считает GPU. Цель железа: **RTX 3080 12 ГБ**, не ниже **10 ток/с**.

Сейчас в репозитории есть **CPU-утилита** `chr`: скачать модель, сжать, разжать, проверить целостность — без GPU. Как гонять: [docs/cpu-roundtrip.md](docs/cpu-roundtrip.md). Какие веса качать под 12 ГБ: [docs/models.md](docs/models.md).

```bash
go test ./...
go build -o chr ./cmd/chr
```

**Сначала про схему:** [docs/ot-konca.md](docs/ot-konca.md) — сколько есть миллисекунд.

**Схема 3080:** [docs/schema.md](docs/schema.md). Один рантайм, два кодека: 4-бит (этап A) и книжка 2×8 (этап B). 70B на 12 ГБ не будет; потолок — **14B в 4 битах** или **32B в ~2 битах**.

Модули:

| Модуль | Документ |
|---|---|
| CPU compress / verify | [docs/cpu-roundtrip.md](docs/cpu-roundtrip.md) |
| Бюджет 12288 МБ | [docs/vram-3080.md](docs/vram-3080.md) |
| Ядро Ampere | [docs/kernel-ampere.md](docs/kernel-ampere.md) |
| Офлайн-компрессор (GPU-план) | [docs/compressor.md](docs/compressor.md) |
| Цикл токена | [docs/token-loop.md](docs/token-loop.md) |

Спеки кодеков для `chr`: [docs/spec/](docs/spec/). Длинный черновик со ссылками: [docs/report.md](docs/report.md).

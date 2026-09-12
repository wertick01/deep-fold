# CPU-проверка сжатия (без GPU)

Утилита `chr`: safetensors → `.chr` → разжать / сравнить с оригиналом. Видеопамять не нужна. Модель в репозиторий не кладётся.

## Собрать и тесты

```bash
go test ./...
go build -o chr ./cmd/chr
```

Тесты сами пишут крошечные safetensors. Веса LLM не качаются.

## После скачивания модели

`$MODEL` — каталог с `model.safetensors` или `model.safetensors.index.json`.

На **Windows** облачный агент диск не видит. 3B (~6,2 ГБ) качается у тебя:

```powershell
# PowerShell, не Downloads
python -m pip install -U huggingface_hub
hf download Qwen/Qwen2.5-3B-Instruct --local-dir C:\dev\models\Qwen2.5-3B-Instruct
```

Или скрипт из репо: `scripts\download-qwen25-3b.ps1`.

Пороги ниже **не** дефолт бинаря: они для живых весов. Unit-тесты гоняют более жёсткие числа.

```bash
# NF4, группа 64, без калибровки
./chr compress --in "$MODEL" --out llama8b.nf4.chr --codec nf4 --quiet
./chr verify  --orig "$MODEL" --chr llama8b.nf4.chr \
    --fail-rmse 0.12 --fail-maxabs 2.0 --json > llama8b.nf4.verify.json

# Книжка 2×8 (residual k-means, seed 0). На 8B это уже минуты–десятки минут на CPU.
./chr compress --in "$MODEL" --out llama8b.vq2.chr --codec vq --seed 0 --iters 20 --chunk 262144
./chr verify  --orig "$MODEL" --chr llama8b.vq2.chr \
    --fail-rmse 0.50 --fail-maxabs 8.0 --json > llama8b.vq2.verify.json
```

`PASS` / код 0 значит: контейнер целый, нормы bit-exact, lossy не взорвался. Это **не** WikiText и не чат.

Полный F32 dump (`chr decode`) для 8B ≈ 32 ГБ — для приёмки не нужен.

Пик RAM: один тензор в float32. `lm_head` / `embed` 8B ≈ 2 ГБ F32 плюс packed выход. На 32B `lm_head` ещё больше; если не влезет — скажи, допишем полосы (в спеке они уже описаны, в этом срезе unit их не гоняет).

## Что внутри

| Команда | Смысл |
|---|---|
| `compress --codec nf4` | QLoRA NF4, группа 64 |
| `compress --codec vq` | две книги 256×8, 2 бит/вес |
| `decode` | один safetensors, **F32** |
| `verify` | orig vs `.chr`, тензор за тензором |

`inv_freq` / rotary не пишутся в `.chr` (это не веса GEMM). Нормы и bias — сырой BF16.

Спеки: [docs/spec/](spec/). Стыки: [spec/stitch.md](spec/stitch.md).

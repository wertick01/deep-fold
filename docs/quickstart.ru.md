# QuickStart

Короткий список команд. Полная справка (`-h`, флаги, ошибки): [`install.ru.md`](install.ru.md).

После `scripts/setup.ps1` или `scripts/setup.sh` команда `deepfold` и `python -m gpu.cli` — одно и то же.
Ниже — `deepfold`. Если её нет на PATH: `python -m gpu.cli` (так и задумано в
conda env `torch-gpu`). После setup активируйте `.venv` или зовите
`.\.venv\Scripts\deepfold.exe`.

Smoke Paris / Berlin / 323 **не** измеряет качество. Tok/s с авторской 3080
на другой карте не обещать.

**Дорабатываются:** TTY / агентный layout в `chat`.

---

## 1. Установка

**Windows**

```powershell
git clone https://github.com/wertick01/deep-fold.git
cd deep-fold
Set-ExecutionPolicy -Scope Process Bypass
powershell -File scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
deepfold doctor
```

**Linux**

```bash
git clone https://github.com/wertick01/deep-fold.git
cd deep-fold
bash scripts/setup.sh
source .venv/bin/activate
deepfold doctor
```

Нужны заранее: драйвер NVIDIA (Ampere или Ada), Python 3.11 или 3.12
(на Ubuntu 22.04 `python3` — это 3.10, поставьте `python3.11`).
Go 1.22+ необязателен: setup сам скачает переносной Go с go.dev, если нет
`chr`. На Windows при отсутствии Build Tools (C++) и CUDA 12.4 setup ставит
их через winget и компилирует `gpu/nf4`. Скрипт **не** ставит драйвер. Код
`doctor`: 0 — можно генерировать; 2 — карта подходящая, install неполный;
3 — этот класс машины generate не умеет.

---

## 2. Скачать модель

Только таблица (не произвольный HuggingFace, не GGUF):

```text
deepfold pull Qwen/Qwen2.5-3B-Instruct --yes
```

Другие id: `Qwen/Qwen2.5-14B-Instruct`, `internlm/internlm2_5-20b-chat`
(нужен `pip install "deepfold[internlm]"`), `Qwen/Qwen2.5-32B-Instruct`.

`pull` печатает готовые `chat` / `run` с путём. Копируйте его.

Куда кладётся дерево: `$DEEPFOLD_MODELS\<имя>`, иначе кэш
(`%LOCALAPPDATA%\deepfold` / `~/.cache/deepfold`).

---

## 3. Сжать (NF4 → `.chr`)

Первый `chat` / `run` упакует сам. Явно:

```text
deepfold compress --in D:\weights\Qwen2.5-3B-Instruct --codec nf4
```

`--codec auto` (дефолт): NF4, если влезает в карту, иначе NF4 overflow (H2).
VQ только с `--codec vq` — это оракул ядра, не разговорная модель.

CPU, минуты (14B/20B/32B дольше). После успеха шарды safetensors можно удалить,
оставив `config.json`, токенизатор и `.chr`. Для `chr verify` шарды ещё нужны.

---

## 4. Проверка целостности и тесты

CLI без сети и без generate:

```text
deepfold test
deepfold test --live
```

`--live` **не** качает и **не** генерирует: есть 3B на диске и doctor
разрешает generate → `LIVE:`, иначе `SKIP:`.

Целостность сжатия (CPU, orig vs `.chr`):

```text
chr verify --orig D:\weights\Qwen2.5-3B-Instruct --chr D:\weights\qwen25-3b.nf4.chr --fail-rmse 0.12 --fail-maxabs 2.0 --json
```

`chr` должен быть на PATH (его собирает setup). Код 0 — PASS, 2 — порог,
1 — ошибка ввода. Это **не** чат и не WikiText.

---

## 5. Разговор

Настоящее окно терминала (не пайп):

```text
deepfold chat --model D:\weights\Qwen2.5-3B-Instruct
```

Enter — отправить, Ctrl+J — новая строка. Ctrl+C останавливает ответ. `/help` `/quit` `/clear` `/stats` `/new` `/chats` `/copy` `/save` `/agent`. Ответы рисуют markdown (жирный, списки, ограды) и переводят LaTeX `$...$` / `$$` в Unicode. По желанию: `--agent --workspace .` — grep/патч/pytest (запись спрашивает; `--agent-trust`). KV между ходами ещё полный prefill; спека: [`spec/agent.md`](spec/agent.md).
У `chat` по умолчанию 256 новых токенов (не дымовой лимит 64).

Скрипт / пайп:

```text
deepfold run --model D:\weights\Qwen2.5-3B-Instruct --prompt "Capital of France?" --max-new-tokens 16
```

`--executor auto` (по умолчанию) берёт Decode V2 на resident NF4 и TokenLoop на overflow / VQ. MMA: `--executor tokenloop`.

---

## 6. Пластина на другом ПК (одна модель → метрики)

Скрипт сам: CPU-тесты CLI, скачивание если надо, NF4 compress, `chr verify`,
жадный smoke Paris / Berlin / 323. **Не** гоняет BF16, VQ и hard-12.

**Windows**

```powershell
powershell -File scripts\plate.ps1 3b
powershell -File scripts\plate.ps1 Qwen/Qwen2.5-3B-Instruct
powershell -File scripts\plate.ps1 14b
powershell -File scripts\plate.ps1 20b
powershell -File scripts\plate.ps1 32b
```

**Linux**

```bash
bash scripts/plate.sh 3b
bash scripts/plate.sh Qwen/Qwen2.5-3B-Instruct
```

Модель: `3b` / `14b` / `20b` / `32b`, HuggingFace id, тег `qwen2.5:3b`, или
локальная папка с `config.json`.

Полезные флаги (после модели):

```text
--dry-run          план, без скачивания и generate
--out DIR          куда писать отчёт
--skip-test        не гонять gpu/cli/test_cli.py
--skip-verify      не гонять chr verify (если шарды уже удалены — скрипт и так SKIP)
--skip-generate    остановиться после verify
--no-download      не ходить в Hub
--force            перепаковать .chr
```

Результат — папка вида `$DEEPFOLD_RUNS/plate-<модель>-<время>` (или `--out`).
Отправьте её целиком: **`SUMMARY.txt`**, **`plate.json`**, при generate ещё
`nf4/summary.csv`, `nf4/messages.csv`, `verify.json`.

Первый generate на чужой карте может ~минуту JIT-ить ядро. 32B — overflow H2,
сжатие и verify долгие.

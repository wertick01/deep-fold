# Установка и команды (другой ПК)

Короткий список команд (установка, pull, compress, тесты, чат, пластина с
метриками): [`quickstart.ru.md`](quickstart.ru.md). Ниже — полный `-h`.

Deepfold — программа **в терминале**: она упаковывает веса модели в один файл
`.chr` и генерирует текст на NVIDIA Ampere или Ada. Это не сайт, не плагин
Ollama и не чат в браузере.

Цифры скорости с авторской RTX 3080 **на другой карте не повторятся**.
Ada (RTX 40xx, `sm_89`) и A100 (`sm_80`) могут генерировать, но `doctor`
напишет `experimental`. Turing, Hopper, Blackwell, AMD и macOS для generate
не подходят.

Программа **не ставит** драйвер NVIDIA и Python.
Если нет `chr`, setup скачивает переносной Go 1.22 с go.dev.
Если нет ядра NF4, на Windows setup через winget ставит Visual Studio 2022
Build Tools (C++) и CUDA Toolkit 12.4 и компилирует `gpu/nf4`.
Красный `doctor` — нормальный отказ установки, не баг ядра.

Справка CLI (`-h`) на английском; ниже она вставлена как есть. После
`pip install -e .` команды `deepfold` и `python -m gpu.cli` — одно и то же.
Если `deepfold` ещё не на PATH, подставьте `python -m gpu.cli`. Авторский
conda env `torch-gpu` **не** ставит команду `deepfold` (сломанное колесо там
убивает ядро); в этой среде всегда `python -m gpu.cli`. На соседнем ПК —
`scripts/setup.ps1`, затем `.\.venv\Scripts\Activate.ps1`. Без активации:
`.\.venv\Scripts\deepfold.exe`.

---

## Минимальный путь (соседний ПК)

```powershell
git clone https://github.com/wertick01/deep-fold.git
cd deep-fold
Set-ExecutionPolicy -Scope Process Bypass
powershell -File scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
deepfold doctor
deepfold pull Qwen/Qwen2.5-3B-Instruct --yes
deepfold chat --model <каталог, который напечатал pull>
```

Linux вместо двух строк setup:

```bash
git clone https://github.com/wertick01/deep-fold.git
cd deep-fold
bash scripts/setup.sh
source .venv/bin/activate
```

Дальше те же `doctor` / `pull` / `chat`. Первый generate на чужой карте может
минуту собрать CUDA-ядро (JIT). Не обещайте tok/s с 3080.

---

## 1. Что нужно заранее

| Что | Зачем | Как проверить |
|---|---|---|
| Драйвер NVIDIA (Ampere или Ada) | GPU | `nvidia-smi` печатает имя карты |
| Python 3.11 или 3.12 | рантайм | Windows: `py -3.11 --version`; Linux: `python3 --version` |
| Go 1.22+ (необязательно) | компрессор `chr`; setup сам скачает с go.dev, если его нет | `go version`, либо пропустить — `scripts/setup.*` / `deepfold setup` |
| Windows: MSVC Build Tools | сборка ядра NF4; setup поставит через winget, если нет | `cl` после `vcvars64.bat` |
| CUDA Toolkit 12.4 (`nvcc`) | компиляция `.cu`; setup поставит через winget, если нет | `nvcc --version`, либо готовый `.pyd` в `gpu/nf4` |
| Linux: `g++` | сборка ядра (setup не делает sudo apt) | `g++ --version` |
| Место на диске | 3B ≈ 6 ГБ BF16 + потом `.chr` | для дыма хватит 3B |

Не используйте conda-окружение `torch-gpu` на чужой машине: оно лабораторное.
Соседу — отдельный `.venv` в каталоге репозитория.

---

## 2. Поставить (один раз)

Скрипт создаёт `.venv`, ставит **CUDA**-torch с индекса `cu124` (не обычный
PyPI: там чаще CPU-колесо), пакет `deepfold[hub,chat]`, собирает `chr`
(Go 1.22+ с PATH или переносной Go 1.22 с go.dev), при отсутствии `cl`/`nvcc`
ставит VS Build Tools и CUDA 12.4 через winget, собирает `gpu/nf4` и вызывает
`doctor`. Winget может показать UAC; для этих двух установщиков надёжнее
PowerShell от администратора.

**Windows (PowerShell):**

```powershell
cd deep-fold
Set-ExecutionPolicy -Scope Process Bypass
powershell -File scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

**Linux:**

```bash
cd deep-fold
bash scripts/setup.sh
source .venv/bin/activate
```

Код выхода `doctor`:

| Код | Значение | Что делать |
|---|---|---|
| **0** | generate возможен | дальше `pull` и `chat` |
| **2** | карта подходящая, не хватает куска install | читать `[fail]` в отчёте: нет `chr`, нет `nvcc`/`cl`, CPU-torch |
| **3** | этот класс машины generate не умеет | compress на CPU может работать; generate — нет (Hopper, Turing, macOS, нет NVIDIA) |
| **1** | не работает ни generate, ни compress | Python / Go / окружение |

Ada и A100 при живом install должны дать **0**, не 3, со строкой
`generate: experimental`.

Если venv уже есть, внутри него можно только напечатать план (без `pip`):

```text
deepfold setup --dry-run
```

`deepfold setup` без `--dry-run` догонит torch и `chr` **в текущем**
интерпретаторе и **откажется**, если это conda `torch-gpu`.

---

## 3. Первый разговор (после doctor = 0)

Скачать allowlist-модель (не произвольный HuggingFace). `pull` сам печатает
готовые команды — копируйте путь оттуда, не выдумывайте.

```text
deepfold pull Qwen/Qwen2.5-3B-Instruct --yes
```

Пример того, что печатает `pull`:

```text
deepfold chat --model D:\weights\Qwen2.5-3B-Instruct
deepfold run --model D:\weights\Qwen2.5-3B-Instruct
```

Куда кладётся дерево:

- если задан `DEEPFOLD_MODELS` — `%DEEPFOLD_MODELS%\Qwen2.5-3B-Instruct`
- на машине автора, если есть `C:\dev\models` — туда же
- иначе — `%DEEPFOLD_HOME%\hf\...` (`%LOCALAPPDATA%\deepfold` / `~/.cache/deepfold`)

Откройте **настоящее окно терминала** (не пайп, не редирект) и запустите
строку `chat`, которую напечатал `pull`.

Первый `chat` / `run` без готового `.chr` вызовет `chr compress` (CPU, минуты).
Потом можно удалить шарды safetensors, оставив `config.json`, токенизатор и
`.chr`.

Одноразовый ответ без чата (скрипты, пайпы):

```text
deepfold run --model D:\weights\Qwen2.5-3B-Instruct --prompt "Say hi in one sentence." --max-new-tokens 32
```

---

## 4. Справка CLI (`-h` / `--help`)

Снято с живого `python -m gpu.cli` (то же, что `deepfold`).
`-h` и `--help` равнозначны.

### Корень

```text
deepfold -h
```

```text
usage: deepfold [-h] <command> ...

Packed CHR0 driver (NF4 or VQ 2-bit): weights stay packed in VRAM for the
whole run. Ampere-family CUDA (sm_86 measured; sm_80/sm_89 experimental).
Turing / Hopper / Blackwell refuse.

positional arguments:
  <command>
    doctor     can deepfold run succeed on this machine? exit 0 if yes
    compress   wrap chr compress (NF4 or VQ 2-bit)
    run        load packed NF4 or VQ weights and generate
    chat       TTY chat session (history + streamed tokens)
    from-ollama
               map an allowlisted Ollama tag to a HuggingFace id (never GGUF)
    pull       download an allowlisted HuggingFace BF16 tree (never GGUF)
    setup      install CUDA torch + build chr in this interpreter (not torch-
               gpu)
    test       run CLI acceptance (no Hub). --live skips unless 3B is on disk

options:
  -h, --help   show this help message and exit
```

Без подкоманды то же help уходит в stderr, код выхода 1.

---

## 5. Команды: зачем, параметры, примеры

### `doctor` — «эта машина вообще сможет run/chat?»

```text
deepfold doctor -h
```

```text
usage: deepfold doctor [-h] [--model MODEL] [--chr-bin CHR_BIN]
                       [--compress-only]

Exit 0 run is possible; 2 broken install on a card that could run; 3 generate
refused by this machine's class but chr compress works; 1 neither.

options:
  -h, --help         show this help message and exit
  --model MODEL      also check this HuggingFace dir's config.json
  --chr-bin CHR_BIN  path to the Go chr binary
  --compress-only    ask only whether chr compress can run (macOS / CPU boxes)
```

```text
deepfold doctor
deepfold doctor --model D:\weights\Qwen2.5-3B-Instruct
deepfold doctor --compress-only
```

Коды выхода — таблица в §2. Ada/A100: 0 + `experimental`, не 3.

### `setup` — догнать torch CUDA и `chr` в **этом** Python

Соседний ПК сначала гоняет `scripts/setup.ps1` / `setup.sh`. Эта команда —
догон внутри уже созданного venv.

```text
deepfold setup -h
```

```text
usage: deepfold setup [-h] [--chr-bin CHR_BIN] [--dry-run]

Catch-up inside an existing venv. Refuses conda env torch-gpu. A neighbor PC
should run scripts/setup.ps1 or scripts/setup.sh first.

options:
  -h, --help         show this help message and exit
  --chr-bin CHR_BIN  path to the Go chr binary
  --dry-run          print the commands; never pip install
```

```text
deepfold setup --dry-run
deepfold setup
```

В conda `torch-gpu` без `--dry-run` код 1 и текст «создайте `.venv` скриптом».

### `pull` — скачать BF16 с HuggingFace (только таблица)

Не GGUF, не Ollama blobs, не произвольный id.

| HuggingFace id | Диск BF16 | Зачем |
|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | ~6.2 ГБ | дым и чат по умолчанию |
| `Qwen/Qwen2.5-14B-Instruct` | ~29.5 ГБ | resident NF4 |
| `internlm/internlm2_5-20b-chat` | ~40 ГБ | internlm; extra `pip install "deepfold[internlm]"` |
| `Qwen/Qwen2.5-32B-Instruct` | ~65 ГБ | overflow H2, не дефолт |

```text
deepfold pull -h
```

```text
usage: deepfold pull [-h] [--dir DIR] [--yes] hf_id

Allowlisted Hub ids from docs/models.md. Arbitrary repos are refused. Confirm
disk (--yes or a TTY). Extra: pip install "deepfold[hub]".

positional arguments:
  hf_id       HuggingFace id, e.g. Qwen/Qwen2.5-3B-Instruct

options:
  -h, --help  show this help message and exit
  --dir DIR   download destination
  --yes, -y   do not ask before snapshot_download (required when stdin is not
              a TTY)
```

```text
deepfold pull Qwen/Qwen2.5-3B-Instruct --yes
deepfold pull Qwen/Qwen2.5-3B-Instruct --yes --dir D:\weights\Qwen2.5-3B-Instruct
deepfold pull meta-llama/Llama-3.1-8B-Instruct
```

Последняя строка должна отказаться (id не в таблице), файл не открывать.
Без `--yes` и без TTY ничего не качается, код 1.
Скрипт setup уже ставит extra `hub`.

### `chat` — разговор в терминале

Нужен настоящий TTY (окно PowerShell / терминал Linux, не редирект).
Веса грузятся один раз. Каждый ход заново префиллит **всю** историю из JSON
в `$DEEPFOLD_HOME/chats` (KV между ходами не копится). У `chat` по умолчанию
256 новых токенов и `--max-seq 2048` (`run` остаётся 64 / 512).

```text
deepfold chat -h
```

```text
usage: deepfold chat [-h] [--model MODEL] [--chr CHR] [--codec {auto,nf4,vq}]
                     [--chr-bin CHR_BIN] [--max-new-tokens MAX_NEW_TOKENS]
                     [--max-seq MAX_SEQ] [--max-resident-mib MAX_RESIDENT_MIB]
                     [--raw] [--no-warmup] [--no-compress] [--quiet] [--debug]

Same load path as run, then a prompt_toolkit session. Enter sends, Ctrl+J newline, Ctrl+C stops a reply. Each turn prefills the whole chat. Needs a TTY; scripts use run --prompt.

options:
  -h, --help            show this help message and exit
  --model MODEL         HuggingFace directory (or $DEEPFOLD_MODEL)
  --chr CHR             packed weights (or $DEEPFOLD_CHR, or a sibling)
  --codec {auto,nf4,vq}
                        when packing: NF4 if it fits, else NF4 overflow (H2);
                        --codec vq is oracle-only
  --chr-bin CHR_BIN     path to the Go chr binary
  --max-new-tokens MAX_NEW_TOKENS
                        tokens to generate per turn (default: 256)
  --max-seq MAX_SEQ     preallocated KV length (default: 2048)
  --max-resident-mib MAX_RESIDENT_MIB
                        HBM cap for NF4 weights in MiB; overflow streams the
                        rest (H2). Default: fully resident if NF4 fits, else
                        auto from VRAM and --max-seq. Canary: fake a small cap
                        on 3B without a 32B file. --codec vq ignores this.
  --raw                 tokenize the prompt as-is, without the model's chat
                        template
  --no-warmup           skip the warmup pass (first token then pays for kernel
                        setup)
  --no-compress         fail instead of packing when no .chr is found
  --quiet               no chr progress output
  --debug               traceback after the report
```

`--max-seq` у `chat` по умолчанию 2048.

```text
deepfold chat --model D:\weights\Qwen2.5-3B-Instruct
deepfold chat --model D:\weights\Qwen2.5-3B-Instruct --chr D:\weights\qwen25-3b.nf4.chr --max-new-tokens 128
deepfold chat --model D:\weights\Qwen2.5-3B-Instruct --max-seq 1024 --no-warmup
deepfold chat --model D:\weights\Qwen2.5-14B-Instruct --agent --workspace C:\dev\deep-fold
```

`--agent` (или `/agent on`) даёт модели список/чтение/запись файлов в
`--workspace` (по умолчанию текущий каталог) и pytest по пути внутри него.
Запись и тесты спрашивают `allow this tool? [y/N]`. Произвольного shell нет.
Для JSON инструментов лучше 14B, чем 3B. `--agent` не сочетается с `--raw`.

Внутри сессии:

| Ввод | Действие |
|---|---|
| Enter | отправить реплику |
| Ctrl+J | новая строка в том же сообщении |
| Ctrl+C | остановить ответ; на пустом промпте дважды — выход |
| `/help` | шпаргалка |
| `/stats` | prefill ms, tok/s и причина стопа |
| `/clear` | сбросить историю |
| `/new` | новый сохранённый чат |
| `/chats` | выбрать сохранённый чат |
| `/copy` | скопировать последний ответ (`/copy all` — весь чат) |
| `/save [path]` | записать последний ответ в файл |
| `/agent on` `/agent off` | инструменты в workspace |
| `/quit` или `/exit` | выйти |
| любая другая `/foo` | отказ, в модель не идёт |

Нужен extra `prompt_toolkit` (`deepfold[chat]`, скрипт setup ставит).
Если запустить не из TTY — код 1 и совет `run --prompt`.
Ответы рисуют markdown (жирный, списки, ограды) и переводят LaTeX `$...$` /
`$$` в Unicode. `/copy` по-прежнему берёт сырой текст модели.

### `run` — один промпт или простой REPL

Для скриптов и пайпов. Интерактивный разговор — `chat`.

```text
deepfold run -h
```

```text
usage: deepfold run [-h] [--model MODEL] [--chr CHR] [--codec {auto,nf4,vq}]
                    [--chr-bin CHR_BIN] [--prompt PROMPT]
                    [--max-new-tokens MAX_NEW_TOKENS] [--max-seq MAX_SEQ]
                    [--max-resident-mib MAX_RESIDENT_MIB] [--raw]
                    [--no-warmup] [--no-compress] [--quiet] [--debug]

Needs two things: a HuggingFace directory (config.json, tokenizer) and one
.chr of packed weights (NF4 or VQ 2-bit). Compresses once if the .chr is
missing. --codec auto (default) packs NF4 when it fits this card, else NF4
overflow (H2). VQ 2-bit is --codec vq only. TTY one-liners: prefer deepfold
chat.
```

Флаги те же, что у `chat`, плюс `--prompt PROMPT` (one-shot вместо stdin REPL).

```text
deepfold run --model D:\weights\Qwen2.5-3B-Instruct --prompt "Capital of France?" --max-new-tokens 16
deepfold run --model D:\weights\Qwen2.5-3B-Instruct --no-compress
deepfold run --model D:\weights\Qwen2.5-3B-Instruct --prompt "Hi" --raw --debug
```

Windows, строки из файла:

```text
Get-Content questions.txt | deepfold run --model D:\weights\Qwen2.5-3B-Instruct
```

Без `--prompt` на TTY — строка `> ` (каждая строка = новый prefill, история
не копится). Для разговора лучше `deepfold chat`.

Ответ модели — stdout, диагностика — stderr:

```text
deepfold run --model DIR --prompt "Hi" > answer.txt
```

### `compress` — только упаковать, без generate

CPU. Нужен `chr` на PATH или `DEEPFOLD_CHR_BIN`.

```text
deepfold compress -h
```

```text
usage: deepfold compress [-h] --in INP [--out OUT] [--chr-bin CHR_BIN]
                         [--codec {auto,nf4,vq}] [--vram-mib VRAM_MIB]
                         [--force] [--quiet]

Packs a HuggingFace BF16/FP16 tree into one .chr. CPU only. Default --codec
auto: NF4 if it fits the card, else NF4 overflow (H2). VQ 2-bit is --codec vq
only (3B greedy canary failed).

options:
  -h, --help            show this help message and exit
  --in INP              HuggingFace directory
  --out OUT             output .chr (default: $DEEPFOLD_HOME/chr/<slug>)
  --chr-bin CHR_BIN     path to the Go chr binary
  --codec {auto,nf4,vq}
                        packed format; auto = NF4 if it fits, else NF4
                        overflow (H2); VQ is --codec vq only
  --vram-mib VRAM_MIB   card size for --codec auto (default: this GPU, else
                        12288)
  --force               repack over an existing .chr
  --quiet               no chr progress output
```

```text
deepfold compress --in D:\weights\Qwen2.5-3B-Instruct --out D:\weights\qwen25-3b.nf4.chr --codec nf4
deepfold compress --in D:\weights\Qwen2.5-3B-Instruct --force
deepfold compress --in D:\weights\Qwen2.5-3B-Instruct --codec auto --vram-mib 12288
```

VQ только с `--codec vq` (на 3B greedy-чат ломается — это оракул ядра).

### `test` — приёмка CLI без Hub и без generate

```text
deepfold test -h
```

```text
usage: deepfold test [-h] [--live] [--chr-bin CHR_BIN]

options:
  -h, --help         show this help message and exit
  --live             check doctor + 3B tree; does not generate and does not
                     download
  --chr-bin CHR_BIN  path to the Go chr binary
```

```text
deepfold test
deepfold test --live
```

`test` гоняет `gpu/cli/test_cli.py` (ноутбук, без сети).
`--live` **не** качает модель и **не** генерирует: если doctor разрешает
generate и на диске есть 3B — печатает `LIVE:`; иначе `SKIP:` (это не
зелёный pass generate).

### `from-ollama` — имя из библиотеки Ollama → тот же HuggingFace id

GGUF из `~/.ollama` **не читается**. Теги: `qwen2.5:3b`, `qwen2.5:3b-instruct`,
`qwen2.5:14b`, `qwen2.5:14b-instruct`. 20B и 32B — через `pull` или `--hf`.

```text
deepfold from-ollama -h
```

```text
usage: deepfold from-ollama [-h] [--hf HF] [--dir DIR] [--run] [--yes] tag

Allowlisted Ollama library names become HuggingFace BF16 trees. The command
never reads ~/.ollama and never loads GGUF.

positional arguments:
  tag         library tag, e.g. qwen2.5:3b

options:
  -h, --help  show this help message and exit
  --hf HF     HuggingFace id; must match this tag, or be a table id if the tag
              is unknown
  --dir DIR   download destination (default: $DEEPFOLD_HOME/hf/<slug>)
  --run       after resolving the tree, invoke deepfold run (compress is still
              first-run of run)
  --yes, -y   do not ask before snapshot_download (required when stdin is not
              a TTY)
```

```text
deepfold from-ollama qwen2.5:3b --yes
deepfold from-ollama qwen2.5:3b --yes --run
deepfold from-ollama qwen2.5:3b --hf Qwen/Qwen2.5-3B-Instruct --yes --dir D:\weights\Qwen2.5-3B-Instruct
```

---

## 6. Переменные окружения

| Переменная | Смысл |
|---|---|
| `DEEPFOLD_MODEL` | папка HuggingFace по умолчанию для `run` / `chat` |
| `DEEPFOLD_CHR` | файл `.chr`, если он есть и заголовок совпадает с моделью |
| `DEEPFOLD_MODELS` | корень деревьев для `pull` и лаборатории |
| `DEEPFOLD_HOME` | кэш (`%LOCALAPPDATA%\deepfold` / `~/.cache/deepfold`); чаты в `chats/` |
| `DEEPFOLD_CHR_BIN` | `chr` / `chr.exe` |
| `DEEPFOLD_RUNS` | дампы лабораторных прогонов |
| `DEEPFOLD_COPY_JOIN` | `1`/`0` — CPU join H2D (Windows по умолчанию да, Linux нет) |

```powershell
set DEEPFOLD_MODELS=D:\weights
set DEEPFOLD_MODEL=D:\weights\Qwen2.5-3B-Instruct
deepfold pull Qwen/Qwen2.5-3B-Instruct --yes
deepfold chat --model %DEEPFOLD_MODEL%
```

```bash
export DEEPFOLD_MODELS=$HOME/models
export DEEPFOLD_MODEL=$HOME/models/Qwen2.5-3B-Instruct
deepfold pull Qwen/Qwen2.5-3B-Instruct --yes
deepfold chat --model "$DEEPFOLD_MODEL"
```

---

## 7. Если сломалось

| Симптом | Что обычно |
|---|---|
| `doctor` код 2, torch cpu | колесо с PyPI; нужен индекс cu124, как в `scripts/setup.*` |
| `doctor` код 2, нет chr | Снова `deepfold setup` или `python -m gpu.cli.go_toolchain` (нужен доступ к go.dev). Либо `go build -o chr.exe ./cmd/chr` |
| `doctor` код 2, нет ядра | Снова `deepfold setup --kernel-only` (winget VS Build Tools + CUDA 12.4). Либо скопировать подходящий `gpu/nf4/chr_nf4_ext*.pyd` |
| `doctor` код 3 на Ada | баг старого контракта; после K4 так быть не должно |
| `chat` «needs a TTY» | запуск из пайпа / IDE без TTY; возьмите окно терминала или `run --prompt` |
| `chat` просит prompt_toolkit | `pip install "deepfold[chat]"` |
| `deepfold` не является командой | Activate `.venv`, или `.\.venv\Scripts\deepfold.exe`, или `python -m gpu.cli`. conda `torch-gpu` команду не ставит. Пикер чатов: `--new`. |
| `pull` unknown id | только четыре строки таблицы выше |
| CUDA OOM | закройте другие GPU-программы; 32B — overflow, не 3B |
| первый запуск минута+ | JIT fatbinary на соседней карте, один раз |

Лабораторные пластины (`python -m gpu.lab.run`) — отдельный стенд, не этот CLI.
Подробности продукта: [`ux.md`](ux.md).

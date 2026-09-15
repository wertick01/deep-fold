# Install and commands (neighbor PC)

Short command list (install, pull, compress, tests, chat, neighbor plate):
[`quickstart.md`](quickstart.md). Full `-h` dumps follow.

Deepfold is a **terminal** program: it packs model weights into one `.chr`
file and generates text on NVIDIA Ampere or Ada. It is not a website, not an
Ollama plugin, and not a browser chat.

Measured tok/s from the author’s RTX 3080 **do not transfer**. Ada (RTX 40xx,
`sm_89`) and A100 (`sm_80`) may generate; `doctor` will say `experimental`.
Turing, Hopper, Blackwell, AMD, and macOS generate are refused.

Deepfold does **not** install the NVIDIA driver, Python, Go, or a CUDA
compiler. A red `doctor` is a normal install refusal, not a kernel bug.

After `pip install -e .`, `deepfold` and `python -m gpu.cli` are the same.
If `deepfold` is not on PATH yet, use `python -m gpu.cli`. `-h` and `--help`
are equivalent. Dumps below were captured from a live `python -m gpu.cli`.

---

## Shortest path (another PC)

```powershell
git clone https://github.com/wertick01/deep-fold.git
cd deep-fold
Set-ExecutionPolicy -Scope Process Bypass
powershell -File scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
deepfold doctor
deepfold pull Qwen/Qwen2.5-3B-Instruct --yes
deepfold chat --model <directory printed by pull>
```

Linux instead of the two setup lines:

```bash
git clone https://github.com/wertick01/deep-fold.git
cd deep-fold
bash scripts/setup.sh
source .venv/bin/activate
```

Then the same `doctor` / `pull` / `chat`. The first generate on a neighbor
card may JIT the CUDA kernel (~1 min). Do not promise 3080 tok/s.

---

## 1. Already on the machine

| Need | Why | Check |
|---|---|---|
| NVIDIA driver (Ampere or Ada) | GPU | `nvidia-smi` prints the card name |
| Python 3.11 or 3.12 | runtime | Windows: `py -3.11 --version`; Linux: `python3 --version` |
| Go 1.22+ | build the `chr` compressor | `go version` |
| Windows: MSVC Build Tools | JIT if there is no prebuilt `.pyd` | `cl` after `vcvars64.bat` |
| Linux: `g++` and `nvcc` | JIT fatbinary (~1 min first run) | `nvcc --version` |
| Disk | 3B ≈ 6 GB BF16, then a `.chr` | 3B is enough for smoke |

Do not use conda env `torch-gpu` on a neighbor box: that is the author’s lab
interpreter. Neighbors get a repo-local `.venv`.

---

## 2. Install once

The script creates `.venv`, installs **CUDA** torch from the `cu124` index
(default PyPI is usually a CPU wheel), `deepfold[hub,chat]`, builds `chr`,
then runs `doctor`.

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

`doctor` exit codes:

| Code | Meaning | Next |
|---|---|---|
| **0** | generate is possible | `pull` then `chat` |
| **2** | this card could run; install is incomplete | read `[fail]`: missing `chr`, `nvcc`/`cl`, or CPU torch |
| **3** | this machine class cannot generate | compress on CPU may still work (Hopper, Turing, macOS, no NVIDIA) |
| **1** | neither generate nor compress | Python / Go / environment |

Ada and A100 with a live install must be **0**, not 3, with
`generate: experimental`.

If the venv already exists, print the plan without `pip`:

```text
deepfold setup --dry-run
```

`deepfold setup` without `--dry-run` catches up torch and `chr` **in this**
interpreter and **refuses** conda env `torch-gpu`.

---

## 3. First conversation (after doctor = 0)

Download an allowlisted model (not an arbitrary HuggingFace id). `pull`
prints ready-to-run commands — copy that path.

```text
deepfold pull Qwen/Qwen2.5-3B-Instruct --yes
```

Example of what `pull` prints:

```text
deepfold chat --model D:\weights\Qwen2.5-3B-Instruct
deepfold run --model D:\weights\Qwen2.5-3B-Instruct
```

Where the tree lands:

- if `DEEPFOLD_MODELS` is set — `%DEEPFOLD_MODELS%\Qwen2.5-3B-Instruct`
- on the author box, if `C:\dev\models` exists — there
- else `$DEEPFOLD_HOME/hf/...` (`%LOCALAPPDATA%\deepfold` / `~/.cache/deepfold`)

Open a **real terminal window** (not a pipe) and run the `chat` line `pull`
printed.

The first `chat` / `run` without a `.chr` runs `chr compress` (CPU, minutes).
After that you may delete the safetensor shards and keep `config.json`, the
tokenizer, and the `.chr`.

One-shot without chat (scripts, pipes):

```text
deepfold run --model D:\weights\Qwen2.5-3B-Instruct --prompt "Say hi in one sentence." --max-new-tokens 32
```

---

## 4. CLI help (`-h` / `--help`)

### Root

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

With no subcommand the same help goes to stderr and the process exits 1.

---

## 5. Commands: flags and examples

### `doctor` — can this machine run / chat?

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

Exit codes: table in §2. Ada/A100: 0 + `experimental`, never 3.

### `setup` — catch up CUDA torch and `chr` in **this** Python

A neighbor PC runs `scripts/setup.ps1` / `setup.sh` first. This command is
catch-up inside an existing venv.

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

Inside conda `torch-gpu` without `--dry-run`: exit 1 and “create a `.venv`
with the script”.

### `pull` — download BF16 from HuggingFace (allowlist only)

Not GGUF, not Ollama blobs, not an arbitrary id.

| HuggingFace id | Disk BF16 | Role |
|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | ~6.2 GB | default smoke / chat |
| `Qwen/Qwen2.5-14B-Instruct` | ~29.5 GB | resident NF4 |
| `internlm/internlm2_5-20b-chat` | ~40 GB | internlm; extra `pip install "deepfold[internlm]"` |
| `Qwen/Qwen2.5-32B-Instruct` | ~65 GB | overflow H2, not the default |

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

The last line must refuse (not in the table) and must not open a file.
Without `--yes` and without a TTY nothing is downloaded, exit 1.
The setup script already installs the `hub` extra.

### `chat` — conversation in a terminal

Needs a real TTY (PowerShell / Linux terminal, not a redirect). Weights load
once. Each turn prefills the **whole** history (KV is not reused across turns
in v1).

```text
deepfold chat -h
```

```text
usage: deepfold chat [-h] [--model MODEL] [--chr CHR] [--codec {auto,nf4,vq}]
                     [--chr-bin CHR_BIN] [--max-new-tokens MAX_NEW_TOKENS]
                     [--max-seq MAX_SEQ] [--max-resident-mib MAX_RESIDENT_MIB]
                     [--raw] [--no-warmup] [--no-compress] [--quiet] [--debug]

Same load path as run, then a prompt_toolkit session. Enter sends, Ctrl+J new
line. Each turn prefills the whole chat. Needs a TTY; scripts use run
--prompt.

options:
  -h, --help            show this help message and exit
  --model MODEL         HuggingFace directory (or $DEEPFOLD_MODEL)
  --chr CHR             packed weights (or $DEEPFOLD_CHR, or a sibling)
  --codec {auto,nf4,vq}
                        when packing: NF4 if it fits, else NF4 overflow (H2);
                        --codec vq is oracle-only
  --chr-bin CHR_BIN     path to the Go chr binary
  --max-new-tokens MAX_NEW_TOKENS
                        tokens to generate per turn (default: 64)
  --max-seq MAX_SEQ     preallocated KV length
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

`--max-seq` defaults to 512.

```text
deepfold chat --model D:\weights\Qwen2.5-3B-Instruct
deepfold chat --model D:\weights\Qwen2.5-3B-Instruct --chr D:\weights\qwen25-3b.nf4.chr --max-new-tokens 128
deepfold chat --model D:\weights\Qwen2.5-3B-Instruct --max-seq 1024 --no-warmup
```

In-session:

| Input | Action |
|---|---|
| Enter | send the turn |
| Ctrl+J | newline in the same message |
| `/help` | this cheat sheet |
| `/clear` | drop history |
| `/stats` | last turn prefill ms and tok/s |
| `/quit` or `/exit` | leave |
| any other `/foo` | refused, not sent to the model |

Needs the `prompt_toolkit` extra (`deepfold[chat]`; the setup script installs
it). Not a TTY → exit 1 and “use `run --prompt`”.

### `run` — one prompt or a thin REPL

For scripts and pipes. Interactive talk is `chat`.

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

Same flags as `chat`, plus `--prompt PROMPT` (one-shot instead of the stdin
REPL).

```text
deepfold run --model D:\weights\Qwen2.5-3B-Instruct --prompt "Capital of France?" --max-new-tokens 16
deepfold run --model D:\weights\Qwen2.5-3B-Instruct --no-compress
deepfold run --model D:\weights\Qwen2.5-3B-Instruct --prompt "Hi" --raw --debug
```

Windows, lines from a file:

```text
Get-Content questions.txt | deepfold run --model D:\weights\Qwen2.5-3B-Instruct
```

Without `--prompt` on a TTY you get `> ` (each line is a new prefill; no
history). Prefer `deepfold chat` for a conversation.

Model text is stdout; diagnostics are stderr:

```text
deepfold run --model DIR --prompt "Hi" > answer.txt
```

### `compress` — pack only, no generate

CPU. Needs `chr` on PATH or `DEEPFOLD_CHR_BIN`.

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

VQ is `--codec vq` only (3B greedy chat collapses — kernel oracle, not a
talking model).

### `test` — CLI acceptance without Hub or generate

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

`test` runs `gpu/cli/test_cli.py` (laptop, no network).
`--live` does **not** download and does **not** generate: if doctor allows
generate and the 3B tree is on disk it prints `LIVE:`; otherwise `SKIP:`
(that is not a green generate pass).

### `from-ollama` — Ollama library name → the same HuggingFace id

It never reads GGUF under `~/.ollama`. Tags: `qwen2.5:3b`,
`qwen2.5:3b-instruct`, `qwen2.5:14b`, `qwen2.5:14b-instruct`. 20B and 32B go
through `pull` or `--hf`.

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

## 6. Environment variables

| Variable | Meaning |
|---|---|
| `DEEPFOLD_MODEL` | default HuggingFace dir for `run` / `chat` |
| `DEEPFOLD_CHR` | `.chr` file if present and the header matches the model |
| `DEEPFOLD_MODELS` | root for `pull` trees and the lab |
| `DEEPFOLD_HOME` | cache (`%LOCALAPPDATA%\deepfold` / `~/.cache/deepfold`) |
| `DEEPFOLD_CHR_BIN` | `chr` / `chr.exe` |
| `DEEPFOLD_RUNS` | lab run dumps |
| `DEEPFOLD_COPY_JOIN` | `1`/`0` — CPU join H2D (Windows on by default, Linux off) |

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

## 7. If it breaks

| Symptom | Usual cause |
|---|---|
| `doctor` exit 2, torch cpu | PyPI wheel; need the cu124 index as in `scripts/setup.*` |
| `doctor` exit 2, no chr | `go build -o chr.exe ./cmd/chr` (Linux: `go build -o chr ./cmd/chr`) |
| `doctor` exit 2, no kernel | no `.pyd`/`.so` and no `cl`/`g++`+`nvcc` for JIT |
| `doctor` exit 3 on Ada | old contract bug; after K4 this must not happen |
| `chat` “needs a TTY” | pipe / IDE without a TTY; use a terminal window or `run --prompt` |
| `chat` asks for prompt_toolkit | `pip install "deepfold[chat]"` |
| `pull` unknown id | only the four table rows above |
| CUDA OOM | close other GPU programs; 32B is overflow, not 3B |
| first run takes a minute+ | one-time fatbinary JIT on the neighbor card |

Lab plates (`python -m gpu.lab.run`) are a separate stand, not this CLI.
Product details: [`ux.md`](ux.md).

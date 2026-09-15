# QuickStart

Short command list. Full `-h` dumps and flags: [`install.md`](install.md).

After `scripts/setup.ps1`, `deepfold` and `python -m gpu.cli` are the same.
If `deepfold` is not on PATH, use `python -m gpu.cli` (this is normal in conda
env `torch-gpu`). After setup, activate `.venv` or call
`.\\.venv\\Scripts\\deepfold.exe`.

Paris / Berlin / 323 smoke is **not** a quality score. Do not expect the
author’s 3080 tok/s on another card.

---

## 1. Install

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

You already need: NVIDIA driver (Ampere or Ada), Python 3.11+, Go 1.22+,
MSVC Build Tools on Windows, `g++` and `nvcc` on Linux. The script does **not**
install the driver. `doctor` 0 = generate possible; 2 = this card could run,
install incomplete; 3 = this machine class cannot generate.

---

## 2. Download a model

Allowlist only (not arbitrary HuggingFace, not GGUF):

```text
deepfold pull Qwen/Qwen2.5-3B-Instruct --yes
```

Also: `Qwen/Qwen2.5-14B-Instruct`, `internlm/internlm2_5-20b-chat`
(`pip install "deepfold[internlm]"`), `Qwen/Qwen2.5-32B-Instruct`.

`pull` prints ready `chat` / `run` lines. Copy that path.

Trees land under `$DEEPFOLD_MODELS\<name>`, else the cache
(`%LOCALAPPDATA%\deepfold` / `~/.cache/deepfold`).

---

## 3. Compress (NF4 → `.chr`)

The first `chat` / `run` packs if needed. Explicitly:

```text
deepfold compress --in D:\weights\Qwen2.5-3B-Instruct --codec nf4
```

`--codec auto` (default): NF4 if it fits, else NF4 overflow (H2).
VQ is `--codec vq` only — kernel oracle, not a talking model.

CPU, minutes (14B/20B/32B longer). After that you may delete safetensor shards
and keep `config.json`, the tokenizer, and the `.chr`. `chr verify` still needs
the shards.

---

## 4. Integrity and tests

CLI with no network and no generate:

```text
deepfold test
deepfold test --live
```

`--live` does **not** download or generate: 3B on disk and doctor allows
generate → `LIVE:`, else `SKIP:`.

Compression integrity (CPU, orig vs `.chr`):

```text
chr verify --orig D:\weights\Qwen2.5-3B-Instruct --chr D:\weights\qwen25-3b.nf4.chr --fail-rmse 0.12 --fail-maxabs 2.0 --json
```

`chr` must be on PATH (setup builds it). Exit 0 = PASS, 2 = threshold, 1 =
input error. This is **not** chat and not WikiText.

---

## 5. Talk

A real terminal window (not a pipe):

```text
deepfold chat --model D:\weights\Qwen2.5-3B-Instruct
```

Enter sends, Ctrl+J newline. Ctrl+C stops a reply. `/help` `/quit` `/clear` `/stats` `/new` `/chats` `/copy` `/save` `/agent`. Replies render markdown (bold, lists, fences) and turn `$...$` / `$$` LaTeX into Unicode. Optional: `--agent --workspace .` for workspace tools (writes and pytest ask first).
Chat default is 256 new tokens (not the smoke cap of 64).

Scripts / pipes:

```text
deepfold run --model D:\weights\Qwen2.5-3B-Instruct --prompt "Capital of France?" --max-new-tokens 16
```

---

## 6. Neighbor plate (one model → metrics)

The script runs: CPU CLI tests, pull if needed, NF4 compress, `chr verify`,
greedy smoke Paris / Berlin / 323. It does **not** run BF16, VQ, or hard-12.

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

Model: `3b` / `14b` / `20b` / `32b`, a HuggingFace id, tag `qwen2.5:3b`, or a
local folder with `config.json`.

Useful flags (after the model):

```text
--dry-run          plan only; no download, no generate
--out DIR          report directory
--skip-test        skip gpu/cli/test_cli.py
--skip-verify      skip chr verify (already SKIP if shards were deleted)
--skip-generate    stop after verify
--no-download      do not hit the Hub
--force            repack .chr
```

Output is `$DEEPFOLD_RUNS/plate-<model>-<time>` (or `--out`). Send the whole
folder: **`SUMMARY.txt`**, **`plate.json`**, and if generate ran
`nf4/summary.csv`, `nf4/messages.csv`, `verify.json`.

First generate on another card may JIT the kernel (~1 min). 32B is H2 overflow;
compress and verify take longer.

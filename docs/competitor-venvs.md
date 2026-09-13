# Isolated competitor venvs (later, not now)

One stack, one interpreter, under `C:\dev\models\venvs\<stack>`. The live lab
uses `C:\Users\Professional\anaconda3\envs\torch-gpu`. **Do not** `pip install`
into `torch-gpu`. A broken `bitsandbytes` there can take `chr_nf4_ext` with it,
and WDDM will not give VRAM back if two 4-bit runtimes share a process.

This file is the recipe. `python -m gpu.lab.competitor --recipe` prints it.
The harness never creates these envs and never installs packages.

Run the commands below only when:

1. The 3080 is **free** (no live 3B `lab.run` / generate / ncu).
2. Disk can take a few more PyTorch trees.
3. You want a real row, not another `SKIP:`.

Until then, `python -m gpu.lab.competitor --detect` is the honest result:
every e2e tok/s cell empty, every microbench `us` / `gb_s` / `tflop_s` /
occupancy / tensor / DRAM / regs / smem cell empty.

CPU llama.cpp tok/s is **not** a GPU competitor. Do not fill `llamacpp-q4`
from a CPU binary.

## Create empty venvs (no packages yet)

Use a Python that is **not** `torch-gpu`. `py -3.11` is enough.

```powershell
$Venvs = 'C:\dev\models\venvs'
$Stacks = @(
  'bitsandbytes-nf4',
  'gptq-marlin',
  'awq',
  'llamacpp-q4',
  'exllamav2-exl2',
  'vllm'
)
New-Item -ItemType Directory -Force -Path $Venvs | Out-Null
foreach ($name in $Stacks) {
  $dir = Join-Path $Venvs $name
  if (-not (Test-Path $dir)) {
    py -3.11 -m venv $dir
  }
}
```

Confirm you did **not** activate `torch-gpu` first:
`python -c "import sys; print(sys.prefix)"` must not end in `envs\torch-gpu`.

## Install, one env at a time, when the GPU is free

Point `--python` at that env's `Scripts\python.exe`. Same card, Qwen2.5-3B,
native artifact per stack. Never convert GGUF/GPTQ/AWQ/EXL2 into `.chr`.

Torch wheels should match the card (`cu124` / sm_86). Exact pins are whatever
the stack's Windows (or Linux) wheel actually builds against; a cu124 mismatch
is a `SKIP`, not a borrowed number.

### `bitsandbytes-nf4`

```powershell
$py = 'C:\dev\models\venvs\bitsandbytes-nf4\Scripts\python.exe'
& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu124
& $py -m pip install transformers accelerate bitsandbytes
```

Artifact: the HF BF16 tree (`DEEPFOLD_MODEL`, already on this box for 3B).
`Linear4bit`, `quant_type='nf4'`.

### `gptq-marlin`

```powershell
$py = 'C:\dev\models\venvs\gptq-marlin\Scripts\python.exe'
& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu124
& $py -m pip install transformers
# Then ONE of: gptqmodel (preferred) or auto-gptq, with a Marlin kernel that
# actually launches on sm_86. Marlin is typically Linux CUDA. On Windows a
# missing wheel is SKIP, never a tok/s copied from a Linux blog.
& $py -m pip install gptqmodel
```

Artifact: a GPTQ dump of the **same** model. `DEEPFOLD_GPTQ` = that directory.

### `awq`

```powershell
$py = 'C:\dev\models\venvs\awq\Scripts\python.exe'
& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu124
& $py -m pip install transformers autoawq
```

Artifact: an AWQ dump of the same model. `DEEPFOLD_AWQ` = that directory.

### `llamacpp-q4` (CUDA only)

The default PyPI `llama-cpp-python` wheel is often **CPU**. That build is a
skip. Force a CUDA-linked build:

```powershell
$py = 'C:\dev\models\venvs\llamacpp-q4\Scripts\python.exe'
& $py -m pip install --upgrade pip
$env:CMAKE_ARGS = '-DGGML_CUDA=on'
& $py -m pip install llama-cpp-python --no-binary llama-cpp-python
```

Then check `llama_supports_gpu_offload()` is True **inside that venv**. If it
is False, the row stays `SKIP`. Do not paste CPU tok/s into `summary.csv`.

Artifact: GGUF Q4_K_M and/or Q4_0 of the same model. `DEEPFOLD_GGUF` = the file.
Record which quant in `notes` when you actually measure.

### `exllamav2-exl2`

```powershell
$py = 'C:\dev\models\venvs\exllamav2-exl2\Scripts\python.exe'
& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu124
& $py -m pip install exllamav2
```

Artifact: an EXL2 dump of the same model. `DEEPFOLD_EXL2` = that directory.
A CPU fallback is not a GPU row.

### `vllm`

vLLM wheels are typically **Linux CUDA**. On this Windows box a missing wheel
is the expected `SKIP`. If you later measure on Linux, still a separate venv,
still not `torch-gpu`:

```powershell
$py = 'C:\dev\models\venvs\vllm\Scripts\python.exe'
& $py -m pip install --upgrade pip
& $py -m pip install vllm
```

Artifact: `DEEPFOLD_VLLM` (HF tree or a dump vLLM can load). A CPU-only engine
is not a GPU competitor.

## How to measure (still later, GPU free)

```powershell
cd C:\dev\deep-fold
$py = 'C:\dev\models\venvs\bitsandbytes-nf4\Scripts\python.exe'
# detect inside THAT venv (this interpreter cannot answer for another):
& $py -m gpu.lab.competitor --detect --stacks bitsandbytes-nf4

# e2e + named microbench grid (child in that venv). One stack on the 3080:
python -m gpu.lab.competitor --stacks bitsandbytes-nf4 --python $py --lab qwen25-3b
```

`--python` is required so a broken stack cannot import into `torch-gpu`.
The linear microbench body is still a skeleton: even after e2e tok/s exists,
`microbench.csv` stays `SKIP` until a kernel timer + ncu pass fills `us`,
`gb_s`, `tflop_s`, `occupancy`, `tensor`, `dram`, `regs`, `smem`. Do not copy
`gpu.nf4.bench` or `docs/runs/ncu/` into a competitor cell — those are our
kernel, not theirs.

## What cannot be measured until the GPU is free

- End-to-end tok/s and TTFT on this 3080 for any competitor
- Linear microbench `[M,K]×[K,N]` for `M=1..256` on Q/K/V/O/gate/up/down
- Nsight occupancy / tensor pipe / DRAM / regs / smem on **their** kernels
- Quantizing GPTQ / AWQ / EXL2 / GGUF of Qwen2.5-3B (those jobs also want the
  card or a long CPU convert)

`python -m gpu.lab.competitor --detect` does none of that. It names the cells.

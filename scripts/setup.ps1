# Neighbor install (Windows). Creates repo-local .venv with CUDA torch.
# Does not touch conda env torch-gpu. Does not install the NVIDIA driver.
#
#   Set-ExecutionPolicy -Scope Process Bypass
#   powershell -File scripts/setup.ps1

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Venv = Join-Path $Repo ".venv"
$VenvPy = Join-Path $Venv "Scripts\python.exe"
$TorchIndex = "https://download.pytorch.org/whl/cu124"

if (-not (Test-Path $VenvPy)) {
    Write-Host "Creating venv at $Venv"
    $made = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv $Venv
        if ($LASTEXITCODE -eq 0 -and (Test-Path $VenvPy)) { $made = $true }
    }
    if (-not $made) {
        & python -m venv $Venv
    }
    if (-not (Test-Path $VenvPy)) {
        throw "venv python missing at $VenvPy (need Python 3.11+)"
    }
}

Write-Host "Using $VenvPy"
& $VenvPy -m pip install --upgrade pip
& $VenvPy -m pip install torch --index-url $TorchIndex
& $VenvPy -m pip install -e ".[hub,chat]"

if (Get-Command go -ErrorAction SilentlyContinue) {
    & go build -o chr.exe ./cmd/chr
} else {
    Write-Host "Go not on PATH: skip chr build. Install Go 1.22+ and re-run, or set DEEPFOLD_CHR_BIN."
}

& $VenvPy -m gpu.cli doctor
Write-Host ""
Write-Host "Next:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  deepfold pull Qwen/Qwen2.5-3B-Instruct --yes"
Write-Host "  deepfold chat --model <that directory>"
Write-Host "Do not expect the 3080 tok/s plate on another card."

# Neighbor install (Windows). Creates repo-local .venv with CUDA torch.
# Does not touch conda env torch-gpu. Does not install the NVIDIA driver.
# If chr is missing, Python fetches portable Go 1.22 from go.dev (not MSI).
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
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit $LASTEXITCODE)" }
& $VenvPy -m pip install torch --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { throw "torch install failed (exit $LASTEXITCODE)" }
& $VenvPy -m pip install -e ".[hub,chat]"
if ($LASTEXITCODE -ne 0) { throw "deepfold extras install failed (exit $LASTEXITCODE)" }

Write-Host "Building chr (PATH Go 1.22+ or portable Go 1.22 from go.dev)"
& $VenvPy -m gpu.cli.go_toolchain
if ($LASTEXITCODE -ne 0) { throw "chr build failed (exit $LASTEXITCODE)" }

& $VenvPy -m gpu.cli doctor
if ($LASTEXITCODE -ne 0) {
    Write-Host "doctor exited $LASTEXITCODE (0 generate possible, 2 install, 3 generate refused by class, 1 neither)."
}

$DeepfoldCmd = Join-Path $Venv "Scripts\deepfold.exe"
Write-Host ""
Write-Host "Next:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  deepfold pull Qwen/Qwen2.5-3B-Instruct --yes"
Write-Host "  deepfold chat --model <that directory>"
if (Test-Path $DeepfoldCmd) {
    Write-Host "Without activate:"
    Write-Host "  $DeepfoldCmd doctor"
} else {
    Write-Host "WARN: deepfold.exe was not created. Use:"
    Write-Host "  $VenvPy -m gpu.cli doctor"
}
Write-Host "Do not expect the 3080 tok/s plate on another card."
Write-Host "conda env torch-gpu never gets a deepfold command; that lab uses python -m gpu.cli."

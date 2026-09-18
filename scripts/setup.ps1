# Neighbor install (Windows). Creates repo-local .venv with CUDA torch.
# Does not touch conda env torch-gpu. Does not install the NVIDIA driver.
# If chr is missing, Python fetches portable Go 1.22 from go.dev (not MSI).
# If the NF4 kernel is missing, winget-installs VS 2022 Build Tools (C++) and
# CUDA Toolkit (12.4 on Ampere; 12.8 then 12.4 on RTX 50 / 5070 Ti).
#
#   Set-ExecutionPolicy -Scope Process Bypass
#   powershell -File scripts/setup.ps1

$ErrorActionPreference = "Stop"

function Test-Python311([string]$Exe) {
    if (-not (Test-Path $Exe)) { return $false }
    & $Exe -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)" | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Get-TorchIndex {
    $cu124 = "https://download.pytorch.org/whl/cu124"
    $cu128 = "https://download.pytorch.org/whl/cu128"
    $raw = [string]$env:DEEPFOLD_TORCH_INDEX
    if ($raw) {
        $raw = $raw.Trim()
        if ($raw -like "http*") { return $raw }
        $key = $raw.ToLowerInvariant()
        if ($key -in @("cu128", "cu129", "sm120")) { return $cu128 }
        if ($key -in @("cu124", "ampere")) { return $cu124 }
    }
    $smiExe = $null
    $hit = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($hit) {
        $smiExe = $hit.Source
    } else {
        $sys = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
        if (Test-Path $sys) { $smiExe = $sys }
    }
    if ($smiExe) {
        $row = $null
        try {
            $row = & $smiExe --query-gpu=name,compute_cap --format=csv,noheader 2>$null |
                Select-Object -First 1
        } catch {
            $row = $null
        }
        if (-not $row) {
            try {
                $row = & $smiExe --query-gpu=name --format=csv,noheader 2>$null |
                    Select-Object -First 1
            } catch {
                $row = $null
            }
        }
        if ($row -match "(?i)rtx\s*50[0-9]{2}" -or $row -match "(?:,\s*)?12\.[01]\s*$") {
            $env:DEEPFOLD_PREFER_SM120_NVCC = "1"
            $env:DEEPFOLD_TORCH_INDEX = "cu128"
            return $cu128
        }
    }
    return $cu124
}

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Venv = Join-Path $Repo ".venv"
$VenvPy = Join-Path $Venv "Scripts\python.exe"
$TorchIndex = Get-TorchIndex

if (-not (Test-Path $VenvPy)) {
    Write-Host "Creating venv at $Venv"
    $made = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($ver in @("3.12", "3.11")) {
            & py "-$ver" -m venv $Venv
            if ($LASTEXITCODE -eq 0 -and (Test-Python311 $VenvPy)) {
                $made = $true
                break
            }
        }
    }
    if (-not $made) {
        & python -m venv $Venv
    }
    if (-not (Test-Path $VenvPy)) {
        throw "venv python missing at $VenvPy (need Python 3.11+)"
    }
}

if (-not (Test-Python311 $VenvPy)) {
    throw "$VenvPy is not Python 3.11+. Delete .venv and re-run with Python 3.11 or 3.12."
}

Write-Host "Using $VenvPy"
Write-Host "torch index: $TorchIndex"
& $VenvPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit $LASTEXITCODE)" }
& $VenvPy -m pip install torch --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { throw "torch install failed (exit $LASTEXITCODE)" }
& $VenvPy -m pip install -e ".[hub,chat]"
if ($LASTEXITCODE -ne 0) { throw "deepfold extras install failed (exit $LASTEXITCODE)" }
& $VenvPy -m pip install ninja
if ($LASTEXITCODE -ne 0) { throw "ninja install failed (exit $LASTEXITCODE)" }

Write-Host "Building chr (PATH Go 1.22+ or portable Go 1.22 from go.dev)"
& $VenvPy -m gpu.cli setup --chr-only
if ($LASTEXITCODE -ne 0) { throw "chr build failed (exit $LASTEXITCODE)" }

Write-Host "NF4 kernel: VS Build Tools (C++) + CUDA nvcc if missing, then compile"
& $VenvPy -m gpu.cli setup --kernel-only
if ($LASTEXITCODE -ne 0) { throw "nf4 kernel build failed (exit $LASTEXITCODE)" }

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

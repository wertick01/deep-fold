# Neighbor plate (Windows). One argument is the model; the rest go to
# python -m gpu.cli.plate. Creates a folder of metrics to send back.
#
#   powershell -File scripts\plate.ps1 3b
#   powershell -File scripts\plate.ps1 Qwen/Qwen2.5-3B-Instruct
#   powershell -File scripts\plate.ps1 32b --skip-verify

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$VenvPy = Join-Path $Repo ".venv\Scripts\python.exe"
if (Test-Path $VenvPy) {
    $Py = $VenvPy
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.11 -m gpu.cli.plate @args
    exit $LASTEXITCODE
} else {
    $Py = "python"
}

& $Py -m gpu.cli.plate @args
exit $LASTEXITCODE

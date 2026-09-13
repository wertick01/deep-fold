# Elevated ncu: unlock GPU performance counters, then profile the four pinned cases.
# Launched with -Verb RunAs. Logs to C:\dev\models\runs\ncu-admin.log
$ErrorActionPreference = "Continue"
$logDir = "C:\dev\models\runs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "ncu-admin.log"
function Log([string]$m) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
    Add-Content -Path $log -Value $line -Encoding UTF8
    Write-Host $line
}
Log "elevated ncu start, admin=$([bool]([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))"
Log "reg add RmProfilingAdminOnly=0"
& reg.exe add "HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak" /v RmProfilingAdminOnly /t REG_DWORD /d 0 /f
Log "reg exit $LASTEXITCODE"
$env:PATH = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.2.0;" + $env:PATH
Set-Location "C:\dev\deep-fold"
$py = "C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe"
Log "python -m gpu.nf4.ncu --profile"
& $py -m gpu.nf4.ncu --profile --out "C:\dev\deep-fold\docs\runs\ncu" --python $py --iters 8
Log "ncu driver exit $LASTEXITCODE"
exit $LASTEXITCODE

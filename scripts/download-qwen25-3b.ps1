# Download Qwen2.5-3B-Instruct (~6.2 GB BF16) onto this Windows machine.
# Run in PowerShell (not the cloud agent). Requires Python 3.
#
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\scripts\download-qwen25-3b.ps1

$ErrorActionPreference = "Stop"

$Dest = if ($env:MODEL_DIR) { $env:MODEL_DIR } else { "C:\dev\models\Qwen2.5-3B-Instruct" }
$Repo = "Qwen/Qwen2.5-3B-Instruct"

Write-Host "Target: $Dest"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

python -m pip install -q -U "huggingface_hub[hf_xet]"
python -c @"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$Repo', local_dir=r'$Dest')
print('ok', r'$Dest')
"@

Write-Host "Done. Point chr at: $Dest"

# Run this in the SAME PowerShell where you set $env:HF_TOKEN
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
.\.venv\Scripts\Activate.ps1

if (-not $env:HF_TOKEN -and -not $env:HUGGING_FACE_HUB_TOKEN) {
    Write-Error "HF_TOKEN is not set in this shell. Set it first, then re-run."
}

$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HUGGING_FACE_HUB_TOKEN = $(if ($env:HUGGING_FACE_HUB_TOKEN) { $env:HUGGING_FACE_HUB_TOKEN } else { $env:HF_TOKEN })

Write-Host "Authenticated as:" -ForegroundColor Cyan
python -c "from huggingface_hub import whoami; print(whoami().get('name'))"

Write-Host "Downloading SmolVLM-Instruct (resumes incomplete files)..." -ForegroundColor Cyan
python -c "from huggingface_hub import snapshot_download; p=snapshot_download('HuggingFaceTB/SmolVLM-Instruct', cache_dir='data_cache'); print(p); print('DOWNLOAD_OK')"

Write-Host "Running CPU pilot (2 steps)..." -ForegroundColor Cyan
python scripts\pilot.py --models smolvlm_2b --steps 2 --out results\pilot_report.json

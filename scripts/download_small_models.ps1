# Download tiny SmolVLMs (skip ONNX). Run in a shell where HF_TOKEN is set if possible.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
.\.venv\Scripts\Activate.ps1
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$models = @(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    "HuggingFaceTB/SmolVLM-500M-Instruct"
)

foreach ($m in $models) {
    Write-Host "Downloading $m (weights only)..." -ForegroundColor Cyan
    python -c @"
from huggingface_hub import snapshot_download
p = snapshot_download(
    '$m',
    cache_dir='data_cache',
    max_workers=2,
    ignore_patterns=['onnx/**', '*.onnx', '*.onnx_data', '*.png', '*.md'],
)
print(p)
print('OK $m')
"@
}

Write-Host "All small models ready." -ForegroundColor Green

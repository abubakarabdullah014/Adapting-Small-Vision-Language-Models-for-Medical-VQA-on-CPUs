# Resumable experiment queue for CPU-only VLM benchmark
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (Test-Path .\.venv\Scripts\Activate.ps1) {
    .\.venv\Scripts\Activate.ps1
}

$queueFile = "results\queue_done.txt"
New-Item -ItemType Directory -Force -Path results | Out-Null
if (-not (Test-Path $queueFile)) { New-Item -ItemType File -Path $queueFile | Out-Null }
$done = Get-Content $queueFile

function Invoke-Config([string]$cfg, [string]$mode) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($cfg)
    if ($done -contains $name) {
        Write-Host "[skip] $name"
        return
    }
    Write-Host "[run] $mode $name"
    if ($mode -eq "train") {
        python src\train.py --config $cfg
    } else {
        python src\evaluate.py --config $cfg
    }
    if ($LASTEXITCODE -eq 0) {
        Add-Content -Path $queueFile -Value $name
    } else {
        Write-Host "[fail] $name exit=$LASTEXITCODE"
        Add-Content -Path "results\queue_failed.txt" -Value $name
    }
}

# 1) Zero-shot + quantization evals first (no training)
Get-ChildItem configs\eval_*_zeroshot_*.yaml | ForEach-Object {
    Invoke-Config $_.FullName "eval"
}

# 2) Training runs (skip qwen25_vl_3b unless pilot marked feasible)
$pilot = "results\pilot_report.json"
$allow3b = $false
if (Test-Path $pilot) {
    $reports = Get-Content $pilot -Raw | ConvertFrom-Json
    foreach ($r in $reports) {
        if ($r.model -eq "qwen25_vl_3b" -and $r.train_feasible -eq $true) { $allow3b = $true }
    }
}

Get-ChildItem configs\train_*.yaml | ForEach-Object {
    if ($_.Name -match "qwen25_vl_3b" -and -not $allow3b) {
        Write-Host "[skip-train-3b] $($_.Name)"
        return
    }
    Invoke-Config $_.FullName "train"
}

# 3) Post-train evals
Get-ChildItem configs\eval_*_lora*.yaml, configs\eval_*_adapter*.yaml, configs\eval_*_ia3*.yaml, configs\eval_*_dora*.yaml |
    ForEach-Object { Invoke-Config $_.FullName "eval" }

Write-Host "Queue finished. Generating tables/figures..."
python analysis\make_tables.py
python analysis\make_figures.py

# Pilot: timing and memory for each VLM on this CPU laptop
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (Test-Path .\.venv\Scripts\Activate.ps1) {
    .\.venv\Scripts\Activate.ps1
}

Write-Host "Generating configs..."
python scripts\generate_configs.py

Write-Host "Running pilot..."
python scripts\pilot.py --steps 3 --out results\pilot_report.json

Write-Host "Pilot complete. See results\pilot_report.md"

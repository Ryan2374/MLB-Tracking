# Run all evaluation models and compile JSON + HTML reports.
# Safe to run in an external Windows Terminal while Cursor stays open.
#
# Usage (from this directory):
#   .\run_full_eval.ps1
#   .\run_full_eval.ps1 -ProjectDir "C:\path\to\mlb_ball_location_mvp"

param(
    [string]$ProjectDir = $PSScriptRoot
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectDir

$venvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual env not found at $venvPython. Run: python -m venv .venv; pip install -r requirements.txt"
}

$env:MLB_EVAL_PROJECT_DIR = $ProjectDir
& $venvPython scripts/run_all_models.py --project-dir $ProjectDir

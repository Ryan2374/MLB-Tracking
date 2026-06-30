# Build MLB_FullEval.exe (PyInstaller one-folder bundle).
# Requires: pip install pyinstaller
#
# Usage:
#   .\scripts\build_eval_exe.ps1

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectDir

$venvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual env not found. Create .venv and pip install -r requirements.txt first."
}

& $venvPython -m pip install pyinstaller

$distDir = Join-Path $ProjectDir "dist\MLB_FullEval"
$exePath = Join-Path $distDir "MLB_FullEval.exe"

& $venvPython -m PyInstaller --noconfirm --clean `
    --name MLB_FullEval `
    --distpath (Join-Path $ProjectDir "dist") `
    --workpath (Join-Path $ProjectDir "build\pyinstaller") `
    --specpath (Join-Path $ProjectDir "build\pyinstaller") `
    --paths $ProjectDir `
    --add-data "prediction;prediction" `
    --add-data "scripts;scripts" `
    --add-data "eval_config.json;." `
    scripts/eval_runner_entry.py

Copy-Item -Force (Join-Path $ProjectDir "eval_config.json") (Join-Path $distDir "eval_config.json")

Write-Host ""
Write-Host "Built: $exePath"
Write-Host "Edit dist\MLB_FullEval\eval_config.json and set project_dir to your inner project root."
Write-Host "Then double-click MLB_FullEval.exe or run it from PowerShell."

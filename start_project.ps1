<#
.SYNOPSIS
    Silent Dropout Detection System - Windows PowerShell launcher
.DESCRIPTION
    Installs dependencies, trains the model, and starts the backend and frontend.
.EXAMPLE
    .\start_project.ps1
#>
Set-StrictMode Pipe
$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "SILENT DROPOUT DETECTION SYSTEM (KOLKATA PILOT)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$PYEXE = "python"
try {
    $ver = & $PYEXE -3 --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Python not found" }
} catch {
    Write-Host "ERROR: Python is not available on PATH. Install Python 3.10+ and restart."
    return
}

Write-Host "[1/6] Setting up virtual environment..."
if (-not (Test-Path "venv")) {
    & $PYEXE -3 -m venv venv
}

Write-Host "[2/6] Installing backend dependencies..."
& venv\Scripts\python -m pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install backend dependencies."
    return
}

Write-Host "[3/6] Generating data + training model (CPU)..."
& venv\Scripts\python -m ml.cli download
& venv\Scripts\python -m ml.cli train --model lightgbm
Write-Host "[3/6] Data & model step complete." -ForegroundColor Green

Write-Host "[4/6] Installing frontend dependencies..."
Push-Location frontend
if (-not (Test-Path "node_modules")) {
    npm install --silent
}
Pop-Location

Write-Host "[5/6] Starting backend API (http://0.0.0.0:8000)..."
if (Test-Path "backend.log") { Remove-Item "backend.log" -Force }
Start-Process -FilePath venv\Scripts\nuvicorn -ArgumentList "--reload main:app --host 0.0.0.0 --port 8000" -NoNewWindow
Write-Host "    (backend.log captures uvicorn output)"

Write-Host "[6/6] Starting frontend dashboard (http://localhost:3000)..."
Push-Location frontend
npm start
Pop-Location

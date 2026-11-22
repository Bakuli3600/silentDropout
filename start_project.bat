@echo off
REM ============================================================================
REM  Silent Dropout Detection System - Windows launcher
REM  Requirements: Python 3.10+ and Node.js 18+ on PATH
REM ============================================================================

setlocal

echo ============================================================
echo SILENT DROPOUT DETECTION SYSTEM (KOLKATA PILOT)
echo ============================================================
echo.

set "PYEXE=py"
%PYEXE% -3 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not available on PATH. Install Python 3.10+ from
    echo https://www.python.org/downloads/ and restart this script.
    goto :eof
)

echo [1/6] Setting up virtual environment...
if not exist "venv" (
    %PYEXE% -3 -m venv venv
)

echo [2/6] Installing backend dependencies...
call venv\Scripts\pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install backend dependencies.
    goto :eof
)

echo [3/6] Generating data + training model (CPU)...
call venv\Scripts\python -m ml.cli download
call venv\Scripts\python -m ml.cli train --model lightgbm
echo [3/6] Data & model step complete.

echo [4/6] Installing frontend dependencies...
cd frontend
if not exist "node_modules" (
    call npm install --silent
)
cd ..

echo [5/6] Starting backend API (http://0.0.0.0:8000)...
if exist "backend.log" del /f /q backend.log
start "Backend" /D. venv\Scripts\nuvicorn --reload main:app --host 0.0.0.0 --port 8000
echo    (backend.log captures uvicorn output)

echo [6/6] Starting frontend dashboard (http://localhost:3000)...
cd frontend
call npm start

endlocal

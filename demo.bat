@echo off
title Guardrail Arena Demo
setlocal

:: Use the directory containing this script as root
set REPO_DIR=%~dp0
cd /d "%REPO_DIR%"

echo.
echo  ============================================
echo   Guardrail Arena -- Starting Demo...
echo  ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

:: Kill any existing server on port 7860
echo [1/4] Checking for existing server on port 7860...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":7860 "') do (
    taskkill /PID %%a /F >nul 2>&1
)

:: Start backend server in new window
echo [2/4] Starting backend server...
start "Guardrail Arena - Server" cmd /k "cd /d %REPO_DIR% && python -m uvicorn app.main:app --host 0.0.0.0 --port 7860"

:: Wait for server to boot
echo [3/4] Waiting for server to start (10s)...
timeout /t 10 /nobreak > nul

:: Open demo runner in browser
echo [4/4] Opening demo runner...
start http://localhost:7860/demo_runner

echo.
echo  ============================================
echo   GUARDRAIL ARENA DEMO READY
echo  ============================================
echo.
echo   Demo Runner:  http://localhost:7860/demo_runner
echo   Local API:    http://localhost:7860
echo   Live Space:   https://varunventra-guardrail-arena.hf.space
echo   GitHub:       https://github.com/sahithsundarw/sentinel
echo.
echo   Demo runner ready -- press SPACE to start the episode
echo   Close the "Guardrail Arena - Server" window to stop.
echo  ============================================
echo.
pause

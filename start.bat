@echo off
title Sentinel — Guardrail Arena
cd /d "%~dp0"

echo Starting Sentinel backend...
start "Sentinel Backend" cmd /k "venv311\Scripts\activate && uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload"

timeout /t 3 /nobreak >nul

echo Opening landing page...
start "" "%~dp0sentinel_landing.html"

echo.
echo Backend running at http://localhost:7860
echo Press any key to close this window (backend keeps running).
pause >nul

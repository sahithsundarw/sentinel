@echo off
title Guardrail Arena Demo
echo.
echo  ============================================
echo   Guardrail Arena - Starting Demo...
echo  ============================================
echo.

:: Start backend server in new window
echo [1/4] Starting backend server...
start "Guardrail Arena - Server" cmd /k "cd /d C:\Users\sahit\OneDrive\Desktop\sentinel && python -m uvicorn app.main:app --host 0.0.0.0 --port 7860"

:: Wait for server to boot
echo [2/4] Waiting for server to start...
timeout /t 5 /nobreak > nul

:: Start frontend dashboard in new window
echo [3/4] Starting dashboard...
start "Guardrail Arena - Dashboard" cmd /k "cd /d C:\Users\sahit\guardrail-dashboard && npm run dev"

:: Populate demo data
echo [4/4] Populating demo data...
python demo_populate.py

:: Open browser
start http://localhost:5173

echo.
echo  ============================================
echo   Demo is live at http://localhost:5173
echo  ============================================
echo.
pause

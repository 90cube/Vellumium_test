@echo off
title Local GPU Worker
echo ====================================================
echo   Local GPU Worker - Queue Processor
echo ====================================================
echo.

if exist .env (
    echo [INFO] Loading .env configuration...
) else (
    echo [WARN] .env file not found, using defaults
)

echo.
echo [INFO] Starting worker...
echo [INFO] Press Ctrl+C to stop
echo.

venv\Scripts\python.exe worker.py

echo.
echo Worker stopped.
pause

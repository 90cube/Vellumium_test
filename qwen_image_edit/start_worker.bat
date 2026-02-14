@echo off
title Qwen Image Edit - Queue Worker
echo ====================================================
echo   Qwen Image Edit - Queue Worker
echo ====================================================
echo.

echo [Starting] worker.py...
..\venv\Scripts\python.exe worker.py

echo.
echo Worker stopped.
pause

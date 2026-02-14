@echo off
title Qwen Image Edit - API Server
echo ====================================================
echo   Qwen Image Edit - API Server
echo   Access via: http://localhost:8200
echo   Docs:       http://localhost:8200/docs
echo ====================================================
echo.

echo [Starting] server.py...
..\venv\Scripts\python.exe server.py

echo.
echo Server stopped.
pause

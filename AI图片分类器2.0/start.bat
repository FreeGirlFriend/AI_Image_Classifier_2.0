@echo off
chcp 65001 >nul 2>&1
title AI Image Classifier

:: Kill any existing process on port 7860
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":7860.*LISTEN"') do taskkill /F /PID %%a >nul 2>&1

"%~dp0venv\Scripts\python.exe" "%~dp0start.py"
if errorlevel 1 (
    echo.
    echo Program exited with error
    pause
)

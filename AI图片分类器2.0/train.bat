@echo off
chcp 65001 >nul 2>&1
title Custom Classifier Trainer
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":7861.*LISTEN"') do taskkill /F /PID %%a >nul 2>&1
"%~dp0venv\Scripts\python.exe" "%~dp0custom_classifier_trainer.py" --ui
if errorlevel 1 (
    echo.
    echo Program exited with error
    pause
)

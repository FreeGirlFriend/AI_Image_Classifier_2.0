@echo off
chcp 65001 >nul 2>&1
title AI Image Classifier - Setup
echo.
echo   AI Image Classifier - Auto Setup
echo   First run may take a few minutes...
echo.
python "%~dp0setup.py"
if errorlevel 1 (
    echo.
    echo Python not found. Please install Python 3.12:
    echo https://www.python.org/downloads/
    echo.
    pause
)

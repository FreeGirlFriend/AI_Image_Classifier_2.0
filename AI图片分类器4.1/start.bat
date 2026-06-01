@echo off
chcp 65001 >nul 2>&1
title AI Image Classifier
setlocal enabledelayedexpansion

:: ============================================================
::  Kill any existing process on port 7860
:: ============================================================
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":7860.*LISTEN"') do taskkill /F /PID %%a >nul 2>&1

:: ============================================================
::  Step 1: Check if venv already exists
:: ============================================================
if exist "%~dp0venv\Scripts\python.exe" goto :run
if exist "%~dp0venv\bin\python.exe" goto :run

:: ============================================================
::  Step 2: No venv - auto-detect Python 3.12 and run setup
:: ============================================================
echo.
echo [Auto Setup] First run detected, setting up environment...
echo.

set "PYTHON_EXE="
set "SETUP_OK=0"

:: Method 1: py launcher (most reliable on Windows)
py -3.12 --version >nul 2>&1
if !errorlevel! equ 0 (
    echo Found Python 3.12 via py launcher
    py -3.12 "%~dp0setup.py" --auto
    if !errorlevel! equ 0 set SETUP_OK=1
)

:: Method 2: python on PATH (check if it is 3.12)
if !SETUP_OK! equ 0 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do (
        echo %%v | findstr /b "3.12" >nul
        if !errorlevel! equ 0 (
            echo Found Python 3.12 on PATH
            python "%~dp0setup.py" --auto
            if !errorlevel! equ 0 set SETUP_OK=1
        )
    )
)

:: Method 3: Common install paths
if !SETUP_OK! equ 0 (
    for %%p in (
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%PROGRAMFILES%\Python312\python.exe"
        "%PROGRAMFILES(x86)%\Python312\python.exe"
        "C:\Python312\python.exe"
        "D:\Python312\python.exe"
    ) do (
        if exist %%p (
            if !SETUP_OK! equ 0 (
                echo Found Python at %%p
                %%p "%~dp0setup.py" --auto
                if !errorlevel! equ 0 set SETUP_OK=1
            )
        )
    )
)

:: Check if setup succeeded
if exist "%~dp0venv\Scripts\python.exe" goto :run
if exist "%~dp0venv\bin\python.exe" goto :run

:: ============================================================
::  Setup failed - show error
:: ============================================================
echo.
echo ==============================================
echo   Setup failed. Please install Python 3.12:
echo   https://www.python.org/downloads/
echo.
echo   After installing, double-click start.bat again.
echo ==============================================
echo.
pause
exit /b 1

:: ============================================================
::  Run the application
:: ============================================================
:run
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" "%~dp0start.py"
) else (
    "%~dp0venv\bin\python.exe" "%~dp0start.py"
)

if errorlevel 1 (
    echo.
    echo Program exited with error
    pause
)
endlocal

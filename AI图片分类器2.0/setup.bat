@echo off
chcp 65001 >nul 2>&1
title AI 图片分类器 - 自动安装
echo.
echo   双击此文件自动安装所有环境
echo   首次运行需要几分钟，请耐心等待
echo.
python "%~dp0setup.py"
if errorlevel 1 (
    echo.
    echo 如果 python 命令不可用，请先安装 Python:
    echo https://www.python.org/downloads/
    echo.
    pause
)

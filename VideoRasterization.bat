@echo off
title VideoRasterization AI Video Colorization
cd /d "%~dp0"
set "PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if exist "%PY311%" (
    "%PY311%" gui\app.py
) else (
    py -3.11 gui\app.py
)
if errorlevel 1 (
    echo.
    echo VideoRasterization failed to start. Check that dependencies are installed.
    pause
)

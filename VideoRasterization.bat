@echo off
title VideoRasterization AI Video Colorization
cd /d "%~dp0"
py -3.11 gui\app.py
if errorlevel 1 (
    echo.
    echo VideoRasterization failed to start. Check that dependencies are installed.
    pause
)

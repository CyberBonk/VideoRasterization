@echo off
title ChromaStudio — AI Video Colorization
cd /d "%~dp0"
py -3.11 gui\app.py
if errorlevel 1 (
    echo.
    echo ChromaStudio failed to start. Check that dependencies are installed.
    pause
)

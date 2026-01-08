@echo off
REM IronSight Command Center - Windows Launcher
REM Simple wrapper for the Python launcher script

setlocal enabledelayedexpansion

REM Change to script directory
cd /d "%~dp0"

REM Check if uv is available first
uv --version >nul 2>&1
if errorlevel 0 (
    echo Found uv package manager - using for better performance
    uv run python run_ironsight.py %*
    goto :end
)

REM Fallback to regular Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Neither uv nor Python is installed or not in PATH
    echo Please install uv (recommended) or Python 3.10-3.12 and try again
    echo.
    echo To install uv: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

REM Run with regular Python
echo Using regular Python (consider installing uv for better performance)
python run_ironsight.py %*

:end
REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit.
    pause >nul
)
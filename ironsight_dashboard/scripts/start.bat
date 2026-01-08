@echo off
REM IronSight Command Center - Legacy Startup Script (Windows)
REM DEPRECATED: Use run_ironsight.bat in the parent directory instead
REM Usage: scripts\start.bat [--dev] [--cpu]

echo.
echo ========================================
echo   DEPRECATED SCRIPT
echo ========================================
echo This script is deprecated. Please use:
echo   run_ironsight.bat
echo.
echo Redirecting to new launcher...
echo.

REM Change to parent directory and run new launcher
cd /d "%~dp0.."
call run_ironsight.bat %*
exit /b %errorlevel%

REM Legacy code below (kept for reference)
REM =======================================

setlocal enabledelayedexpansion

REM Default settings
set DEV_MODE=false
set CPU_ONLY=false
set PORT=8501

REM Parse arguments
:parse_args
if "%~1"=="" goto :done_parsing
if "%~1"=="--dev" (
    set DEV_MODE=true
    shift
    goto :parse_args
)
if "%~1"=="--cpu" (
    set CPU_ONLY=true
    shift
    goto :parse_args
)
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-h" goto :show_help
if "%~1"=="--help" goto :show_help
echo Unknown option: %~1
exit /b 1

:show_help
echo IronSight Command Center Startup Script
echo.
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   --dev     Run in development mode with auto-reload
echo   --cpu     Run without GPU (CPU-only mode)
echo   --port N  Use port N (default: 8501)
echo   -h        Show this help message
exit /b 0

:done_parsing

echo ========================================
echo   IronSight Command Center
echo ========================================
echo.

REM Change to project directory
cd /d "%~dp0.."

REM Check Python version
echo Checking Python version...
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo Python version: %PYTHON_VERSION%

REM Check CUDA availability
if "%CPU_ONLY%"=="false" (
    echo Checking CUDA availability...
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
    if !errorlevel! equ 0 (
        for /f "delims=" %%g in ('python -c "import torch; print(torch.cuda.get_device_name(0))" 2^>nul') do set GPU_NAME=%%g
        echo CUDA available: !GPU_NAME!
    ) else (
        echo Warning: CUDA not available, running in CPU mode
        set CPU_ONLY=true
    )
)

REM Set environment variables
if "%CPU_ONLY%"=="true" (
    set CUDA_VISIBLE_DEVICES=
    echo Running in CPU-only mode
)

REM Load .env file if exists
if exist ".env" (
    echo Loading environment from .env...
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
)

REM Check for required model files
echo Checking model files...
if not defined NAFNET_MODEL_PATH set NAFNET_MODEL_PATH=..\NAFNet-GoPro-width64.pth
if exist "%NAFNET_MODEL_PATH%" (
    echo NAFNet model found
) else (
    echo Warning: NAFNet model not found at %NAFNET_MODEL_PATH%
)

REM Create logs directory
if not exist "logs" mkdir logs

REM Build Streamlit command
set STREAMLIT_CMD=streamlit run src/app.py
set STREAMLIT_CMD=%STREAMLIT_CMD% --server.port=%PORT%
set STREAMLIT_CMD=%STREAMLIT_CMD% --server.address=0.0.0.0

if "%DEV_MODE%"=="true" (
    echo Running in development mode with auto-reload
    set STREAMLIT_CMD=%STREAMLIT_CMD% --server.runOnSave=true
) else (
    set STREAMLIT_CMD=%STREAMLIT_CMD% --server.headless=true
)

echo.
echo Starting IronSight Command Center on port %PORT%...
echo Dashboard URL: http://localhost:%PORT%
echo.

REM Run Streamlit
%STREAMLIT_CMD%

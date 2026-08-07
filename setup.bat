@echo off
rem Bootstraps the environment, then launches the setup program, which writes
rem ~/.buzz/config.toml.
rem
rem Reuses .venv if it already exists and is Python 3.12 or later; otherwise finds a
rem system Python that is, creates .venv with it, and installs requirements.txt.
setlocal enabledelayedexpansion

set "DIR=%~dp0"
set "VENV=%DIR%.venv"
set "PYTHON="

if exist "%VENV%\Scripts\python.exe" (
    "%VENV%\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "PYTHON=%VENV%\Scripts\python.exe"
    ) else (
        echo Existing .venv is older than Python 3.12; recreating it.
    )
)

if not defined PYTHON (
    set "SYSTEM_PYTHON="
    where py >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
        if !ERRORLEVEL! EQU 0 set "SYSTEM_PYTHON=py -3"
    )
    if not defined SYSTEM_PYTHON (
        for %%C in (python py) do (
            if not defined SYSTEM_PYTHON (
                where %%C >nul 2>&1
                if !ERRORLEVEL! EQU 0 (
                    %%C -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
                    if !ERRORLEVEL! EQU 0 set "SYSTEM_PYTHON=%%C"
                )
            )
        )
    )
    if not defined SYSTEM_PYTHON (
        echo No Python 3.12 or later found on PATH. Install Python 3.12+ from python.org and try again.
        exit /b 1
    )
    echo Creating virtual environment at %VENV% using !SYSTEM_PYTHON!...
    !SYSTEM_PYTHON! -m venv "%VENV%"
    set "PYTHON=%VENV%\Scripts\python.exe"
)

echo Installing requirements...
"%PYTHON%" -m pip install -r "%DIR%requirements.txt"

set "PYTHONPATH=%DIR%lib"
"%PYTHON%" -m buzz.setup

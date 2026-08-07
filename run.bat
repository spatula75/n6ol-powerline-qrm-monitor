@echo off
rem Runs the monitor with no special arguments, using the .venv setup.bat created.
setlocal

set "DIR=%~dp0"
set "VENV=%DIR%.venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "CONFIG_DIR=%USERPROFILE%\.buzz"

if not exist "%PYTHON%" (
    echo No virtual environment found at %VENV%. Run setup.bat first.
    exit /b 1
)

if not exist "%CONFIG_DIR%" (
    echo No configuration found at %CONFIG_DIR%. Run setup.bat first.
    exit /b 1
)

set "PYTHONPATH=%DIR%lib"
"%PYTHON%" -m buzz.main

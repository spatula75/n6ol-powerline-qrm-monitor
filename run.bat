@echo off
rem Runs the monitor with no special arguments, using the .venv setup.bat created.
setlocal

set "DIR=%~dp0"
set "VENV=%DIR%.venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "CONFIG_DIR=%USERPROFILE%\.buzz"
set "CONFIG_FILE=%CONFIG_DIR%\config.toml"

if not exist "%PYTHON%" (
    echo No virtual environment found at %VENV%. Run setup.bat first.
    exit /b 1
)

rem The file, not just the directory: FinishScreen.save() creates %CONFIG_DIR%
rem right before writing config.toml into it, so a setup run that crashed or was
rem killed between those two steps leaves the directory behind with no config
rem inside it.  Starting on defaults nobody chose in that state is exactly what
rem this check exists to refuse.
if not exist "%CONFIG_FILE%" (
    echo No configuration found at %CONFIG_FILE%. Run setup.bat first.
    exit /b 1
)

set "PYTHONPATH=%DIR%lib"
"%PYTHON%" -m buzz.main

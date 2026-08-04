@echo off
rem Launches the setup program, which writes ~/.buzz/config.toml.
setlocal
set PYTHONPATH=%~dp0lib
python -m buzz.setup
endlocal

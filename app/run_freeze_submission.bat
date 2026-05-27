@echo off
if "%PYTHON_BIN%"=="" set "PYTHON_BIN=python"
"%PYTHON_BIN%" "%~dp0code\src\cli.py" --app-root "%~dp0." freeze %*

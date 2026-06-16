@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

if /I "%~1"=="--help" goto :help
if /I "%~1"=="/?" goto :help

if /I "%~1"=="--sana" (
  set "VENV=.venv-sana"
  set "BACKEND=SANA-Streaming"
) else (
  set "VENV=.venv"
  set "BACKEND=FLUX"
)

if not exist "%VENV%\Scripts\python.exe" (
  echo ERROR: %VENV% was not found.
  if /I "%~1"=="--sana" ( echo Run install.bat --sana first. ) else ( echo Run install.bat first. )
  exit /b 1
)

set "PYTHONPATH=%ROOT%src"

echo.
echo Starting StreamForge Operator Console (%BACKEND% env: %VENV%)...
echo Open http://127.0.0.1:8765 in a browser.
if /I "%~1"=="--sana" echo In the console, set Backend = SANA-Streaming (temporal).
echo Press Ctrl+C in this window to stop the server.
echo.

"%VENV%\Scripts\python.exe" scripts\web.py
exit /b %ERRORLEVEL%

:help
echo Usage:
echo   startup.bat          Start the console under .venv (FLUX backend).
echo   startup.bat --sana   Start the console under .venv-sana (enables the SANA-Streaming backend).
echo.
echo Starts the local StreamForge Operator Console at:
echo   http://127.0.0.1:8765
echo.
echo Run install.bat (or install.bat --sana) first if the venv does not exist.
exit /b 0

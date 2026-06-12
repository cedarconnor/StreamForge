@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

if /I "%~1"=="--help" goto :help
if /I "%~1"=="/?" goto :help

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv was not found.
  echo Run install.bat first.
  exit /b 1
)

set "PYTHONPATH=%ROOT%src"

echo.
echo Starting StreamForge Operator Console...
echo Open http://127.0.0.1:8765 in a browser.
echo Press Ctrl+C in this window to stop the server.
echo.

".venv\Scripts\python.exe" scripts\web.py
exit /b %ERRORLEVEL%

:help
echo Usage:
echo   startup.bat
echo.
echo Starts the local StreamForge Operator Console at:
echo   http://127.0.0.1:8765
echo.
echo Run install.bat first if .venv does not exist.
exit /b 0

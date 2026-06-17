@echo off
setlocal
rem ============================================================================
rem  StreamForge launcher - SANA-Streaming backend (.venv-sana)
rem  Kills any StreamForge server still listening on port 8765, then starts a
rem  fresh one under the SANA venv (its transformers==4.57.3 pin needs the
rem  separate .venv-sana). Use this for the temporal video-to-video backend.
rem ============================================================================

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "VENV=.venv-sana"

if not exist "%VENV%\Scripts\python.exe" (
  echo ERROR: %VENV% was not found. Run install.bat --sana first.
  exit /b 1
)

echo Checking for a stale StreamForge server on port 8765...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8765" ^| findstr LISTENING') do (
  echo   stopping stale server PID %%a
  taskkill /F /PID %%a >nul 2>&1
)

set "PYTHONPATH=%ROOT%src"

echo.
echo Starting StreamForge Operator Console (SANA env: .venv-sana)...
echo Open http://127.0.0.1:8765 in a browser.
echo In the console, set Backend = SANA-Streaming (temporal).
echo Press Ctrl+C in this window to stop the server.
echo.

"%VENV%\Scripts\python.exe" scripts\web.py
exit /b %ERRORLEVEL%

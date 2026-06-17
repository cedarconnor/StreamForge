@echo off
setlocal
rem ============================================================================
rem  StreamForge launcher - FLUX.2-klein backend (.venv)
rem  Kills any StreamForge server still listening on port 8765, then starts a
rem  fresh one. Use this for the image-to-image FLUX backend.
rem ============================================================================

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "VENV=.venv"

if not exist "%VENV%\Scripts\python.exe" (
  echo ERROR: %VENV% was not found. Run install.bat first.
  exit /b 1
)

echo Checking for a stale StreamForge server on port 8765...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8765" ^| findstr LISTENING') do (
  echo   stopping stale server PID %%a
  taskkill /F /PID %%a >nul 2>&1
)

set "PYTHONPATH=%ROOT%src"

echo.
echo Starting StreamForge Operator Console (FLUX env: .venv)...
echo Open http://127.0.0.1:8765 in a browser.
echo In the console, set Backend = FLUX.2-klein (img2img).
echo Press Ctrl+C in this window to stop the server.
echo.

"%VENV%\Scripts\python.exe" scripts\web.py
exit /b %ERRORLEVEL%

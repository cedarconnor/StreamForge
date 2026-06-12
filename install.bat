@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

if /I "%~1"=="--help" goto :help
if /I "%~1"=="/?" goto :help

echo.
echo StreamForge install
echo ====================

where py >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python launcher "py" was not found. Install Python 3.11 for Windows first.
  exit /b 1
)

py -3.11 --version >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python 3.11 was not found by the Python launcher.
  echo Install Python 3.11, then rerun install.bat.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating .venv with Python 3.11...
  py -3.11 -m venv .venv
  if errorlevel 1 exit /b 1
) else (
  echo Reusing existing .venv.
)

echo Upgrading pip tooling...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1

echo Installing CUDA PyTorch wheels...
".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 exit /b 1

echo Installing StreamForge requirements...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo Installing StreamForge package in editable mode...
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 exit /b 1

if /I "%~1"=="--models" (
  echo Downloading model weights...
  ".venv\Scripts\python.exe" scripts\download_models.py
  if errorlevel 1 exit /b 1
) else (
  echo Skipping model download. Run install.bat --models to fetch weights.
)

echo.
echo Install complete.
echo Run startup.bat to launch the Operator Console.
exit /b 0

:help
echo Usage:
echo   install.bat          Create/update .venv and install dependencies.
echo   install.bat --models Create/update .venv, install dependencies, and download model weights.
echo.
echo Requirements:
echo   - Windows with Python 3.11 available through the "py" launcher.
echo   - Network access for Python packages.
echo   - CUDA 12.8 compatible NVIDIA driver for the default torch install.
exit /b 0

@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

if /I "%~1"=="--help" goto :help
if /I "%~1"=="/?" goto :help
if /I "%~1"=="--sana" goto :sana

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

:sana
echo.
echo StreamForge SANA-Streaming install
echo ==================================
echo This builds a SEPARATE .venv-sana (transformers 4.57.3 would clash with the FLUX .venv)
echo and clones NVlabs/Sana into external\Sana at the pinned revision. ~10 GB of weights
echo download to the HF cache on first run.
echo.

where git >nul 2>nul
if errorlevel 1 ( echo ERROR: git was not found. Install git first. & exit /b 1 )
where py >nul 2>nul
if errorlevel 1 ( echo ERROR: Python launcher "py" was not found. & exit /b 1 )

if not exist "external\Sana\.git" (
  echo Cloning NVlabs/Sana into external\Sana ...
  git clone https://github.com/NVlabs/Sana external\Sana
  if errorlevel 1 exit /b 1
)
echo Pinning external\Sana to the verified revision ...
git -C external\Sana checkout 51baa3c99acd1632ca8c4b2e45e893e8be08a819
if errorlevel 1 exit /b 1

if not exist ".venv-sana\Scripts\python.exe" (
  echo Creating .venv-sana with Python 3.11 ...
  py -3.11 -m venv .venv-sana
  if errorlevel 1 exit /b 1
) else (
  echo Reusing existing .venv-sana.
)
set "SPY=.venv-sana\Scripts\python.exe"

echo Upgrading pip tooling (setuptools<80 for the pure-python mmcv build) ...
"%SPY%" -m pip install --upgrade pip wheel "setuptools<80"
if errorlevel 1 exit /b 1

echo Installing CUDA PyTorch + matched Triton (proven stack) ...
"%SPY%" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 exit /b 1
"%SPY%" -m pip install "triton-windows==3.6.0.post26"
if errorlevel 1 exit /b 1

echo Installing SANA inference + console dependencies ...
"%SPY%" -m pip install "transformers==4.57.3" diffusers accelerate safetensors huggingface_hub einops omegaconf pyrallis sentencepiece protobuf "imageio[ffmpeg]" opencv-python numpy pillow pyyaml termcolor timm pytz qwen-vl-utils ftfy fastapi "uvicorn[standard]"
if errorlevel 1 exit /b 1

echo Installing flash-linear-attention (split package; --no-deps to keep triton-windows) ...
"%SPY%" -m pip install flash-linear-attention --no-deps
if errorlevel 1 exit /b 1
"%SPY%" -m pip install fla-core --no-deps
if errorlevel 1 exit /b 1

echo Installing mmcv 1.7.2 (pure-python, no CUDA ops) ...
"%SPY%" -m pip install --no-build-isolation "mmcv==1.7.2"
if errorlevel 1 exit /b 1

echo Installing StreamForge package into .venv-sana (editable, --no-deps) ...
"%SPY%" -m pip install -e . --no-deps
if errorlevel 1 exit /b 1

echo.
echo SANA install complete.
echo Run startup.bat --sana to launch the console under .venv-sana,
echo then set Backend = SANA-Streaming in the Operator Console.
exit /b 0

:help
echo Usage:
echo   install.bat          Create/update .venv (FLUX backend) and install dependencies.
echo   install.bat --models Create/update .venv, install dependencies, and download FLUX weights.
echo   install.bat --sana   Build .venv-sana + clone NVlabs/Sana for the SANA-Streaming backend.
echo.
echo Requirements:
echo   - Windows with Python 3.11 available through the "py" launcher.
echo   - Network access for Python packages.
echo   - CUDA 12.8 compatible NVIDIA driver for the default torch install.
exit /b 0

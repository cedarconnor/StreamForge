# StreamForge — Real-Time FLUX Live-Restyle Engine

Self-hosted, commercially-clean real-time live-restyle engine: live video in → FLUX.2-klein-4B (Apache-2.0) img2img restyle → media server (Resolume) via Spout/NDI, on a single RTX A6000. A **sacred output clock** runs at show framerate while the AI cadence adapts underneath it.

- Design: [`streamforge_design_v1.2.md`](streamforge_design_v1.2.md)

## Quick Start on Windows

From a Command Prompt or PowerShell window in the repo root:

```bat
install.bat
startup.bat
```

Then open `http://127.0.0.1:8765`.

`install.bat` creates `.venv`, installs CUDA PyTorch for CUDA 12.8, installs StreamForge dependencies, and installs the package in editable mode. Use `install.bat --models` when you also want to download the pinned model weights into `models/`.

`startup.bat` starts the local Operator Console. The console validates webcam, NDI, Spout, file, and synthetic inputs before starting the live pipeline, and shows real-time input/output previews.

## Development

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install -r requirements.txt
pytest -m "not gpu and not model"   # pure-logic suite, no GPU/weights needed
```

## Operator Console

Run the local web console directly:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\web.py
```

Open `http://127.0.0.1:8765`. The console validates webcam, NDI, Spout, file, and synthetic inputs before starting the live pipeline, then shows live input/output previews while the runner is active.

StreamForge uses fit-fill-and-crop aspect handling instead of stretching frames. Use `Auto preserve` for source-ratio-safe internal dimensions, or choose an explicit canvas such as `16:9` or `1:1` when cropped output is intentional.

Model weights live in `models/` (git-ignored). Run `.\.venv\Scripts\python.exe scripts\download_models.py` or `install.bat --models` to fetch the pinned manifest and freeze exact revisions into `manifest.yaml`.

NDI and Spout depend on local runtime support. NDI send/receive uses the Python NDI bindings; Spout is same-machine GPU texture sharing and requires a compatible Windows/graphics environment.

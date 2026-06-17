# StreamForge — Real-Time FLUX Live-Restyle Engine

![StreamForge Operator Console](assets/operator-console.png)

Self-hosted, commercially-clean real-time live-restyle engine: live video in → FLUX.2-klein-4B (Apache-2.0) img2img restyle → media server (Resolume) via Spout/NDI, on a single RTX A6000. A **sacred output clock** runs at show framerate while the AI cadence adapts underneath it.

Two runtime backends share the same operator console, output clock, and sink pipeline: **FLUX** (one-frame-in/one-frame-out img2img) and **SANA-Streaming** (a chunk-causal temporal video-to-video model with native temporal memory — see [SANA_STREAMING.md](SANA_STREAMING.md)).

## Quick Start on Windows

From a Command Prompt or PowerShell window in the repo root:

```bat
install.bat
start-flux.bat
```

Then open `http://127.0.0.1:8765`.

`install.bat` creates `.venv`, installs CUDA PyTorch for CUDA 12.8, installs StreamForge dependencies (including NDI/Spout input support), and installs the package in editable mode. Use `install.bat --models` when you also want to download the pinned model weights into `models/`.

### Launchers

| Script | Backend | Environment |
|---|---|---|
| `start-flux.bat` | FLUX (img2img) | `.venv` |
| `start-sana.bat` | SANA-Streaming (temporal) | `.venv-sana` |

Each launcher **kills any stale console still listening on port 8765**, then starts a fresh one under the correct environment and opens at `http://127.0.0.1:8765`. (`startup.bat [--sana]` is the older single launcher without the pre-kill.) The console validates webcam, NDI, Spout, file, and synthetic inputs before starting the live pipeline, and shows real-time input/output previews.

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

Open `http://127.0.0.1:8765`. The console shows live input/output previews while the runner is active.

**Workflow:** pick an **Input Type** + Source → choose a **Backend** → write a **Prompt** → **Validate** (opens the source once and reports the input/output sizes) → **Start** (the first start loads the model, then the Output preview goes live) → tune the **Live Control** sliders while running. A built-in **Quickstart** panel walks through this, every control has a hover **tooltip**, and each Run-Health metric is annotated.

**Inputs** — all four live inputs are verified end-to-end under both backends via the console:

| Input Type | Source field | Notes |
|---|---|---|
| Webcam | device index (`0`, `1`, …) | USB / built-in camera |
| File | a video path | use **Browse…** (native file dialog) or the **recent** dropdown; a chosen clip **loops forever** |
| NDI | sender name (blank = first found) | a video stream on the local network |
| Spout | sender name | a GPU texture shared by another Windows app |

StreamForge uses fit-fill-and-crop aspect handling instead of stretching frames. Use `Auto preserve` for source-ratio-safe internal dimensions, or choose an explicit canvas such as `16:9` or `1:1` when cropped output is intentional.

![StreamForge aspect crop validation](assets/aspect-crop-validation.png)

Model weights live in `models/` (git-ignored). Run `.\.venv\Scripts\python.exe scripts\download_models.py` or `install.bat --models` to fetch the pinned manifest and freeze exact revisions into `manifest.yaml`.

NDI and Spout depend on local runtime support. NDI send/receive uses the Python NDI bindings (`ndi-python`); Spout is same-machine GPU texture sharing (`SpoutGL`) and requires a compatible Windows/graphics environment — both are installed by `install.bat`. No NDI/Spout source handy? `scripts/ndi_test_sender.py` and `scripts/spout_test_sender.py` broadcast a moving test pattern you can point the console at (the Spout sender also needs `glfw`, included in `install.bat`):

```bat
.venv\Scripts\python scripts\ndi_test_sender.py     :: broadcasts NDI "StreamForge-Test"
.venv\Scripts\python scripts\spout_test_sender.py   :: shares Spout "StreamForge"
```

## SANA-Streaming backend (temporal)

StreamForge also runs **SANA-Streaming** (Apache-2.0), a chunk-causal video-to-video model with native temporal memory, as a second runtime backend. It installs into a separate `.venv-sana` (its dependency pins differ from the FLUX stack):

```bat
install.bat --sana
startup.bat --sana
```

`install.bat --sana` clones `NVlabs/Sana` into `external/Sana` at the pinned revision, builds `.venv-sana` (torch 2.11 + cu128 + `triton-windows`, SANA inference deps, `flash-linear-attention`, pure-python `mmcv`, NDI/Spout input support), and installs StreamForge into it. ~10 GB of weights (DiT + Gemma-2-2B + LTX-2 VAE) download to the Hugging Face cache on first run. `start-sana.bat` (or `startup.bat --sana`) launches the console under `.venv-sana`.

In the console, set **Backend → SANA-Streaming (temporal)**, pick `SANA_FAST` or `SANA_BALANCED`, choose a ≤512² aspect, then Validate/Start. The **SANA Live Control** panel tunes steps / flow shift / motion / seed live; cached-blocks / sink-token / prompt "resync" the recurrent state. All four live inputs (webcam, file, NDI, Spout) are verified under SANA as well as FLUX.

![StreamForge SANA-Streaming backend](assets/sana-console.png)

Verified on the RTX A6000 (BF16 + fused GDN Triton kernels, native Windows): ~22 fps @384×640 and ~21 fps @512² at step-4, ~27 fps at step-2 — real-time-capable at ≤512². Full build notes, performance, architecture, and caveats are in [SANA_STREAMING.md](SANA_STREAMING.md).

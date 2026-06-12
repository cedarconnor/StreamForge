# StreamForge — Real-Time FLUX Live-Restyle Engine

Self-hosted, commercially-clean real-time live-restyle engine: live video in → FLUX.2-klein-4B (Apache-2.0) img2img restyle → media server (Resolume) via Spout/NDI, on a single RTX A6000. A **sacred output clock** runs at show framerate while the AI cadence adapts underneath it.

- Design: [`streamforge_design_v1.2.md`](streamforge_design_v1.2.md)
- Build plan: [`docs/superpowers/plans/2026-06-04-streamforge-realtime-restyle-engine.md`](docs/superpowers/plans/2026-06-04-streamforge-realtime-restyle-engine.md)

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

Run the local web console:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\web.py
```

Open `http://127.0.0.1:8765`. The console validates webcam, NDI, Spout, file, and synthetic inputs before starting the live pipeline, then shows live input/output previews while the runner is active.

StreamForge uses fit-fill-and-crop aspect handling instead of stretching frames. Use `Auto preserve` for source-ratio-safe internal dimensions, or choose an explicit canvas such as `16:9` or `1:1` when cropped output is intentional.

Model weights live in `models/` (git-ignored). Run `python scripts/download_models.py` to fetch the pinned manifest and freeze exact revisions into `manifest.yaml`.

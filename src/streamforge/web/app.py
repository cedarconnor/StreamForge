from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from streamforge.runner import RunnerConfig, StreamForgeRunner


class RunnerConfigIn(BaseModel):
    source_type: str = "webcam"
    source_name: str = "0"
    sink: str = "null"
    fps: int = 30
    seconds: float = 0.0
    prompt: str = "vivid oil painting, thick impasto brushstrokes"
    mode: str = "img2img"
    preset: str = "BALANCED"
    in_res: str = "auto"
    color: str = "full"
    fill: str = "off"
    max_extrap_ms: float = 120.0
    flow_max_side: int = 384
    compile_transformer: bool = False
    tiny_vae: bool = False
    spout_flip: bool = True
    backend: str = "flux"
    cached_blocks: int = 2
    sink_token: bool = True

    def to_config(self) -> RunnerConfig:
        allowed = {f.name for f in fields(RunnerConfig)}
        data: dict[str, Any] = self.model_dump()
        return RunnerConfig(**{k: v for k, v in data.items() if k in allowed})


class ControlIn(BaseModel):
    ref_strength: float | None = None
    text_magnitude: float | None = None
    steps: int | None = None
    seed: int | None = None
    prompt: str | None = None
    mode: str | None = None
    # SANA live knobs (ignored by the FLUX path)
    flow_shift: float | None = None
    motion_score: int | None = None
    num_cached_blocks: int | None = None
    sink_token: bool | None = None


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}


def _list_video_files() -> list[dict[str, str]]:
    """Video files the File source can loop. Scans TestFile/ (recursively) and the
    working dir (shallow), both relative to wherever the server was launched."""
    base = Path.cwd()
    roots = [(base / "TestFile", True), (base, False)]
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for root, recursive in roots:
        if not root.is_dir():
            continue
        for p in (root.rglob("*") if recursive else root.glob("*")):
            if p.suffix.lower() in VIDEO_EXTS and p.is_file():
                rp = str(p.resolve())
                if rp not in seen:
                    seen.add(rp)
                    out.append({"name": p.name, "path": rp})
    out.sort(key=lambda d: d["name"].lower())
    return out[:100]


def _native_pick_file() -> dict[str, str]:
    """Open a native OS file dialog (in a subprocess so it never touches the server's
    event loop / Tk main-thread rules) and return the chosen absolute path. Works because
    the console is a localhost tool — the dialog opens on the same machine as the browser."""
    code = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
        "p = filedialog.askopenfilename(title='StreamForge - select a video to loop',\n"
        "    filetypes=[('Video', '*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.mpg *.mpeg *.wmv'),\n"
        "               ('All files', '*.*')])\n"
        "r.destroy()\n"
        "print(p or '')\n"
    )
    try:
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        return {"path": res.stdout.strip()}
    except Exception as e:  # subprocess/timeout/Tk failure -> report, don't crash the server
        return {"path": "", "error": f"{type(e).__name__}: {e}"}


def create_app(runner: StreamForgeRunner | None = None) -> FastAPI:
    app = FastAPI(title="StreamForge Operator Console")
    app.state.runner = runner or StreamForgeRunner()
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir, check_dir=False), name="static")

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/api/sources")
    def sources():
        webcam = [{"type": "webcam", "name": "0", "label": "Webcam 0"}]
        ndi = []
        spout = []
        try:
            from streamforge.sources.ndi_source import NDISource
            ndi = [{"type": "ndi", "name": name, "label": name} for name in NDISource.list_sources()]
        except Exception as e:
            ndi = [{"type": "ndi", "name": "", "label": f"NDI unavailable: {e}", "disabled": True}]
        try:
            from streamforge.sources.spout_source import SpoutSource
            spout = [{"type": "spout", "name": name, "label": name} for name in SpoutSource.list_sources()]
        except Exception as e:
            spout = [{"type": "spout", "name": "", "label": f"Spout unavailable: {e}", "disabled": True}]
        return {"sources": webcam + ndi + spout}

    @app.get("/api/files")
    def files():
        return {"files": _list_video_files()}

    @app.post("/api/browse")
    def browse():
        return _native_pick_file()

    @app.post("/api/validate")
    def validate(config: RunnerConfigIn):
        return app.state.runner.validate(config.to_config())

    @app.post("/api/run/start")
    def start(config: RunnerConfigIn):
        app.state.runner.start(config.to_config())
        return {"ok": True}

    @app.post("/api/run/stop")
    def stop():
        app.state.runner.stop()
        return {"ok": True}

    @app.post("/api/control")
    def control(body: ControlIn):
        return app.state.runner.apply_control(**body.model_dump(exclude_none=True))

    @app.get("/api/status")
    def status():
        return app.state.runner.status()

    @app.get("/preview/input.jpg")
    def preview_input():
        data = app.state.runner.latest_input_jpeg()
        if data is None:
            return Response(status_code=204)
        return Response(content=data, media_type="image/jpeg")

    @app.get("/preview/output.jpg")
    def preview_output():
        data = app.state.runner.latest_output_jpeg()
        if data is None:
            return Response(status_code=204)
        return Response(content=data, media_type="image/jpeg")

    return app


app = create_app()

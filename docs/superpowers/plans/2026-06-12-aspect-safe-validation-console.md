# Aspect-Safe Validation Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove aspect-ratio stretching from StreamForge and add a simplified Operator Console web UI that validates NDI, Spout, or webcam input and shows live input/output previews plus runtime health.

**Architecture:** Add pure aspect/resize utilities first, then route runtime downscale/upscale and validation scripts through them so all shape changes preserve aspect. Add first-class source classes for webcam/NDI/Spout, extract `scripts/live.py` orchestration into a reusable runner service, expose that service through a small FastAPI backend, and serve a dense static Operator Console that polls status and preview images.

**Tech Stack:** Python 3.11, PyTorch, OpenCV, Pillow, FastAPI, Uvicorn, pytest, existing StreamForge `Source`/`Sink`/`InferenceWorker`/`RealtimeClock` abstractions. Spec: `docs/superpowers/specs/2026-06-12-aspect-safe-validation-console-design.md`.

---

## Context You Need

- Run commands from `D:\StreamForge`.
- Use `.\.venv\Scripts\python.exe` for Python.
- Set `PYTHONPATH=src` when running tests directly from PowerShell:
  `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_aspect.py -v`
- The current pure suite passes: `50 passed`.
- The distortion bug is confirmed in `scripts/restyle_clip.py`: the line `interpolate(t, size=(args.res, args.res))` stretches a 640x480 source into 512x512.
- The live runtime can also distort when `EagerRuntime(internal_hw=(224, 384))` receives 4:3 input, because `runtime_eager.py` directly interpolates to the fixed 16:9 internal canvas.
- `src/streamforge/fill/*` already contains the RIFE/RAFT frame-fill work. Do not rewrite it unless a changed interface requires a narrow adjustment.
- `.superpowers/` is ignored and contains only local browser planning scratch.

## File Structure

| File | Responsibility |
|---|---|
| `requirements.txt` | Add FastAPI/Uvicorn/httpx dependencies for backend and API tests. |
| `src/streamforge/aspect.py` | Pure size, fit-plan, crop, snap, and tensor resize utilities. |
| `tests/test_aspect.py` | Tests for 4:3/16:9/1:1 planning and tensor fitting. |
| `src/streamforge/diffusion/runtime_eager.py` | Use aspect-safe internal resizing instead of direct stretch. |
| `scripts/restyle_clip.py` | Stop square-stretching validation frames; use aspect policy. |
| `scripts/verify_track_a.py` | Report aspect policy and derived internal size. |
| `tests/test_runtime_aspect.py` | Runtime resize tests with fake dispatch, no model load. |
| `src/streamforge/sources/base.py` | Add `SourceStatus` metadata contract and optional `status()` method. |
| `src/streamforge/sources/webcam.py` | Webcam source via OpenCV `VideoCapture`. |
| `src/streamforge/sources/ndi_source.py` | NDI receiver source class. |
| `src/streamforge/sources/spout_source.py` | Spout receiver source class. |
| `tests/test_input_sources.py` | Fake-module tests for source metadata and frame conversion. |
| `src/streamforge/sinks/ndi_sink.py` | Set explicit NDI video dimensions/stride/aspect metadata. |
| `tests/test_ndi_sink.py` | Fake NDI sender test for video frame metadata. |
| `src/streamforge/runner.py` | Reusable pipeline runner service used by CLI and web API. |
| `tests/test_runner.py` | Validate/start/status/stop service behavior with fake sources/runtime/sink. |
| `scripts/live.py` | Thin CLI wrapper around `StreamForgeRunner`. |
| `src/streamforge/web/app.py` | FastAPI app, API routes, static UI mount. |
| `src/streamforge/web/static/index.html` | Operator Console markup. |
| `src/streamforge/web/static/app.js` | Browser logic for validate/start/stop/status/preview polling. |
| `src/streamforge/web/static/styles.css` | Dense console layout and crop overlays. |
| `scripts/web.py` | Local web server entrypoint. |
| `tests/test_web_api.py` | API tests with fake runner dependencies. |
| `README.md` and `docs/TESTING_OUTPUTS.md` | Document the console and aspect-safe validation workflow. |

---

## Task 1: Add Web Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependencies**

Append these lines to `requirements.txt`:

```text
fastapi
uvicorn
httpx
```

- [ ] **Step 2: Install dependencies**

Run: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`

Expected: FastAPI, Uvicorn, and httpx install successfully without replacing the CUDA-specific Torch build.

- [ ] **Step 3: Verify imports**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -c "import fastapi, uvicorn, httpx; print('web deps ok')"`

Expected: prints `web deps ok`.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore(web): add FastAPI console dependencies"
```

---

## Task 2: Aspect Planning and Tensor Fit Utilities

**Files:**
- Create: `src/streamforge/aspect.py`
- Test: `tests/test_aspect.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_aspect.py
import torch

from streamforge.aspect import FitMode, Size, fit_tensor, plan_fit, snap_to_multiple_for_aspect


def test_fill_crop_4x3_into_16x9_crops_top_bottom():
    plan = plan_fit(Size(640, 480), Size(384, 216), FitMode.FILL_CROP)
    assert plan.resized == Size(384, 288)
    assert plan.crop_left == 0
    assert plan.crop_top == 36
    assert plan.crop_width == 384
    assert plan.crop_height == 216
    assert plan.crop_direction == "top_bottom"


def test_fill_crop_16x9_into_4x3_crops_sides():
    plan = plan_fit(Size(1920, 1080), Size(640, 480), FitMode.FILL_CROP)
    assert plan.resized == Size(853, 480)
    assert plan.crop_left == 106
    assert plan.crop_top == 0
    assert plan.crop_width == 640
    assert plan.crop_height == 480
    assert plan.crop_direction == "sides"


def test_same_ratio_has_no_crop():
    plan = plan_fit(Size(640, 480), Size(320, 240), FitMode.FILL_CROP)
    assert plan.resized == Size(320, 240)
    assert plan.crop_left == 0 and plan.crop_top == 0
    assert plan.crop_direction == "none"


def test_fit_tensor_preserves_geometry_by_cropping_not_stretching():
    img = torch.zeros(1, 1, 4, 4)
    img[:, :, 1:3, 1:3] = 1.0
    out, plan = fit_tensor(img, Size(4, 2), FitMode.FILL_CROP)
    assert out.shape == (1, 1, 2, 4)
    assert plan.crop_direction == "top_bottom"
    assert float(out.max()) == 1.0


def test_snap_to_multiple_for_aspect_keeps_4x3_ratio():
    size = snap_to_multiple_for_aspect(Size(640, 480), max_side=384, multiple=16)
    assert size == Size(384, 288)
```

- [ ] **Step 2: Run to verify failure**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_aspect.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'streamforge.aspect'`.

- [ ] **Step 3: Implement**

```python
# src/streamforge/aspect.py
"""Aspect-ratio planning and tensor fitting utilities.

All resize plans use one uniform scale factor. If source and target ratios differ, content is
cropped after scaling; it is never non-uniformly stretched.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch
import torch.nn.functional as F


class FitMode(str, Enum):
    FILL_CROP = "fill_crop"


@dataclass(frozen=True)
class Size:
    width: int
    height: int

    @property
    def aspect(self) -> float:
        return self.width / self.height


@dataclass(frozen=True)
class FitPlan:
    source: Size
    target: Size
    resized: Size
    crop_left: int
    crop_top: int
    crop_width: int
    crop_height: int
    scale: float
    crop_direction: str


def _validate_size(size: Size, label: str) -> None:
    if size.width <= 0 or size.height <= 0:
        raise ValueError(f"{label} width and height must be positive, got {size.width}x{size.height}")


def plan_fit(source: Size, target: Size, mode: FitMode = FitMode.FILL_CROP) -> FitPlan:
    _validate_size(source, "source")
    _validate_size(target, "target")
    if mode != FitMode.FILL_CROP:
        raise ValueError(f"unsupported fit mode: {mode}")
    scale = max(target.width / source.width, target.height / source.height)
    resized_w = max(target.width, int(math.ceil(source.width * scale)))
    resized_h = max(target.height, int(math.ceil(source.height * scale)))
    crop_left = max(0, (resized_w - target.width) // 2)
    crop_top = max(0, (resized_h - target.height) // 2)
    if resized_w > target.width:
        direction = "sides"
    elif resized_h > target.height:
        direction = "top_bottom"
    else:
        direction = "none"
    return FitPlan(
        source=source,
        target=target,
        resized=Size(resized_w, resized_h),
        crop_left=crop_left,
        crop_top=crop_top,
        crop_width=target.width,
        crop_height=target.height,
        scale=scale,
        crop_direction=direction,
    )


def fit_tensor(
    image_bchw: torch.Tensor,
    target: Size,
    mode: FitMode = FitMode.FILL_CROP,
    antialias: bool = True,
) -> tuple[torch.Tensor, FitPlan]:
    if image_bchw.ndim != 4:
        raise ValueError(f"expected BCHW tensor, got shape {tuple(image_bchw.shape)}")
    source = Size(width=int(image_bchw.shape[-1]), height=int(image_bchw.shape[-2]))
    plan = plan_fit(source, target, mode)
    resized = F.interpolate(
        image_bchw,
        size=(plan.resized.height, plan.resized.width),
        mode="bilinear",
        align_corners=False,
        antialias=antialias,
    )
    top = plan.crop_top
    left = plan.crop_left
    cropped = resized[:, :, top:top + plan.crop_height, left:left + plan.crop_width]
    return cropped, plan


def snap_to_multiple_for_aspect(source: Size, max_side: int, multiple: int = 16) -> Size:
    _validate_size(source, "source")
    if max_side < multiple:
        raise ValueError(f"max_side must be >= multiple, got max_side={max_side}, multiple={multiple}")
    if source.width >= source.height:
        width = (max_side // multiple) * multiple
        height = max(multiple, round((width / source.aspect) / multiple) * multiple)
    else:
        height = (max_side // multiple) * multiple
        width = max(multiple, round((height * source.aspect) / multiple) * multiple)
    return Size(width=int(width), height=int(height))
```

- [ ] **Step 4: Verify tests pass**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_aspect.py -v`

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/streamforge/aspect.py tests/test_aspect.py
git commit -m "feat(aspect): add fill-crop resize planning"
```

---

## Task 3: Use Aspect-Safe Fitting in Runtime and Validation Scripts

**Files:**
- Modify: `src/streamforge/diffusion/runtime_eager.py`
- Modify: `scripts/restyle_clip.py`
- Modify: `scripts/verify_track_a.py`
- Test: `tests/test_runtime_aspect.py`

- [ ] **Step 1: Write failing runtime aspect tests**

```python
# tests/test_runtime_aspect.py
import torch

from streamforge.control import EngineParams
from streamforge.diffusion.runtime_eager import EagerRuntime


class FakeRuntime(EagerRuntime):
    def __init__(self, internal_hw):
        self.device = "cpu"
        self.internal_hw = self._snap_hw(internal_hw)
        self.fit_plan = None

    def _dispatch(self, image_bchw, params):
        return image_bchw


def test_internal_resize_preserves_source_shape_after_roundtrip():
    rt = FakeRuntime(internal_hw=(224, 384))
    src = torch.rand(1, 3, 480, 640)
    out = rt.restyle(src, EngineParams(denoise_strength=0.5, guidance=1.0))
    assert out.shape == src.shape


def test_internal_resize_records_crop_when_ratio_differs():
    rt = FakeRuntime(internal_hw=(224, 384))
    src = torch.rand(1, 3, 480, 640)
    rt.restyle(src, EngineParams(denoise_strength=0.5, guidance=1.0))
    assert rt.fit_plan.crop_direction == "top_bottom"
```

- [ ] **Step 2: Run to verify the second test fails**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_runtime_aspect.py -v`

Expected: FAIL because `EagerRuntime` does not record an aspect fit plan.

- [ ] **Step 3: Patch `EagerRuntime.restyle()`**

In `src/streamforge/diffusion/runtime_eager.py`, import aspect helpers:

```python
from streamforge.aspect import FitMode, Size, fit_tensor
```

Then replace the internal resize block in `restyle()` with:

```python
        if self.internal_hw is not None:
            out_h, out_w = image_bchw.shape[-2], image_bchw.shape[-1]
            ih, iw = self.internal_hw
            if (ih, iw) != (out_h, out_w):
                small, plan = fit_tensor(
                    image_bchw.to(self.device),
                    Size(width=iw, height=ih),
                    FitMode.FILL_CROP,
                    antialias=True,
                )
                self.fit_plan = plan
                styled = self._dispatch(small, params)
                return torch.nn.functional.interpolate(
                    styled, size=(out_h, out_w), mode="bilinear", align_corners=False)
```

Add `self.fit_plan = None` in `__init__` immediately after the line that assigns `self.internal_hw`.

- [ ] **Step 4: Patch `scripts/restyle_clip.py`**

Add imports:

```python
from streamforge.aspect import FitMode, Size, fit_tensor, snap_to_multiple_for_aspect
```

Replace `--res` with:

```python
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--target", default="source", choices=["source", "square", "16x9"],
                    help="source preserves clip ratio; square/16x9 use fill-crop instead of stretch")
```

Replace the direct square interpolate with:

```python
            if args.target == "source":
                target = snap_to_multiple_for_aspect(Size(width=t.shape[-1], height=t.shape[-2]), args.max_side)
            elif args.target == "square":
                target = Size(args.max_side, args.max_side)
            else:
                target = Size(args.max_side, round(args.max_side * 9 / 16))
            t, plan = fit_tensor(t, target, FitMode.FILL_CROP)
            print(f"frame {idx}: source={plan.source.width}x{plan.source.height} "
                  f"target={plan.target.width}x{plan.target.height} crop={plan.crop_direction}")
```

- [ ] **Step 5: Patch `scripts/verify_track_a.py`**

After reading the first frame and parsing `--in-res`, print an explicit warning when source and internal ratios differ:

```python
    source_ratio = in_w / in_h
    internal_ratio = w_i / h_i
    if abs(source_ratio - internal_ratio) > 0.01:
        print(f"aspect policy: fill-crop source {in_w}x{in_h} into internal {w_i}x{h_i}")
    else:
        print(f"aspect policy: preserve source ratio at internal {w_i}x{h_i}")
```

- [ ] **Step 6: Verify tests**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_aspect.py tests/test_runtime_aspect.py -v`

Expected: all tests pass.

- [ ] **Step 7: Verify still-frame script no longer square-stretches by default**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe scripts\restyle_clip.py --count 1 --every 1 --max-side 512 --target source --preset PRESERVE`

Expected: the printed target is `512x384` for the 640x480 test clip, not `512x512`.

- [ ] **Step 8: Commit**

```bash
git add src/streamforge/diffusion/runtime_eager.py scripts/restyle_clip.py scripts/verify_track_a.py tests/test_runtime_aspect.py
git commit -m "fix(aspect): preserve ratio in runtime and visual gates"
```

---

## Task 4: Source Metadata and Input Source Classes

**Files:**
- Modify: `src/streamforge/sources/base.py`
- Create: `src/streamforge/sources/webcam.py`
- Create: `src/streamforge/sources/ndi_source.py`
- Create: `src/streamforge/sources/spout_source.py`
- Modify: `src/streamforge/sources/__init__.py`
- Test: `tests/test_input_sources.py`

- [ ] **Step 1: Extend the source contract**

Patch `src/streamforge/sources/base.py`:

```python
@dataclass(frozen=True)
class SourceStatus:
    name: str
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    available: bool = False
    last_frame_age_ms: float | None = None
    error: str | None = None


class Source(ABC):
    capabilities: Capabilities = Capabilities()

    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def read(self) -> GpuFrame | None:
        """Return the next frame, or None at end-of-stream."""

    @abstractmethod
    def close(self) -> None:
        pass

    def status(self) -> SourceStatus:
        return SourceStatus(name=type(self).__name__, available=False)
```

- [ ] **Step 2: Implement `WebcamSource`**

```python
# src/streamforge/sources/webcam.py
from __future__ import annotations

import time

import torch

from streamforge.frame import GpuFrame
from streamforge.sources.base import Capabilities, Source, SourceStatus


class WebcamSource(Source):
    capabilities = Capabilities(has_motion_vectors=False)

    def __init__(self, index: int = 0, fps: int = 30, device: str = "cuda"):
        self.index = index
        self.fps = fps
        self._device = device if torch.cuda.is_available() else "cpu"
        self._cap = None
        self._seq = 0
        self._last_frame_t: float | None = None
        self._error: str | None = None

    def open(self) -> None:
        import cv2
        self._cap = cv2.VideoCapture(self.index)
        self._seq = 0
        if not self._cap.isOpened():
            self._error = f"webcam {self.index} is not available"

    def read(self) -> GpuFrame | None:
        import cv2
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, bgr = self._cap.read()
        if not ok:
            self._error = f"webcam {self.index} returned no frame"
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self._device)
        self._last_frame_t = time.perf_counter()
        frame = GpuFrame(tensor=t, seq=self._seq, pts=self._seq / self.fps,
                         width=t.shape[-1], height=t.shape[-2])
        self._seq += 1
        return frame

    def status(self) -> SourceStatus:
        width = height = None
        if self._cap is not None:
            import cv2
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        age = ((time.perf_counter() - self._last_frame_t) * 1000.0
               if self._last_frame_t is not None else None)
        return SourceStatus(
            name=f"Webcam {self.index}",
            width=width,
            height=height,
            fps=float(self.fps),
            available=self._cap is not None and self._cap.isOpened(),
            last_frame_age_ms=age,
            error=self._error,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
```

- [ ] **Step 3: Implement `NDISource`**

```python
# src/streamforge/sources/ndi_source.py
from __future__ import annotations

import time

import numpy as np
import torch

from streamforge.frame import GpuFrame
from streamforge.sources.base import Capabilities, Source, SourceStatus


class NDISource(Source):
    capabilities = Capabilities(has_motion_vectors=False)

    def __init__(self, name: str = "", fps: int = 30, device: str = "cuda", timeout_ms: int = 1000):
        self.name = name
        self.fps = fps
        self.timeout_ms = timeout_ms
        self._device = device if torch.cuda.is_available() else "cpu"
        self._ndi = None
        self._finder = None
        self._recv = None
        self._source_name: str | None = None
        self._seq = 0
        self._last_frame_t: float | None = None
        self._size: tuple[int, int] | None = None
        self._error: str | None = None

    @staticmethod
    def list_sources() -> list[str]:
        import NDIlib as ndi
        if not ndi.initialize():
            return []
        finder = ndi.find_create_v2()
        try:
            ndi.find_wait_for_sources(finder, 1000)
            return [s.ndi_name for s in ndi.find_get_current_sources(finder)]
        finally:
            ndi.find_destroy(finder)
            ndi.destroy()

    def open(self) -> None:
        import NDIlib as ndi
        self._ndi = ndi
        if not ndi.initialize():
            self._error = "NDI runtime failed to initialize"
            return
        self._finder = ndi.find_create_v2()
        ndi.find_wait_for_sources(self._finder, 2000)
        sources = ndi.find_get_current_sources(self._finder)
        selected = None
        for src in sources:
            if not self.name or self.name.lower() in src.ndi_name.lower():
                selected = src
                break
        if selected is None:
            self._error = f"NDI source matching '{self.name}' not found"
            return
        rc = ndi.RecvCreateV3()
        rc.color_format = ndi.RECV_COLOR_FORMAT_RGBX_RGBA
        self._recv = ndi.recv_create_v3(rc)
        ndi.recv_connect(self._recv, selected)
        self._source_name = selected.ndi_name
        self._seq = 0

    def read(self) -> GpuFrame | None:
        if self._ndi is None or self._recv is None:
            return None
        tpe, v, a, m = self._ndi.recv_capture_v2(self._recv, self.timeout_ms)
        if tpe == self._ndi.FRAME_TYPE_VIDEO:
            try:
                rgba = np.copy(v.data)
                rgb = rgba[:, :, :3]
                tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self._device)
                self._size = (tensor.shape[-1], tensor.shape[-2])
                self._last_frame_t = time.perf_counter()
                frame = GpuFrame(tensor=tensor, seq=self._seq, pts=self._seq / self.fps,
                                 width=tensor.shape[-1], height=tensor.shape[-2])
                self._seq += 1
                return frame
            finally:
                self._ndi.recv_free_video_v2(self._recv, v)
        if tpe == self._ndi.FRAME_TYPE_AUDIO:
            self._ndi.recv_free_audio_v2(self._recv, a)
        return None

    def status(self) -> SourceStatus:
        age = ((time.perf_counter() - self._last_frame_t) * 1000.0
               if self._last_frame_t is not None else None)
        width = self._size[0] if self._size else None
        height = self._size[1] if self._size else None
        return SourceStatus(
            name=self._source_name or self.name or "NDI",
            width=width,
            height=height,
            fps=float(self.fps),
            available=self._recv is not None and self._error is None,
            last_frame_age_ms=age,
            error=self._error,
        )

    def close(self) -> None:
        if self._ndi is not None and self._recv is not None:
            self._ndi.recv_destroy(self._recv)
        if self._ndi is not None and self._finder is not None:
            self._ndi.find_destroy(self._finder)
        if self._ndi is not None:
            self._ndi.destroy()
        self._recv = None
        self._finder = None
```

- [ ] **Step 4: Implement `SpoutSource`**

```python
# src/streamforge/sources/spout_source.py
from __future__ import annotations

import time

import numpy as np
import torch

from streamforge.frame import GpuFrame
from streamforge.sources.base import Capabilities, Source, SourceStatus

GL_RGBA = 0x1908


class SpoutSource(Source):
    capabilities = Capabilities(has_motion_vectors=False)

    def __init__(self, name: str = "StreamForge", fps: int = 30, device: str = "cuda", invert: bool = True):
        self.name = name
        self.fps = fps
        self.invert = invert
        self._device = device if torch.cuda.is_available() else "cpu"
        self._receiver = None
        self._buffer = None
        self._seq = 0
        self._last_frame_t: float | None = None
        self._size: tuple[int, int] | None = None
        self._error: str | None = None

    @staticmethod
    def list_sources() -> list[str]:
        import SpoutGL
        receiver = SpoutGL.SpoutReceiver()
        try:
            return list(receiver.getSenderList())
        finally:
            receiver.releaseReceiver()

    def open(self) -> None:
        import SpoutGL
        self._receiver = SpoutGL.SpoutReceiver()
        if not self._receiver.setReceiverName(self.name):
            self._error = f"Spout sender '{self.name}' not found"
        self._seq = 0

    def read(self) -> GpuFrame | None:
        if self._receiver is None:
            return None
        width = int(self._receiver.getSenderWidth())
        height = int(self._receiver.getSenderHeight())
        if width <= 0 or height <= 0:
            return None
        needed = width * height * 4
        if self._buffer is None or len(self._buffer) != needed:
            self._buffer = bytearray(needed)
        ok = self._receiver.receiveImage(self._buffer, GL_RGBA, self.invert, 0)
        if not ok:
            return None
        rgba = np.frombuffer(self._buffer, dtype=np.uint8).reshape((height, width, 4))
        rgb = np.copy(rgba[:, :, :3])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self._device)
        self._size = (width, height)
        self._last_frame_t = time.perf_counter()
        frame = GpuFrame(tensor=tensor, seq=self._seq, pts=self._seq / self.fps,
                         width=width, height=height)
        self._seq += 1
        return frame

    def status(self) -> SourceStatus:
        age = ((time.perf_counter() - self._last_frame_t) * 1000.0
               if self._last_frame_t is not None else None)
        return SourceStatus(
            name=self.name,
            width=self._size[0] if self._size else None,
            height=self._size[1] if self._size else None,
            fps=float(self.fps),
            available=self._receiver is not None and self._error is None,
            last_frame_age_ms=age,
            error=self._error,
        )

    def close(self) -> None:
        if self._receiver is not None:
            self._receiver.releaseReceiver()
            self._receiver = None
```

- [ ] **Step 5: Export source classes**

```python
# src/streamforge/sources/__init__.py
from streamforge.sources.base import Capabilities, Source, SourceStatus
from streamforge.sources.file_source import FileSource
from streamforge.sources.synthetic import SyntheticSource
from streamforge.sources.webcam import WebcamSource

__all__ = [
    "Capabilities",
    "FileSource",
    "Source",
    "SourceStatus",
    "SyntheticSource",
    "WebcamSource",
]
```

Do not import `NDISource` or `SpoutSource` here; those modules import optional runtime libraries.

- [ ] **Step 6: Add fake-module tests**

Create `tests/test_input_sources.py`:

```python
import sys
import types

import numpy as np


class FakeCV2:
    COLOR_BGR2RGB = 1
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    class VideoCapture:
        def __init__(self, index):
            self.index = index
            self.opened = True
            self.frame = np.zeros((4, 6, 3), dtype=np.uint8)

        def isOpened(self):
            return self.opened

        def read(self):
            return True, self.frame

        def get(self, prop):
            return {3: 6, 4: 4}.get(prop, 0)

        def release(self):
            self.opened = False

    @staticmethod
    def cvtColor(frame, code):
        return frame[:, :, ::-1]


def test_webcam_source_reads_bgr_frame_and_reports_status(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    from streamforge.sources.webcam import WebcamSource
    source = WebcamSource(index=0, fps=30, device="cpu")
    source.open()
    frame = source.read()
    status = source.status()
    source.close()
    assert frame.tensor.shape == (1, 3, 4, 6)
    assert frame.width == 6 and frame.height == 4
    assert status.available is True
    assert status.width == 6 and status.height == 4


class FakeNDIVideo:
    def __init__(self):
        self.data = np.zeros((4, 6, 4), dtype=np.uint8)


class FakeNDI:
    RECV_COLOR_FORMAT_RGBX_RGBA = 1
    FRAME_TYPE_VIDEO = 2
    FRAME_TYPE_AUDIO = 3

    def initialize(self):
        return True

    def find_create_v2(self):
        return object()

    def find_wait_for_sources(self, finder, timeout_ms):
        return True

    def find_get_current_sources(self, finder):
        return [types.SimpleNamespace(ndi_name="Camera A")]

    def find_destroy(self, finder):
        pass

    class RecvCreateV3:
        color_format = None

    def recv_create_v3(self, config):
        return object()

    def recv_connect(self, recv, source):
        pass

    def recv_capture_v2(self, recv, timeout_ms):
        return self.FRAME_TYPE_VIDEO, FakeNDIVideo(), None, None

    def recv_free_video_v2(self, recv, video):
        pass

    def recv_free_audio_v2(self, recv, audio):
        pass

    def recv_destroy(self, recv):
        pass

    def destroy(self):
        pass


def test_ndi_source_reads_rgba_frame_and_reports_status(monkeypatch):
    monkeypatch.setitem(sys.modules, "NDIlib", FakeNDI())
    from streamforge.sources.ndi_source import NDISource
    source = NDISource(name="Camera", fps=30, device="cpu")
    source.open()
    frame = source.read()
    status = source.status()
    source.close()
    assert frame.tensor.shape == (1, 3, 4, 6)
    assert frame.width == 6 and frame.height == 4
    assert status.available is True
    assert status.width == 6 and status.height == 4


class FakeSpoutReceiver:
    def __init__(self):
        self.released = False

    def getSenderList(self):
        return ["Camera A"]

    def setReceiverName(self, name):
        return True

    def getSenderWidth(self):
        return 6

    def getSenderHeight(self):
        return 4

    def receiveImage(self, buffer, gl_format, invert, host_fbo):
        arr = np.zeros((4, 6, 4), dtype=np.uint8)
        buffer[:] = arr.tobytes()
        return True

    def releaseReceiver(self):
        self.released = True


class FakeSpoutGL:
    SpoutReceiver = FakeSpoutReceiver


def test_spout_source_reads_rgba_buffer_and_reports_status(monkeypatch):
    monkeypatch.setitem(sys.modules, "SpoutGL", FakeSpoutGL)
    from streamforge.sources.spout_source import SpoutSource
    source = SpoutSource(name="Camera A", fps=30, device="cpu")
    source.open()
    frame = source.read()
    status = source.status()
    source.close()
    assert frame.tensor.shape == (1, 3, 4, 6)
    assert frame.width == 6 and frame.height == 4
    assert status.available is True
    assert status.width == 6 and status.height == 4
```

- [ ] **Step 7: Verify tests**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_input_sources.py -v`

Expected: source tests pass without real webcam, NDI, or Spout runtimes.

- [ ] **Step 8: Commit**

```bash
git add src/streamforge/sources tests/test_input_sources.py
git commit -m "feat(sources): add webcam NDI and Spout input sources"
```

---

## Task 5: Explicit NDI Sink Frame Metadata

**Files:**
- Modify: `src/streamforge/sinks/ndi_sink.py`
- Test: `tests/test_ndi_sink.py`

- [ ] **Step 1: Write failing fake NDI metadata test**

```python
# tests/test_ndi_sink.py
import types

import torch

from streamforge.frame import GpuFrame


class FakeVideoFrame:
    pass


class FakeNDI:
    FOURCC_VIDEO_TYPE_RGBA = "RGBA"

    def __init__(self):
        self.sent = None

    def initialize(self): return True
    def SendCreate(self): return types.SimpleNamespace(ndi_name="")
    def send_create(self, settings): return object()
    def VideoFrameV2(self): return FakeVideoFrame()
    def send_send_video_v2(self, sender, frame): self.sent = frame
    def send_destroy(self, sender): pass
    def destroy(self): pass


def test_ndi_sink_sets_dimensions_stride_and_aspect(monkeypatch):
    fake = FakeNDI()
    monkeypatch.setitem(__import__("sys").modules, "NDIlib", fake)
    from streamforge.sinks.ndi_sink import NDISink
    sink = NDISink(fps=30)
    sink.open()
    frame = GpuFrame(tensor=torch.zeros(1, 3, 4, 6), seq=0, pts=0.0, width=6, height=4)
    sink.send(frame)
    assert fake.sent.xres == 6
    assert fake.sent.yres == 4
    assert fake.sent.line_stride_in_bytes == 6 * 4
    assert fake.sent.picture_aspect_ratio == 1.5
```

- [ ] **Step 2: Run to verify failure**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_ndi_sink.py -v`

Expected: FAIL because `xres`, `yres`, `line_stride_in_bytes`, and `picture_aspect_ratio` are not set.

- [ ] **Step 3: Patch `NDISink.send()`**

After constructing the contiguous `rgba` array and before `send_send_video_v2`, set:

```python
        self._vframe.xres = int(w)
        self._vframe.yres = int(h)
        self._vframe.line_stride_in_bytes = int(w * 4)
        self._vframe.picture_aspect_ratio = float(w / h)
        self._vframe.frame_rate_N = int(self.fps * 1000)
        self._vframe.frame_rate_D = 1000
```

Keep the existing `self._vframe.FourCC = self._ndi.FOURCC_VIDEO_TYPE_RGBA`.

- [ ] **Step 4: Verify**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_ndi_sink.py tests/test_sinks.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/streamforge/sinks/ndi_sink.py tests/test_ndi_sink.py
git commit -m "fix(ndi): send explicit frame dimensions and aspect metadata"
```

---

## Task 6: Reusable Runner Service

**Files:**
- Create: `src/streamforge/runner.py`
- Modify: `scripts/live.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write runner service tests**

```python
# tests/test_runner.py
import torch

from streamforge.frame import GpuFrame
from streamforge.runner import RunnerConfig, StreamForgeRunner


class StubSource:
    def __init__(self):
        self.opened = False
        self.closed = False
        self.frame = GpuFrame(torch.zeros(1, 3, 4, 6), 0, 0.0, 6, 4)

    def open(self): self.opened = True
    def read(self): return self.frame
    def close(self): self.closed = True
    def status(self):
        from streamforge.sources.base import SourceStatus
        return SourceStatus(name="stub", width=6, height=4, fps=30.0, available=True)


class StubRuntime:
    def set_prompt(self, prompt): self.prompt = prompt
    def restyle(self, tensor, params): return tensor + 0.25


class StubSink:
    def __init__(self): self.frames = []
    def open(self): pass
    def send(self, frame): self.frames.append(frame)
    def close(self): pass


def test_validate_reads_first_frame_and_returns_aspect():
    runner = StreamForgeRunner(source_factory=lambda cfg: StubSource(),
                               runtime_factory=lambda cfg: StubRuntime(),
                               sink_factory=lambda cfg: StubSink())
    result = runner.validate(RunnerConfig(source_type="webcam", source_name="0"))
    assert result["ok"] is True
    assert result["source"]["width"] == 6
    assert result["source"]["height"] == 4
    assert result["aspect"]["source_ratio"] == 1.5


def test_start_status_stop_with_fake_runtime():
    sink = StubSink()
    runner = StreamForgeRunner(source_factory=lambda cfg: StubSource(),
                               runtime_factory=lambda cfg: StubRuntime(),
                               sink_factory=lambda cfg: sink)
    runner.start(RunnerConfig(source_type="webcam", source_name="0", seconds=0.2, fps=5))
    status = runner.status()
    assert status["running"] is True
    runner.stop()
    assert runner.status()["running"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_runner.py -v`

Expected: FAIL with `ModuleNotFoundError: streamforge.runner`.

- [ ] **Step 3: Implement runner dataclasses and service**

Create `src/streamforge/runner.py` with:

```python
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch

from streamforge.aspect import FitMode, Size, plan_fit, snap_to_multiple_for_aspect
from streamforge.clock import FrameBuffer, RealtimeClock
from streamforge.color import ColorPipeline
from streamforge.control import TwoAxisControl
from streamforge.metrics import jitter_ms
from streamforge.worker import InferenceWorker


@dataclass(frozen=True)
class RunnerConfig:
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


def default_source_factory(config: RunnerConfig):
    if config.source_type == "webcam":
        from streamforge.sources.webcam import WebcamSource
        return WebcamSource(index=int(config.source_name or "0"), fps=config.fps)
    if config.source_type == "ndi":
        from streamforge.sources.ndi_source import NDISource
        return NDISource(name=config.source_name, fps=config.fps)
    if config.source_type == "spout":
        from streamforge.sources.spout_source import SpoutSource
        return SpoutSource(name=config.source_name, fps=config.fps)
    if config.source_type == "file":
        from streamforge.sources.file_source import FileSource
        return FileSource(config.source_name, fps=config.fps)
    raise ValueError(f"unknown source type {config.source_type!r}")


def default_runtime_factory(config: RunnerConfig):
    from streamforge.diffusion.runtime_eager import EagerRuntime
    internal_hw = None
    if config.in_res not in ("off", "native", "auto", ""):
        w_i, h_i = (int(v) for v in config.in_res.lower().split("x"))
        internal_hw = (h_i, w_i)
    return EagerRuntime(mode=config.mode, internal_hw=internal_hw)


def default_sink_factory(config: RunnerConfig):
    if config.sink == "null":
        from streamforge.sinks.null_sink import NullSink
        return NullSink()
    if config.sink == "spout":
        from streamforge.sinks.spout_sink import SpoutSink
        return SpoutSink()
    if config.sink == "ndi":
        from streamforge.sinks.ndi_sink import NDISink
        return NDISink(fps=config.fps)
    raise ValueError(f"unknown sink {config.sink!r}")


class StreamForgeRunner:
    def __init__(
        self,
        source_factory: Callable[[RunnerConfig], object] = default_source_factory,
        runtime_factory: Callable[[RunnerConfig], object] = default_runtime_factory,
        sink_factory: Callable[[RunnerConfig], object] = default_sink_factory,
    ):
        self.source_factory = source_factory
        self.runtime_factory = runtime_factory
        self.sink_factory = sink_factory
        self._lock = threading.Lock()
        self._running = False
        self._config: RunnerConfig | None = None
        self._timestamps: list[float] = []
        self._infer_ms: list[float] = []
        self._latest_input = None
        self._latest_output = None
        self._worker = None
        self._clock = None
        self._sink = None

    def validate(self, config: RunnerConfig) -> dict:
        source = self.source_factory(config)
        source.open()
        try:
            frame = source.read()
            if frame is None:
                return {"ok": False, "error": "source returned no frame", "source": asdict(source.status())}
            source_size = Size(width=frame.width, height=frame.height)
            internal = snap_to_multiple_for_aspect(source_size, max_side=384, multiple=16)
            plan = plan_fit(source_size, internal, FitMode.FILL_CROP)
            return {
                "ok": True,
                "source": asdict(source.status()),
                "aspect": {
                    "source_ratio": source_size.aspect,
                    "internal": asdict(internal),
                    "crop_direction": plan.crop_direction,
                },
            }
        finally:
            source.close()

    def start(self, config: RunnerConfig) -> None:
        self.stop()
        source = self.source_factory(config)
        runtime = self.runtime_factory(config)
        runtime.set_prompt(config.prompt)
        sink = self.sink_factory(config)
        sink.open()
        color = None if config.color == "off" else ColorPipeline(range_mode=config.color)
        fb = FrameBuffer()
        params = TwoAxisControl.preset(config.preset).to_engine_params()
        filler = None
        flow = None
        if config.fill == "warp":
            from streamforge.fill.filler import FrameFiller
            from streamforge.fill.flow import RaftFlow
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            flow = RaftFlow(device=dev, max_side=config.flow_max_side)
            filler = FrameFiller(max_extrap_ms=config.max_extrap_ms)

        def timing(stage: str, ms: float) -> None:
            if stage == "infer":
                with self._lock:
                    self._infer_ms.append(ms)

        def emit(frame) -> None:
            if frame is None:
                return
            out = frame
            if color is not None:
                out = frame.with_tensor(color.apply(frame.tensor))
            with self._lock:
                self._latest_output = out
                self._timestamps.append(time.perf_counter())
            sink.send(out)

        worker = InferenceWorker(source, runtime, fb, params_provider=lambda: params,
                                 on_timing=timing, flow=flow, filler=filler)
        clock = RealtimeClock(config.fps, fb, emit, filler=filler)
        self._worker = worker
        self._clock = clock
        self._sink = sink
        self._config = config
        self._running = True
        worker.start()
        threading.Thread(target=clock.run, daemon=True).start()
        if config.seconds > 0:
            threading.Timer(config.seconds, self.stop).start()

    def stop(self) -> None:
        if self._clock is not None:
            self._clock.stop()
        if self._worker is not None:
            self._worker.stop()
        if self._sink is not None:
            self._sink.close()
        self._clock = None
        self._worker = None
        self._sink = None
        self._running = False

    def status(self) -> dict:
        with self._lock:
            timestamps = list(self._timestamps)
            infer_ms = list(self._infer_ms)
        clock = self._clock
        return {
            "running": self._running,
            "config": asdict(self._config) if self._config else None,
            "jitter_ms": jitter_ms(timestamps),
            "infer_ms_last": infer_ms[-1] if infer_ms else None,
            "emitted": len(timestamps),
            "repeats": clock.repeat_count if clock else 0,
            "filled": clock.filled_count if clock else 0,
        }
```

- [ ] **Step 4: Refactor `scripts/live.py` to call the runner**

Replace the direct orchestration in `scripts/live.py` with CLI parsing that builds `RunnerConfig`, calls `runner.start(config)`, waits until `seconds` elapses, then prints `runner.status()`. Preserve the existing `--test-pattern` branch because it is a sink/color diagnostic and does not need the model runner.

- [ ] **Step 5: Verify**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_runner.py tests/test_clock.py tests/test_worker_fill.py -v`

Expected: runner and existing clock/worker tests pass.

- [ ] **Step 6: Smoke CLI**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe scripts\live.py --sink null --source file --clip "D:\StreamForge\TestFile\DriveVideo.mp4" --seconds 1 --fps 5 --mode img2img --in-res auto`

Expected: script starts and stops cleanly. If model weights are not available in the current environment, it fails during runtime initialization with a clear error rather than before argument parsing.

- [ ] **Step 7: Commit**

```bash
git add src/streamforge/runner.py scripts/live.py tests/test_runner.py
git commit -m "feat(runner): extract reusable live pipeline service"
```

---

## Task 7: FastAPI Backend and Preview Encoding

**Files:**
- Create: `src/streamforge/web/__init__.py`
- Create: `src/streamforge/web/app.py`
- Create: `scripts/web.py`
- Test: `tests/test_web_api.py`

- [ ] **Step 1: Write API tests**

```python
# tests/test_web_api.py
from fastapi.testclient import TestClient

from streamforge.web.app import create_app


class FakeRunner:
    def __init__(self): self.started = False
    def validate(self, config):
        return {"ok": True, "source": {"width": 640, "height": 480}, "aspect": {"crop_direction": "none"}}
    def start(self, config):
        self.started = True
    def stop(self):
        self.started = False
    def status(self):
        return {"running": self.started, "emitted": 0, "repeats": 0, "filled": 0}


def test_validate_endpoint():
    app = create_app(FakeRunner())
    client = TestClient(app)
    res = client.post("/api/validate", json={"source_type": "webcam", "source_name": "0"})
    assert res.status_code == 200
    assert res.json()["ok"] is True


def test_start_status_stop():
    runner = FakeRunner()
    client = TestClient(create_app(runner))
    assert client.post("/api/run/start", json={"source_type": "webcam", "source_name": "0"}).status_code == 200
    assert client.get("/api/status").json()["running"] is True
    assert client.post("/api/run/stop").status_code == 200
    assert client.get("/api/status").json()["running"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_web_api.py -v`

Expected: FAIL with `ModuleNotFoundError: streamforge.web`.

- [ ] **Step 3: Implement FastAPI app**

```python
# src/streamforge/web/__init__.py
"""StreamForge local operator console web app."""
```

```python
# src/streamforge/web/app.py
from __future__ import annotations

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

    def to_config(self) -> RunnerConfig:
        allowed = {f.name for f in fields(RunnerConfig)}
        data: dict[str, Any] = self.model_dump()
        return RunnerConfig(**{k: v for k, v in data.items() if k in allowed})


def create_app(runner: StreamForgeRunner | None = None) -> FastAPI:
    app = FastAPI(title="StreamForge Operator Console")
    app.state.runner = runner or StreamForgeRunner()
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

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

    @app.get("/api/status")
    def status():
        return app.state.runner.status()

    @app.get("/preview/input.jpg")
    def preview_input():
        return Response(status_code=204)

    @app.get("/preview/output.jpg")
    def preview_output():
        return Response(status_code=204)

    return app


app = create_app()
```

```python
# scripts/web.py
from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("streamforge.web.app:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify API tests**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_web_api.py -v`

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/streamforge/web scripts/web.py tests/test_web_api.py
git commit -m "feat(web): add FastAPI operator console API"
```

---

## Task 8: Operator Console Static UI

**Files:**
- Create: `src/streamforge/web/static/index.html`
- Create: `src/streamforge/web/static/styles.css`
- Create: `src/streamforge/web/static/app.js`
- Test: `tests/test_web_static.py`

- [ ] **Step 1: Write static asset test**

```python
# tests/test_web_static.py
from pathlib import Path


def test_operator_console_assets_exist_and_reference_api():
    root = Path("src/streamforge/web/static")
    html = (root / "index.html").read_text()
    js = (root / "app.js").read_text()
    css = (root / "styles.css").read_text()
    assert "StreamForge" in html
    assert "/api/validate" in js
    assert "/api/run/start" in js
    assert "/api/status" in js
    assert "aspect-ratio" in css
```

- [ ] **Step 2: Run to verify failure**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_web_static.py -v`

Expected: FAIL because static assets do not exist.

- [ ] **Step 3: Add `index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>StreamForge Operator Console</title>
  <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
  <main class="console">
    <aside class="rail">
      <h1>StreamForge</h1>
      <label>Input Type
        <select id="sourceType">
          <option value="webcam">Webcam</option>
          <option value="ndi">NDI</option>
          <option value="spout">Spout</option>
          <option value="file">File</option>
        </select>
      </label>
      <label>Source
        <input id="sourceName" value="0">
      </label>
      <label>Aspect Policy
        <select id="inRes">
          <option value="auto">Auto preserve</option>
          <option value="native">Native</option>
          <option value="384x288">4:3 384x288</option>
          <option value="384x224">16:9 384x224 crop</option>
        </select>
      </label>
      <label>Prompt
        <textarea id="prompt">vivid oil painting, thick impasto brushstrokes</textarea>
      </label>
      <label>Preset
        <select id="preset">
          <option>PRESERVE</option>
          <option>SUBTLE</option>
          <option selected>BALANCED</option>
          <option>FOLLOW</option>
          <option>FORCE</option>
        </select>
      </label>
      <div class="buttons">
        <button id="validate">Validate</button>
        <button id="start">Start</button>
        <button id="stop">Stop</button>
      </div>
    </aside>
    <section class="previews">
      <div class="preview-card">
        <header><span>Input</span><b id="inputMeta">not validated</b></header>
        <div class="preview-box"><img id="inputPreview" alt="Input preview"></div>
      </div>
      <div class="preview-card">
        <header><span>Output</span><b id="outputMeta">idle</b></header>
        <div class="preview-box crop"><img id="outputPreview" alt="Output preview"></div>
      </div>
    </section>
    <aside class="health">
      <h2>Run Health</h2>
      <dl>
        <dt>State</dt><dd id="state">idle</dd>
        <dt>Emitted</dt><dd id="emitted">0</dd>
        <dt>Repeats</dt><dd id="repeats">0</dd>
        <dt>Filled</dt><dd id="filled">0</dd>
        <dt>Jitter</dt><dd id="jitter">0.0 ms</dd>
        <dt>Inference</dt><dd id="infer">none</dd>
      </dl>
      <pre id="messages"></pre>
    </aside>
  </main>
  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Add `styles.css`**

```css
:root {
  color-scheme: dark;
  --bg: #111315;
  --panel: #1b1f23;
  --line: #343a40;
  --text: #eef1f3;
  --muted: #a8b0b7;
  --accent: #32d6a0;
  --warn: #f5ba4f;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: Inter, Segoe UI, Arial, sans-serif; }
.console { min-height: 100vh; display: grid; grid-template-columns: 300px 1fr 280px; gap: 1px; background: var(--line); }
.rail, .health, .previews { background: var(--bg); padding: 18px; }
h1, h2 { margin: 0 0 16px; letter-spacing: 0; }
label { display: grid; gap: 6px; margin-bottom: 12px; color: var(--muted); font-size: 12px; }
input, select, textarea, button { width: 100%; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); color: var(--text); padding: 9px; font: inherit; }
textarea { min-height: 84px; resize: vertical; }
.buttons { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
button { cursor: pointer; }
button:hover { border-color: var(--accent); }
.previews { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-content: start; }
.preview-card { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); overflow: hidden; }
.preview-card header { display: flex; justify-content: space-between; padding: 10px 12px; border-bottom: 1px solid var(--line); color: var(--muted); }
.preview-box { aspect-ratio: 16 / 9; display: grid; place-items: center; background: #070809; position: relative; overflow: hidden; }
.preview-box img { width: 100%; height: 100%; object-fit: cover; }
.preview-box.crop::before, .preview-box.crop::after { content: ""; position: absolute; left: 0; right: 0; height: 18px; background: rgba(245, 186, 79, .25); pointer-events: none; }
.preview-box.crop::before { top: 0; }
.preview-box.crop::after { bottom: 0; }
dl { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; margin: 0; }
dt { color: var(--muted); }
dd { margin: 0; text-align: right; }
pre { white-space: pre-wrap; color: var(--warn); }
@media (max-width: 1050px) { .console { grid-template-columns: 1fr; } .previews { grid-template-columns: 1fr; } }
```

- [ ] **Step 5: Add `app.js`**

```javascript
function config() {
  return {
    source_type: document.getElementById("sourceType").value,
    source_name: document.getElementById("sourceName").value,
    in_res: document.getElementById("inRes").value,
    prompt: document.getElementById("prompt").value,
    preset: document.getElementById("preset").value,
    sink: "null",
    mode: "img2img",
    fps: 30,
    seconds: 0
  };
}

function showMessage(value) {
  document.getElementById("messages").textContent =
    typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(`${url} failed: ${res.status}`);
  return res.json();
}

async function validate() {
  const data = await postJson("/api/validate", config());
  showMessage(data);
  if (data.ok && data.source) {
    document.getElementById("inputMeta").textContent = `${data.source.width}x${data.source.height}`;
  }
}

async function start() {
  await postJson("/api/run/start", config());
  showMessage("started");
  await refreshStatus();
}

async function stop() {
  await postJson("/api/run/stop", {});
  showMessage("stopped");
  await refreshStatus();
}

async function refreshStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  document.getElementById("state").textContent = data.running ? "running" : "idle";
  document.getElementById("emitted").textContent = data.emitted ?? 0;
  document.getElementById("repeats").textContent = data.repeats ?? 0;
  document.getElementById("filled").textContent = data.filled ?? 0;
  document.getElementById("jitter").textContent = `${(data.jitter_ms ?? 0).toFixed(2)} ms`;
  document.getElementById("infer").textContent =
    data.infer_ms_last == null ? "none" : `${data.infer_ms_last.toFixed(1)} ms`;
  const stamp = Date.now();
  document.getElementById("inputPreview").src = `/preview/input.jpg?t=${stamp}`;
  document.getElementById("outputPreview").src = `/preview/output.jpg?t=${stamp}`;
}

document.getElementById("validate").addEventListener("click", () => validate().catch(err => showMessage(err.message)));
document.getElementById("start").addEventListener("click", () => start().catch(err => showMessage(err.message)));
document.getElementById("stop").addEventListener("click", () => stop().catch(err => showMessage(err.message)));
window.setInterval(() => refreshStatus().catch(() => {}), 1000);
refreshStatus().catch(() => {});
```

- [ ] **Step 6: Verify static tests**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_web_static.py -v`

Expected: `1 passed`.

- [ ] **Step 7: Commit**

```bash
git add src/streamforge/web/static tests/test_web_static.py
git commit -m "feat(web): add operator console static UI"
```

---

## Task 9: Preview JPEG Endpoints

**Files:**
- Modify: `src/streamforge/runner.py`
- Modify: `src/streamforge/web/app.py`
- Test: `tests/test_preview_encoding.py`

- [ ] **Step 1: Write preview tests**

```python
# tests/test_preview_encoding.py
import torch
from PIL import Image

from streamforge.frame import GpuFrame
from streamforge.runner import frame_to_jpeg


def test_frame_to_jpeg_returns_valid_image_bytes():
    frame = GpuFrame(tensor=torch.zeros(1, 3, 4, 6), seq=0, pts=0.0, width=6, height=4)
    data = frame_to_jpeg(frame)
    assert data.startswith(b"\xff\xd8")
    img = Image.open(__import__("io").BytesIO(data))
    assert img.size == (6, 4)
```

- [ ] **Step 2: Run to verify failure**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_preview_encoding.py -v`

Expected: FAIL because `frame_to_jpeg` does not exist.

- [ ] **Step 3: Add JPEG helper and latest preview getters**

Add to `src/streamforge/runner.py`:

```python
def frame_to_jpeg(frame, quality: int = 80) -> bytes:
    import io
    from PIL import Image
    arr = (frame.tensor[0].detach().clamp(0, 1).float().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
```

Add methods to `StreamForgeRunner`:

```python
    def latest_input_jpeg(self) -> bytes | None:
        with self._lock:
            frame = self._latest_input
        return frame_to_jpeg(frame) if frame is not None else None

    def latest_output_jpeg(self) -> bytes | None:
        with self._lock:
            frame = self._latest_output
        return frame_to_jpeg(frame) if frame is not None else None
```

Update `validate()` to store `_latest_input = frame` under the lock. Update the worker/pipeline path later if input preview during a live run needs every source frame; MVP validation preview is acceptable first.

- [ ] **Step 4: Patch preview endpoints**

In `src/streamforge/web/app.py`, replace preview endpoint bodies:

```python
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
```

- [ ] **Step 5: Verify**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_preview_encoding.py tests/test_web_api.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/streamforge/runner.py src/streamforge/web/app.py tests/test_preview_encoding.py
git commit -m "feat(web): serve latest input and output preview JPEGs"
```

---

## Task 10: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/TESTING_OUTPUTS.md`

- [ ] **Step 1: Add README section**

Add this section to `README.md`:

```markdown
## Operator Console

Run the local web console:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\web.py
```

Open `http://127.0.0.1:8765`. The console validates webcam, NDI, or Spout inputs before starting
the live pipeline. Aspect handling is fill-and-crop only; StreamForge does not stretch source frames.
Use `Auto preserve` for source-ratio-safe internal dimensions, or choose an explicit canvas when
you want a cropped 16:9 or square output.
```

- [ ] **Step 2: Add testing docs**

Add to `docs/TESTING_OUTPUTS.md`:

```markdown
## Web console validation

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\web.py
```

Open `http://127.0.0.1:8765`, choose an input, and click Validate. For the 640x480 test clip or a
4:3 webcam, the source preview should remain 4:3 unless an explicit output canvas is selected. If
the selected canvas differs, the UI should show a crop overlay instead of stretching.
```

- [ ] **Step 3: Run the full pure suite**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests pass.

- [ ] **Step 4: Start the web server**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe scripts\web.py`

Expected: Uvicorn starts at `http://127.0.0.1:8765`.

- [ ] **Step 5: Browser smoke test**

Open `http://127.0.0.1:8765` in the in-app browser. Verify:

- the Operator Console loads,
- Validate calls `/api/validate`,
- status polling updates without console errors,
- preview boxes stay fixed and do not overlap controls.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/TESTING_OUTPUTS.md
git commit -m "docs(web): document aspect-safe operator console"
```

---

## Final Verification Commands

Run these before claiming implementation complete:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\restyle_clip.py --count 1 --every 1 --max-side 512 --target source --preset PRESERVE
.\.venv\Scripts\python.exe scripts\web.py
```

For the second command, expected output includes a 4:3 target such as `512x384` for the 640x480 test clip. For the web command, manually stop the server after the browser smoke test.

## Self-Review

- **Spec coverage:** Task 2 covers core aspect utilities. Task 3 covers runtime and validation scripts. Task 4 covers webcam/NDI/Spout sources. Task 5 covers NDI output metadata. Task 6 covers reusable runner service. Tasks 7-9 cover API, static Operator Console, and preview validation. Task 10 covers docs and full verification.
- **Placeholder scan:** The plan contains no unresolved markers, incomplete sections, or unspecified test commands.
- **Type consistency:** `Size(width, height)`, `FitMode.FILL_CROP`, `fit_tensor(image_bchw, target, mode, antialias)`, `RunnerConfig`, and `StreamForgeRunner` are used consistently across tests, API, and CLI.
- **Risk handling:** Optional NDI/Spout runtime imports are isolated to their source modules and source listing route; fake tests cover behavior without requiring installed sender runtimes.

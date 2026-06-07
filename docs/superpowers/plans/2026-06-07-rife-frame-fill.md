# RIFE Frame-Fill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn StreamForge's ~12–14 fresh fps AI restyle into smooth 30/50 fps output by forward-warping the latest styled frame along source motion (optical-flow extrapolation), replacing today's repeated-frame judder — with zero added latency.

**Architecture:** The `InferenceWorker` (its own thread) measures motion between consecutive **source** frames with torchvision `raft_small` and hands a `FrameFiller` an anchor = (latest styled tensor, flow field, timestamp). On each **stale** clock tick the `RealtimeClock` asks the filler for a frame extrapolated to *now* via `grid_sample`, feeding the existing `select_fill(ai, warped, held, raw)` `"warped"` tier. Interpolation was rejected (adds latency); this is extrapolation only.

**Tech Stack:** PyTorch 2.11, torchvision 0.26 (`raft_small`, BSD weights), the existing `streamforge` package. Spec: `docs/superpowers/specs/2026-06-07-rife-frame-fill-design.md`.

---

## Context You Need (read before Task 1)

- **Run Python as** `.\.venv\Scripts\python.exe` from `D:\StreamForge`. Tests need `src` on the path:
  PowerShell `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest ...`.
- **Existing output path** (already built):
  - `src/streamforge/frame.py`: `GpuFrame(tensor, seq, pts, width, height, colorspace)` frozen dataclass with `.with_tensor(t) -> GpuFrame`. Tensors are NCHW float in [0,1] on device.
  - `src/streamforge/clock.py`: `FrameBuffer` (single-slot, `publish` / `get_with_freshness() -> Freshness(value, is_fresh)`); `select_fill(ai, warped, held, raw) -> FillResult(value, source)` with priority ai>warped>held>raw; `RealtimeClock(fps, frame_buffer, emit)` whose `_tick_once` currently emits the latest value and counts repeats. `run_for_ticks(n)` is the synchronous test mode.
  - `src/streamforge/worker.py`: `InferenceWorker(source, runtime, frame_buffer, params_provider, on_timing=None)` runs `runtime.restyle(frame.tensor, params)` on a thread and publishes `frame.with_tensor(out)`.
  - `scripts/live.py`: wires source → worker → clock → `emit` (color + sink).
- **Flow/displacement model (important):** `raft_small` returns flow = full displacement between two consecutive frames at flow-res, in pixels. `warp_forward(image, flow, dt, fps, max_disp)` computes `disp = flow · fps · dt` where **`fps` is the AI/anchor rate** (frames per second at which styled anchors are produced, ~12–14). So `flow · fps` = displacement-per-second, `× dt` = displacement at `dt` seconds past the anchor. `FrameFiller` EMA-refines `fps` from real anchor timestamps.
- **`models/` and `out/` are gitignored.** `.gitignore` has no inline comments (comments on their own lines only).
- The branch is `feat/rife-frame-fill`.

## File Structure

| File | Responsibility |
|---|---|
| `src/streamforge/fill/__init__.py` | package marker |
| `src/streamforge/fill/warp.py` | `warp_forward(...)` — pure forward-warp via `grid_sample`; resizes/scales flow to image res. No model, no GPU required. |
| `src/streamforge/fill/filler.py` | `FrameFiller` — thread-safe anchor (styled, flow, t); EMA anchor-rate; `fill(now)` warps or returns `None` past the time cap. |
| `src/streamforge/fill/flow.py` | `RaftFlow` — lazy `raft_small` (fp16-capable, eval, no grad); `estimate(prev, cur)` at flow-res. |
| `src/streamforge/clock.py` *(modify)* | `RealtimeClock(..., filler=None)`; wire `select_fill`; add `_held`, `filled_count`. |
| `src/streamforge/worker.py` *(modify)* | optional `flow`/`filler`; keep `prev_src`; set anchor each AI frame. |
| `scripts/live.py` *(modify)* | `--fill {off,warp}` (default off), `--max-extrap-ms` (default 120), `--flow-max-side` (default 384); build + inject. |
| `tests/test_warp.py`, `tests/test_filler.py`, `tests/test_clock_fill.py`, `tests/test_worker_fill.py`, `tests/test_flow.py` | unit + smoke tests |

---

## Task 1: `warp_forward` pure function

**Files:**
- Create: `src/streamforge/fill/__init__.py`
- Create: `src/streamforge/fill/warp.py`
- Test: `tests/test_warp.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_warp.py
import torch
from streamforge.fill.warp import warp_forward


def test_identity_when_flow_zero():
    img = torch.rand(1, 3, 8, 10)
    flow = torch.zeros(1, 2, 8, 10)
    out = warp_forward(img, flow, dt=1.0, fps=1.0)
    assert torch.allclose(out, img, atol=1e-5)


def test_positive_x_flow_shifts_content_right():
    # a bright vertical line at column 2; +x flow moves content to the right
    img = torch.zeros(1, 3, 4, 8); img[..., 2] = 1.0
    flow = torch.zeros(1, 2, 4, 8); flow[:, 0] = 2.0   # +2 px/frame in x
    out = warp_forward(img, flow, dt=1.0, fps=1.0)      # disp = 2*1*1 = 2 px
    col = out[0, 0].sum(dim=0).argmax().item()
    assert col == 4                                     # 2 + 2


def test_max_disp_clamps():
    img = torch.zeros(1, 3, 4, 16); img[..., 2] = 1.0
    flow = torch.zeros(1, 2, 4, 16); flow[:, 0] = 10.0
    out = warp_forward(img, flow, dt=1.0, fps=1.0, max_disp=3.0)
    col = out[0, 0].sum(dim=0).argmax().item()
    assert col == 5                                     # 2 + 3 (clamped)


def test_flow_resized_to_image_res():
    # flow at half the image resolution must be upscaled AND magnitude-scaled
    img = torch.zeros(1, 3, 4, 8); img[..., 1] = 1.0
    flow = torch.zeros(1, 2, 2, 4); flow[:, 0] = 1.0    # 1 px at flow-res (width 4)
    out = warp_forward(img, flow, dt=1.0, fps=1.0)      # scaled by 8/4=2 -> disp 2 px
    col = out[0, 0].sum(dim=0).argmax().item()
    assert col == 3                                     # 1 + 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_warp.py -v`
Expected: FAIL — `ModuleNotFoundError: streamforge.fill.warp`.

- [ ] **Step 3: Implement**

```python
# src/streamforge/fill/__init__.py
"""Motion-extrapolation frame-fill (design §6.0 'warped' tier)."""
```

```python
# src/streamforge/fill/warp.py
"""Forward-warp an image along an optical-flow field via grid_sample (pure, testable).

disp = flow * fps * dt, where `fps` is the AI/anchor rate and `flow` is the full inter-anchor
displacement (pixels at flow-res). Content moves by +disp, so we backward-sample by -disp.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def warp_forward(image: torch.Tensor, flow: torch.Tensor, dt: float, fps: float,
                 max_disp: float | None = None) -> torch.Tensor:
    b, c, h, w = image.shape
    fh, fw = flow.shape[-2], flow.shape[-1]
    if (fh, fw) != (h, w):
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False).clone()
        flow[:, 0] *= w / fw   # x magnitude scales with width ratio
        flow[:, 1] *= h / fh   # y magnitude scales with height ratio
    disp = flow * (fps * dt)
    if max_disp is not None:
        disp = disp.clamp(-max_disp, max_disp)
    yy, xx = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype), indexing="ij")
    src_x = xx[None] - disp[:, 0]   # backward sample
    src_y = yy[None] - disp[:, 1]
    gx = 2.0 * src_x / max(w - 1, 1) - 1.0
    gy = 2.0 * src_y / max(h - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)   # [B,H,W,2]
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
```

- [ ] **Step 4: Run to verify they pass**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_warp.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/streamforge/fill/__init__.py src/streamforge/fill/warp.py tests/test_warp.py
git commit -m "feat(fill): warp_forward optical-flow warp (pure, tested)"
```

---

## Task 2: `FrameFiller` thread-safe extrapolator

**Files:**
- Create: `src/streamforge/fill/filler.py`
- Test: `tests/test_filler.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_filler.py
import torch
from streamforge.fill.filler import FrameFiller


def test_none_before_anchor():
    f = FrameFiller(max_extrap_ms=120, fps=10)
    assert f.fill(now=1.0) is None


def test_warps_within_cap():
    f = FrameFiller(max_extrap_ms=200, fps=10)
    img = torch.zeros(1, 3, 4, 8); img[..., 2] = 1.0
    flow = torch.zeros(1, 2, 4, 8); flow[:, 0] = 2.0
    f.set_anchor(img, flow, t=0.0)
    out = f.fill(now=0.1)                 # disp = 2 * 10 * 0.1 = 2 px
    assert out is not None
    assert out[0, 0].sum(dim=0).argmax().item() == 4


def test_none_past_time_cap():
    f = FrameFiller(max_extrap_ms=120, fps=10)
    f.set_anchor(torch.zeros(1, 3, 4, 8), torch.zeros(1, 2, 4, 8), t=0.0)
    assert f.fill(now=0.2) is None        # 200ms > 120ms cap


def test_ema_refines_fps_from_anchor_times():
    f = FrameFiller(max_extrap_ms=500, fps=100.0)   # bad nominal
    img = torch.zeros(1, 3, 4, 8); flow = torch.zeros(1, 2, 4, 8)
    f.set_anchor(img, flow, t=0.0)
    f.set_anchor(img, flow, t=0.1)        # 0.1s interval -> 10 fps observed
    assert 9.0 < f.fps < 100.0            # pulled toward 10 from 100 by the EMA
```

- [ ] **Step 2: Run to verify they fail**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_filler.py -v`
Expected: FAIL — `ModuleNotFoundError: streamforge.fill.filler`.

- [ ] **Step 3: Implement**

```python
# src/streamforge/fill/filler.py
"""FrameFiller — thread-safe latest-anchor + forward extrapolation (the motion analog of
FrameBuffer). The worker writes anchors; the clock reads warped frames. `fps` is the AI/anchor
rate, seeded by the nominal value and EMA-refined from real anchor timestamps."""
from __future__ import annotations
import threading
import torch
from streamforge.fill.warp import warp_forward


class FrameFiller:
    def __init__(self, max_extrap_ms: float = 120.0, fps: float = 13.0,
                 max_disp: float | None = None, ema: float = 0.5):
        self.max_extrap_s = max_extrap_ms / 1000.0
        self.fps = float(fps)
        self.max_disp = max_disp
        self._ema = ema
        self._lock = threading.Lock()
        self._styled: torch.Tensor | None = None
        self._flow: torch.Tensor | None = None
        self._t: float | None = None

    def set_anchor(self, styled: torch.Tensor, flow: torch.Tensor, t: float) -> None:
        with self._lock:
            if self._t is not None:
                dt = t - self._t
                if dt > 1e-4:
                    inst = 1.0 / dt
                    self.fps = self._ema * inst + (1.0 - self._ema) * self.fps
            self._styled, self._flow, self._t = styled, flow, t

    def fill(self, now: float) -> torch.Tensor | None:
        with self._lock:
            styled, flow, t, fps = self._styled, self._flow, self._t, self.fps
        if styled is None or flow is None or t is None:
            return None
        dt = now - t
        if dt < 0 or dt > self.max_extrap_s:
            return None
        return warp_forward(styled, flow, dt, fps, self.max_disp)
```

- [ ] **Step 4: Run to verify they pass**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_filler.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/streamforge/fill/filler.py tests/test_filler.py
git commit -m "feat(fill): FrameFiller thread-safe forward extrapolator"
```

---

## Task 3: Wire fill into `RealtimeClock`

**Files:**
- Modify: `src/streamforge/clock.py` (the `RealtimeClock` class)
- Test: `tests/test_clock_fill.py`

The new behavior: a fresh tick emits the AI frame and records it as `held`; a stale tick asks the filler for a warped tensor, wraps it in the held frame's metadata, and falls back to `held` when the filler returns `None`. With `filler=None` behavior is identical to today (the existing `tests/test_clock.py` must stay green).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_clock_fill.py
import torch
from streamforge.clock import FrameBuffer, RealtimeClock
from streamforge.frame import GpuFrame


class _StubFiller:
    def __init__(self, out): self._out = out; self.calls = 0
    def fill(self, now): self.calls += 1; return self._out


def _frame(val=0.0):
    return GpuFrame(tensor=torch.full((1, 3, 2, 2), val), seq=1, pts=0.0, width=2, height=2)


def test_fresh_tick_emits_ai_not_warped():
    fb = FrameBuffer(); emitted = []
    f = _frame(0.0); fb.publish(f)
    clk = RealtimeClock(30, fb, emitted.append, filler=_StubFiller(torch.ones(1, 3, 2, 2)))
    clk.run_for_ticks(1)
    assert emitted[0] is f and clk.filled_count == 0


def test_stale_tick_emits_warped_with_held_metadata():
    fb = FrameBuffer(); emitted = []
    f = _frame(0.0); fb.publish(f)
    warped_t = torch.ones(1, 3, 2, 2)
    clk = RealtimeClock(30, fb, emitted.append, filler=_StubFiller(warped_t))
    clk.run_for_ticks(2)                       # tick1 fresh, tick2 stale
    assert emitted[0] is f
    assert torch.equal(emitted[1].tensor, warped_t)
    assert emitted[1].seq == f.seq             # warped frame carries held metadata
    assert clk.filled_count == 1


def test_stale_tick_holds_when_filler_returns_none():
    fb = FrameBuffer(); emitted = []
    f = _frame(0.0); fb.publish(f)
    clk = RealtimeClock(30, fb, emitted.append, filler=_StubFiller(None))
    clk.run_for_ticks(2)
    assert emitted[1] is f                      # held
    assert clk.filled_count == 0 and clk.repeat_count == 1


def test_no_filler_is_backcompat_repeat():
    fb = FrameBuffer(); emitted = []
    f = _frame(0.0); fb.publish(f)
    clk = RealtimeClock(30, fb, emitted.append)   # filler=None
    clk.run_for_ticks(2)
    assert emitted[0] is f and emitted[1] is f
    assert clk.repeat_count == 1 and clk.filled_count == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_clock_fill.py -v`
Expected: FAIL — `TypeError` (`RealtimeClock` has no `filler`) / `AttributeError` (`filled_count`).

- [ ] **Step 3: Implement — replace the `RealtimeClock` class in `src/streamforge/clock.py`**

Replace the existing `class RealtimeClock` (the `__init__` and `_tick_once`) with:

```python
class RealtimeClock:
    """Emits one frame per tick at the target cadence. On a stale tick (no fresh AI frame), an
    optional FrameFiller supplies a motion-extrapolated frame; otherwise the last frame is held."""

    def __init__(self, fps: int, frame_buffer: FrameBuffer, emit: Callable[[Any], None],
                 filler: Any = None):
        self.period = 1.0 / fps
        self.fb = frame_buffer
        self.emit = emit
        self.filler = filler
        self.repeat_count = 0
        self.filled_count = 0
        self._held: Any = None
        self._running = False

    def _tick_once(self) -> None:
        now = time.perf_counter()
        fr = self.fb.get_with_freshness()
        if fr.is_fresh and fr.value is not None:
            self._held = fr.value
            self.emit(fr.value)
            return
        self.repeat_count += 1
        warped_t = self.filler.fill(now) if self.filler is not None else None
        warped = (self._held.with_tensor(warped_t)
                  if warped_t is not None and self._held is not None else None)
        result = select_fill(ai=None, warped=warped, held=self._held, raw=None)
        if result.source == "warped":
            self.filled_count += 1
        self.emit(result.value)
```

Keep the existing `run_for_ticks`, `run`, and `stop` methods unchanged.

- [ ] **Step 4: Run to verify new + existing clock tests pass**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_clock_fill.py tests/test_clock.py -v`
Expected: all passed (4 new + 4 existing).

- [ ] **Step 5: Commit**

```bash
git add src/streamforge/clock.py tests/test_clock_fill.py
git commit -m "feat(fill): wire FrameFiller into RealtimeClock (warped tier)"
```

---

## Task 4: `RaftFlow` optical-flow estimator

**Files:**
- Create: `src/streamforge/fill/flow.py`
- Test: `tests/test_flow.py`

`raft_small` returns a pyramid of predictions; the last is the final flow `[1,2,fh,fw]` in flow-res pixels. Inputs are normalized to [-1,1]; both spatial dims must be divisible by 8.

- [ ] **Step 1: Write the smoke test (GPU; skips without CUDA)**

```python
# tests/test_flow.py
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RAFT needs CUDA")
def test_raft_detects_rightward_shift():
    from streamforge.fill.flow import RaftFlow
    base = torch.rand(1, 3, 128, 128)
    shifted = torch.zeros_like(base); shifted[..., 8:] = base[..., :-8]   # content moved +8 in x
    rf = RaftFlow(device="cuda", max_side=128)
    flow = rf.estimate(base, shifted)                  # prev=base, cur=shifted
    assert flow.shape[1] == 2
    assert flow[:, 0, 32:96, 32:96].mean().item() > 0  # central x-flow is positive
```

- [ ] **Step 2: Run to verify it fails**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_flow.py -v`
Expected: FAIL — `ModuleNotFoundError: streamforge.fill.flow` (or SKIP if no CUDA — in that case verify failure on a CUDA box; the import error still surfaces at collection).

- [ ] **Step 3: Implement**

```python
# src/streamforge/fill/flow.py
"""Optical flow via torchvision raft_small (BSD weights). Estimates flow between two source frames
at a low flow-res (longest side <= max_side, dims divisible by 8) to stay cheap; runs on the worker
thread at AI rate. Returns flow in flow-res pixels; warp_forward rescales to the styled image res."""
from __future__ import annotations
import torch
import torch.nn.functional as F


class RaftFlow:
    def __init__(self, device: str = "cuda", max_side: int = 384):
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        self.device = device
        self.max_side = max_side
        self.model = raft_small(weights=Raft_Small_Weights.C_T_V2, progress=False).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        scale = min(1.0, self.max_side / max(h, w))
        fh = max(8, int(round(h * scale / 8)) * 8)
        fw = max(8, int(round(w * scale / 8)) * 8)
        x = F.interpolate(x.to(self.device), size=(fh, fw), mode="bilinear", align_corners=False)
        return x * 2.0 - 1.0   # [0,1] -> [-1,1], RAFT convention

    @torch.no_grad()
    def estimate(self, prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        return self.model(self._prep(prev), self._prep(cur))[-1]   # [1,2,fh,fw]
```

- [ ] **Step 4: Run to verify it passes (on a CUDA box)**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_flow.py -v`
Expected: 1 passed (downloads raft_small weights on first run; or 1 skipped without CUDA).

- [ ] **Step 5: Commit**

```bash
git add src/streamforge/fill/flow.py tests/test_flow.py
git commit -m "feat(fill): RaftFlow raft_small optical-flow estimator"
```

---

## Task 5: Integrate flow + filler into `InferenceWorker`

**Files:**
- Modify: `src/streamforge/worker.py`
- Test: `tests/test_worker_fill.py`

The worker keeps the previous source frame; once it has two, it estimates flow and sets the filler anchor with the **styled output tensor** (warp target) + the **source flow** (motion). Flow only runs when both `flow` and `filler` are provided, so the existing call sites are unaffected.

- [ ] **Step 1: Write the failing test (stubs; no GPU/model)**

```python
# tests/test_worker_fill.py
import torch
from streamforge.clock import FrameBuffer
from streamforge.frame import GpuFrame
from streamforge.worker import InferenceWorker


class _StubSource:
    def __init__(self, frames): self._frames = frames; self._i = 0
    def open(self): pass
    def read(self):
        if self._i >= len(self._frames):
            return None
        f = self._frames[self._i]; self._i += 1; return f
    def close(self): pass


class _StubRuntime:
    def restyle(self, tensor, params): return tensor * 0.5


class _StubFlow:
    def __init__(self): self.calls = 0
    def estimate(self, prev, cur): self.calls += 1; return torch.zeros(1, 2, 2, 2)


class _RecordingFiller:
    def __init__(self): self.anchors = []
    def set_anchor(self, styled, flow, t): self.anchors.append((styled, flow, t))


def _frame(i):
    return GpuFrame(tensor=torch.rand(1, 3, 4, 4), seq=i, pts=float(i), width=4, height=4)


def test_worker_sets_anchor_from_second_frame():
    frames = [_frame(0), _frame(1), _frame(2)]
    fb = FrameBuffer(); flow = _StubFlow(); filler = _RecordingFiller()
    w = InferenceWorker(_StubSource(frames), _StubRuntime(), fb,
                        params_provider=lambda: None, flow=flow, filler=filler)
    w.start(); w.stop()
    assert flow.calls == 2                       # frames 2 and 3 (prev available)
    assert len(filler.anchors) == 2
    styled, vel, t = filler.anchors[0]
    assert styled.shape == (1, 3, 4, 4)          # the styled OUTPUT tensor
    assert vel.shape == (1, 2, 2, 2)


def test_worker_without_fill_still_publishes():
    frames = [_frame(0)]
    fb = FrameBuffer()
    w = InferenceWorker(_StubSource(frames), _StubRuntime(), fb, params_provider=lambda: None)
    w.start(); w.stop()
    assert fb.get_latest() is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_worker_fill.py -v`
Expected: FAIL — `InferenceWorker.__init__` got an unexpected keyword `flow`.

- [ ] **Step 3: Implement — update `InferenceWorker` in `src/streamforge/worker.py`**

Change `__init__` to accept `flow=None, filler=None`, store them and `self._prev_src = None`. Replace `_loop` so it estimates flow and sets the anchor when both are present:

```python
    def __init__(
        self,
        source,
        runtime,
        frame_buffer: FrameBuffer,
        params_provider: Callable[[], EngineParams],
        on_timing: Optional[Callable[[str, float], None]] = None,
        flow=None,
        filler=None,
    ):
        self.source = source
        self.runtime = runtime
        self.fb = frame_buffer
        self.params_provider = params_provider
        self.on_timing = on_timing
        self.flow = flow
        self.filler = filler
        self._prev_src = None
        self._t: Optional[threading.Thread] = None
        self._running = False

    def _loop(self) -> None:
        import torch

        self.source.open()
        try:
            while self._running:
                f = self.source.read()
                if f is None:
                    break
                t0 = time.perf_counter()
                out = self.runtime.restyle(f.tensor, self.params_provider())
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000.0
                if self.flow is not None and self.filler is not None:
                    if self._prev_src is not None:
                        velocity = self.flow.estimate(self._prev_src, f.tensor)
                        self.filler.set_anchor(out, velocity, time.perf_counter())
                    self._prev_src = f.tensor
                self.fb.publish(f.with_tensor(out))
                if self.on_timing:
                    self.on_timing("infer", ms)
        finally:
            self.source.close()
```

`start()` and `stop()` are unchanged.

- [ ] **Step 4: Run to verify it passes**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_worker_fill.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/streamforge/worker.py tests/test_worker_fill.py
git commit -m "feat(fill): InferenceWorker computes source flow + sets anchor"
```

---

## Task 6: Wire `--fill` into `scripts/live.py`

**Files:**
- Modify: `scripts/live.py`

- [ ] **Step 1: Add the CLI args**

In `main()`'s argparse block (after the `--tiny-vae` arg), add:

```python
    ap.add_argument("--fill", choices=["off", "warp"], default="off",
                    help="output frame-fill: 'warp' = RAFT motion-extrapolation of stale frames")
    ap.add_argument("--max-extrap-ms", type=float, default=120.0,
                    help="cap how far past the last AI frame to extrapolate (ms)")
    ap.add_argument("--flow-max-side", type=int, default=384,
                    help="flow-res longest side for RAFT (smaller = faster)")
```

- [ ] **Step 2: Build flow + filler and inject**

Replace the two lines that build the worker and clock:

```python
    fb = FrameBuffer()
    worker = InferenceWorker(source, runtime, fb, params_provider=lambda: params)
    clk = RealtimeClock(args.fps, fb, emit)
```

with:

```python
    fb = FrameBuffer()
    flow = filler = None
    if args.fill == "warp":
        from streamforge.fill.filler import FrameFiller
        try:
            from streamforge.fill.flow import RaftFlow
            flow = RaftFlow(device=dev, max_side=args.flow_max_side)
            filler = FrameFiller(max_extrap_ms=args.max_extrap_ms)
            print(f"[fill] motion-extrapolation ON (max_extrap={args.max_extrap_ms}ms, "
                  f"flow_max_side={args.flow_max_side})")
        except Exception as e:
            print(f"[fill] disabled (flow init failed): {e}")
            flow = filler = None
    worker = InferenceWorker(source, runtime, fb, params_provider=lambda: params,
                             flow=flow, filler=filler)
    clk = RealtimeClock(args.fps, fb, emit, filler=filler)
```

- [ ] **Step 3: Report filled frames**

Replace the final summary `print(...)` (the `done: emitted=...` line) with:

```python
    fresh = len(timestamps) - clk.repeat_count
    print(f"done: emitted={len(timestamps)} jitter={jitter_ms(timestamps):.2f}ms "
          f"repeats={clk.repeat_count} filled={clk.filled_count} fresh_AI~={fresh} "
          f"(AI~={fresh/args.seconds:.1f}fps)")
```

- [ ] **Step 4: Smoke-run end-to-end (null sink, short)**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe scripts\live.py --sink null --source file --mode img2img --in-res 384x224 --tiny-vae --preset PRESERVE --fps 30 --seconds 10 --fill warp`
Expected: prints `[fill] motion-extrapolation ON ...` then a `done:` line where **`filled` > 0 and `repeats` is much lower than with `--fill off`** (most stale ticks are now filled, not repeated). No exceptions.

- [ ] **Step 5: A/B baseline (optional sanity)**

Run the same command with `--fill off` and confirm `filled=0` with a high `repeats`. The `--fill warp` run should show `repeats` dropping by roughly the number that became `filled`.

- [ ] **Step 6: Commit**

```bash
git add scripts/live.py
git commit -m "feat(fill): --fill warp wiring + filled-frame metrics in live runner"
```

---

## Task 7: Full suite + docs + memory

**Files:**
- Modify: `PERFORMANCE_FINDINGS.md`
- Modify: `C:\Users\Cedar\.claude\projects\D--StreamForge\memory\streamforge-plan.md`

- [ ] **Step 1: Run the entire test suite**

Run: `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest -q`
Expected: all green (prior suite + `test_warp` 4, `test_filler` 4, `test_clock_fill` 4, `test_worker_fill` 2, `test_flow` 1-or-skip).

- [ ] **Step 2: Append a RIFE result section to `PERFORMANCE_FINDINGS.md`**

Add a short section recording: motion-extrapolation frame-fill shipped (RAFT source-flow forward-warp), the measured `--fill warp` vs `--fill off` numbers from Task 6 (emitted / fresh_AI / repeats→filled / jitter), and that output is now smooth 30/50 fps without touching the transformer.

- [ ] **Step 3: Update the memory file**

In `streamforge-plan.md`, add one paragraph: RIFE frame-fill executed — extrapolation (not interpolation, per user), RAFT-small source-flow forward-warp into the `select_fill` "warped" tier, `--fill warp` flag, files `src/streamforge/fill/{warp,filler,flow}.py` + clock/worker/live wiring, on branch `feat/rife-frame-fill`. Record the measured filled/repeats/jitter. Note the deferred fast-follow (dedicated fill thread if clock-thread warp regresses jitter).

- [ ] **Step 4: Commit**

```bash
git add PERFORMANCE_FINDINGS.md
git commit -m "docs(fill): RIFE frame-fill results + perf findings"
```

---

## Risks & Notes
- **Clock-thread jitter:** the warp (`grid_sample`, ~1–2 ms) runs on the emit thread, which already does Spout readback (~31 ms jitter noted in memory). Task 6's `jitter_ms` print is the check; if it regresses materially, the fast-follow is a dedicated fill thread that pre-renders the next warped frame (out of scope here, YAGNI).
- **First fill before EMA:** with a single anchor the nominal `fps` (13) is used; the EMA corrects within ~2 AI frames. Acceptable.
- **AI rate cost:** RAFT (~10–20 ms at flow-res) runs on the worker per AI frame, nudging fresh fps ~14→~12. Accepted tradeoff for smooth output.
- **Sources without motion vectors:** RAFT covers file/camera. GPU motion vectors from render feeds (`Source.Capabilities.has_motion_vectors`) are a future faster path, out of scope.

## Self-Review (completed by plan author)
- **Spec coverage:** warp (T1), filler+cap+EMA (T2), clock/select_fill wiring (T3), RAFT flow (T4), worker flow+anchor (T5), live `--fill`/cap/flow-res + metrics (T6), tests across T1–T5, docs/memory (T7). Degradation paths: RAFT load failure (T6 try/except), past-cap hold (T2/T3), flow-disabled back-compat (T3/T5 default-None).
- **Type consistency:** `warp_forward(image, flow, dt, fps, max_disp=None)`; `FrameFiller(max_extrap_ms, fps, max_disp, ema).set_anchor(styled, flow, t)` / `.fill(now)->tensor|None`; `RaftFlow(device, max_side).estimate(prev, cur)->[1,2,fh,fw]`; `RealtimeClock(fps, fb, emit, filler=None)` with `filled_count`; `InferenceWorker(..., flow=None, filler=None)`. Used identically across tasks.
- **No placeholders:** every code/test step is complete and runnable; every run step has an exact command + expected result.
- **Refinement vs spec:** `fps` is the AI/anchor rate (EMA-refined), not a fixed source/output fps — this is the correct scaling for smooth motion and supersedes the spec's fixed-fps phrasing; everything else matches the approved spec.

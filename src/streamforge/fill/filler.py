"""FrameFiller — thread-safe latest-anchor + forward extrapolation (the motion analog of
FrameBuffer). The worker writes anchors; the clock reads warped frames. `fps` is the AI/anchor
rate, seeded by the nominal value and EMA-refined from real anchor timestamps."""
from __future__ import annotations
import threading
import torch
from streamforge.fill.warp import warp_forward


class FrameFiller:
    def __init__(self, max_extrap_ms: float = 120.0, fps: float = 13.0,
                 max_disp: float | None = None, ema: float = 0.5,
                 fixed_fps: float | None = None):
        self.max_extrap_s = max_extrap_ms / 1000.0
        # fixed_fps pins the warp scale to a known rate (the display fps) instead of inferring it
        # from anchor arrival times. SANA sets one anchor per CHUNK (~1/s) but passes a per-INPUT-
        # frame flow, so the EMA-from-arrival heuristic would collapse the scale — pin it instead.
        self._fixed_fps = float(fixed_fps) if fixed_fps is not None else None
        self.fps = self._fixed_fps if self._fixed_fps is not None else float(fps)
        self.max_disp = max_disp
        self._ema = ema
        self._lock = threading.Lock()
        self._styled: torch.Tensor | None = None
        self._flow: torch.Tensor | None = None
        self._t: float | None = None

    def set_anchor(self, styled: torch.Tensor, flow: torch.Tensor, t: float) -> None:
        with self._lock:
            if self._fixed_fps is None and self._t is not None:
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

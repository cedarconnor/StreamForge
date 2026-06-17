from __future__ import annotations

import time

import numpy as np
import torch

from streamforge.frame import GpuFrame
from streamforge.sources.base import Capabilities, Source, SourceStatus


class WebcamSource(Source):
    capabilities = Capabilities(has_motion_vectors=False)

    def __init__(self, index: int = 0, fps: int = 30, device: str = "cuda",
                 backend: int | None = None, read_timeout: float = 2.0):
        self.index = index
        self.fps = fps
        self.backend = backend          # None => try DirectShow, then MSMF, then default
        self.read_timeout = read_timeout  # tolerate transient grab failures this long before None
        self._device = device if torch.cuda.is_available() else "cpu"
        self._cap = None
        self._seq = 0
        self._last_frame_t: float | None = None
        self._error: str | None = None

    def open(self) -> None:
        import cv2
        self._seq = 0
        self._error = None
        # DirectShow (CAP_DSHOW) is markedly more stable than the default Media Foundation
        # backend for OpenCV webcams on Windows (MSMF throws 0xC00D3704 grab errors on many
        # cameras). Prefer it, fall back to MSMF, then the library default.
        candidates = [self.backend] if self.backend is not None else [
            getattr(cv2, "CAP_DSHOW", None), getattr(cv2, "CAP_MSMF", None), None]
        for b in candidates:
            cap = cv2.VideoCapture(self.index) if b is None else cv2.VideoCapture(self.index, b)
            if cap.isOpened():
                self._cap = cap
                return
            cap.release()
        self._cap = None
        self._error = f"webcam {self.index} is not available"

    def read(self) -> GpuFrame | None:
        import cv2
        if self._cap is None or not self._cap.isOpened():
            return None
        # Retry transient grab failures (a single MSMF/DSHOW hiccup must not be read as
        # end-of-stream, which would stop the worker); only give up after read_timeout.
        deadline = time.perf_counter() + self.read_timeout
        ok, bgr = self._cap.read()
        while (not ok or bgr is None) and time.perf_counter() < deadline:
            time.sleep(0.005)
            ok, bgr = self._cap.read()
        if not ok or bgr is None:
            self._error = f"webcam {self.index} returned no frame"
            return None
        self._error = None
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self._device)
        self._last_frame_t = time.perf_counter()
        frame = GpuFrame(tensor=t, seq=self._seq, pts=self._seq / self.fps,
                         width=t.shape[-1], height=t.shape[-2])
        self._seq += 1
        return frame

    def status(self) -> SourceStatus:
        width = height = None
        available = False
        if self._cap is not None:
            import cv2
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            available = self._cap.isOpened()
        age = ((time.perf_counter() - self._last_frame_t) * 1000.0
               if self._last_frame_t is not None else None)
        return SourceStatus(
            name=f"Webcam {self.index}",
            width=width,
            height=height,
            fps=float(self.fps),
            available=available,
            last_frame_age_ms=age,
            error=self._error,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()

from __future__ import annotations

import time

import numpy as np
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
        self._error = None
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

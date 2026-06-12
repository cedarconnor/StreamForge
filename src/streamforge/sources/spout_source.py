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
        self._error = None
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

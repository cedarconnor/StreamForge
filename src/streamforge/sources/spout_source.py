from __future__ import annotations

import time

import numpy as np
import torch

from streamforge.frame import GpuFrame
from streamforge.sources.base import Capabilities, Source, SourceStatus

GL_RGBA = 0x1908


class SpoutSource(Source):
    capabilities = Capabilities(has_motion_vectors=False)

    def __init__(self, name: str = "StreamForge", fps: int = 30, device: str = "cuda",
                 invert: bool = True, read_timeout: float = 2.0):
        self.name = name
        self.fps = fps
        self.invert = invert
        self.read_timeout = read_timeout  # block this long for a frame before signalling end-of-stream
        self._device = device if torch.cuda.is_available() else "cpu"
        self._receiver = None
        self._buffer: bytearray | None = None
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
        # Spout shares a GPU texture; receiving the pixels needs an active GL context. Create a
        # hidden one bound to THIS thread (open()/read() run on the same worker thread).
        self._receiver.createOpenGL()
        self._receiver.setReceiverName(self.name)
        self._buffer = bytearray(4)  # probe buffer; resized to w*h*4 once the sender is detected
        self._seq = 0

    def read(self) -> GpuFrame | None:
        """Block up to read_timeout for one frame. Spout only reports the sender's size AFTER the
        first receiveImage+isUpdated(), so we poll: detect size, resize the buffer, then capture."""
        rcv = self._receiver
        if rcv is None:
            return None
        deadline = time.perf_counter() + self.read_timeout
        while time.perf_counter() < deadline:
            ok = rcv.receiveImage(self._buffer, GL_RGBA, self.invert, 0)
            if rcv.isUpdated():  # new/changed sender -> learn its dimensions, resize, retry
                w, h = int(rcv.getSenderWidth()), int(rcv.getSenderHeight())
                if w > 0 and h > 0:
                    self._buffer = bytearray(w * h * 4)
                continue
            w, h = int(rcv.getSenderWidth()), int(rcv.getSenderHeight())
            if ok and w > 0 and h > 0 and len(self._buffer) == w * h * 4:
                rgba = np.frombuffer(self._buffer, dtype=np.uint8).reshape((h, w, 4))
                rgb = np.copy(rgba[:, :, :3])
                tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self._device)
                self._size = (w, h)
                self._last_frame_t = time.perf_counter()
                frame = GpuFrame(tensor=tensor, seq=self._seq, pts=self._seq / self.fps,
                                 width=w, height=h)
                self._seq += 1
                return frame
            time.sleep(0.003)
        if self._size is None:
            self._error = f"Spout sender '{self.name}' not found"
        return None

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
            try:
                self._receiver.closeOpenGL()
            except Exception:
                pass
            self._receiver.releaseReceiver()
            self._receiver = None

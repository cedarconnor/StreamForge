"""SpoutSink — publish frames as a Spout source (same-machine GPU texture share, design §8.2).

v1 uses the robust readback path (tensor -> CPU RGBA bytes -> SpoutGL.sendImage). True
zero-copy (CUDA->GL interop) is a later optimization; the profiling showed the transformer,
not IO, is the budget, so a readback here is not the bottleneck.

Notes:
- Spout is Windows-only and needs an NVIDIA/AMD GPU.
- Resolume Arena commonly needs a vertical flip -> `flip=True` (bInvert) by default.
- open()/send() must run on the same thread (the GL context is thread-affine). In the live
  app the RealtimeClock owns the emit thread, so the sink lives there.
"""
from __future__ import annotations

import numpy as np

from streamforge.frame import GpuFrame
from streamforge.sinks.base import Sink

GL_RGBA = 0x1908  # OpenGL enum; avoids a hard PyOpenGL dependency


class SpoutSink(Sink):
    def __init__(self, name: str = "StreamForge", flip: bool = True):
        self.name = name
        self.flip = flip
        self._sender = None

    def open(self) -> None:
        import SpoutGL
        self._sender = SpoutGL.SpoutSender()
        self._sender.setSenderName(self.name)
        self._sender.createOpenGL()   # hidden GL context so we can send headless

    def send(self, frame: GpuFrame) -> None:
        if frame is None or self._sender is None:
            return
        t = frame.tensor[0].clamp(0, 1)                      # (3, H, W) in [0,1]
        rgb = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # (H, W, 3)
        h, w, _ = rgb.shape
        rgba = np.dstack([rgb, np.full((h, w, 1), 255, np.uint8)])       # add opaque alpha
        self._sender.sendImage(rgba.tobytes(), w, h, GL_RGBA, self.flip, 0)

    def close(self) -> None:
        if self._sender is not None:
            self._sender.releaseSender()
            self._sender.closeOpenGL()
            self._sender = None

"""SyntheticSource — a deterministic moving test pattern.

Lets us benchmark the whole pipeline with no capture hardware and with perfectly
repeatable input (so latency numbers are comparable across runs).
"""
from __future__ import annotations

import torch

from streamforge.frame import GpuFrame
from streamforge.sources.base import Source, Capabilities


class SyntheticSource(Source):
    capabilities = Capabilities(has_motion_vectors=False)

    def __init__(self, width: int, height: int, fps: int, device: str = "cuda"):
        self.width = width
        self.height = height
        self.fps = fps
        self._device = device if torch.cuda.is_available() else "cpu"
        self._seq = 0

    def open(self) -> None:
        self._seq = 0

    def read(self) -> GpuFrame:
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, self.height),
            torch.linspace(0, 1, self.width),
            indexing="ij",
        )
        phase = (self._seq % self.fps) / self.fps
        r = (xx + phase) % 1.0
        g = (yy + phase) % 1.0
        b = torch.full_like(r, phase)
        t = torch.stack([r, g, b])[None].to(self._device)
        f = GpuFrame(tensor=t, seq=self._seq, pts=self._seq / self.fps,
                     width=self.width, height=self.height)
        self._seq += 1
        return f

    def close(self) -> None:
        pass

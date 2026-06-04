"""FileSink — writes each frame as a PNG, for visual output validation."""
from __future__ import annotations

import pathlib

from streamforge.frame import GpuFrame
from streamforge.sinks.base import Sink


class FileSink(Sink):
    def __init__(self, out_dir: str):
        self.out_dir = pathlib.Path(out_dir)

    def open(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def send(self, frame: GpuFrame) -> None:
        from PIL import Image
        arr = (frame.tensor[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(arr).save(self.out_dir / f"frame_{frame.seq:06d}.png")

    def close(self) -> None:
        pass

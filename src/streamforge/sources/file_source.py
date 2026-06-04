"""FileSource — loops a video file. Repeatable input that exercises the real
decode/upload path (unlike SyntheticSource, which generates on-device)."""
from __future__ import annotations

import torch

from streamforge.frame import GpuFrame
from streamforge.sources.base import Source, Capabilities


class FileSource(Source):
    capabilities = Capabilities(has_motion_vectors=False)

    def __init__(self, path: str, fps: int, device: str = "cuda"):
        self.path = path
        self.fps = fps
        self._device = device if torch.cuda.is_available() else "cpu"
        self._cap = None
        self._seq = 0

    def open(self) -> None:
        import cv2
        self._cap = cv2.VideoCapture(self.path)
        self._seq = 0

    def read(self) -> GpuFrame | None:
        import cv2
        ok, bgr = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
            ok, bgr = self._cap.read()
            if not ok:
                return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self._device)
        f = GpuFrame(tensor=t, seq=self._seq, pts=self._seq / self.fps,
                     width=t.shape[-1], height=t.shape[-2])
        self._seq += 1
        return f

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()

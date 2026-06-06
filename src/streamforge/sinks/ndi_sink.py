"""NDISink — publish frames as an NDI source over the network/localhost (design §8.3).

Brought forward from Phase 9 so the NDI path is testable now. v1 uses the readback path
(tensor -> CPU RGBA -> NDI send). Uses ndi-python, whose wheel bundles the NDI runtime, so no
separate NDI SDK install is required. Any NDI receiver (NDI Studio Monitor, OBS/DistroAV, or
scripts/ndi_receiver.py) on the same machine/network can view it.

Phase-9 items still TODO: receive-side FrameSync (clock-drift), NVENC, 2h soak.
"""
from __future__ import annotations

import numpy as np

from streamforge.frame import GpuFrame
from streamforge.sinks.base import Sink


class NDISink(Sink):
    def __init__(self, name: str = "StreamForge", fps: int = 30):
        self.name = name
        self.fps = fps
        self._ndi = None
        self._sender = None
        self._vframe = None

    def open(self) -> None:
        import NDIlib as ndi
        self._ndi = ndi
        if not ndi.initialize():
            raise RuntimeError("NDI runtime failed to initialize")
        settings = ndi.SendCreate()
        settings.ndi_name = self.name
        self._sender = ndi.send_create(settings)
        if self._sender is None:
            raise RuntimeError("ndi.send_create failed")
        self._vframe = ndi.VideoFrameV2()
        self._vframe.frame_rate_N = int(self.fps * 1000)
        self._vframe.frame_rate_D = 1000

    def send(self, frame: GpuFrame) -> None:
        if frame is None or self._sender is None:
            return
        t = frame.tensor[0].clamp(0, 1)
        rgb = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)   # (H, W, 3)
        h, w, _ = rgb.shape
        rgba = np.ascontiguousarray(np.dstack([rgb, np.full((h, w, 1), 255, np.uint8)]))
        self._vframe.data = rgba
        self._vframe.FourCC = self._ndi.FOURCC_VIDEO_TYPE_RGBA
        self._ndi.send_send_video_v2(self._sender, self._vframe)

    def close(self) -> None:
        if self._sender is not None:
            self._ndi.send_destroy(self._sender)
            self._sender = None
        if self._ndi is not None:
            self._ndi.destroy()

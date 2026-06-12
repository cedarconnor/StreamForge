from __future__ import annotations

import time

import numpy as np
import torch

from streamforge.frame import GpuFrame
from streamforge.sources.base import Capabilities, Source, SourceStatus


class NDISource(Source):
    capabilities = Capabilities(has_motion_vectors=False)

    def __init__(self, name: str = "", fps: int = 30, device: str = "cuda", timeout_ms: int = 1000):
        self.name = name
        self.fps = fps
        self.timeout_ms = timeout_ms
        self._device = device if torch.cuda.is_available() else "cpu"
        self._ndi = None
        self._finder = None
        self._recv = None
        self._source_name: str | None = None
        self._seq = 0
        self._last_frame_t: float | None = None
        self._size: tuple[int, int] | None = None
        self._error: str | None = None

    @staticmethod
    def list_sources() -> list[str]:
        import NDIlib as ndi
        if not ndi.initialize():
            return []
        finder = ndi.find_create_v2()
        try:
            ndi.find_wait_for_sources(finder, 1000)
            return [s.ndi_name for s in ndi.find_get_current_sources(finder)]
        finally:
            ndi.find_destroy(finder)
            ndi.destroy()

    def open(self) -> None:
        import NDIlib as ndi
        self._ndi = ndi
        self._error = None
        if not ndi.initialize():
            self._error = "NDI runtime failed to initialize"
            return
        self._finder = ndi.find_create_v2()
        ndi.find_wait_for_sources(self._finder, 2000)
        sources = ndi.find_get_current_sources(self._finder)
        selected = None
        for src in sources:
            if not self.name or self.name.lower() in src.ndi_name.lower():
                selected = src
                break
        if selected is None:
            self._error = f"NDI source matching '{self.name}' not found"
            return
        rc = ndi.RecvCreateV3()
        rc.color_format = ndi.RECV_COLOR_FORMAT_RGBX_RGBA
        self._recv = ndi.recv_create_v3(rc)
        ndi.recv_connect(self._recv, selected)
        self._source_name = selected.ndi_name
        self._seq = 0

    def read(self) -> GpuFrame | None:
        if self._ndi is None or self._recv is None:
            return None
        tpe, v, a, _m = self._ndi.recv_capture_v2(self._recv, self.timeout_ms)
        if tpe == self._ndi.FRAME_TYPE_VIDEO:
            try:
                rgba = np.copy(v.data)
                rgb = rgba[:, :, :3]
                tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self._device)
                self._size = (tensor.shape[-1], tensor.shape[-2])
                self._last_frame_t = time.perf_counter()
                frame = GpuFrame(tensor=tensor, seq=self._seq, pts=self._seq / self.fps,
                                 width=tensor.shape[-1], height=tensor.shape[-2])
                self._seq += 1
                return frame
            finally:
                self._ndi.recv_free_video_v2(self._recv, v)
        if tpe == self._ndi.FRAME_TYPE_AUDIO:
            self._ndi.recv_free_audio_v2(self._recv, a)
        return None

    def status(self) -> SourceStatus:
        age = ((time.perf_counter() - self._last_frame_t) * 1000.0
               if self._last_frame_t is not None else None)
        width = self._size[0] if self._size else None
        height = self._size[1] if self._size else None
        return SourceStatus(
            name=self._source_name or self.name or "NDI",
            width=width,
            height=height,
            fps=float(self.fps),
            available=self._recv is not None and self._error is None,
            last_frame_age_ms=age,
            error=self._error,
        )

    def close(self) -> None:
        if self._ndi is not None and self._recv is not None:
            self._ndi.recv_destroy(self._recv)
        if self._ndi is not None and self._finder is not None:
            self._ndi.find_destroy(self._finder)
        if self._ndi is not None:
            self._ndi.destroy()
        self._recv = None
        self._finder = None

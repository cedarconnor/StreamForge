import sys
import types

import torch

from streamforge.frame import GpuFrame


class FakeVideoFrame:
    pass


class FakeNDI:
    FOURCC_VIDEO_TYPE_RGBA = "RGBA"

    def __init__(self):
        self.sent = None

    def initialize(self):
        return True

    def SendCreate(self):
        return types.SimpleNamespace(ndi_name="")

    def send_create(self, settings):
        return object()

    def VideoFrameV2(self):
        return FakeVideoFrame()

    def send_send_video_v2(self, sender, frame):
        self.sent = frame

    def send_destroy(self, sender):
        pass

    def destroy(self):
        pass


def test_ndi_sink_sets_dimensions_stride_and_aspect(monkeypatch):
    fake = FakeNDI()
    monkeypatch.setitem(sys.modules, "NDIlib", fake)
    from streamforge.sinks.ndi_sink import NDISink
    sink = NDISink(fps=30)
    sink.open()
    frame = GpuFrame(tensor=torch.zeros(1, 3, 4, 6), seq=0, pts=0.0, width=6, height=4)
    sink.send(frame)
    assert fake.sent.xres == 6
    assert fake.sent.yres == 4
    assert fake.sent.line_stride_in_bytes == 6 * 4
    assert fake.sent.picture_aspect_ratio == 1.5

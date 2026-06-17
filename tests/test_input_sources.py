import sys
import types

import numpy as np


class FakeCV2:
    COLOR_BGR2RGB = 1
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    class VideoCapture:
        def __init__(self, index):
            self.index = index
            self.opened = True
            self.frame = np.zeros((4, 6, 3), dtype=np.uint8)

        def isOpened(self):
            return self.opened

        def read(self):
            return True, self.frame

        def get(self, prop):
            return {3: 6, 4: 4}.get(prop, 0)

        def release(self):
            self.opened = False

    @staticmethod
    def cvtColor(frame, code):
        return frame[:, :, ::-1]


def test_webcam_source_reads_bgr_frame_and_reports_status(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    from streamforge.sources.webcam import WebcamSource
    source = WebcamSource(index=0, fps=30, device="cpu")
    source.open()
    frame = source.read()
    status = source.status()
    source.close()
    assert frame.tensor.shape == (1, 3, 4, 6)
    assert frame.width == 6 and frame.height == 4
    assert status.available is True
    assert status.width == 6 and status.height == 4


class FlakyCV2:
    """A webcam that fails the first two grabs (transient MSMF 0xC00D3704) then succeeds."""
    COLOR_BGR2RGB = 1
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_DSHOW = 700
    CAP_MSMF = 1400

    class VideoCapture:
        def __init__(self, index, backend=None):
            self.index = index
            self.backend = backend
            self.opened = True
            self.frame = np.zeros((4, 6, 3), dtype=np.uint8)
            self.calls = 0

        def isOpened(self):
            return self.opened

        def read(self):
            self.calls += 1
            if self.calls <= 2:
                return False, None
            return True, self.frame

        def get(self, prop):
            return {3: 6, 4: 4}.get(prop, 0)

        def release(self):
            self.opened = False

    @staticmethod
    def cvtColor(frame, code):
        return frame[:, :, ::-1]


def test_webcam_source_tolerates_transient_grab_failures(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", FlakyCV2)
    from streamforge.sources.webcam import WebcamSource
    source = WebcamSource(index=0, fps=30, device="cpu", read_timeout=1.0)
    source.open()
    frame = source.read()  # first two grabs fail; must retry, not return None (which stops the run)
    source.close()
    assert frame is not None
    assert frame.width == 6 and frame.height == 4


class FakeNDIVideo:
    def __init__(self):
        self.data = np.zeros((4, 6, 4), dtype=np.uint8)


class FakeNDI:
    RECV_COLOR_FORMAT_RGBX_RGBA = 1
    FRAME_TYPE_VIDEO = 2
    FRAME_TYPE_AUDIO = 3

    def initialize(self):
        return True

    def find_create_v2(self):
        return object()

    def find_wait_for_sources(self, finder, timeout_ms):
        return True

    def find_get_current_sources(self, finder):
        return [types.SimpleNamespace(ndi_name="Camera A")]

    def find_destroy(self, finder):
        pass

    class RecvCreateV3:
        color_format = None

    def recv_create_v3(self, config):
        return object()

    def recv_connect(self, recv, source):
        pass

    def recv_capture_v2(self, recv, timeout_ms):
        return self.FRAME_TYPE_VIDEO, FakeNDIVideo(), None, None

    def recv_free_video_v2(self, recv, video):
        pass

    def recv_free_audio_v2(self, recv, audio):
        pass

    def recv_destroy(self, recv):
        pass

    def destroy(self):
        pass


def test_ndi_source_reads_rgba_frame_and_reports_status(monkeypatch):
    monkeypatch.setitem(sys.modules, "NDIlib", FakeNDI())
    from streamforge.sources.ndi_source import NDISource
    source = NDISource(name="Camera", fps=30, device="cpu")
    source.open()
    frame = source.read()
    status = source.status()
    source.close()
    assert frame.tensor.shape == (1, 3, 4, 6)
    assert frame.width == 6 and frame.height == 4
    assert status.available is True
    assert status.width == 6 and status.height == 4


class FakeSpoutReceiver:
    """Models SpoutGL: a GL context is created, and the sender's size is only known after the
    first receiveImage()+isUpdated() — so the source must resize its buffer then capture."""

    def __init__(self):
        self.released = False
        self.gl_open = False
        self._reads = 0

    def getSenderList(self):
        return ["Camera A"]

    def createOpenGL(self):
        self.gl_open = True
        return True

    def closeOpenGL(self):
        self.gl_open = False

    def setReceiverName(self, name):
        return True

    def isUpdated(self):
        # real SpoutGL reports a new/changed sender once, right after the first receive
        self._reads += 1
        return self._reads == 1

    def getSenderWidth(self):
        return 6

    def getSenderHeight(self):
        return 4

    def receiveImage(self, buffer, gl_format, invert, host_fbo):
        arr = np.zeros((4, 6, 4), dtype=np.uint8)
        if len(buffer) == arr.size:  # only fills a correctly-sized buffer, like the real API
            buffer[:] = arr.tobytes()
        return True

    def releaseReceiver(self):
        self.released = True


class FakeSpoutGL:
    SpoutReceiver = FakeSpoutReceiver


def test_spout_source_reads_rgba_buffer_and_reports_status(monkeypatch):
    monkeypatch.setitem(sys.modules, "SpoutGL", FakeSpoutGL)
    from streamforge.sources.spout_source import SpoutSource
    source = SpoutSource(name="Camera A", fps=30, device="cpu")
    source.open()
    frame = source.read()
    status = source.status()
    source.close()
    assert frame.tensor.shape == (1, 3, 4, 6)
    assert frame.width == 6 and frame.height == 4
    assert status.available is True
    assert status.width == 6 and status.height == 4

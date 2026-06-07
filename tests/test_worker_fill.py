import torch
from streamforge.clock import FrameBuffer
from streamforge.frame import GpuFrame
from streamforge.worker import InferenceWorker


class _StubSource:
    def __init__(self, frames): self._frames = frames; self._i = 0
    def open(self): pass
    def read(self):
        if self._i >= len(self._frames):
            return None
        f = self._frames[self._i]; self._i += 1; return f
    def close(self): pass


class _StubRuntime:
    def restyle(self, tensor, params): return tensor * 0.5


class _StubFlow:
    def __init__(self): self.calls = 0
    def estimate(self, prev, cur): self.calls += 1; return torch.zeros(1, 2, 2, 2)


class _RecordingFiller:
    def __init__(self): self.anchors = []
    def set_anchor(self, styled, flow, t): self.anchors.append((styled, flow, t))


def _frame(i):
    return GpuFrame(tensor=torch.rand(1, 3, 4, 4), seq=i, pts=float(i), width=4, height=4)


def _run_to_exhaustion(w):
    # drive the loop synchronously (stub source returns None -> loop breaks); avoids the
    # start()/stop() thread race. Threading is exercised by the live e2e smoke.
    w._running = True
    w._loop()


def test_worker_sets_anchor_from_second_frame():
    frames = [_frame(0), _frame(1), _frame(2)]
    fb = FrameBuffer(); flow = _StubFlow(); filler = _RecordingFiller()
    w = InferenceWorker(_StubSource(frames), _StubRuntime(), fb,
                        params_provider=lambda: None, flow=flow, filler=filler)
    _run_to_exhaustion(w)
    assert flow.calls == 2                       # frames 2 and 3 (prev available)
    assert len(filler.anchors) == 2
    styled, vel, t = filler.anchors[0]
    assert styled.shape == (1, 3, 4, 4)          # the styled OUTPUT tensor
    assert vel.shape == (1, 2, 2, 2)


def test_worker_without_fill_still_publishes():
    frames = [_frame(0)]
    fb = FrameBuffer()
    w = InferenceWorker(_StubSource(frames), _StubRuntime(), fb, params_provider=lambda: None)
    _run_to_exhaustion(w)
    assert fb.get_latest() is not None

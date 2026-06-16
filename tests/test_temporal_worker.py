import torch  # noqa: F401  pre-import so the worker thread's `import torch` is instant (avoids a startup race)

from streamforge.clock import RealtimeClock
from streamforge.worker_temporal import QueueFrameBuffer, TemporalInferenceWorker


def test_queue_drains_all_n_not_just_last():
    """The overwrite-bug guard: a single-slot FrameBuffer would yield only [4]."""
    qb = QueueFrameBuffer(maxlen=8)
    for i in range(5):
        qb.publish(i)
    got = []
    fr = qb.get_with_freshness()
    while fr.is_fresh:
        got.append(fr.value)
        fr = qb.get_with_freshness()
    assert got == [0, 1, 2, 3, 4]


def test_queue_drops_oldest_on_overflow():
    qb = QueueFrameBuffer(maxlen=3)
    for i in range(5):
        qb.publish(i)
    got = []
    fr = qb.get_with_freshness()
    while fr.is_fresh:
        got.append(fr.value)
        fr = qb.get_with_freshness()
    assert got == [2, 3, 4]  # oldest (0,1) dropped


def test_clock_emits_queued_then_holds():
    qb = QueueFrameBuffer(maxlen=8)
    qb.publish("a")
    qb.publish("b")
    emitted = []
    clk = RealtimeClock(fps=10, frame_buffer=qb, emit=emitted.append)
    clk.run_for_ticks(4)
    assert emitted[:2] == ["a", "b"]
    assert emitted[2] == "b" and emitted[3] == "b"  # holds last value on underrun


# --- worker integration with fakes (mirrors tests/test_live_control.py) ---

class _Frame:
    def __init__(self, tensor):
        self.tensor = tensor

    def with_tensor(self, t):
        return _Frame(t)


class _Src:
    def __init__(self, n):
        self.n = n
        self.reads = 0

    def open(self):
        pass

    def read(self):
        self.reads += 1
        return _Frame(self.reads) if self.reads <= self.n else None

    def close(self):
        pass


class _Runtime:
    def __init__(self):
        self.loaded = False
        self.resets = 0

    def load_once(self):
        self.loaded = True

    def reset_state(self):
        self.resets += 1

    def push_frame(self, tensor, params=None):
        # emit 2 frames per input (exercises the multi-frame drain)
        return [f"{tensor}a", f"{tensor}b"]


def test_worker_pushes_all_chunk_frames_to_queue():
    qb = QueueFrameBuffer(maxlen=16)
    rt = _Runtime()
    w = TemporalInferenceWorker(_Src(3), rt, qb)
    w.start()
    w._t.join(timeout=5.0)  # fake source exhausts after 3 frames -> loop breaks naturally
    w.stop()
    assert rt.loaded is True
    drained = []
    fr = qb.get_with_freshness()
    while fr.is_fresh:
        drained.append(fr.value.tensor)
        fr = qb.get_with_freshness()
    # 3 input frames x 2 emitted each = 6 frames
    assert drained == ["1a", "1b", "2a", "2b", "3a", "3b"]

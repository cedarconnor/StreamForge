import time

import torch

from streamforge.frame import GpuFrame
from streamforge.runner import RunnerConfig, StreamForgeRunner


class StubSource:
    def __init__(self):
        self.opened = False
        self.closed = False
        self.reads = 0
        self.frame = GpuFrame(torch.zeros(1, 3, 4, 6), 0, 0.0, 6, 4)

    def open(self):
        self.opened = True

    def read(self):
        if self.reads > 2:
            return None
        self.reads += 1
        return self.frame

    def close(self):
        self.closed = True

    def status(self):
        from streamforge.sources.base import SourceStatus
        return SourceStatus(name="stub", width=6, height=4, fps=30.0, available=True)


class StubRuntime:
    def set_prompt(self, prompt):
        self.prompt = prompt

    def restyle(self, tensor, params):
        return tensor + 0.25


class StubSink:
    def __init__(self):
        self.frames = []
        self.closed = False

    def open(self):
        pass

    def send(self, frame):
        self.frames.append(frame)

    def close(self):
        self.closed = True


def test_validate_reads_first_frame_and_returns_aspect():
    runner = StreamForgeRunner(source_factory=lambda cfg: StubSource(),
                               runtime_factory=lambda cfg: StubRuntime(),
                               sink_factory=lambda cfg: StubSink())
    result = runner.validate(RunnerConfig(source_type="webcam", source_name="0"))
    assert result["ok"] is True
    assert result["source"]["width"] == 6
    assert result["source"]["height"] == 4
    assert result["aspect"]["source_ratio"] == 1.5


def test_start_status_stop_with_fake_runtime():
    sink = StubSink()
    runner = StreamForgeRunner(source_factory=lambda cfg: StubSource(),
                               runtime_factory=lambda cfg: StubRuntime(),
                               sink_factory=lambda cfg: sink)
    runner.start(RunnerConfig(source_type="webcam", source_name="0", seconds=0.2, fps=5))
    time.sleep(0.05)
    status = runner.status()
    assert status["running"] is True
    assert status["fresh_ai"] >= 0
    runner.stop()
    assert runner.status()["running"] is False
    assert sink.closed is True

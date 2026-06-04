"""NullSink — discards frames; used to bench upstream stages in isolation."""
from __future__ import annotations

from streamforge.frame import GpuFrame
from streamforge.sinks.base import Sink


class NullSink(Sink):
    def __init__(self) -> None:
        self.count = 0

    def open(self) -> None:
        self.count = 0

    def send(self, frame: GpuFrame) -> None:
        self.count += 1

    def close(self) -> None:
        pass

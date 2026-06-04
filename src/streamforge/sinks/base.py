"""Sink interface — the pluggable output side (design §8.5).

The same pipeline serves SpoutSink / SyphonSink / NDISink without touching the core.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from streamforge.frame import GpuFrame


class Sink(ABC):
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def send(self, frame: GpuFrame) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

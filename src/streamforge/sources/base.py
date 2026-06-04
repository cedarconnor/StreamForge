"""Source interface — the pluggable input side (symmetric with Sink).

Design v1.2 under-specified input; this makes it first-class. Sources differ by capability:
a render feed (UE5/Notch/disguise) can hand over motion vectors and depth (the §9 "sleeper
win" for temporal coherence), while a camera cannot.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from streamforge.frame import GpuFrame


@dataclass(frozen=True)
class Capabilities:
    has_motion_vectors: bool = False
    has_depth: bool = False


class Source(ABC):
    capabilities: Capabilities = Capabilities()

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> GpuFrame | None:
        """Return the next frame, or None at end-of-stream."""

    @abstractmethod
    def close(self) -> None: ...

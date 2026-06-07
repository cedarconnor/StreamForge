"""The sacred output clock and the latest-frame handoff (design §6.0).

RealtimeClock owns the output cadence (30/50 fps) and NEVER blocks on inference. Each
tick it emits the best available frame. The FrameBuffer is the thread-safe single-slot
handoff between the InferenceWorker (publisher) and the clock (consumer): the clock learns
whether the latest frame is fresh or a repeat, which is exactly the §3.1 cadence signal.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Freshness:
    value: Any
    is_fresh: bool


class FrameBuffer:
    """Thread-safe single-slot latest-frame handoff. Publisher overwrites; consumer reads
    the latest and learns if it is new (fresh) or unchanged since last read (a repeat)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._val: Any = None
        self._consumed = True

    def publish(self, value: Any) -> None:
        with self._lock:
            self._val = value
            self._consumed = False

    def get_latest(self) -> Any:
        """Peek at the latest value without consuming freshness."""
        with self._lock:
            return self._val

    def get_with_freshness(self) -> Freshness:
        """Read the latest value and report whether it is fresh (new since last read)."""
        with self._lock:
            fresh = not self._consumed
            self._consumed = True
            return Freshness(self._val, fresh)


@dataclass(frozen=True)
class FillResult:
    value: Any
    source: str   # "ai" | "warped" | "held" | "raw"


def select_fill(ai: Any, warped: Any, held: Any, raw: Any) -> FillResult:
    """Frame-fill priority from design §6.0: freshest available wins."""
    if ai is not None:
        return FillResult(ai, "ai")
    if warped is not None:
        return FillResult(warped, "warped")
    if held is not None:
        return FillResult(held, "held")
    return FillResult(raw, "raw")


class RealtimeClock:
    """Emits one frame per tick at the target cadence; counts repeats (no fresh AI frame)."""

    def __init__(self, fps: int, frame_buffer: FrameBuffer, emit: Callable[[Any], None],
                 filler: Any = None):
        self.period = 1.0 / fps
        self.fb = frame_buffer
        self.emit = emit
        self.filler = filler
        self.repeat_count = 0
        self.filled_count = 0
        self._held: Any = None
        self._running = False

    def _tick_once(self) -> None:
        now = time.perf_counter()
        fr = self.fb.get_with_freshness()
        if fr.is_fresh and fr.value is not None:
            self._held = fr.value
            self.emit(fr.value)
            return
        self.repeat_count += 1
        warped_t = self.filler.fill(now) if self.filler is not None else None
        warped = (self._held.with_tensor(warped_t)
                  if warped_t is not None and self._held is not None else None)
        result = select_fill(ai=None, warped=warped, held=self._held, raw=None)
        if result.source == "warped":
            self.filled_count += 1
        self.emit(result.value)

    def run_for_ticks(self, n: int) -> None:
        """Synchronous test mode: n ticks, no real sleeping."""
        for _ in range(n):
            self._tick_once()

    def run(self) -> None:
        """Real mode: wall-clock paced loop. Call stop() from another thread to end."""
        self._running = True
        next_t = time.perf_counter()
        while self._running:
            self._tick_once()
            next_t += self.period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

    def stop(self) -> None:
        self._running = False

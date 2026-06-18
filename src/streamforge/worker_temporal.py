"""Temporal inference worker + bounded drain queue (design §4.3).

The single-slot FrameBuffer is perfect for FLUX (1 frame in -> 1 frame out: publisher overwrites,
consumer reads newest). But a SANA chunk emits N frames at once; a tight publish loop would
overwrite all but the last before the clock consumes them. So the temporal path puts a small
bounded FIFO between the worker and the clock, drained one frame per clock tick.

QueueFrameBuffer exposes the SAME get_with_freshness()/get_latest() contract RealtimeClock already
calls, so the clock needs ZERO edits — is_fresh becomes "a queued frame was available this tick".
"""
from __future__ import annotations

import collections
import threading
import time
from typing import Callable, Optional

from streamforge.clock import Freshness


class QueueFrameBuffer:
    """Bounded FIFO with the clock's freshness contract. Drains one per tick; drops OLDEST on
    overflow (favor latency over completeness for a live show). Holds the last value when empty."""

    def __init__(self, maxlen: int = 3):
        self._q: collections.deque = collections.deque(maxlen=maxlen)
        self._last = None
        self._lock = threading.Lock()

    def publish(self, value) -> None:
        with self._lock:
            self._q.append(value)  # deque(maxlen) drops oldest on overflow

    def get_with_freshness(self) -> Freshness:
        with self._lock:
            if self._q:
                self._last = self._q.popleft()
                return Freshness(self._last, True)
            return Freshness(self._last, False)

    def get_latest(self):
        with self._lock:
            return self._q[-1] if self._q else self._last


class TemporalInferenceWorker:
    """Drives runtime.push_frame on its own thread; each chunk's 0..N frames go to the queue.

    Reads a live control each frame: a WARM change (control.needs_reset) triggers
    runtime.reset_state(); HOT params are passed into push_frame for per-chunk application.
    """

    def __init__(self, source, runtime, frame_buffer: QueueFrameBuffer,
                 control=None, on_timing: Optional[Callable[[str, float], None]] = None,
                 flow=None, filler=None, on_input: Optional[Callable] = None,
                 display_fps: float = 30.0):
        self.source = source
        self.runtime = runtime
        self.fb = frame_buffer
        self.control = control
        self.on_timing = on_timing
        self.flow = flow
        self.filler = filler
        self.on_input = on_input
        self.display_fps = float(display_fps)
        self._prev_src = None
        self._t: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self) -> None:
        import torch

        self.source.open()
        self.runtime.load_once()
        try:
            while self._running:
                f = self.source.read()
                if f is None:
                    break
                if self.on_input is not None:
                    self.on_input(f)  # live Input preview tracks the source, not the validate frame
                if self.control is not None and self.control.needs_reset:
                    self.runtime.reset_state()
                    self.control.clear_reset()
                params = self.control.params() if self.control is not None else None
                t0 = time.perf_counter()
                outs = (self.runtime.push_frame(f.tensor, params)
                        if params is not None else self.runtime.push_frame(f.tensor))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000.0
                for out in outs:
                    self.fb.publish(f.with_tensor(out))
                # Frame-fill anchor: a chunk produces N real frames in one ~burst, so the queue
                # starves between chunks. Anchor the filler to the chunk's LAST (freshest) frame
                # with the latest per-input-frame motion; the clock warps it forward to bridge the
                # gap. anchor_t is offset to ~when that last frame will display (the queue drains
                # one per tick at display_fps), so warp dt starts at 0 there, not at emit time.
                if outs and self.flow is not None and self.filler is not None \
                        and self._prev_src is not None:
                    velocity = self.flow.estimate(self._prev_src, f.tensor)
                    anchor_t = time.perf_counter() + max(0, len(outs) - 1) / self.display_fps
                    self.filler.set_anchor(outs[-1], velocity, anchor_t)
                self._prev_src = f.tensor  # keep prev as the immediately-preceding input frame
                if outs and self.on_timing:
                    self.on_timing("infer", ms)
        finally:
            self.source.close()

    def stop(self) -> None:
        self._running = False
        if self._t is not None:
            self._t.join(timeout=5.0)

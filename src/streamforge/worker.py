"""InferenceWorker — runs the diffusion runtime opportunistically on its own thread.

This is the explicit resolution of the design §3 ("single orchestration thread") vs §6.0
("clock never blocks on inference") tension: the worker runs on a separate thread and
publishes its latest result to a FrameBuffer; the RealtimeClock + Sink live on the emit
thread. CUDA ops release the GIL, so the two threads overlap on the GPU. The output clock
therefore can never be blocked by a slow denoise.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from streamforge.clock import FrameBuffer
from streamforge.control import EngineParams


class InferenceWorker:
    def __init__(
        self,
        source,
        runtime,
        frame_buffer: FrameBuffer,
        params_provider: Callable[[], EngineParams],
        on_timing: Optional[Callable[[str, float], None]] = None,
        flow=None,
        filler=None,
    ):
        self.source = source
        self.runtime = runtime
        self.fb = frame_buffer
        self.params_provider = params_provider
        self.on_timing = on_timing
        self.flow = flow
        self.filler = filler
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
        try:
            while self._running:
                f = self.source.read()
                if f is None:
                    break
                t0 = time.perf_counter()
                out = self.runtime.restyle(f.tensor, self.params_provider())
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000.0
                if self.flow is not None and self.filler is not None:
                    if self._prev_src is not None:
                        velocity = self.flow.estimate(self._prev_src, f.tensor)
                        self.filler.set_anchor(out, velocity, time.perf_counter())
                    self._prev_src = f.tensor
                self.fb.publish(f.with_tensor(out))
                if self.on_timing:
                    self.on_timing("infer", ms)
        finally:
            self.source.close()

    def stop(self) -> None:
        self._running = False
        if self._t is not None:
            self._t.join(timeout=5.0)

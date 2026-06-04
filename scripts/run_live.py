"""Live runner / app skeleton (seed of app.py).

Wires Source -> InferenceWorker -> FrameBuffer -> RealtimeClock -> Sink with the AI cadence
decoupled from the output clock (design §6.0). Supports a FAKE slow runtime so the decoupling
can be validated on real hardware/threads BEFORE the model is integrated (Task 5.2 gate):
the output clock must stay rock-stable even when the "AI" runs far slower than the clock.

Examples:
  # Phase-5 gate: 50 fps output while "AI" only manages ~12 fps -> stable clock, many repeats
  python scripts/run_live.py --fps 50 --seconds 10 --res 512 --fake-ai-fps 12 --sink null
"""
from __future__ import annotations

import argparse
import threading
import time

from streamforge.clock import FrameBuffer, RealtimeClock
from streamforge.control import TwoAxisControl
from streamforge.metrics import jitter_ms
from streamforge.sources.synthetic import SyntheticSource
from streamforge.sinks.null_sink import NullSink
from streamforge.worker import InferenceWorker


class FakeSlowRuntime:
    """Passthrough runtime that sleeps to emulate a fixed AI frame time. Model-independent."""

    def __init__(self, ai_fps: float):
        self._delay = 1.0 / ai_fps if ai_fps > 0 else 0.0

    def set_prompt(self, prompt: str) -> None:
        pass

    def restyle(self, image_bchw, params):
        if self._delay:
            time.sleep(self._delay)
        return image_bchw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--fake-ai-fps", type=float, default=12.0)
    ap.add_argument("--sink", choices=["null"], default="null")
    args = ap.parse_args()

    source = SyntheticSource(args.res, args.res, args.fps)
    runtime = FakeSlowRuntime(args.fake_ai_fps)
    fb = FrameBuffer()
    params = TwoAxisControl.preset("BALANCED").to_engine_params()
    worker = InferenceWorker(source, runtime, fb, params_provider=lambda: params)

    sink = NullSink()
    sink.open()
    timestamps: list[float] = []

    def emit(frame) -> None:
        sink.send(frame)             # NullSink tolerates None (pre-first-AI-frame ticks)
        timestamps.append(time.perf_counter())

    clk = RealtimeClock(args.fps, fb, emit)
    worker.start()
    threading.Timer(args.seconds, clk.stop).start()
    clk.run()                        # blocks until stop()
    worker.stop()
    sink.close()

    emitted = len(timestamps)
    expected = int(args.fps * args.seconds)
    print(f"output: emitted={emitted} expected~={expected} "
          f"jitter={jitter_ms(timestamps):.2f}ms repeats={clk.repeat_count} "
          f"(AI fresh frames ~= {emitted - clk.repeat_count})")


if __name__ == "__main__":
    main()

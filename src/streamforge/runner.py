from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch

from streamforge.aspect import FitMode, Size, plan_fit, snap_to_multiple_for_aspect
from streamforge.clock import FrameBuffer, RealtimeClock
from streamforge.color import ColorPipeline
from streamforge.control import TwoAxisControl
from streamforge.metrics import jitter_ms
from streamforge.worker import InferenceWorker


@dataclass(frozen=True)
class RunnerConfig:
    source_type: str = "webcam"
    source_name: str = "0"
    sink: str = "null"
    fps: int = 30
    seconds: float = 0.0
    prompt: str = "vivid oil painting, thick impasto brushstrokes"
    mode: str = "img2img"
    preset: str = "BALANCED"
    in_res: str = "auto"
    color: str = "full"
    fill: str = "off"
    max_extrap_ms: float = 120.0
    flow_max_side: int = 384
    compile_transformer: bool = False
    tiny_vae: bool = False
    spout_flip: bool = True


def default_source_factory(config: RunnerConfig):
    if config.source_type == "webcam":
        from streamforge.sources.webcam import WebcamSource
        return WebcamSource(index=int(config.source_name or "0"), fps=config.fps)
    if config.source_type == "ndi":
        from streamforge.sources.ndi_source import NDISource
        return NDISource(name=config.source_name, fps=config.fps)
    if config.source_type == "spout":
        from streamforge.sources.spout_source import SpoutSource
        return SpoutSource(name=config.source_name, fps=config.fps)
    if config.source_type == "file":
        from streamforge.sources.file_source import FileSource
        return FileSource(config.source_name, fps=config.fps)
    if config.source_type == "synthetic":
        from streamforge.sources.synthetic import SyntheticSource
        size = int(config.source_name or "512")
        return SyntheticSource(size, size, config.fps)
    raise ValueError(f"unknown source type {config.source_type!r}")


def _parse_internal_hw(config: RunnerConfig) -> tuple[int, int] | None:
    if config.in_res.lower() in ("off", "native", "auto", ""):
        return None
    w_i, h_i = (int(v) for v in config.in_res.lower().split("x"))
    return h_i, w_i


def default_runtime_factory(config: RunnerConfig):
    from streamforge.diffusion.runtime_eager import EagerRuntime
    return EagerRuntime(
        mode=config.mode,
        internal_hw=_parse_internal_hw(config),
        compile_transformer=config.compile_transformer,
        tiny_vae=config.tiny_vae,
    )


def default_sink_factory(config: RunnerConfig):
    if config.sink == "null":
        from streamforge.sinks.null_sink import NullSink
        return NullSink()
    if config.sink == "spout":
        from streamforge.sinks.spout_sink import SpoutSink
        return SpoutSink(flip=config.spout_flip)
    if config.sink == "ndi":
        from streamforge.sinks.ndi_sink import NDISink
        return NDISink(fps=config.fps)
    raise ValueError(f"unknown sink {config.sink!r}")


class StreamForgeRunner:
    def __init__(
        self,
        source_factory: Callable[[RunnerConfig], object] = default_source_factory,
        runtime_factory: Callable[[RunnerConfig], object] = default_runtime_factory,
        sink_factory: Callable[[RunnerConfig], object] = default_sink_factory,
    ):
        self.source_factory = source_factory
        self.runtime_factory = runtime_factory
        self.sink_factory = sink_factory
        self._lock = threading.Lock()
        self._running = False
        self._config: RunnerConfig | None = None
        self._timestamps: list[float] = []
        self._infer_ms: list[float] = []
        self._latest_input = None
        self._latest_output = None
        self._worker = None
        self._clock = None
        self._sink = None
        self._clock_thread: threading.Thread | None = None

    def validate(self, config: RunnerConfig) -> dict:
        source = self.source_factory(config)
        source.open()
        try:
            frame = source.read()
            if frame is None:
                return {"ok": False, "error": "source returned no frame", "source": asdict(source.status())}
            with self._lock:
                self._latest_input = frame
            source_size = Size(width=frame.width, height=frame.height)
            internal = snap_to_multiple_for_aspect(source_size, max_side=384, multiple=16)
            plan = plan_fit(source_size, internal, FitMode.FILL_CROP)
            return {
                "ok": True,
                "source": asdict(source.status()),
                "aspect": {
                    "source_ratio": source_size.aspect,
                    "internal": asdict(internal),
                    "crop_direction": plan.crop_direction,
                },
            }
        finally:
            source.close()

    def start(self, config: RunnerConfig) -> None:
        self.stop()
        source = self.source_factory(config)
        runtime = self.runtime_factory(config)
        runtime.set_prompt(config.prompt)
        sink = self.sink_factory(config)
        sink.open()
        color = None if config.color == "off" else ColorPipeline(range_mode=config.color)
        fb = FrameBuffer()
        params = TwoAxisControl.preset(config.preset).to_engine_params()
        filler = None
        flow = None
        if config.fill == "warp":
            from streamforge.fill.filler import FrameFiller
            from streamforge.fill.flow import RaftFlow
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            flow = RaftFlow(device=dev, max_side=config.flow_max_side)
            filler = FrameFiller(max_extrap_ms=config.max_extrap_ms)

        def timing(stage: str, ms: float) -> None:
            if stage == "infer":
                with self._lock:
                    self._infer_ms.append(ms)

        def emit(frame) -> None:
            if frame is None:
                return
            out = frame
            if color is not None:
                out = frame.with_tensor(color.apply(frame.tensor))
            with self._lock:
                self._latest_output = out
                self._timestamps.append(time.perf_counter())
            sink.send(out)

        worker = InferenceWorker(source, runtime, fb, params_provider=lambda: params,
                                 on_timing=timing, flow=flow, filler=filler)
        clock = RealtimeClock(config.fps, fb, emit, filler=filler)
        self._worker = worker
        self._clock = clock
        self._sink = sink
        self._config = config
        self._timestamps = []
        self._infer_ms = []
        self._running = True
        worker.start()
        self._clock_thread = threading.Thread(target=clock.run, daemon=True)
        self._clock_thread.start()
        if config.seconds > 0:
            threading.Timer(config.seconds, self.stop).start()

    def stop(self) -> None:
        if self._clock is not None:
            self._clock.stop()
        if self._worker is not None:
            self._worker.stop()
        if self._clock_thread is not None:
            self._clock_thread.join(timeout=5.0)
        if self._sink is not None:
            self._sink.close()
        self._clock = None
        self._worker = None
        self._sink = None
        self._clock_thread = None
        self._running = False

    def status(self) -> dict:
        with self._lock:
            timestamps = list(self._timestamps)
            infer_ms = list(self._infer_ms)
        clock = self._clock
        repeats = clock.repeat_count if clock else 0
        emitted = len(timestamps)
        return {
            "running": self._running,
            "config": asdict(self._config) if self._config else None,
            "jitter_ms": jitter_ms(timestamps),
            "infer_ms_last": infer_ms[-1] if infer_ms else None,
            "emitted": emitted,
            "repeats": repeats,
            "filled": clock.filled_count if clock else 0,
            "fresh_ai": max(0, emitted - repeats),
        }

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, replace
from typing import Callable

import torch

from streamforge.aspect import FitMode, Size, plan_fit, snap_to_multiple_for_aspect
from streamforge.clock import FrameBuffer, RealtimeClock
from streamforge.color import ColorPipeline
from streamforge.control import EngineParams, TwoAxisControl
from streamforge.metrics import jitter_ms
from streamforge.sana_control import SanaControl, SanaLiveControl
from streamforge.worker import InferenceWorker
from streamforge.worker_temporal import QueueFrameBuffer, TemporalInferenceWorker


def frame_to_jpeg(frame, quality: int = 80) -> bytes:
    import io
    from PIL import Image
    arr = (frame.tensor[0].detach().clamp(0, 1).float().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class LiveControl:
    """Thread-safe live restyle controls. `params()` feeds the worker every frame;
    `apply()` mutates the underlying TwoAxisControl and recomputes EngineParams under a lock."""

    def __init__(self, ctrl: TwoAxisControl):
        self._lock = threading.Lock()
        self._ctrl = ctrl
        self._params = ctrl.to_engine_params()

    def params(self) -> EngineParams:
        with self._lock:
            return self._params

    def apply(self, *, ref_strength=None, text_magnitude=None, steps=None, seed=None) -> None:
        with self._lock:
            updates: dict = {}
            if ref_strength is not None:
                updates["ref_strength"] = float(ref_strength)
            if text_magnitude is not None:
                updates["text_magnitude"] = float(text_magnitude)
            if steps is not None:
                updates["steps"] = max(1, int(steps))
            if seed is not None:
                updates["seed"] = int(seed)
            if updates:
                self._ctrl = replace(self._ctrl, **updates)
                self._params = self._ctrl.to_engine_params()

    def as_dict(self) -> dict:
        with self._lock:
            c = self._ctrl
        return {"ref_strength": c.ref_strength, "text_magnitude": c.text_magnitude,
                "steps": c.steps, "seed": c.seed}


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
    backend: str = "flux"            # "flux" (EagerRuntime) | "sana_streaming" (temporal)
    cached_blocks: int = 2           # SANA: GDN/KV window
    sink_token: bool = True          # SANA: attention-sink stability
    resync_every: int = 8            # SANA: re-anchor temporal state every N chunks (0 = off)


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
    if config.backend == "sana_streaming":
        from streamforge.diffusion.runtime_sana_streaming import SanaStreamingRuntime
        return SanaStreamingRuntime(
            internal_hw=_parse_internal_hw(config),
            num_cached_blocks=config.cached_blocks,
            sink_token=config.sink_token,
            resync_every=config.resync_every,
        )
    if config.backend == "flux":
        from streamforge.diffusion.runtime_eager import EagerRuntime
        return EagerRuntime(
            mode=config.mode,
            internal_hw=_parse_internal_hw(config),
            compile_transformer=config.compile_transformer,
            tiny_vae=config.tiny_vae,
        )
    raise ValueError(f"unknown backend {config.backend!r}")


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
        self._control: LiveControl | None = None
        self._runtime = None
        self._prompt: str | None = None
        self._mode: str | None = None
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
            explicit_hw = _parse_internal_hw(config)
            multiple = 32 if config.backend == "sana_streaming" else 16  # SANA VAE stride is 32
            if explicit_hw is None:
                internal = snap_to_multiple_for_aspect(source_size, max_side=384, multiple=multiple)
            else:
                explicit_h, explicit_w = explicit_hw
                internal = Size(width=explicit_w, height=explicit_h)
            if config.backend == "sana_streaming":
                from streamforge.diffusion.runtime_sana_streaming import validate_dims
                try:
                    validate_dims(internal.width, internal.height, 9)  # spatial check (frames at runtime)
                except ValueError as e:
                    return {"ok": False, "error": str(e), "source": asdict(source.status())}
            plan = plan_fit(source_size, internal, FitMode.FILL_CROP)
            source_status = asdict(source.status())
            source_status.update({"width": frame.width, "height": frame.height, "available": True})
            return {
                "ok": True,
                "source": source_status,
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
        self._runtime = runtime
        self._prompt = config.prompt
        self._mode = config.mode
        temporal = getattr(runtime, "temporal", False)
        if temporal:
            # Default a non-SANA preset to SANA_FAST (steps=2): the profiler shows steps=2 clears
            # real-time (1.14x) while steps=4 is 0.84x (behind). Operators who want the higher-
            # quality 4-step path select SANA_BALANCED explicitly.
            sana_preset = config.preset if config.preset.startswith("SANA") else "SANA_FAST"
            self._control = SanaLiveControl(SanaControl.preset(sana_preset))
            fb = QueueFrameBuffer(maxlen=64)   # holds a couple of chunk bursts (~24 frames each)
        else:
            self._control = LiveControl(TwoAxisControl.preset(config.preset))
            fb = FrameBuffer()
        filler = None
        flow = None
        if config.fill == "warp":
            from streamforge.fill.filler import FrameFiller
            from streamforge.fill.flow import RaftFlow
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            flow = RaftFlow(device=dev, max_side=config.flow_max_side)
            if temporal:
                # SANA emits a burst of frames per chunk and starves the queue between chunks, so
                # bridge a longer gap than FLUX's per-frame fill. Anchors arrive per-chunk but the
                # flow is per-input-frame, so pin the warp scale to the display rate (fixed_fps);
                # clamp displacement so a long extrapolation can't smear the frame wildly.
                filler = FrameFiller(max_extrap_ms=max(config.max_extrap_ms, 400.0),
                                     fixed_fps=float(config.fps), max_disp=64.0)
            else:
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

        def on_input(frame) -> None:
            with self._lock:
                self._latest_input = frame  # keep the Input preview live during the run

        if temporal:
            worker = TemporalInferenceWorker(source, runtime, fb, control=self._control,
                                             on_timing=timing, flow=flow, filler=filler,
                                             on_input=on_input, display_fps=float(config.fps))
        else:
            worker = InferenceWorker(source, runtime, fb, params_provider=self._control.params,
                                     on_timing=timing, flow=flow, filler=filler,
                                     on_input=on_input)
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
        self._control = None
        self._runtime = None
        self._clock_thread = None
        self._running = False
        # Release the runtime's GPU memory so repeated Start/Stop cycles don't accumulate VRAM
        # (each FLUX/SANA load is ~13-27 GB; without this they stack until the card is full and
        # inference slows to a crawl under memory pressure).
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _control_snapshot(self) -> dict | None:
        if self._control is None:
            return None
        d = self._control.as_dict()
        d["prompt"] = self._prompt
        d["mode"] = self._mode
        d["backend"] = self._config.backend if self._config else "flux"
        if getattr(self._runtime, "temporal", False):
            d["resync_every"] = getattr(self._runtime, "resync_every", 0)
        return d

    def apply_control(self, *, ref_strength=None, text_magnitude=None, steps=None,
                      seed=None, prompt=None, mode=None, flow_shift=None, motion_score=None,
                      num_cached_blocks=None, sink_token=None, resync_every=None) -> dict:
        if self._control is None or self._runtime is None:
            return {}
        if getattr(self._runtime, "temporal", False):
            # SANA: HOT knobs apply live; num_cached_blocks/sink_token are WARM (engine rebuilt on
            # the worker's next reset_state, which reads these runtime attrs); prompt re-encodes.
            if num_cached_blocks is not None:
                self._runtime.num_cached_blocks = int(num_cached_blocks)
            if sink_token is not None:
                self._runtime.sink_token = bool(sink_token)
            if resync_every is not None:  # HOT: just changes when the next auto re-anchor fires
                self._runtime.resync_every = max(0, int(resync_every))
            self._control.apply(step=steps, flow_shift=flow_shift, motion_score=motion_score,
                                seed=seed, num_cached_blocks=num_cached_blocks, sink_token=sink_token)
            if prompt is not None:
                self._runtime.set_prompt(prompt)
                self._prompt = prompt
        else:
            self._control.apply(ref_strength=ref_strength, text_magnitude=text_magnitude,
                                steps=steps, seed=seed)
            if prompt is not None:
                self._runtime.set_prompt(prompt)
                self._prompt = prompt
            if mode is not None:
                self._runtime.set_mode(mode)
                self._mode = mode
        return self._control_snapshot() or {}

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
            "control": self._control_snapshot(),
        }

    def latest_input_jpeg(self) -> bytes | None:
        with self._lock:
            frame = self._latest_input
        return frame_to_jpeg(frame) if frame is not None else None

    def latest_output_jpeg(self) -> bytes | None:
        with self._lock:
            frame = self._latest_output
        return frame_to_jpeg(frame) if frame is not None else None

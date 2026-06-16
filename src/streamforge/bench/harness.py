"""Benchmark harness (design §5.2) — a required subsystem, not a milestone.

Runs N frames through source -> runtime, recording per-stage latency percentiles, output
jitter, VRAM high-water, missed deadlines, and frame repeats. Average FPS is useless for
shows; this reports the tail.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from streamforge.metrics import LatencyStats, jitter_ms


@dataclass
class StageTimer:
    samples_ms: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples_ms.append(ms)

    @property
    def stats(self) -> LatencyStats:
        return LatencyStats.from_samples_ms(self.samples_ms)


@dataclass
class BenchReport:
    stages: dict[str, LatencyStats]
    output_jitter_ms: float
    vram_peak_gb: float
    missed_deadlines: int
    frame_repeats: int
    frames: int


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _default_params():
    from streamforge.control import TwoAxisControl
    return TwoAxisControl.preset("BALANCED").to_engine_params()


class BenchHarness:
    """Runs N frames through source->runtime, recording per-stage latency, jitter, VRAM, misses."""

    def __init__(self, source, runtime, frames: int, fps: int, prompt: str = "test"):
        self.source = source
        self.runtime = runtime
        self.frames = frames
        self.fps = fps
        self.prompt = prompt
        self.budget_ms = 1000.0 / fps

    def run(self) -> BenchReport:
        if getattr(self.runtime, "temporal", False):
            return self._run_temporal()
        timers = {k: StageTimer() for k in ("read", "infer")}
        out_ts: list[float] = []
        misses = 0
        repeats = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.source.open()
        self.runtime.set_prompt(self.prompt)
        last_seq = -1
        params = _default_params()
        for _ in range(self.frames):
            t0 = _now_ms()
            f = self.source.read()
            timers["read"].add(_now_ms() - t0)
            if f is None:
                break
            t1 = _now_ms()
            self.runtime.restyle(f.tensor, params)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = _now_ms() - t1
            timers["infer"].add(dt)
            if dt > self.budget_ms:
                misses += 1
            if f.seq == last_seq:
                repeats += 1
            last_seq = f.seq
            out_ts.append(_now_ms() / 1000.0)
        self.source.close()
        vram = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
        return BenchReport(
            stages={k: t.stats for k, t in timers.items()},
            output_jitter_ms=jitter_ms(out_ts),
            vram_peak_gb=round(vram, 2),
            missed_deadlines=misses,
            frame_repeats=repeats,
            frames=len(out_ts),
        )

    def _run_temporal(self) -> BenchReport:
        """Bench a TemporalDiffusionRuntime: load once, feed frames, count drained outputs.
        `infer` records per-chunk latency (only frames that produced output); `frames` is the
        total emitted output frame count."""
        timers = {k: StageTimer() for k in ("read", "infer")}
        out_ts: list[float] = []
        misses = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.source.open()
        self.runtime.load_once()
        self.runtime.set_prompt(self.prompt)
        out_count = 0
        for _ in range(self.frames):
            t0 = _now_ms()
            f = self.source.read()
            timers["read"].add(_now_ms() - t0)
            if f is None:
                break
            t1 = _now_ms()
            outs = self.runtime.push_frame(f.tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = _now_ms() - t1
            if outs:
                timers["infer"].add(dt)
                if dt > self.budget_ms * len(outs):  # budget per emitted frame
                    misses += 1
                for _o in outs:
                    out_count += 1
                    out_ts.append(_now_ms() / 1000.0)
        self.source.close()
        vram = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
        return BenchReport(
            stages={k: t.stats for k, t in timers.items()},
            output_jitter_ms=jitter_ms(out_ts),
            vram_peak_gb=round(vram, 2),
            missed_deadlines=misses,
            frame_repeats=0,
            frames=out_count,
        )

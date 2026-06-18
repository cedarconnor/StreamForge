"""SANA-Streaming live control surface (Phase-1).

SANA's knobs do not map onto the FLUX two-axis model (ref_strength / text_magnitude).
They are governed instead by their interaction with the GDN recurrent state, in three classes:

  HOT  (apply live, per chunk, no reset): flow_shift, motion_score, seed, step
  WARM (require reset_state(); brief reseed/flash): prompt, num_cached_blocks, sink_token
  COLD (require a run restart): height/width, backend, cfg_scale (streaming is pinned 1.0)

SanaLiveControl mirrors LiveControl (runner.py) but over SanaParams; the temporal worker reads
params() each frame and consumes needs_reset to call runtime.reset_state() after a WARM change.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, replace

# Knob safety classes (see module docstring). cfg_scale/height/width/backend are COLD (run restart).
HOT = {"flow_shift", "motion_score", "seed", "step"}
WARM = {"prompt", "num_cached_blocks", "sink_token"}


@dataclass(frozen=True)
class SanaParams:
    """What the engine reads per chunk."""
    step: int = 4
    cfg_scale: float = 1.0
    flow_shift: float = 8.0
    num_cached_blocks: int = 2
    sink_token: bool = True
    motion_score: int = 0
    seed: int = 7


@dataclass(frozen=True)
class SanaControl:
    step: int = 4
    num_cached_blocks: int = 2
    sink_token: bool = True
    flow_shift: float = 8.0
    motion_score: int = 0
    seed: int = 7
    cfg_scale: float = 1.0

    def to_params(self) -> SanaParams:
        return SanaParams(
            step=max(1, int(self.step)),
            cfg_scale=float(self.cfg_scale),
            flow_shift=float(self.flow_shift),
            num_cached_blocks=int(self.num_cached_blocks),
            sink_token=bool(self.sink_token),
            motion_score=max(0, int(self.motion_score)),
            seed=int(self.seed),
        )

    @classmethod
    def preset(cls, name: str) -> "SanaControl":
        # The two presets differ only by `step` — the profiler showed steps is the real speed lever
        # (DiT is 55-67% of a chunk and scales with steps+1 forwards), while num_cached_blocks is
        # minor. Both keep cached=2 for temporal coherence. FAST (step=2) is real-time (1.14x);
        # BALANCED (step=4) is the higher-quality 0.84x path (pair it with Frame-fill).
        table = {
            "SANA_FAST": dict(step=2, num_cached_blocks=2),
            "SANA_BALANCED": dict(step=4, num_cached_blocks=2),
        }
        return cls(**table[name])


class SanaLiveControl:
    """Thread-safe live SANA controls. params() feeds the worker every frame; apply() mutates
    the control. A WARM-class change sets needs_reset so the worker clears the GDN state."""

    def __init__(self, ctrl: SanaControl):
        self._lock = threading.Lock()
        self._ctrl = ctrl
        self._params = ctrl.to_params()
        self.needs_reset = False

    def params(self) -> SanaParams:
        with self._lock:
            return self._params

    def apply(self, **kw) -> None:
        with self._lock:
            updates = {k: v for k, v in kw.items() if v is not None and hasattr(self._ctrl, k)}
            if any(k in WARM for k in updates):
                self.needs_reset = True
            if updates:
                self._ctrl = replace(self._ctrl, **updates)
                self._params = self._ctrl.to_params()

    def clear_reset(self) -> None:
        with self._lock:
            self.needs_reset = False

    def as_dict(self) -> dict:
        with self._lock:
            c = self._ctrl
        return {
            "step": c.step, "num_cached_blocks": c.num_cached_blocks, "sink_token": c.sink_token,
            "flow_shift": c.flow_shift, "motion_score": c.motion_score, "seed": c.seed,
            "cfg_scale": c.cfg_scale,
        }

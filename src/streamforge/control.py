"""Two-axis live control (design §4.2).

Instead of a single "strength" knob, live behavior is governed by two orthogonal axes:
  - ref_strength    : how closely output follows the INPUT frame's structure (higher = faithful)
  - text_magnitude  : how strongly the PROMPT/style asserts itself (higher = more stylized)

These map onto the FLUX.2-klein mechanics confirmed empirically (Phase-0 still-image gate):
  - ref_strength  -> img2img denoise strength (high ref_strength => less noise added => faithful)
  - text_magnitude-> prompt-EMBEDDING interpolation/scaling, NOT guidance_scale.
    The distilled klein-4B IGNORES guidance_scale ("ignored for step-wise distilled models"),
    so the EagerRuntime implements text_magnitude as lerp(neutral_embeds, prompt_embeds, tm)
    (tm>1 exaggerates). The `guidance` field below is retained only for non-distilled paths
    (base 4B / bake-off variants that DO respond to guidance) and is ignored by distilled.

The functional form here is fixed; the numeric bounds are Phase-3 calibration targets.
"""
from __future__ import annotations

from dataclasses import dataclass


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t)


@dataclass(frozen=True)
class EngineParams:
    denoise_strength: float   # img2img noise added to the encoded input latent
    guidance: float           # FLUX.2 distilled guidance scalar
    steps: int = 4
    seed: int = 7


# Calibrated bounds (Phase-3 tuning targets these constants; functional form is fixed).
DENOISE_MIN, DENOISE_MAX = 0.30, 0.95   # PRESERVE..FORCE map into this band
GUIDANCE_MIN, GUIDANCE_MAX = 1.0, 5.0   # distilled guidance scalar usable band


@dataclass(frozen=True)
class TwoAxisControl:
    ref_strength: float
    text_magnitude: float
    steps: int = 4
    seed: int = 7

    def to_engine_params(self) -> EngineParams:
        rs = _clamp(self.ref_strength)
        tm = _clamp(self.text_magnitude)
        # high ref_strength -> low denoise (stay faithful to input structure)
        denoise = _lerp(DENOISE_MAX, DENOISE_MIN, rs)
        guidance = _lerp(GUIDANCE_MIN, GUIDANCE_MAX, tm)
        return EngineParams(denoise_strength=denoise, guidance=guidance,
                            steps=self.steps, seed=self.seed)

    @classmethod
    def preset(cls, name: str) -> "TwoAxisControl":
        table = {
            "PRESERVE": (0.90, 0.20),
            "SUBTLE":   (0.75, 0.45),
            "BALANCED": (0.55, 0.60),
            "FOLLOW":   (0.30, 0.85),
            "FORCE":    (0.10, 1.00),
        }
        rs, tm = table[name]
        return cls(ref_strength=rs, text_magnitude=tm)

"""Pure latency/jitter metrics. No GPU, no torch — unit-tested in isolation.

Average FPS is useless for shows (design §5.2); tail latency and output jitter decide
show-safety, so these are the primitives the benchmark harness rolls up.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass


def percentile(samples: list[float], p: float) -> float:
    """Linear-interpolated percentile (p in 0..100)."""
    if not samples:
        raise ValueError("empty samples")
    s = sorted(samples)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def jitter_ms(timestamps_s: list[float]) -> float:
    """Std-dev of inter-frame intervals, in ms. Low = smooth output clock."""
    if len(timestamps_s) < 3:
        return 0.0
    intervals = [(b - a) * 1000.0 for a, b in zip(timestamps_s, timestamps_s[1:])]
    return statistics.pstdev(intervals)


@dataclass(frozen=True)
class LatencyStats:
    p50: float
    p95: float
    p99: float
    worst: float
    mean: float
    n: int

    @classmethod
    def from_samples_ms(cls, samples: list[float]) -> "LatencyStats":
        return cls(
            p50=percentile(samples, 50),
            p95=percentile(samples, 95),
            p99=percentile(samples, 99),
            worst=max(samples),
            mean=sum(samples) / len(samples),
            n=len(samples),
        )

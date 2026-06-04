"""Render a BenchReport to a human table and to JSON (for archiving bake-off results)."""
from __future__ import annotations

import json

from streamforge.bench.harness import BenchReport


def to_table(r: BenchReport) -> str:
    lines = [
        f"frames={r.frames} jitter={r.output_jitter_ms:.2f}ms vram={r.vram_peak_gb}GB "
        f"missed={r.missed_deadlines} repeats={r.frame_repeats}",
        f"{'stage':8s} {'p50':>7} {'p95':>7} {'p99':>7} {'worst':>8}",
    ]
    for name, s in r.stages.items():
        lines.append(f"{name:8s} {s.p50:7.1f} {s.p95:7.1f} {s.p99:7.1f} {s.worst:8.1f}")
    return "\n".join(lines)


def to_json(r: BenchReport) -> str:
    return json.dumps(
        {
            "frames": r.frames,
            "jitter_ms": r.output_jitter_ms,
            "vram_gb": r.vram_peak_gb,
            "missed": r.missed_deadlines,
            "repeats": r.frame_repeats,
            "stages": {k: vars(v) for k, v in r.stages.items()},
        },
        indent=2,
    )

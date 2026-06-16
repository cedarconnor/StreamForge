"""Benchmark the SANA temporal backend on the A6000 (run under .venv-sana, [gpu]).

Drives SanaStreamingRuntime through the temporal BenchHarness path. `infer` = per-chunk latency;
output FPS ~= (frames per chunk) / chunk-latency. Reloads the model per config (resolution is
fixed per runtime), so keep the config list short.
"""
from __future__ import annotations

from streamforge.bench.harness import BenchHarness
from streamforge.diffusion.runtime_sana_streaming import SanaStreamingRuntime
from streamforge.sources.file_source import FileSource

VIDEO = "D:/StreamForge/TestFile/DriveVideo.mp4"
PROMPT = "a cinematic oil painting, vivid impasto brushstrokes"


def bench(h: int, w: int, steps: int, frames: int = 120):
    src = FileSource(VIDEO, fps=16)
    rt = SanaStreamingRuntime(internal_hw=(h, w), default_steps=steps)
    rep = BenchHarness(src, rt, frames=frames, fps=16, prompt=PROMPT).run()
    inf = rep.stages["infer"]
    frames_per_chunk = (rt._base_chunk_frames - 1) * 8 + 1
    fps = (frames_per_chunk * 1000.0 / inf.p50) if inf.p50 > 0 else 0.0
    print(f"{w}x{h} step{steps}: out={rep.frames}f  chunk p50={inf.p50:.0f}ms p95={inf.p95:.0f}ms "
          f"worst={inf.worst:.0f}ms  ~{fps:.1f} fps  jitter={rep.output_jitter_ms:.1f}ms  "
          f"vram={rep.vram_peak_gb}GB  misses={rep.missed_deadlines}")


def main():
    bench(384, 640, 4)
    bench(384, 640, 2)
    bench(512, 512, 4)


if __name__ == "__main__":
    main()

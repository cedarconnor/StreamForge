"""Measure the two unverified assumptions in the perf review:
  (1) step count  — PRESERVE (~1 step) vs BALANCED (~2 steps) on the img2img path
  (2) resolution  — 512x512 vs rectangular show formats (512x288, 384x224)

Self-contained timing (warmup + per-frame cuda.synchronize), so numbers are comparable to the
recorded 358 ms BALANCED@512 baseline. Dims are snapped to a multiple of 16 (FLUX latent packing)
and the ACTUAL dims used are printed. Writes out/levers.json.

  python scripts/measure_levers.py
  python scripts/measure_levers.py --frames 24 --warmup 4
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import torch

from streamforge.control import TwoAxisControl
from streamforge.diffusion.runtime_eager import EagerRuntime
from streamforge.sources.synthetic import SyntheticSource


def _snap16(x: int) -> int:
    return max(16, round(x / 16) * 16)


def _steps_for(preset: str) -> int:
    """Effective transformer steps the img2img loop will actually run for this preset."""
    p = TwoAxisControl.preset(preset).to_engine_params()
    init = min(int(p.steps * p.denoise_strength), p.steps)
    return max(init, 1)  # loop always runs >=1 step after t_start slice


def measure(rt: EagerRuntime, w: int, h: int, params, frames: int, warmup: int) -> dict:
    src = SyntheticSource(w, h, fps=30)
    src.open()
    # warmup (compile/alloc/caches)
    for _ in range(warmup):
        f = src.read()
        rt.restyle(f.tensor, params)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(frames):
        f = src.read()
        t0 = time.perf_counter()
        rt.restyle(f.tensor, params)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    src.close()
    samples.sort()
    p50 = statistics.median(samples)
    p95 = samples[min(len(samples) - 1, int(0.95 * len(samples)))]
    vram = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    return {"p50_ms": round(p50, 1), "p95_ms": round(p95, 1),
            "fps": round(1000.0 / p50, 2), "vram_gb": round(vram, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--prompt", default="vivid oil painting, thick impasto brushstrokes")
    ap.add_argument("--compile", action="store_true",
                    help="compile the transformer (needs higher --warmup to absorb compile time)")
    ap.add_argument("--compile-mode", default="default",
                    help="torch.compile mode: default | reduce-overhead (CUDA graphs) | max-autotune")
    ap.add_argument("--tiny-vae", action="store_true", help="use taef2 tiny VAE")
    ap.add_argument("--quant", default="none",
                    help="none | int8ao (TorchAO INT8 weight-only) | 8bit | 4bit (bnb)")
    ap.add_argument("--res", default="512x512,512x288,384x224",
                    help="comma list of WxH internal resolutions to sweep")
    args = ap.parse_args()

    # one runtime (model loaded once); img2img is the production-fast path
    rt = EagerRuntime(mode="img2img", compile_transformer=args.compile,
                      compile_mode=args.compile_mode, tiny_vae=args.tiny_vae,
                      quant=None if args.quant == "none" else args.quant)
    rt.set_prompt(args.prompt)

    resolutions = [tuple(int(v) for v in r.lower().split("x")) for r in args.res.split(",")]
    presets = ["BALANCED", "PRESERVE"]

    rows = []
    print(f"{'res':>10} {'preset':>9} {'steps':>5} {'p50_ms':>8} {'p95_ms':>8} {'fps':>6} {'vram':>6}")
    for (w0, h0) in resolutions:
        w, h = _snap16(w0), _snap16(h0)
        for preset in presets:
            params = TwoAxisControl.preset(preset).to_engine_params()
            steps = _steps_for(preset)
            r = measure(rt, w, h, params, args.frames, args.warmup)
            r.update({"res": f"{w}x{h}", "preset": preset, "steps": steps})
            rows.append(r)
            print(f"{w}x{h:<5} {preset:>9} {steps:>5} {r['p50_ms']:>8} {r['p95_ms']:>8} "
                  f"{r['fps']:>6} {r['vram_gb']:>6}")

    base = next((x for x in rows if x["res"] == "512x512" and x["preset"] == "BALANCED"), None)
    ref = base["p50_ms"] if base else 359.5  # recorded 512x512 BALANCED eager baseline
    print(f"\nspeedup vs 512x512 BALANCED eager ({ref:.0f}ms):")
    for r in rows:
        print(f"  {r['res']:>9} {r['preset']:<9} {ref/r['p50_ms']:.2f}x  ({r['fps']} fps)")

    pathlib.Path("out").mkdir(exist_ok=True)
    out = pathlib.Path("out") / "levers.json"
    out.write_text(json.dumps({"baseline": base, "rows": rows}, indent=2))
    print("\nwrote", out)


if __name__ == "__main__":
    main()

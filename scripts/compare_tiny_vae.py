"""Compare full FLUX.2 VAE vs taef2 on the img2img fast path: QUALITY (side-by-side) + SPEED.

Loads the transformer once, restyles a real frame with the full VAE, then hot-swaps in taef2 and
restyles the SAME frame again (same seed/preset/res), so the only variable is the VAE. Saves a
[source | full-VAE | taef2] strip and prints p50 latency + fps for each.

  python scripts/compare_tiny_vae.py --in-res 384x224 --preset PRESERVE --compile
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch
from PIL import Image

from streamforge.control import TwoAxisControl
from streamforge.diffusion.runtime_eager import EagerRuntime
from streamforge.diffusion.tiny_vae import build_taef2
from streamforge.sources.file_source import FileSource


def _np(t):
    return (t[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")


def _timed(rt, frame, params, n, warmup):
    for _ in range(warmup):
        rt.restyle(frame.tensor, params)
    torch.cuda.synchronize()
    s = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = rt.restyle(frame.tensor, params)
        torch.cuda.synchronize()
        s.append((time.perf_counter() - t0) * 1000)
    return out, statistics.median(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=r"D:\StreamForge\TestFile\DriveVideo.mp4")
    ap.add_argument("--in-res", default="384x224")
    ap.add_argument("--preset", default="PRESERVE")
    ap.add_argument("--prompt", default="vivid oil painting, thick impasto brushstrokes")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--out", default=r"out\tiny_vae_compare.png")
    args = ap.parse_args()

    w, h = (int(v) for v in args.in_res.lower().split("x"))
    src = FileSource(args.clip, fps=30); src.open(); frame = src.read(); src.close()
    params = TwoAxisControl.preset(args.preset).to_engine_params()

    rt = EagerRuntime(mode="img2img", internal_hw=(h, w), compile_transformer=args.compile)
    rt.set_prompt(args.prompt)

    full_out, full_ms = _timed(rt, frame, params, args.frames, args.warmup)
    print(f"full VAE : {full_ms:6.1f} ms  ({1000/full_ms:.1f} fps)")

    rt.pipe.vae = build_taef2(device=rt.device, dtype=rt.dtype)  # hot-swap, same transformer
    tiny_out, tiny_ms = _timed(rt, frame, params, args.frames, args.warmup)
    print(f"taef2    : {tiny_ms:6.1f} ms  ({1000/tiny_ms:.1f} fps)   speedup {full_ms/tiny_ms:.2f}x")

    strip = np.concatenate([_np(frame.tensor), _np(full_out), _np(tiny_out)], axis=1)
    Image.fromarray(strip).save(args.out)
    print("wrote", args.out, "[ source | full-VAE | taef2 ]")


if __name__ == "__main__":
    main()

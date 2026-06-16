"""Validate SanaStreamingRuntime end-to-end (run under .venv-sana, [gpu]).

Feeds source frames one at a time through push_frame (the path the TemporalInferenceWorker uses),
collects the emitted frames, and saves an output clip for visual inspection of VAE-chunk seams.
"""
from __future__ import annotations

import time

import imageio.v3 as iio
import numpy as np
import torch

from streamforge.diffusion.runtime_sana_streaming import SanaStreamingRuntime

VIDEO = "D:/StreamForge/TestFile/DriveVideo.mp4"
OUT = "D:/StreamForge/out/sana_runtime/runtime_smoke.mp4"
N_IN = 48


def main():
    rt = SanaStreamingRuntime(internal_hw=(384, 640), default_steps=4)
    rt.load_once()
    rt.set_prompt("a cinematic oil painting, vivid impasto brushstrokes")

    src = iio.imread(VIDEO, index=None)[:N_IN]
    outs = []
    t0 = time.perf_counter()
    for fr in src:
        t = torch.from_numpy(np.asarray(fr)).float().div(255).permute(2, 0, 1).unsqueeze(0)
        outs.extend(rt.push_frame(t))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    print(f"fed {len(src)} frames -> {len(outs)} output frames in {dt:.2f}s "
          f"({len(outs)/dt:.1f} fps out)")
    assert outs, "no frames emitted"
    o = outs[0]
    print(f"out frame: shape={tuple(o.shape)} dtype={o.dtype} device={o.device} "
          f"range=[{o.min():.3f},{o.max():.3f}]")
    assert o.dim() == 4 and o.shape[1] == 3, f"expected [1,3,H,W], got {tuple(o.shape)}"

    import pathlib
    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    arr = (torch.cat(outs, dim=0).detach().float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
    iio.imwrite(OUT, arr, fps=16)
    print(f"PASS: saved {OUT}")


if __name__ == "__main__":
    main()

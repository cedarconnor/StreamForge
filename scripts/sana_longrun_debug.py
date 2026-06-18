"""Reproduce the SANA long-run colored-noise divergence.

The single-chunk test renders fine; the user's screenshot showed ~250 chunks
(Emitted 22614 / Repeats 18418). This drives the REAL push_frame path for many
chunks (looping DriveVideo) at the user's 384x288 and logs per-chunk output stats
+ saves periodic frames, to find WHEN/IF the stream diverges to noise.

Run under .venv-sana.
"""
from __future__ import annotations

import pathlib

import imageio.v3 as iio
import numpy as np
import torch

from streamforge.diffusion.runtime_sana_streaming import SanaStreamingRuntime

VIDEO = "D:/StreamForge/TestFile/DriveVideo.mp4"
OUTDIR = pathlib.Path("D:/StreamForge/out/sana_debug/longrun")
PROMPT = "vivid oil painting, thick impasto brushstrokes"
N_CHUNKS = 220
SAVE_AT = {0, 1, 2, 5, 10, 20, 40, 70, 110, 160, 219}
import os
RESYNC_EVERY = int(os.environ.get("RESYNC_EVERY", "0"))  # 0 = never (baseline)


def main():
    global OUTDIR
    if RESYNC_EVERY:
        OUTDIR = OUTDIR.parent / f"longrun_resync{RESYNC_EVERY}"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rt = SanaStreamingRuntime(internal_hw=(288, 384), default_steps=4)  # 4:3 384x288 like the user
    rt.load_once()
    rt.set_prompt(PROMPT)

    src = iio.imread(VIDEO, index=None)
    src01 = [torch.from_numpy(np.asarray(fr)).float().div(255).permute(2, 0, 1).unsqueeze(0)
             for fr in src]
    n = len(src01)

    print(f"pixel/chunk={rt._pixel_per_chunk}  src_frames={n}  target_chunks={N_CHUNKS}")
    chunk = 0
    fed = 0
    i = 0
    while chunk < N_CHUNKS:
        if RESYNC_EVERY and chunk > 0 and chunk % RESYNC_EVERY == 0 and not rt._buf:
            rt.reset_state()  # re-anchor temporal state before drift accumulates
        outs = rt.push_frame(src01[i % n])  # real shipped path
        i += 1
        fed += 1
        if not outs:
            continue
        # a chunk just emitted
        o = torch.cat(outs, dim=0).detach().float()
        std = o.std().item()
        mn, mx, mean = o.min().item(), o.max().item(), o.mean().item()
        flag = "  <-- low-variance/diverged?" if std < 0.05 or std > 0.40 else ""
        print(f"chunk {chunk:4d}  out[min={mn:+.2f} max={mx:+.2f} mean={mean:.2f} std={std:.3f}]{flag}")
        if chunk in SAVE_AT:
            f0 = outs[0][0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            iio.imwrite(OUTDIR / f"chunk_{chunk:04d}.png", (f0 * 255).astype("uint8"))
        chunk += 1
    torch.cuda.synchronize()
    print("done")


if __name__ == "__main__":
    main()

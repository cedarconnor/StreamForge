"""End-to-end check of the Track-A internal-resolution path: load a real frame, restyle it
with the AI running at low internal res, confirm the output comes back at SOURCE resolution,
and save a before/after strip as visual proof.

  python scripts/verify_track_a.py --clip D:\\StreamForge\\TestFile\\DriveVideo.mp4 --in-res 384x224
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from streamforge.control import TwoAxisControl
from streamforge.diffusion.runtime_eager import EagerRuntime
from streamforge.sources.file_source import FileSource


def _to_np(t: torch.Tensor) -> np.ndarray:
    return (t[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=r"D:\StreamForge\TestFile\DriveVideo.mp4")
    ap.add_argument("--in-res", default="384x224")
    ap.add_argument("--preset", default="PRESERVE")
    ap.add_argument("--prompt", default="vivid oil painting, thick impasto brushstrokes")
    ap.add_argument("--out", default=r"out\track_a_before_after.png")
    args = ap.parse_args()

    w_i, h_i = (int(v) for v in args.in_res.lower().split("x"))

    src = FileSource(args.clip, fps=30)
    src.open()
    frame = src.read()
    src.close()
    in_h, in_w = frame.tensor.shape[-2], frame.tensor.shape[-1]
    print(f"source frame: {in_w}x{in_h}")
    source_ratio = in_w / in_h
    internal_ratio = w_i / h_i
    if abs(source_ratio - internal_ratio) > 0.01:
        print(f"aspect policy: fill-crop source {in_w}x{in_h} into internal {w_i}x{h_i}")
    else:
        print(f"aspect policy: preserve source ratio at internal {w_i}x{h_i}")

    rt = EagerRuntime(mode="img2img", internal_hw=(h_i, w_i))
    rt.set_prompt(args.prompt)
    params = TwoAxisControl.preset(args.preset).to_engine_params()
    print(f"internal diffusion res (snapped): {rt.internal_hw[1]}x{rt.internal_hw[0]}  "
          f"preset={args.preset} denoise={params.denoise_strength:.2f}")

    out = rt.restyle(frame.tensor, params)
    out_h, out_w = out.shape[-2], out.shape[-1]
    print(f"output frame: {out_w}x{out_h}")
    assert (out_h, out_w) == (in_h, in_w), "output must come back at SOURCE resolution"
    print("PASS: low-res diffusion, full-res output")

    before, after = _to_np(frame.tensor), _to_np(out)
    strip = np.concatenate([before, after], axis=1)
    Image.fromarray(strip).save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

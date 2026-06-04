"""Task 1.2 / 1.3 visual gate: restyle frames from the test clip via EagerRuntime.

Saves source/restyled pairs for several frames so we can eyeball quality and confirm
structure adherence holds across motion (not just frame 0).
"""
from __future__ import annotations

import argparse
import pathlib

import cv2
import torch

from streamforge.diffusion.runtime_eager import EagerRuntime
from streamforge.control import TwoAxisControl

CLIP = r"D:\StreamForge\TestFile\DriveVideo.mp4"
OUT = pathlib.Path("out/clip")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--every", type=int, default=80, help="restyle 1 of every N frames")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--preset", default="BALANCED")
    ap.add_argument("--prompt", default="vivid oil painting, thick impasto brushstrokes, expressive color")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rt = EagerRuntime()
    rt.set_prompt(args.prompt)
    params = TwoAxisControl.preset(args.preset).to_engine_params()

    cap = cv2.VideoCapture(CLIP)
    saved = 0
    idx = 0
    while saved < args.count:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).cuda()
            t = torch.nn.functional.interpolate(t, size=(args.res, args.res))
            out = rt.restyle(t, params)
            src_np = (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            out_np = (out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            cv2.imwrite(str(OUT / f"f{idx:04d}_src.png"), cv2.cvtColor(src_np, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(OUT / f"f{idx:04d}_out.png"), cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR))
            print(f"restyled frame {idx}")
            saved += 1
        idx += 1
    cap.release()
    print(f"saved {saved} pairs to {OUT}")


if __name__ == "__main__":
    main()

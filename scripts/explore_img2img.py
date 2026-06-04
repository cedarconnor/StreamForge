"""Exploration (Task 1.2 dev): work out the FLUX.2-klein img2img/restyle mechanism.

The distilled model ignores guidance_scale and has no `strength` param. This probes the
native `image=` (edit/reference) path on a real frame from the test clip, to see whether it
restyles while holding structure. Output saved to out/ for eyeball inspection.
"""
from __future__ import annotations

import pathlib

import cv2
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

OUT = pathlib.Path("out")
CLIP = r"D:\StreamForge\TestFile\DriveVideo.mp4"
PROMPT = "turn this into a vivid oil painting, thick impasto brushstrokes, expressive color"


def first_frame(size=512) -> Image.Image:
    cap = cv2.VideoCapture(CLIP)
    ok, bgr = cap.read()
    cap.release()
    assert ok, "could not read clip"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).resize((size, size))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    pipe = Flux2KleinPipeline.from_pretrained("models/transformer", torch_dtype=torch.bfloat16).to("cuda")
    src = first_frame(512)
    src.save(OUT / "src_frame.png")
    torch.cuda.reset_peak_memory_stats()
    out = pipe(
        image=src,
        prompt=PROMPT,
        num_inference_steps=4,
        generator=torch.Generator("cuda").manual_seed(7),
    ).images[0]
    out.save(OUT / "edit_ref.png")
    print("edit_ref OK:", out.size, "peak VRAM GB:", round(torch.cuda.max_memory_allocated() / 1e9, 2))


if __name__ == "__main__":
    main()

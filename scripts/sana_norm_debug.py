"""Diagnostic: is the SANA colored-noise output a normalization-boundary bug?

Hypothesis: StreamForge frames are [0,1] (GpuFrame contract), but the SANA LTX2 VAE
operates in [-1,1] pixel space (reference read_video does Normalize(0.5,0.5); save_video
does 127.5*x+127.5). The runtime feeds [0,1] to vae_encode and clamp(0,1)s the decode.

This script drives ONE chunk two ways and reports evidence:
  A) CURRENT  : encode([0,1]) -> decode -> clamp(0,1)            (the shipped path)
  B) FIXED    : encode(x*2-1) -> decode -> (x+1)/2 -> clamp(0,1) (reference-matched)
Run under .venv-sana.
"""
from __future__ import annotations

import pathlib

import imageio.v3 as iio
import numpy as np
import torch

from streamforge.diffusion.runtime_sana_streaming import SanaStreamingRuntime

VIDEO = "D:/StreamForge/TestFile/DriveVideo.mp4"
OUTDIR = pathlib.Path("D:/StreamForge/out/sana_debug")
PROMPT = "a cinematic oil painting, vivid impasto brushstrokes"


def stats(name, t):
    f = t.detach().float()
    print(f"  {name:22s} shape={tuple(f.shape)} "
          f"min={f.min():.3f} max={f.max():.3f} mean={f.mean():.3f} std={f.std():.3f}")


def build_window(rt, frames01):
    window = [rt._fit(t) for t in frames01[:rt._pixel_per_chunk]]
    # [pixel_frames,1,3,H,W] -> [1,3,nf,H,W]
    return torch.cat(window, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(rt.device, rt.vae_dtype)


def run_once(rt, video01, *, normalize_input, denorm_output, seed=7):
    cfg = rt._config
    rt.reset_state()                          # fresh engine + kv-cache + buffer
    rt._gen.manual_seed(seed)                 # identical noise for both paths
    video = video01.clone()
    if normalize_input:
        video = video * 2.0 - 1.0             # [0,1] -> [-1,1]
    embeds = rt._vae_encode(cfg.vae.vae_type, rt._vae, video,
                            sample_posterior=False, device=rt.device).to(rt.vae_dtype)
    m = embeds.shape[2]
    noise = torch.randn(1, rt.vae_latent_dim, m, rt._latent_h, rt._latent_w,
                        generator=rt._gen, device=rt.device)
    lat = rt.engine.push_chunk(noise, embeds, 4).to(rt.vae_dtype)
    samples = rt._vae_decode(cfg.vae.vae_type, rt._vae, lat)  # raw, BEFORE any clamp
    raw = samples[:, :, 0]                                    # frame 0, [1,3,H,W]
    if denorm_output:
        shown = ((raw + 1.0) / 2.0).clamp(0, 1)
    else:
        shown = raw.clamp(0, 1)
    return embeds, raw, shown


def save(name, shown):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    arr = (shown[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    p = OUTDIR / name
    iio.imwrite(p, arr)
    print(f"  saved {p}")


def main():
    rt = SanaStreamingRuntime(internal_hw=(384, 640), default_steps=4)
    rt.load_once()
    rt.set_prompt(PROMPT)

    src = iio.imread(VIDEO, index=None)
    frames01 = [torch.from_numpy(np.asarray(fr)).float().div(255).permute(2, 0, 1).unsqueeze(0)
                for fr in src[:rt._pixel_per_chunk]]
    video01 = build_window(rt, frames01)
    print(f"\nINPUT video01 (what push_frame feeds today):")
    stats("video01", video01)

    print("\n[A] CURRENT path  encode([0,1]) -> decode -> clamp(0,1):")
    eA, rawA, shownA = run_once(rt, video01, normalize_input=False, denorm_output=False)
    stats("embeds", eA); stats("raw decode", rawA); stats("shown", shownA)
    save("A_current.png", shownA)

    print("\n[B] FIXED path    encode([-1,1]) -> decode -> (x+1)/2:")
    eB, rawB, shownB = run_once(rt, video01, normalize_input=True, denorm_output=True)
    stats("embeds", eB); stats("raw decode", rawB); stats("shown", shownB)
    save("B_fixed.png", shownB)

    # also save the input frame 0 (denormalized to [0,1] for viewing) for reference
    save("input0.png", video01[:, :, 0].clamp(0, 1))
    print("\nDone.")


if __name__ == "__main__":
    main()

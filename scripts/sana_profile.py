"""Profile where SANA-Streaming's per-chunk time goes: VAE-encode vs DiT vs VAE-decode.

Drives the real runtime internals (same math as runtime_sana_streaming.push_frame) with a
torch.cuda.synchronize() around each stage, so the 832 ms/chunk the operator sees is split into
its three components. Measures steps=4 and steps=2 so we can see exactly how much the live
"Steps" knob actually buys (the DiT runs steps+1 forwards; encode/decode are fixed per chunk).

Run under .venv-sana:  PYTHONPATH=src .venv-sana/Scripts/python.exe scripts/sana_profile.py
"""
from __future__ import annotations

import statistics as st
import time

import imageio.v3 as iio
import numpy as np
import torch

from streamforge.diffusion.runtime_sana_streaming import SanaStreamingRuntime

VIDEO = "D:/StreamForge/TestFile/DriveVideo.mp4"
PROMPT = "vivid oil painting, thick impasto brushstrokes"
INTERNAL_HW = (288, 384)   # the user's 4:3 384x288
DISPLAY_FPS = 30
WARMUP = 3                 # skip while the bounded KV window fills to steady state
TARGET = 15                # measured chunks per steps setting


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _fmt(x):
    return f"mean={st.mean(x):6.1f}  median={st.median(x):6.1f}  min={min(x):6.1f}  max={max(x):6.1f}"


def main():
    rt = SanaStreamingRuntime(internal_hw=INTERNAL_HW, default_steps=4, resync_every=0)
    rt.load_once()
    rt.set_prompt(PROMPT)

    src = iio.imread(VIDEO, index=None)
    src01 = [torch.from_numpy(np.asarray(fr)).float().div(255).permute(2, 0, 1).unsqueeze(0)
             for fr in src]
    n = len(src01)
    ppc = rt._pixel_per_chunk
    pixels_per_chunk_s = ppc / DISPLAY_FPS  # seconds of 30fps video produced per chunk
    print(f"internal_hw={INTERNAL_HW}  latent={rt._latent_h}x{rt._latent_w}  pixel/chunk={ppc}  "
          f"src_frames={n}  ({pixels_per_chunk_s*1000:.0f} ms of video per chunk)")

    for steps in (4, 2):
        rt.reset_state()  # fresh engine + cache for this setting
        enc_t, dit_t, dec_t, tot_t = [], [], [], []
        buf, i, measured = [], 0, 0
        while measured < WARMUP + TARGET:
            buf.append(rt._fit(src01[i % n]))
            i += 1
            if len(buf) < ppc:
                continue
            window, buf = buf[:ppc], buf[ppc:]
            video = torch.cat(window, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(rt.device, rt.vae_dtype)
            video = video * 2.0 - 1.0

            _sync(); t0 = time.perf_counter()
            embeds = rt._vae_encode(rt._config.vae.vae_type, rt._vae, video,
                                    sample_posterior=False, device=rt.device).to(rt.vae_dtype)
            _sync(); t1 = time.perf_counter()
            m = embeds.shape[2]
            noise = torch.randn(1, rt.vae_latent_dim, m, rt._latent_h, rt._latent_w,
                                generator=rt._gen, device=rt.device)
            lat = rt.engine.push_chunk(noise, embeds, steps).to(rt.vae_dtype)
            _sync(); t2 = time.perf_counter()
            samples = rt._vae_decode(rt._config.vae.vae_type, rt._vae, lat)
            _sync(); t3 = time.perf_counter()

            measured += 1
            if measured <= WARMUP:
                continue
            enc_t.append((t1 - t0) * 1000)
            dit_t.append((t2 - t1) * 1000)
            dec_t.append((t3 - t2) * 1000)
            tot_t.append((t3 - t0) * 1000)

        total = st.mean(tot_t)
        rt_ratio = pixels_per_chunk_s / (total / 1000.0)
        print(f"\n=== steps={steps}  (n={len(tot_t)} chunks) ===")
        print(f"  VAE-encode : {_fmt(enc_t)} ms   ({st.mean(enc_t)/total*100:4.1f}% of chunk)")
        print(f"  DiT        : {_fmt(dit_t)} ms   ({st.mean(dit_t)/total*100:4.1f}% of chunk, {steps}+1 fwd)")
        print(f"  VAE-decode : {_fmt(dec_t)} ms   ({st.mean(dec_t)/total*100:4.1f}% of chunk)")
        print(f"  TOTAL      : {_fmt(tot_t)} ms")
        print(f"  -> {rt_ratio:.2f}x real-time  ({'AHEAD' if rt_ratio >= 1 else 'BEHIND'}; "
              f"need {total/1000.0:.2f}s, have {pixels_per_chunk_s:.2f}s of video)")

    print("\ndone")


if __name__ == "__main__":
    main()

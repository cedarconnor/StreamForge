"""Build + validate the classic img2img path (no reference-token concat).

Step 1: encode->decode ROUNDTRIP to prove the latent-space handling (vae.encode ->
patchify -> bn-normalize -> pack, and its inverse) is correct.
Step 2: img2img at a couple of strengths via flow-matching scale_noise + truncated schedule.

Reuses the pipeline's own helpers so the math matches exactly.
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_timesteps, compute_empirical_mu

OUT = pathlib.Path("out/classic")
CLIP = r"D:\StreamForge\TestFile\DriveVideo.mp4"
RES = 512


def first_frame(size=RES) -> Image.Image:
    cap = cv2.VideoCapture(CLIP)
    ok, bgr = cap.read()
    cap.release()
    assert ok
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((size, size))


def encode_to_gen_latent(pipe, pil, device):
    """image -> (packed latents, latent_ids, pre-pack latent shape) in the generation space."""
    pre = pipe.image_processor.preprocess(pil, height=RES, width=RES, resize_mode="crop")
    pre = pre.to(device, pipe.vae.dtype)
    lat = pipe._encode_vae_image(pre, generator=torch.Generator(device).manual_seed(0))  # (1,128,32,32) bn-normalized
    ids = pipe._prepare_latent_ids(lat).to(device)
    packed = pipe._pack_latents(lat)  # (1,1024,128)
    return packed, ids


def decode_gen_latent(pipe, packed, ids):
    lh = 2 * (RES // (pipe.vae_scale_factor * 2))      # 64
    lat = pipe._unpack_latents_with_ids(packed, ids, lh // 2, lh // 2)  # (1,128,32,32)
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
    std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(lat.device, lat.dtype)
    lat = lat * std + mean
    lat = pipe._unpatchify_latents(lat)               # (1,32,64,64)
    img = pipe.vae.decode(lat, return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type="pil")[0]


def img2img(pipe, packed, ids, prompt_embeds, text_ids, strength, steps, device, seed=7):
    image_seq_len = packed.shape[1]
    sigmas = np.linspace(1.0, 1 / steps, steps)
    if getattr(pipe.scheduler.config, "use_flow_sigmas", False):
        sigmas = None
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=steps)
    timesteps, steps = retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas, mu=mu)
    # truncate schedule to the requested strength
    init = min(int(steps * strength), steps)
    t_start = max(steps - init, 0)
    timesteps = timesteps[t_start:]
    pipe.scheduler.set_begin_index(t_start)
    noise = torch.randn(packed.shape, generator=torch.Generator(device).manual_seed(seed),
                        device=device, dtype=packed.dtype)
    latents = pipe.scheduler.scale_noise(packed, timesteps[:1], noise).to(pipe.transformer.dtype)
    for t in timesteps:
        ts = t.expand(latents.shape[0]).to(latents.dtype) / 1000
        noise_pred = pipe.transformer(
            hidden_states=latents, timestep=ts, guidance=None,
            encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=ids,
            return_dict=False,
        )[0]
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return decode_gen_latent(pipe, latents, ids)


@torch.no_grad()
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    pipe = Flux2KleinPipeline.from_pretrained("models/transformer", torch_dtype=torch.bfloat16).to(dev)
    pipe.set_progress_bar_config(disable=True)
    src = first_frame()
    src.save(OUT / "src.png")

    # 1) roundtrip
    packed, ids = encode_to_gen_latent(pipe, src, dev)
    rt = decode_gen_latent(pipe, packed, ids)
    rt.save(OUT / "roundtrip.png")
    a = np.asarray(src).astype(np.float32); b = np.asarray(rt).astype(np.float32)
    print(f"roundtrip mean abs err (0-255): {np.abs(a - b).mean():.1f}  (low = latent handling correct)")

    # 2) img2img at two strengths
    embeds, tids = pipe.encode_prompt("vivid oil painting, thick impasto brushstrokes", device=dev)
    for s in (0.6, 0.9):
        out = img2img(pipe, packed, ids, embeds, tids, strength=s, steps=4, device=dev)
        out.save(OUT / f"img2img_s{int(s*100)}.png")
        print(f"img2img strength={s} -> out/classic/img2img_s{int(s*100)}.png")


if __name__ == "__main__":
    main()

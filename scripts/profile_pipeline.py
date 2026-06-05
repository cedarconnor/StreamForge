"""Where does the per-frame time go? Decide if taef2 (VAE swap) is worth the integration.

Times: full VAE encode of a 512 ref, full VAE decode, a txt2img 4-step (transformer-ish
floor, no reference tokens), and the full image= restyle. The gap between restyle and
txt2img isolates the reference-encode + concatenated-token cost.
"""
from __future__ import annotations

import time

import torch
from PIL import Image
import numpy as np
from diffusers import Flux2KleinPipeline


def _t(fn, n=5):
    fn()  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


def main() -> None:
    dev = "cuda"
    pipe = Flux2KleinPipeline.from_pretrained("models/transformer", torch_dtype=torch.bfloat16).to(dev)
    pipe.set_progress_bar_config(disable=True)
    embeds, _ = pipe.encode_prompt("vivid oil painting, thick impasto", device=dev)
    img = Image.fromarray((np.random.rand(512, 512, 3) * 255).astype("uint8"))
    img_t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].float().div(255).mul(2).sub(1).to(dev, torch.bfloat16)

    # VAE encode/decode in isolation
    def enc():
        with torch.no_grad():
            return pipe.vae.encode(img_t).latent_dist.sample()
    lat = enc()

    def dec():
        with torch.no_grad():
            return pipe.vae.decode(lat).sample

    t_enc = _t(enc); t_dec = _t(dec)

    def txt2img():
        return pipe(prompt_embeds=embeds, height=512, width=512, num_inference_steps=4,
                    generator=torch.Generator(dev).manual_seed(7)).images[0]

    def restyle():
        return pipe(image=img, prompt_embeds=embeds, height=512, width=512, num_inference_steps=4,
                    generator=torch.Generator(dev).manual_seed(7)).images[0]

    t_txt = _t(txt2img, n=5); t_restyle = _t(restyle, n=5)
    print(f"VAE encode (512):   {t_enc:7.1f} ms")
    print(f"VAE decode (512):   {t_dec:7.1f} ms")
    print(f"txt2img 4-step:     {t_txt:7.1f} ms   (transformer, no reference tokens)")
    print(f"image= restyle:     {t_restyle:7.1f} ms   (reference encode + concat tokens + decode)")
    print(f"reference overhead: {t_restyle - t_txt:7.1f} ms   (restyle - txt2img)")


if __name__ == "__main__":
    main()

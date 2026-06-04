---
license: mit
---

# 🍰 Tiny AutoEncoder for FLUX.2

[TAEF2](https://github.com/madebyollin/taesd) is very tiny autoencoder which uses the same "latent API" as FLUX.2's VAE.
FLUX.2 is useful for real-time previewing of the FLUX.2 generation process, as well as general resource-constrained encoding/decoding.

This repo contains `.safetensors` versions of the TAEF2 weights.

## Using in 🧨 diffusers

**NOTE**: Unlike TAEF1, TAEF2's architecture [isn't properly integrated](https://github.com/madebyollin/taesd/issues/35#issuecomment-3765620926) into Diffusers yet.
So for now you'll want some wrapper code:

```sh
pip install git+https://www.github.com/huggingface/diffusers # needed for Klein support as of 2026-01-18
wget -nc -nv https://raw.githubusercontent.com/madebyollin/taesd/refs/heads/main/taesd.py -O taesd.py
wget -nc -nv https://huggingface.co/madebyollin/taef2/resolve/main/taef2.safetensors -O taef2.safetensors
```

```python
# Construction
from taesd import TAESD
import torch
import safetensors.torch as stt
from diffusers.utils.accelerate_utils import apply_forward_hook

def convert_diffusers_sd_to_taesd(sd):
    out = {}
    for k, v in sd.items():
        encdec, _layers, index, *suffix = k.split(".")
        offset = 0
        if encdec == "decoder":
            offset = +1
        out[".".join([encdec, str(int(index)+offset), *suffix])] = v
    return out

class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class DiffusersTAEF2Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.bfloat16
        self.taesd = TAESD(encoder_path=None, decoder_path=None, latent_channels=32, arch_variant="flux_2").to(self.dtype)
        self.taesd.load_state_dict(convert_diffusers_sd_to_taesd(stt.load_file("taef2.safetensors")))
        self.bn = torch.nn.BatchNorm2d(128, affine=False, eps=0.0) # default bn
        self.config = DotDict(batch_norm_eps=self.bn.eps)

    @apply_forward_hook
    def encode(self, x):
        return DotDict(latent_dist=DotDict(sample=lambda : self.taesd.encoder(x.to(self.dtype).mul(0.5).add_(0.5)).to(x.dtype)))

    @apply_forward_hook
    def decode(self, x, return_dict=True):
        x = self.taesd.decoder(x.to(self.dtype)).mul(2).sub_(1).clamp_(-1, 1).to(x.dtype)
        return dict(sample=x) if return_dict else x,

taef2_diffusers = DiffusersTAEF2Wrapper().eval().requires_grad_(False)

# Usage
from diffusers import Flux2KleinPipeline

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype)
pipe.vae = taef2_diffusers
pipe.enable_sequential_cpu_offload() # pipe.enable_model_cpu_offload() # pipe = pipe.to(device)

prompt = "A slice of delicious New York-style berry cheesecake"
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device="cpu").manual_seed(0)
).images[0]
image.save("flux-klein.png")
image
```


![image](https://cdn-uploads.huggingface.co/production/uploads/630447d40547362a22a969a2/liXpUtbsz-g8ZgYeuJLk8.png)

## Quality Comparisons

These compare TAEF2, the [full FLUX.2 VAE](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/vae/config.json), and the alternate [FAL FLUX.2 Tiny](https://huggingface.co/fal/FLUX.2-Tiny-AutoEncoder) AE.

![image](https://cdn-uploads.huggingface.co/production/uploads/630447d40547362a22a969a2/H2wjbbH1rf4x1GF8RKphB.png)

![image](https://cdn-uploads.huggingface.co/production/uploads/630447d40547362a22a969a2/RXKUncrfj4dGtjGiUVUN9.png)

![image](https://cdn-uploads.huggingface.co/production/uploads/630447d40547362a22a969a2/oCwOP1wkagSTM71Xn_rDH.png)

![image](https://cdn-uploads.huggingface.co/production/uploads/630447d40547362a22a969a2/OcC8pVRIGW7b7JKTfVfJg.png)
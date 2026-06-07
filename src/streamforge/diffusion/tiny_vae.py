"""TAEF2 tiny-VAE wrapper — a drop-in replacement for FLUX.2-klein's `pipe.vae`.

Why: after Track-A (low internal res + 1-step + compile) cut the transformer ~4x, the full VAE
encode+decode is now ~33% of the fast path (~30 ms of ~90 ms @384x224). taef2 is a tiny conv
autoencoder trained on FLUX.2's *normalized* latent space, so it slots into the exact same
pipeline math.

Latent-space contract (verified against pipeline_flux2_klein.py):
  encode path:  vae.encode -> _patchify -> (x - bn.mean)/bn.std
  decode path:  x*bn.std + bn.mean -> _unpatchify -> vae.decode
The full VAE's `bn` denormalizes to its native latent space. taef2 is trained directly on the
NORMALIZED latents, so this wrapper supplies an IDENTITY bn (mean=0, var=1, eps=0) -> both
normalize/denormalize steps become no-ops and taef2 round-trips in the diffusion space.

Wrapper must also expose `latent_dist.mode()` because `_encode_vae_image` calls
`retrieve_latents(..., sample_mode="argmax")` -> `.mode()` (the upstream README wrapper only had
`.sample`, since its example was txt2img and never encoded).

taesd.py is vendored in `_taesd.py` (MIT, madebyollin).
"""
from __future__ import annotations

import torch

from streamforge.diffusion._taesd import TAESD


class _DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class _LatentDist:
    """taef2 is deterministic: mode() and sample() both return the encoder output."""

    def __init__(self, z: torch.Tensor):
        self._z = z

    def mode(self) -> torch.Tensor:        # used by sample_mode="argmax" (img2img encode)
        return self._z

    def sample(self, generator=None) -> torch.Tensor:  # used by sample_mode="sample"
        return self._z


class _EncoderOutput:
    def __init__(self, z: torch.Tensor):
        self.latent_dist = _LatentDist(z)


def _convert_diffusers_sd_to_taesd(sd: dict) -> dict:
    """Map `encoder.layers.N.suffix` (safetensors) -> `encoder.N.suffix` (nn.Sequential index).
    Decoder gets +1 because its Sequential starts with a param-less Clamp() at index 0."""
    out = {}
    for k, v in sd.items():
        encdec, _layers, index, *suffix = k.split(".")
        offset = 1 if encdec == "decoder" else 0
        out[".".join([encdec, str(int(index) + offset), *suffix])] = v
    return out


class TAEF2VAE(torch.nn.Module):
    """Drop-in for `Flux2KleinPipeline.vae`: provides encode/decode/bn/config matching the
    attributes the klein pipeline (and StreamForge's img2img loop) touches."""

    def __init__(self, weights_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        import safetensors.torch as stt
        self.taesd = TAESD(encoder_path=None, decoder_path=None,
                           latent_channels=32, arch_variant="flux_2").to(dtype)
        self.taesd.load_state_dict(_convert_diffusers_sd_to_taesd(stt.load_file(weights_path)))
        # IDENTITY bn (mean=0, var=1, eps=0) so the pipeline's normalize/denormalize are no-ops.
        self.bn = torch.nn.BatchNorm2d(128, affine=False, eps=0.0)
        self.config = _DotDict(batch_norm_eps=self.bn.eps)
        self.eval().requires_grad_(False).to(device)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> _EncoderOutput:
        # taesd encoder expects [0,1]; pipeline feeds images in [-1,1].
        z = self.taesd.encoder(x.to(self.dtype).mul(0.5).add(0.5)).to(x.dtype)
        return _EncoderOutput(z)

    @torch.no_grad()
    def decode(self, x: torch.Tensor, return_dict: bool = True):
        out = self.taesd.decoder(x.to(self.dtype)).mul(2).sub(1).clamp(-1, 1).to(x.dtype)
        if return_dict:
            return _DotDict(sample=out)
        return (out,)


def build_taef2(weights_path: str = "models/vae_tiny/taef2.safetensors",
                device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> TAEF2VAE:
    return TAEF2VAE(weights_path, device=device, dtype=dtype)

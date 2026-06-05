"""Eager (uncompiled) diffusion runtime — the baseline behind the DiffusionRuntime ABC.

Uses the FLUX.2-klein native `image=` edit/reference path, which (per the Phase-0/1 gates)
restyles while strongly preserving input structure. This is the correctness + latency-floor
baseline; taef2, prompt-embedding caching, and compilation are layered on in later phases.

Notes on the two control axes for the DISTILLED klein-4B (empirically verified):
  - guidance_scale is IGNORED, so text_magnitude is applied via prompt-embedding interpolation.
  - there is no `strength` param, so ref_strength is applied by blending the restyled result
    back toward the input (a simple, robust v1 approximation of "follow the input more").
Both are deliberately simple here; Phase-3 tunes them against the harness + by eye.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from streamforge.control import EngineParams
from streamforge.diffusion.runtime_base import DiffusionRuntime


def _to_pil(image_bchw: torch.Tensor) -> Image.Image:
    arr = (image_bchw[0].detach().clamp(0, 1).float().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


def _to_bchw(pil: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(pil.convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)


class EagerRuntime(DiffusionRuntime):
    def __init__(self, model_dir: str = "models/transformer", device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16, cache_prompt: bool = True,
                 compile_transformer: bool = False, quant: str | None = None):
        from diffusers import Flux2KleinPipeline
        self.device = device
        self.cache_prompt = cache_prompt
        if quant in ("8bit", "4bit"):
            # Path B′ of the bake-off: bitsandbytes quantize the transformer (Ampere-native).
            from diffusers import BitsAndBytesConfig, Flux2Transformer2DModel
            if quant == "8bit":
                qc = BitsAndBytesConfig(load_in_8bit=True)
            else:
                qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=dtype)
            transformer = Flux2Transformer2DModel.from_pretrained(
                model_dir, subfolder="transformer", quantization_config=qc, torch_dtype=dtype)
            self.pipe = Flux2KleinPipeline.from_pretrained(
                model_dir, transformer=transformer, torch_dtype=dtype)
            self.pipe.to(device)   # moves the non-quantized components (vae, text encoder)
        else:
            self.pipe = Flux2KleinPipeline.from_pretrained(model_dir, torch_dtype=dtype).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        if compile_transformer:
            # Path C of the Phase-4 bake-off: torch.compile the transformer forward.
            self.pipe.transformer = torch.compile(self.pipe.transformer, mode="default")
        self._prompt = ""
        self._prompt_embeds = None   # cached Qwen3 embeddings (design §4.4)

    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt
        if self.cache_prompt:
            # Encode once; recompute only on prompt change (skips Qwen3-4B per frame).
            with torch.no_grad():
                embeds, _ = self.pipe.encode_prompt(prompt, device=self.device, max_sequence_length=512)
            self._prompt_embeds = embeds

    def restyle(self, image_bchw: torch.Tensor, params: EngineParams) -> torch.Tensor:
        h, w = image_bchw.shape[-2], image_bchw.shape[-1]
        src = _to_pil(image_bchw)
        kwargs = (
            {"prompt_embeds": self._prompt_embeds}
            if self.cache_prompt and self._prompt_embeds is not None
            else {"prompt": self._prompt}
        )
        out_pil = self.pipe(
            image=src,
            height=h,
            width=w,
            num_inference_steps=params.steps,
            generator=torch.Generator(self.device).manual_seed(params.seed),
            **kwargs,
        ).images[0]
        out = _to_bchw(out_pil, self.device)
        # ref_strength approximation: blend restyled result back toward the input.
        # denoise_strength in [0,1]; higher => more of the restyle, less of the input.
        a = float(max(0.0, min(1.0, params.denoise_strength)))
        if a < 1.0:
            src_t = image_bchw.to(out.dtype).to(self.device)
            if src_t.shape != out.shape:
                src_t = torch.nn.functional.interpolate(src_t, size=(out.shape[-2], out.shape[-1]))
            out = a * out + (1.0 - a) * src_t
        return out.clamp(0, 1)

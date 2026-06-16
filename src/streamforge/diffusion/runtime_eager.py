"""Eager (uncompiled) diffusion runtime — the baseline behind the DiffusionRuntime ABC.

Two restyle modes, both validated on the A6000 (each a distinct cadence×quality×control point):

  mode="edit" (default): FLUX.2-klein native `image=` reference path. STRONG structure
    preservation + strong style, but ~1032 ms @512² (reference tokens are concatenated,
    ~doubling the sequence). ref_strength approximated by blending the result toward the input.

  mode="img2img": classic flow-matching loop (encode -> noise@strength -> denoise SAME tokens,
    no concat). ~600 ms @512² (~40% faster) and a REAL ref_strength knob via denoise strength,
    but high stylization loses input structure (the design §7.2 classic-img2img weakness).

For the DISTILLED klein-4B: guidance_scale is IGNORED; text influence comes from the prompt
embeddings (cached via encode_prompt). Both paths use the same cached embeddings.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from streamforge.aspect import FitMode, Size, fit_tensor
from streamforge.control import EngineParams
from streamforge.diffusion.runtime_base import DiffusionRuntime


def _to_pil(image_bchw: torch.Tensor) -> Image.Image:
    arr = (image_bchw[0].detach().clamp(0, 1).float().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


def _to_bchw(pil: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(pil.convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)


def effective_embeds(prompt_embeds, neutral_embeds, tm: float):
    """Lerp/extrapolate prompt embeddings toward neutral by text_magnitude `tm`.

    tm == 1 -> plain prompt (no-op, regression-safe). tm == 0 -> neutral. tm > 1 -> exaggerate.
    Falls back to the plain prompt embeds when neutral is missing or shapes don't match.
    """
    if prompt_embeds is None:
        return None
    if neutral_embeds is None or abs(tm - 1.0) < 1e-3 \
            or neutral_embeds.shape != prompt_embeds.shape:
        return prompt_embeds
    return neutral_embeds + (prompt_embeds - neutral_embeds) * tm


class EagerRuntime(DiffusionRuntime):
    def __init__(self, model_dir: str = "models/transformer", device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16, cache_prompt: bool = True,
                 compile_transformer: bool = False, quant: str | None = None,
                 mode: str = "edit", internal_hw: tuple[int, int] | None = None,
                 compile_mode: str = "default", tiny_vae: bool = False):
        from diffusers import Flux2KleinPipeline
        self.device = device
        self.dtype = dtype
        self.cache_prompt = cache_prompt
        self.mode = mode
        # Run diffusion at a low INTERNAL resolution (snapped to /16 for FLUX latent packing),
        # then upscale the result back to the source frame's size. Biggest Windows-native speed
        # lever: 384x224 ~3x faster than 512² (Track-A perf finding). None = native source res.
        self.internal_hw = self._snap_hw(internal_hw) if internal_hw else None
        self.fit_plan = None
        if quant in ("8bit", "4bit"):
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
            self.pipe.to(device)
        else:
            self.pipe = Flux2KleinPipeline.from_pretrained(model_dir, torch_dtype=dtype).to(device)
            if quant == "int8ao":
                # TorchAO INT8 weight-only + torch.compile (fused dequant+matmul, needs Triton).
                # Weight-only avoids the W8A8 dynamic-activation kernel's M>16 constraint, which
                # the transformer's small adaLN/modulation linears (M=1) violate.
                from torchao.quantization import quantize_, Int8WeightOnlyConfig
                quantize_(self.pipe.transformer, Int8WeightOnlyConfig())
        if tiny_vae:
            # Swap the full FLUX.2 VAE for taef2. vae_scale_factor (already cached, =8) is unchanged
            # since taef2 also downsamples 8x; the wrapper's identity bn keeps the latent math valid.
            from streamforge.diffusion.tiny_vae import build_taef2
            self.pipe.vae = build_taef2(device=device, dtype=dtype)
        self.pipe.set_progress_bar_config(disable=True)
        if compile_transformer:
            # mode="reduce-overhead" => Inductor cudagraph-trees: the transformer's kernel-launch
            # sequence is captured once (static shape, needs a few warmup iters) and replayed as a
            # single launch, killing the per-kernel Python/launch overhead in the fixed floor.
            self.pipe.transformer = torch.compile(self.pipe.transformer, mode=compile_mode)
        self._prompt = ""
        self._prompt_embeds = None
        self._neutral_embeds = None
        self._text_ids = None

    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt
        if self.cache_prompt:
            with torch.no_grad():
                embeds, text_ids = self.pipe.encode_prompt(prompt, device=self.device, max_sequence_length=512)
                neutral, _ = self.pipe.encode_prompt("", device=self.device, max_sequence_length=512)
            self._prompt_embeds, self._text_ids = embeds, text_ids
            self._neutral_embeds = neutral

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def _effective_embeds(self, tm: float):
        return effective_embeds(self._prompt_embeds, self._neutral_embeds, tm)

    @staticmethod
    def _snap_hw(hw: tuple[int, int]) -> tuple[int, int]:
        h, w = hw
        return (max(16, round(h / 16) * 16), max(16, round(w / 16) * 16))

    @torch.no_grad()
    def restyle(self, image_bchw: torch.Tensor, params: EngineParams) -> torch.Tensor:
        # Downscale to internal diffusion res (if set), restyle, upscale back to source size.
        if self.internal_hw is not None:
            out_h, out_w = image_bchw.shape[-2], image_bchw.shape[-1]
            ih, iw = self.internal_hw
            if (ih, iw) != (out_h, out_w):
                small, plan = fit_tensor(
                    image_bchw.to(self.device),
                    Size(width=iw, height=ih),
                    FitMode.FILL_CROP,
                    antialias=True,
                )
                self.fit_plan = plan
                styled = self._dispatch(small, params)
                return torch.nn.functional.interpolate(
                    styled, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return self._dispatch(image_bchw, params)

    def _dispatch(self, image_bchw: torch.Tensor, params: EngineParams) -> torch.Tensor:
        if self.mode == "img2img":
            return self._restyle_img2img(image_bchw, params)
        return self._restyle_edit(image_bchw, params)

    # --- mode="edit": native image= reference path -------------------------------------------
    def _restyle_edit(self, image_bchw: torch.Tensor, params: EngineParams) -> torch.Tensor:
        h, w = image_bchw.shape[-2], image_bchw.shape[-1]
        src = _to_pil(image_bchw)
        embeds = self._effective_embeds(params.text_magnitude)
        kwargs = (
            {"prompt_embeds": embeds}
            if self.cache_prompt and embeds is not None
            else {"prompt": self._prompt}
        )
        out_pil = self.pipe(
            image=src, height=h, width=w, num_inference_steps=params.steps,
            generator=torch.Generator(self.device).manual_seed(params.seed), **kwargs,
        ).images[0]
        out = _to_bchw(out_pil, self.device)
        a = float(max(0.0, min(1.0, params.denoise_strength)))
        if a < 1.0:
            src_t = image_bchw.to(out.dtype).to(self.device)
            if src_t.shape != out.shape:
                src_t = torch.nn.functional.interpolate(src_t, size=(out.shape[-2], out.shape[-1]))
            out = a * out + (1.0 - a) * src_t
        return out.clamp(0, 1)

    # --- mode="img2img": classic flow-matching loop, no reference concat ----------------------
    def _restyle_img2img(self, image_bchw: torch.Tensor, params: EngineParams) -> torch.Tensor:
        from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_timesteps, compute_empirical_mu
        p = self.pipe
        h, w = image_bchw.shape[-2], image_bchw.shape[-1]
        img_m1p1 = (image_bchw.to(self.device, self.dtype) * 2 - 1)
        lat = p._encode_vae_image(img_m1p1, generator=torch.Generator(self.device).manual_seed(params.seed))
        ids = p._prepare_latent_ids(lat).to(self.device)
        packed = p._pack_latents(lat)

        steps = params.steps
        sigmas = np.linspace(1.0, 1 / steps, steps)
        if getattr(p.scheduler.config, "use_flow_sigmas", False):
            sigmas = None
        mu = compute_empirical_mu(image_seq_len=packed.shape[1], num_steps=steps)
        timesteps, steps = retrieve_timesteps(p.scheduler, steps, self.device, sigmas=sigmas, mu=mu)
        init = min(int(steps * params.denoise_strength), steps)
        t_start = max(steps - init, 0)
        timesteps = timesteps[t_start:]
        p.scheduler.set_begin_index(t_start)

        noise = torch.randn(packed.shape, generator=torch.Generator(self.device).manual_seed(params.seed),
                            device=self.device, dtype=packed.dtype)
        latents = p.scheduler.scale_noise(packed, timesteps[:1], noise).to(p.transformer.dtype)
        embeds = self._effective_embeds(params.text_magnitude)
        if embeds is None:
            embeds = p.encode_prompt(self._prompt, device=self.device)[0]
        text_ids = self._text_ids if self._text_ids is not None else \
            p.encode_prompt(self._prompt, device=self.device)[1]
        for t in timesteps:
            ts = t.expand(latents.shape[0]).to(latents.dtype) / 1000
            noise_pred = p.transformer(
                hidden_states=latents, timestep=ts, guidance=None,
                encoder_hidden_states=embeds, txt_ids=text_ids, img_ids=ids, return_dict=False,
            )[0]
            latents = p.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        lh = 2 * (h // (p.vae_scale_factor * 2))
        lw = 2 * (w // (p.vae_scale_factor * 2))
        lat = p._unpack_latents_with_ids(latents, ids, lh // 2, lw // 2)
        mean = p.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
        std = torch.sqrt(p.vae.bn.running_var.view(1, -1, 1, 1) + p.vae.config.batch_norm_eps).to(lat.device, lat.dtype)
        lat = lat * std + mean
        lat = p._unpatchify_latents(lat)
        img = p.vae.decode(lat, return_dict=False)[0]
        return p.image_processor.postprocess(img, output_type="pt")[0].unsqueeze(0).to(self.device).clamp(0, 1)

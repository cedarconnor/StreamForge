"""Stage breakdown of the img2img fast path, to decide whether CUDA graphs (launch overhead)
or a tiny VAE / INT4 (compute) is the right next lever.

Times, with cuda events (averaged over N frames, after warmup): VAE encode, pack/prep, the
transformer denoise loop, unpack+denorm, VAE decode. If the non-transformer stages are large and
look launch-bound -> full-pipeline CUDA graph helps. If they're VAE *compute* -> taef2. If the
transformer dominates -> INT4/WSL.

  python scripts/profile_img2img.py --in-res 384x224 --preset PRESERVE --compile
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from streamforge.control import TwoAxisControl
from streamforge.diffusion.runtime_eager import EagerRuntime
from streamforge.sources.synthetic import SyntheticSource


class _Ev:
    def __init__(self):
        self.acc = {}
        self.n = 0

    def lap(self, name, start, end):
        self.acc[name] = self.acc.get(name, 0.0) + start.elapsed_time(end)


def profile(rt: EagerRuntime, w, h, params, frames, warmup):
    from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_timesteps, compute_empirical_mu
    p = rt.pipe
    src = SyntheticSource(w, h, fps=30)
    src.open()
    ev = _Ev()

    def one(image_bchw, record):
        e = {k: torch.cuda.Event(enable_timing=True) for k in
             ("a", "b", "c", "d", "e", "f")}
        img_m1p1 = (image_bchw.to(rt.device, rt.dtype) * 2 - 1)
        e["a"].record()
        lat = p._encode_vae_image(img_m1p1, generator=torch.Generator(rt.device).manual_seed(params.seed))
        e["b"].record()
        ids = p._prepare_latent_ids(lat).to(rt.device)
        packed = p._pack_latents(lat)
        steps = params.steps
        sigmas = np.linspace(1.0, 1 / steps, steps)
        if getattr(p.scheduler.config, "use_flow_sigmas", False):
            sigmas = None
        mu = compute_empirical_mu(image_seq_len=packed.shape[1], num_steps=steps)
        timesteps, steps = retrieve_timesteps(p.scheduler, steps, rt.device, sigmas=sigmas, mu=mu)
        init = min(int(steps * params.denoise_strength), steps)
        t_start = max(steps - init, 0)
        timesteps = timesteps[t_start:]
        p.scheduler.set_begin_index(t_start)
        noise = torch.randn(packed.shape, generator=torch.Generator(rt.device).manual_seed(params.seed),
                            device=rt.device, dtype=packed.dtype)
        latents = p.scheduler.scale_noise(packed, timesteps[:1], noise).to(p.transformer.dtype)
        embeds, text_ids = rt._prompt_embeds, rt._text_ids
        e["c"].record()
        for t in timesteps:
            ts = t.expand(latents.shape[0]).to(latents.dtype) / 1000
            noise_pred = p.transformer(hidden_states=latents, timestep=ts, guidance=None,
                                       encoder_hidden_states=embeds, txt_ids=text_ids,
                                       img_ids=ids, return_dict=False)[0]
            latents = p.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        e["d"].record()
        lh = 2 * (h // (p.vae_scale_factor * 2)); lw = 2 * (w // (p.vae_scale_factor * 2))
        lat = p._unpack_latents_with_ids(latents, ids, lh // 2, lw // 2)
        mean = p.vae.bn.running_mean.view(1, -1, 1, 1).to(lat.device, lat.dtype)
        std = torch.sqrt(p.vae.bn.running_var.view(1, -1, 1, 1) + p.vae.config.batch_norm_eps).to(lat.device, lat.dtype)
        lat = lat * std + mean
        lat = p._unpatchify_latents(lat)
        e["e"].record()
        img = p.vae.decode(lat, return_dict=False)[0]
        _ = p.image_processor.postprocess(img, output_type="pt")
        e["f"].record()
        torch.cuda.synchronize()
        if record:
            ev.lap("vae_encode", e["a"], e["b"])
            ev.lap("prep/pack", e["b"], e["c"])
            ev.lap("transformer_loop", e["c"], e["d"])
            ev.lap("unpack+denorm", e["d"], e["e"])
            ev.lap("vae_decode+post", e["e"], e["f"])
            ev.n += 1

    with torch.no_grad():
        for _ in range(warmup):
            one(src.read().tensor, record=False)
        for _ in range(frames):
            one(src.read().tensor, record=True)
    src.close()
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-res", default="384x224")
    ap.add_argument("--preset", default="PRESERVE")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument("--tiny-vae", action="store_true")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=8)
    args = ap.parse_args()

    w, h = (int(v) for v in args.in_res.lower().split("x"))
    h, w = EagerRuntime._snap_hw((h, w))
    rt = EagerRuntime(mode="img2img", internal_hw=(h, w), tiny_vae=args.tiny_vae,
                      compile_transformer=args.compile, compile_mode=args.compile_mode)
    rt.set_prompt("vivid oil painting, thick impasto brushstrokes")
    params = TwoAxisControl.preset(args.preset).to_engine_params()

    ev = profile(rt, w, h, params, args.frames, args.warmup)
    total = sum(ev.acc.values()) / ev.n
    print(f"\nstage breakdown @ {w}x{h} {args.preset} compile={args.compile}({args.compile_mode}) "
          f"over {ev.n} frames:")
    for k, v in ev.acc.items():
        ms = v / ev.n
        print(f"  {k:>18}: {ms:7.2f} ms  ({100*ms/total:4.1f}%)")
    print(f"  {'TOTAL':>18}: {total:7.2f} ms  ({1000/total:.1f} fps)")


if __name__ == "__main__":
    main()

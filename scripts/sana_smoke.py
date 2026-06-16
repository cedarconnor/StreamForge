"""Phase-1 Task 2 acceptance + Phase-0 Task 5 smoke (run under .venv-sana, [gpu]).

Loads the SANA streaming stack once, then:
  - runs the stock batch sampler.sample() as the reference
  - runs SanaStreamingEngine incrementally (one chunk at a time, persistent KV cache)
  - asserts the two match within 1e-2 MAE (continuity preserved across calls)
  - decodes the incremental latents to prove TENSORS out, no MP4/PIL/disk on the hot path

Run: PYTHONPATH set by _bootstrap; just `.venv-sana\\Scripts\\python.exe scripts\\sana_smoke.py`
"""
from __future__ import annotations

import importlib.util
import time

import torch

from streamforge.diffusion.sana._bootstrap import _SANA, ensure_sana_importable

ensure_sana_importable()

import pyrallis  # noqa: E402
from accelerate import Accelerator  # noqa: E402

from diffusion.model.builder import (  # noqa: E402
    get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode,
)
from diffusion.model.utils import get_weight_dtype  # noqa: E402
from diffusion.scheduler.sana_streaming_sampler import SANAStreamingSampler  # noqa: E402

from streamforge.diffusion.sana.engine import SanaStreamingEngine  # noqa: E402

CONFIG = str(_SANA / "configs" / "sana_streaming" / "sana_streaming_2b_720p.yaml")
VIDEO = "D:/StreamForge/TestFile/DriveVideo.mp4"
HEIGHT, WIDTH, NUM_FRAMES, STEPS = 384, 640, 49, 4


def _load_script_module():
    path = _SANA / "inference_video_scripts" / "v2v" / "inference_sana_streaming.py"
    spec = importlib.util.spec_from_file_location("sana_stream_script", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_continuity_check():
    S = _load_script_module()
    config = pyrallis.parse(config_class=S.InferenceConfig, config_path=CONFIG, args=[])
    accelerator = Accelerator(mixed_precision=config.model.mixed_precision)
    device = accelerator.device
    weight_dtype = get_weight_dtype(config.model.mixed_precision)
    vae_dtype = get_weight_dtype(config.vae.weight_dtype)
    vae_stride = config.vae.vae_stride
    latent_t = (NUM_FRAMES - 1) // vae_stride[0] + 1
    latent_h = HEIGHT // vae_stride[1]
    latent_w = WIDTH // vae_stride[2]
    latent_size = config.model.image_size // config.vae.vae_downsample_rate
    flow_shift = config.scheduler.inference_flow_shift or config.scheduler.flow_shift

    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device=device, dtype=vae_dtype, config=config.vae)
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(config.text_encoder.text_encoder_name, device=device)
    neg_embeds, _ = S.encode_prompt(tokenizer, text_encoder, "", config, device, use_chi_prompt=False)
    model = S.load_model(config, latent_size, device, weight_dtype, S.DEFAULTS["long_streaming"]["model_path"])
    model, text_encoder = accelerator.prepare(model, text_encoder)
    prompt_embeds, prompt_mask = S.encode_prompt(tokenizer, text_encoder, "a cinematic oil painting",
                                                 config, device, use_chi_prompt=True)

    video = S.read_video(VIDEO, latent_h * vae_stride[1], latent_w * vae_stride[2], NUM_FRAMES)
    video = video.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=vae_dtype)
    image_vae_embeds = vae_encode(config.vae.vae_type, vae, video, sample_posterior=False, device=device).to(vae_dtype)

    gen = torch.Generator(device=device).manual_seed(7)
    noise = torch.randn(1, config.vae.vae_latent_dim, latent_t, latent_h, latent_w, device=device, generator=gen)
    base_chunk_frames = 24 // vae_stride[0]

    def mk(cls):
        return cls(model, condition=prompt_embeds, uncondition=neg_embeds, cfg_scale=1.0, flow_shift=flow_shift,
                   model_kwargs={"data_info": {"image_vae_embeds": image_vae_embeds}, "mask": prompt_mask},
                   base_chunk_frames=base_chunk_frames, num_cached_blocks=2, cache_strategy="fixed_rope",
                   efficient_cache=False, sink_token=True)

    ref = mk(SANAStreamingSampler).sample(noise.clone(), steps=STEPS).to(vae_dtype)

    eng = mk(SanaStreamingEngine)
    eng.begin(latent_h, latent_w)
    seg = eng.create_autoregressive_segments(latent_t)
    outs = []
    t0 = time.perf_counter()
    for i in range(len(seg) - 1):
        s, e = seg[i], seg[i + 1]
        outs.append(eng.push_chunk(noise[:, :, s:e].clone(), image_vae_embeds[:, :, s:e], STEPS))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    inc = torch.cat(outs, dim=2).to(vae_dtype)

    mae = (inc.float() - ref.float()).abs().mean().item()
    samples = vae_decode(config.vae.vae_type, vae, inc)
    return {"chunks": len(seg) - 1, "latent_t": latent_t, "incr_time_s": dt, "mae": mae,
            "decoded_shape": tuple(samples.shape), "decoded_device": str(samples.device)}


def main():
    r = run_continuity_check()
    print(f"chunks={r['chunks']} latent_t={r['latent_t']} incr_time={r['incr_time_s']:.2f}s "
          f"MAE(incremental vs batch)={r['mae']:.6f}")
    print(f"decoded samples shape={r['decoded_shape']} device={r['decoded_device']}")
    assert r["mae"] < 1e-2, f"CONTINUITY FAIL: MAE {r['mae']:.4f} >= 1e-2"
    print("PASS: incremental matches batch within 1e-2; tensors out, no disk on hot path")


if __name__ == "__main__":
    main()

"""SanaStreamingRuntime — the temporal StreamForge backend wrapping SanaStreamingEngine.

Design notes:
- validate_dims() and the class body do NO heavy imports, so the pure-logic dims test runs in
  the main .venv. The SANA stack (torch, engine, builder) loads lazily in load_once().
- push_frame() buffers source frames, and once a chunk's worth of pixels is accumulated,
  VAE-encodes them, drives engine.push_chunk() (DiT continuity is bit-exact across chunks),
  VAE-decodes, and returns the per-frame tensors.
- KNOWN caveat: the LTX-2 causal VAE is encoded/decoded per chunk window here, so cross-chunk
  VAE seams are possible (the DiT path is continuous; the VAE is the residual risk — verify
  visually / tune in the bench). The first cut favors a working live pipeline.
"""
from __future__ import annotations

import threading

from streamforge.diffusion.runtime_base import TemporalDiffusionRuntime

# load_once() runs on the worker thread, and a re-Start while the first ~30s load is still in
# flight spawns a SECOND worker (runner.stop() only joins with a 5s timeout), so two load_once()
# run concurrently. accelerate/transformers are not thread-safe across that: the Accelerator
# state singleton races (one reads device=None -> AttributeError at accelerator.py:589) and the
# big-model meta-device materialization races (-> "Cannot copy out of meta tensor" in the VAE /
# text encoder). This lock serializes the WHOLE load across all instances so loads never overlap.
_ACCEL_INIT_LOCK = threading.Lock()

SPATIAL_STRIDE = 32      # vae_stride[1:] — width/height must be divisible by this
TEMPORAL_STRIDE = 8      # vae_stride[0] — num_frames must satisfy (n-1) % 8 == 0
DEFAULT_CONFIG = "configs/sana_streaming/sana_streaming_2b_720p.yaml"
DEFAULT_MODEL = "hf://Efficient-Large-Model/SANA-Streaming/dit/sana_streaming_ar.pth"


def validate_dims(width: int, height: int, num_frames: int) -> tuple[int, int, int]:
    """Enforce SANA's VAE divisibility (recon: vae_stride [8,32,32]). Raises ValueError on a bad
    size so the operator console fails at Validate, not mid-show."""
    if width % SPATIAL_STRIDE != 0:
        raise ValueError(f"width {width} must be divisible by {SPATIAL_STRIDE}")
    if height % SPATIAL_STRIDE != 0:
        raise ValueError(f"height {height} must be divisible by {SPATIAL_STRIDE}")
    if (num_frames - 1) % TEMPORAL_STRIDE != 0:
        raise ValueError(f"num_frames {num_frames} must satisfy (n-1) % {TEMPORAL_STRIDE} == 0 "
                         f"(e.g. 9, 17, 25, 33, ...)")
    return (width, height, num_frames)


class SanaStreamingRuntime(TemporalDiffusionRuntime):
    def __init__(self, internal_hw: tuple[int, int] | None = None, control=None,
                 config_path: str | None = None, model_path: str | None = None,
                 num_cached_blocks: int = 2, sink_token: bool = True, default_steps: int = 4,
                 resync_every: int = 8):
        # internal_hw is (height, width) like EagerRuntime; default 384x640 (both /32, real-time)
        self.internal_hw = internal_hw or (384, 640)
        self.control = control
        self.config_path = config_path
        self.model_path = model_path or DEFAULT_MODEL
        self.num_cached_blocks = num_cached_blocks
        self.sink_token = sink_token
        self.default_steps = default_steps
        # SANA-Streaming is validated for bounded clips (~969 frames ~= 40 chunks); driven as an
        # UNBOUNDED live stream its RoPE positions + autoregressive KV state leave the training
        # distribution and the output drifts to colored mush after ~10-15 chunks. Re-anchor the
        # temporal state every `resync_every` chunks to hold coherence (0 disables -> raw drift).
        self.resync_every = resync_every
        self._chunk_count = 0
        self._loaded = False
        self._prompt = ""
        self.engine = None

    # --- loading -------------------------------------------------------------------------------
    def load_once(self) -> None:
        if self._loaded:
            return
        import importlib.util

        import torch

        from streamforge.diffusion.sana._bootstrap import _SANA, ensure_sana_importable

        ensure_sana_importable()  # MUST run before any `from diffusion...` import
        torch.set_grad_enabled(False)  # inference-only runtime (matches the SANA reference script)

        import pyrallis
        from accelerate import Accelerator

        from diffusion.model.builder import (
            get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode,
        )
        from diffusion.model.utils import get_weight_dtype

        from streamforge.diffusion.sana.engine import SanaStreamingEngine
        cfg_path = self.config_path or str(_SANA / DEFAULT_CONFIG)
        script = _SANA / "inference_video_scripts" / "v2v" / "inference_sana_streaming.py"
        spec = importlib.util.spec_from_file_location("sana_stream_script", str(script))
        S = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S)
        self._S = S
        self._vae_encode = vae_encode
        self._vae_decode = vae_decode

        config = pyrallis.parse(config_class=S.InferenceConfig, config_path=cfg_path, args=[])
        self._config = config
        # Serialize the ENTIRE model load, not just Accelerator: accelerate/transformers load big
        # models on the meta device then materialize, so two concurrent load_once() (a re-Start
        # before the first ~30s load finishes) corrupt each other's shared init -> Accelerator
        # device=None at one site OR "Cannot copy out of meta tensor" in the text encoder at the
        # next. One global lock over the whole load makes the two runs strictly sequential.
        with _ACCEL_INIT_LOCK:
            accelerator = Accelerator(mixed_precision=config.model.mixed_precision)
            self.device = accelerator.device
            self._weight_dtype = get_weight_dtype(config.model.mixed_precision)
            self.vae_dtype = get_weight_dtype(config.vae.weight_dtype)
            self._vae_stride = config.vae.vae_stride
            self.vae_latent_dim = config.vae.vae_latent_dim
            latent_size = config.model.image_size // config.vae.vae_downsample_rate
            self._flow_shift = (config.scheduler.inference_flow_shift
                                if config.scheduler.inference_flow_shift is not None
                                else config.scheduler.flow_shift)

            self._vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained,
                                device=self.device, dtype=self.vae_dtype, config=config.vae)
            self._tokenizer, text_encoder = get_tokenizer_and_text_encoder(
                config.text_encoder.text_encoder_name, device=self.device)
            model = S.load_model(config, latent_size, self.device, self._weight_dtype, self.model_path)
            self._model, self._text_encoder = accelerator.prepare(model, text_encoder)
            self._neg_embeds, _ = S.encode_prompt(self._tokenizer, self._text_encoder, "",
                                                  config, self.device, use_chi_prompt=False)
        self._engine_cls = SanaStreamingEngine
        self._base_chunk_frames = 24 // self._vae_stride[0]
        # causal VAE requires (N-1) % temporal_stride == 0; this yields exactly base_chunk_frames
        # latent frames per chunk with no leftover "pending" temporal state. (3-1)*8+1 = 17.
        self._pixel_per_chunk = (self._base_chunk_frames - 1) * self._vae_stride[0] + 1
        h, w = self.internal_hw
        self._latent_h = h // self._vae_stride[1]
        self._latent_w = w // self._vae_stride[2]
        self._buf: list = []
        self._gen = torch.Generator(device=self.device).manual_seed(7)
        self._loaded = True
        if self._prompt:
            self.set_prompt(self._prompt)

    # --- prompt + state ------------------------------------------------------------------------
    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt
        if not self._loaded:
            return
        self._prompt_embeds, self._prompt_mask = self._S.encode_prompt(
            self._tokenizer, self._text_encoder, prompt, self._config, self.device, use_chi_prompt=True)
        self.reset_state()

    def reset_state(self) -> None:
        if not self._loaded:
            return
        self.engine = self._engine_cls(
            self._model, condition=self._prompt_embeds, uncondition=self._neg_embeds,
            cfg_scale=1.0, flow_shift=self._flow_shift,
            model_kwargs={"data_info": {}, "mask": self._prompt_mask},
            base_chunk_frames=self._base_chunk_frames, num_cached_blocks=self.num_cached_blocks,
            cache_strategy="fixed_rope", efficient_cache=False, sink_token=self.sink_token)
        self.engine.begin(self._latent_h, self._latent_w)
        self._buf = []
        self._chunk_count = 0

    # --- streaming -----------------------------------------------------------------------------
    def _fit(self, image_bchw):
        import torch

        from streamforge.aspect import FitMode, Size, fit_tensor
        h, w = self.internal_hw
        if image_bchw.shape[-2:] == (h, w):
            return image_bchw.to(self.device)
        small, _ = fit_tensor(image_bchw.to(self.device), Size(width=w, height=h),
                              FitMode.FILL_CROP, antialias=True)
        return small

    def push_frame(self, image_bchw, params=None) -> list:
        import torch
        if not self._loaded:
            self.load_once()
        if self.engine is None:  # set_prompt not called -> nothing to condition on
            return []
        self._buf.append(self._fit(image_bchw))
        if len(self._buf) < self._pixel_per_chunk:
            return []

        window = self._buf[:self._pixel_per_chunk]
        self._buf = self._buf[self._pixel_per_chunk:]
        video = torch.cat(window, dim=0).permute(1, 0, 2, 3).unsqueeze(0).to(self.device, self.vae_dtype)
        # StreamForge frames are [0,1] (GpuFrame contract); SANA's LTX2 VAE was trained on [-1,1]
        # pixels (reference read_video: Normalize(0.5,0.5)). Feeding [0,1] gives off-distribution
        # image_vae_embeds -> weak per-chunk V2V conditioning -> the stream drifts to abstract mush
        # over a long run. Map to [-1,1] before encode.
        video = video * 2.0 - 1.0
        embeds = self._vae_encode(self._config.vae.vae_type, self._vae, video,
                                  sample_posterior=False, device=self.device).to(self.vae_dtype)
        m = embeds.shape[2]
        steps = self.default_steps
        if params is not None:
            steps = params.step
            self.engine.flow_shift = params.flow_shift
        noise = torch.randn(1, self.vae_latent_dim, m, self._latent_h, self._latent_w,
                            generator=self._gen, device=self.device)
        lat = self.engine.push_chunk(noise, embeds, steps).to(self.vae_dtype)
        samples = self._vae_decode(self._config.vae.vae_type, self._vae, lat)
        # decoder returns [-1,1] (reference save_video: 127.5*x+127.5); map back to the [0,1] the
        # rest of StreamForge (preview/sink/fill) expects. clamp(0,1) alone would crush the whole
        # negative half to black (under-exposed output).
        out = [((samples[:, :, i] + 1.0) / 2.0).clamp(0, 1) for i in range(samples.shape[2])]
        self._chunk_count += 1
        if self.resync_every and self._chunk_count >= self.resync_every:
            self.reset_state()  # re-anchor temporal state before drift accumulates (resets count)
        return out

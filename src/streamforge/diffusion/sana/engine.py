"""SanaStreamingEngine — drive SANA's streaming sampler one chunk at a time.

SANA's `SANAStreamingSampler.sample()` is batch-shaped: it takes the whole clip, allocates the
KV cache for all chunks upfront, and loops internally. This subclass extracts that per-chunk body
(sana_streaming_sampler.py:268-368) into `push_chunk()`, persisting `kv_cache` + the absolute
`chunk_indices` across calls so frames can be fed live and unbounded.

Why this works: the cross-chunk dependency flows entirely through `kv_cache` (the GDN recurrent
state + KV window), not the latents tensor — each chunk's model forward operates only on its own
slice with RoPE positioned by absolute frame indices. The window is bounded by `num_cached_blocks`,
so memory stays bounded for an unbounded stream.

Acceptance (tests/test_sana_engine.py + scripts/sana_smoke.py): incremental output must match a
single batch sample() within 1e-2 MAE (same kernels, same cache math).
"""
from __future__ import annotations

import torch

from streamforge.diffusion.sana._bootstrap import ensure_sana_importable

ensure_sana_importable()

from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402

from diffusion.scheduler.sana_streaming_sampler import SANAStreamingSampler  # noqa: E402

try:  # diffusers output type used by the sampler (optional isinstance guard)
    from diffusers.models.modeling_outputs import Transformer2DModelOutput  # noqa: E402
except Exception:  # pragma: no cover
    Transformer2DModelOutput = None


class SanaStreamingEngine(SANAStreamingSampler):
    """Incremental driver: begin() once, then push_chunk() per arriving chunk."""

    def begin(self, height: int, width: int) -> None:
        """Reset all streaming state for a new clip / prompt."""
        self._spatial_hw = height * width
        self._chunk_idx = 0
        self._kv_cache: list = []
        self._chunk_indices: list = [0]
        self.model_kwargs.pop("data_info", None)  # supplied per-chunk via push_chunk

    @torch.no_grad()
    def push_chunk(self, chunk_latents: torch.Tensor, chunk_vae_embeds: torch.Tensor,
                   steps: int) -> torch.Tensor:
        """chunk_latents: [B,C,nf,h,w] noise for THIS chunk. chunk_vae_embeds: matching input
        VAE encoding. Returns the denoised latents for this chunk; persists state."""
        device = self.condition.device
        do_cfg = self.cfg_scale > 1
        ci = self._chunk_idx
        nf = chunk_latents.shape[2]
        start_f = self._chunk_indices[-1]
        end_f = start_f + nf
        self._chunk_indices.append(end_f)

        # fresh per-block cache slot for this chunk, then accumulate prior state into it
        self._kv_cache.append([[None] * 6 for _ in range(self.num_model_blocks)])
        chunk_kv_cache, _, sink_num, num_cached_frames = self.accumulate_kv_cache(self._kv_cache, ci)

        prompt_embeds = self.condition
        if do_cfg:
            prompt_embeds = torch.cat([self.uncondition, prompt_embeds], dim=0)
        mask = self.mask

        scheduler = FlowMatchEulerDiscreteScheduler(shift=self.flow_shift)
        timesteps = self._timesteps_for_steps(scheduler, steps, device)
        cache_start_chunk_idx = max(ci - self.num_cached_blocks, 0) if self.num_cached_blocks > 0 else 0

        frame_index = None
        if sink_num > 0:
            sink_fi = torch.arange(sink_num, device=device)
            non_sink_count = num_cached_frames - sink_num + nf
            window_start_f = end_f - non_sink_count
            remaining_fi = torch.arange(window_start_f, end_f, device=device)
            frame_index = torch.cat([sink_fi, remaining_fi], dim=0)
            rope_start_f, rope_end_f = 0, end_f
        else:
            rope_start_f = self._chunk_indices[cache_start_chunk_idx]
            rope_end_f = end_f

        local_data_info = {"image_vae_embeds": chunk_vae_embeds}
        latents = chunk_latents

        for t in timesteps:
            lmi = torch.cat([latents] * 2) if do_cfg else latents
            timestep = t.expand(lmi.shape[0])
            noise_pred, _ = self.model(
                lmi, timestep, prompt_embeds, start_f=rope_start_f, end_f=rope_end_f,
                frame_index=frame_index, save_kv_cache=False, kv_cache=chunk_kv_cache,
                mask=mask, data_info=local_data_info, **self.model_kwargs,
            )
            if Transformer2DModelOutput is not None and isinstance(noise_pred, Transformer2DModelOutput):
                noise_pred = noise_pred[0]
            if do_cfg:
                u, c = noise_pred.chunk(2)
                noise_pred = u + self.cfg_scale * (c - u)
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # post-denoise forward at t=0 saves this chunk's clean-latent cache for the next chunk
        lmi = torch.cat([latents] * 2) if do_cfg else latents
        timestep = torch.zeros(lmi.shape[0], device=device)
        _, updated_kv_cache = self.model(
            lmi, timestep, prompt_embeds, start_f=rope_start_f, end_f=rope_end_f,
            frame_index=frame_index, save_kv_cache=True, kv_cache=chunk_kv_cache,
            mask=mask, data_info=local_data_info, **self.model_kwargs,
        )
        self._kv_cache[ci] = updated_kv_cache
        self._promote_fixed_rope_full_history_cache(self._kv_cache, ci)

        # bound memory: drop chunks older than the cache window (keep list indices stable)
        if self.num_cached_blocks > 0:
            old = ci - self.num_cached_blocks - 1
            if old >= 0:
                self._kv_cache[old] = [[None] * 6 for _ in range(self.num_model_blocks)]

        self._chunk_idx += 1
        return latents

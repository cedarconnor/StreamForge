# Phase 0–1 execution notes

## Environment (verified 2026-06-04)
- GPU: NVIDIA RTX A6000, 48 GB, driver 595.79. `torch.cuda.is_available() = True`.
- venv `.venv` (Python 3.11.9): torch 2.11.0+cu128, diffusers 0.38.0, transformers 4.56.1
  (pinned), pydantic 2.13.4, opencv 4.13, pytest 9.0.3, huggingface_hub 0.36.2.
- diffusers 0.38.0 exposes the FLUX.2 stack: `Flux2KleinPipeline`, `Flux2KleinInpaintPipeline`,
  `Flux2KleinKVPipeline`, `AutoencoderKLFlux2`, `Flux2Transformer2DModel`, plus modular
  AutoBlocks. **No `Flux2KleinImg2ImgPipeline`** — img2img is reference-conditioned / manual.

## Test suite
- 31/31 pure-logic tests green (no GPU/weights needed): `.\.venv\Scripts\python.exe -m pytest`.

## Phase-5 decoupling gate (Task 5.2) — PASS (model-independent, real threads)
- `run_live.py --fps 50 --seconds 10 --res 512 --fake-ai-fps 12 --sink null`
- Result: emitted=501 (~500 expected), **output jitter = 0.15 ms** (threshold <2 ms),
  repeats=386, fresh AI ~115 (~11.5 fps) — sacred output clock holds under slow AI. ✓

## Pipeline API facts (from installed diffusers 0.38.0 introspection)
- `Flux2KleinPipeline.__call__`: accepts `image`, `prompt`, `height`, `width`,
  `num_inference_steps`, `sigmas`, `guidance_scale` (default 4.0), `prompt_embeds`,
  `max_sequence_length=512`, `text_encoder_out_layers=(9,18,27)`. **No `strength` param** ->
  `text_magnitude`->`guidance_scale`; `ref_strength` needs a MANUAL encode->noise->denoise
  loop (image= is reference-conditioning, not strength img2img).
- Test clip: D:\StreamForge\TestFile\DriveVideo.mp4 = 640x480, 30 fps, 436 frames.

## Pending GPU/model gates (records to fill)
- [ ] Task 0.4 full-stack still-image gate: full-VAE std=__, taef2 std=__, peak VRAM=__ GB.
- [ ] Task 2.4 eager baseline (512²): infer p50/p95/p99=__/__/__ ms, jitter=__, VRAM=__.
- [ ] Task 3.2 MVP floor (taef2+caches, 512²): infer p95=__ ms, prompt-swap latency=__ ms.
- [ ] Task 1.3 structure adherence: edge-IoU PRESERVE/SUBTLE/BALANCED/FOLLOW/FORCE = __.

## Nunchaku / Path B
- See NUNCHAKU_DECISION.md — literal Nunchaku-4B unavailable; bake-off substitutes existing
  Ampere 4-bit/INT8 quants (SDNQ-4bit, INT8-Comfy).

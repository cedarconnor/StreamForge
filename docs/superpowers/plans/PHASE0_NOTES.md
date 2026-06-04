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

## Pending GPU/model gates (records to fill)
- [ ] Task 0.4 full-stack still-image gate: full-VAE std=__, taef2 std=__, peak VRAM=__ GB.
- [ ] Task 2.4 eager baseline (512²): infer p50/p95/p99=__/__/__ ms, jitter=__, VRAM=__.
- [ ] Task 3.2 MVP floor (taef2+caches, 512²): infer p95=__ ms, prompt-swap latency=__ ms.
- [ ] Task 1.3 structure adherence: edge-IoU PRESERVE/SUBTLE/BALANCED/FOLLOW/FORCE = __.

## Nunchaku / Path B
- See NUNCHAKU_DECISION.md — literal Nunchaku-4B unavailable; bake-off substitutes existing
  Ampere 4-bit/INT8 quants (SDNQ-4bit, INT8-Comfy).

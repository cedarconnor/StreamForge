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

## Task 0.4 still-image gate — PASS (2026-06-04)
- Full-VAE text->image: 1024x1024, std=59.4 (coherent), 4 steps ~1.5s, **peak VRAM 18.6 GB**.
- Repo is self-contained: model_index.json bundles Qwen3 text_encoder, Qwen2TokenizerFast,
  FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel, AutoencoderKLFlux2; is_distilled=true.

## ⚠ CRITICAL FINDING: distilled 4B IGNORES guidance_scale
- diffusers warns: "Guidance scale 4.0 is ignored for step-wise distilled models."
- => `text_magnitude` CANNOT map to guidance_scale (refutes the introspection assumption;
  this is exactly review-gap #5). The distilled klein-4B is CFG-baked-in.
- Mechanism for `text_magnitude` instead: **prompt-embedding interpolation/scaling** —
  embeds = lerp(neutral_embeds, prompt_embeds, text_magnitude), tm>1 to exaggerate.
- `ref_strength` still maps to img2img denoise strength via a manual/latents+sigmas loop.

## taef2 (tiny VAE) packaging
- models/vae_tiny ships a SINGLE `taef2.safetensors` and NO taesd.py; diffusers
  AutoencoderTiny does not natively support taef2 (flux_2, 32-ch). Phase-3 needs the
  madebyollin taesd wrapper (single-file load). Deferred to Phase 3 (the §4.3 optimization).

## Pending GPU/model gates (records to fill)
- [ ] Task 0.4 full-stack still-image gate: full-VAE std=59.4 ✓, taef2 std=__ (Phase 3), peak VRAM=18.6 GB ✓.
- [x] Task 2.4 eager baseline (512²): infer p50/p95/p99 = 1195/1219/1486 ms (~0.8 fps),
      VRAM 16.7 GB. Includes per-frame Qwen3 re-encode (no cache) + full VAE. Standalone
      single-encode denoise was ~0.75s/4steps -> prompt-cache+taef2 (Phase 3) is the big win.
- [x] Task 3.x prompt cache (512²): p50 1195 -> 1049 ms (~12%; transformer dominates, not text encode).

## ⚠ Profiling overturns the design §4.3 taef2 assumption (2026-06-05)
`scripts/profile_pipeline.py` @512²:
  VAE encode 27ms + decode 51ms = ~79ms (**only ~8%** of the 1032ms restyle).
  Transformer 4-step (txt2img) = 601ms. image= restyle = 1032ms.
  **Reference-token overhead = 431ms** (image= concatenates reference tokens, ~doubling seq len).
Implications:
  - **taef2 is NOT worth integrating** for this model/path — saves <~60ms (~2%). Design §4.3
    ("full VAE decode is a disproportionate share") is FALSE here. Phase-3 taef2 DEFERRED with evidence.
  - Per-frame budget is **transformer-bound** => the real cadence lever is Phase 4 (quantize/compile).
  - The image= edit path costs ~72% more than txt2img. A classic img2img mode (encode->noise@strength
    ->denoise SAME tokens, no concat) would cut ~431ms AND provide the real ref_strength knob, at some
    structure-adherence cost. Worth prototyping as an alternate runtime / Phase-4 input.
- [ ] Task 1.3 structure adherence: edge-IoU PRESERVE/SUBTLE/BALANCED/FOLLOW/FORCE = __.

## Bake-off Path C (torch.compile) — BLOCKED on Windows (2026-06-05)
- `torch.compile(transformer)` fails: `torch._inductor.exc.TritonMissing` — Triton/inductor
  is not available on Windows by default. Unblock options: `pip install triton-windows`
  (version must match torch 2.11+cu128; uncertain), or run under WSL/Linux. Left the
  `compile_transformer` flag in EagerRuntime (works on Linux / with triton-windows).
- => On native Windows, the viable Phase-4 paths are Path A (TensorRT) and Path B′
  (existing Ampere INT8/4-bit quants), not torch.compile.

## Nunchaku / Path B
- See NUNCHAKU_DECISION.md — literal Nunchaku-4B unavailable; bake-off substitutes existing
  Ampere 4-bit/INT8 quants (SDNQ-4bit, INT8-Comfy).

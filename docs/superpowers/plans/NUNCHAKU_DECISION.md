# Nunchaku / Path-B decision (Task 0.5)

**Date:** 2026-06-04
**Probe:** `scripts/check_nunchaku_4b.py` (HF model search, no GPU/download)

## Finding
- **No pre-quantized Nunchaku/SVDQuant build of `FLUX.2-klein-4B` exists** on HF. Named
  Nunchaku quants exist only for the **non-commercial 9B / 9B-kv** (`tonera/...`).
- FLUX.2 architecture support in Nunchaku is still in-progress (PR #926 / discussion #8).
- **However, Ampere-viable 4-bit / INT8 alternatives for the 4B already exist** and can serve
  Path B's *intent* (shrink the transformer, lift A6000 throughput without FP8):
  - `Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic` (SDNQ 4-bit; a different SVD-ish method)
  - `Norton0924/FLUX.2-klein-4B-4bit` and `-8bit`
  - `Bedovyy/FLUX.2-klein-4B-INT8-Comfy` (INT8 — native on Ampere)
  - GGUF (`OlegSkutte`, `Aatricks` ternary) — runs, but NOT the fast kernel path.
  - `Bedovyy/FLUX.2-klein-4b-nvfp4mixed` — NVFP4, **Blackwell only**, not the A6000.

## Decision
**Path B (literal Nunchaku/SVDQuant INT4) = DEFERRED for the 4B.** Pre-quantized Nunchaku is
unavailable; betting on it requires either self-quantizing (only if Nunchaku exposes
SVDQuant kernels for FLUX.2-klein-4B on sm_86 — unverified) or waiting on PR #926.

**Bake-off (Phase 4) revised paths:**
- Path A: TensorRT (FP16/INT8) — safe baseline.
- **Path B′ (substitute): test the existing Ampere 4-bit/INT8 quants** above (SDNQ-4bit,
  INT8-Comfy) for cadence×quality, instead of literal Nunchaku.
- Path C: torch.compile — fallback.
- (Re-add literal Nunchaku if PR #926 lands a 4B kernel before Phase 4.)

## Still to verify at the GPU gate
- [ ] Does a released Nunchaku version expose SVDQuant kernels for FLUX.2-klein-4B on sm_86?
- [ ] Smoke-test `Disty0/...SDNQ-4bit` and/or `Bedovyy/...INT8-Comfy` on the A6000: renders? latency? VRAM? quality at projection scale?

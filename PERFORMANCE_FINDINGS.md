# StreamForge — Performance

Real-time FLUX.2-klein-4B img2img restyle on one RTX A6000 (sm_86, 48 GB), Windows-native,
Spout/NDI out. The output clock is decoupled from the AI (worker → FrameBuffer → RealtimeClock),
so output cadence (30/50 fps) never blocks on inference.

## Adopted stack (the fast path)
`EagerRuntime(mode="img2img", internal_hw=(224,384), compile_transformer=True, tiny_vae=True)` at
384×224 PRESERVE (1 step):

| stage | ms |
|---|---:|
| transformer (1 step) | ~58 |
| VAE encode (taef2) | 6.6 |
| VAE decode + post (taef2) | 3.1 |
| prep / pack | 1.7 |
| **total** | **~70.7 → 14.1 fps fresh** |

From a 359 ms baseline → 70.7 ms = **5.1×**, all Windows-native. Levers (each measured via
`scripts/measure_levers.py` / `scripts/profile_img2img.py`):
- **Internal resolution** (diffuse low-res, upscale output) — biggest lever; 512²→384×224 ≈ 3×.
  There is a fixed ~35–89 ms floor (VAE + dispatch) that shrinks with resolution but never vanishes.
- **Classic img2img** (encode→noise→denoise, no reference concat) — ~40% faster than the `image=`
  edit path AND gives a real `ref_strength` knob. Step count = `int(steps × denoise)`, so
  PRESERVE = 1 step, BALANCED = 2.
- **torch.compile** at fixed shape — ~1.25×, stable (p95 ≈ p50, no recompile pathology).
- **taef2 tiny VAE** — VAE 29.9 → 9.7 ms (1.28×); worth it only after the transformer was cut
  (regime flip — in the slow regime the VAE was <10% and not worth it). Latent contract: taef2 is
  trained on the *normalized* FLUX.2 latent space, so the wrapper supplies an identity batch-norm
  to make the pipeline normalize/denormalize a no-op. Quality verified side-by-side.

The path is now ~84% transformer — cleanly transformer-bound.

## RIFE frame-fill (perceived fps)
The transformer step can't be made faster on this hardware (see "Ruled out"), so smoothness comes
from output-side **motion extrapolation** — NOT the RIFE network (extrapolation was chosen for zero
added latency; true RIFE needs the next frame). `raft_small` (torchvision, BSD) measures source
motion on the worker thread; `RealtimeClock` forward-warps the latest styled frame to each stale
tick (the `select_fill` "warped" tier), holding past `--max-extrap-ms` (default 120). Flag:
`--fill warp`. Measured (null sink, 10 s, 384×224 PRESERVE, uncompiled): **filled = 159 vs 0**
baseline; fresh AI 10 → 6.4 fps (RAFT cost on the worker, tune `--flow-max-side`); ~10 ms
clock-thread warp jitter (fast-follow: a dedicated fill thread if Spout jitter regresses).
Spec/plan: `docs/superpowers/specs/2026-06-07-rife-frame-fill-design.md`,
`docs/superpowers/plans/2026-06-07-rife-frame-fill.md`.

## Ruled out (do NOT retry on this hardware)
The transformer is **memory-bandwidth-bound** at batch=1 / ~336 tokens (reads ~8 GB bf16 weights
per step). That single fact explains every dead end below — nothing precision-based reduces the
bottleneck; only fewer/lighter weight reads (INT4) would.

- **FP8 quant** — A6000 is sm_86 (Ampere): no FP8 tensor cores (FP8 starts at Ada/Hopper). Dead.
- **bitsandbytes 8/4-bit** — VRAM tool, *slower* (dequant overhead); VRAM was never the constraint.
- **torchao INT8 weight-only + torch.compile** — ~10 s/step pathology, identical on Windows AND
  WSL/Linux (same torch 2.11 / triton 3.6 / sm_86). The torchao+compile+FLUX.2 combo is broken,
  not the OS.
- **CUDA graphs** (`reduce-overhead`) — ~1%; only ~2% of the path is dispatch glue.
- **TensorRT FP16/INT8** — no speedup at any precision (FP16 55.9 ms / cosine 0.999; INT8 56.1 ms /
  cosine 0.278; bf16 ~58 ms). TRT's autotuner kept FP16 over INT8 (INT8 not faster at these
  batch-1 shapes); INT8 also collapses quality on this distilled model. Investigated thoroughly,
  then removed from the repo.
- **Nunchaku SVDQuant INT4** — the one untried fresh-fps lever (real INT4 tensor cores), but no
  commercial 4B weights exist yet and engine support is pending upstream. Re-evaluate when available.

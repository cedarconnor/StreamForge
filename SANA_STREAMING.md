# StreamForge × SANA-Streaming — Build Notes & Results

Status: **Phase 0 (env/kernel proof) + Phase 1 (StreamForge integration) COMPLETE.** SANA-Streaming
runs as a second StreamForge runtime backend (`backend="sana_streaming"`) — a stateful, chunk-causal
video-to-video engine — alongside the existing FLUX.2-klein img2img path, on the RTX A6000 under
native Windows (no WSL2/conda).

These notes consolidate the verification done on 2026-06-16. The original design doc is
`docs/streamforge-sana-streaming-design.md`; the task plans are under `docs/superpowers/plans/`.

---

## What SANA-Streaming is

Real-time streaming **video-to-video** editing. Chunk-causal autoregressive over time, with a GDN
(Gated DeltaNet) linear-attention recurrent state plus a Hybrid DiT that mixes softmax attention
into a minority (~25%) of blocks. Unlike FLUX (one frame in → one frame out, no temporal memory),
SANA carries recurrent state across frames and emits decoded frames per autoregressive chunk.

- Model: `SanaMSVideoV2V_2000M` (2B), `attn_type V2VStateCachedBiGDNAttention`, BF16.
- VAE: LTX-2 causal VAE (`Lightricks/LTX-2`, subfolder `vae`), temporal+spatial stride `[8,32,32]`.
- Text encoder: Gemma-2-2B-it (`Efficient-Large-Model/gemma-2-2b-it` mirror, ungated).
- License: Apache-2.0.

---

## Phase 0 — environment & kernel proof (all gates PASSED)

Run entirely outside the `streamforge` package (separate `.venv-sana`, a cloned `external/Sana`,
and `scripts/`), so a kernel failure would cost zero integration code.

| Gate | Result |
|---|---|
| 1 — Triton on Windows/A6000 | ✅ `triton-windows 3.6.0.post26` compiles jit + autotune + `torch.compile` kernels (`scripts/sana_gate_triton.py`) |
| 3 — **SANA's own fused GDN kernels** (the go/no-go) | ✅ `USE_CHUNKWISE_GDN=1` produced a valid clip on the A6000 |
| 4 — softmax attention | ✅ SANA falls back to SDPA natively when flash-attn is absent; **no FA2 wheel** matches torch 2.11+cu128+cp311, and none is needed (`scripts/sana_gate_sdpa.py`) |
| 0 — recon | ✅ entry point, config, import graph, dim rules, repos all resolved (`scripts/sana_recon.md`) |

**flash-attn finding:** no prebuilt FA2 wheel exists for our stack (lldacing maxes at torch 2.8;
ussoewwin's torch-2.11 wheels are cu130/cp313). SDPA's mem-efficient backend runs in BF16 and SANA
guards every flash-attn import, so SDPA is the path. Do not chase FA2 unless a measured bottleneck.

---

## Phase 1 — StreamForge integration (all 9 tasks COMPLETE)

| Task | What |
|---|---|
| T1 | `manifest-sana.yaml` (pinned SHAs) + `download_sana_models.py` + `sana/_bootstrap.py` (path/shim/env injection) |
| **T2** | **`SanaStreamingEngine`** — drives the batch sampler one chunk at a time with persistent KV cache. **Bit-exact to batch `sample()` (MAE = 0.000000)** — the spike, fully de-risked |
| T3 | `TemporalDiffusionRuntime` protocol (temporal flag, `load_once`/`reset_state`/`push_frame`, no-op `set_mode`) |
| T4 | `SanaControl`/`SanaLiveControl` — HOT/WARM/COLD knob taxonomy, `SANA_FAST`/`SANA_BALANCED` presets |
| T5 | `QueueFrameBuffer` + `TemporalInferenceWorker` — bounded drain queue honoring the clock's `get_with_freshness()` contract (zero clock edits); overwrite-bug guard test |
| T6 | `SanaStreamingRuntime` + `validate_dims` (W/H ÷32, frames 8k+1). `push_frame` buffers 17-pixel windows aligned to the causal VAE temporal stride |
| T7 | Runner `backend` branch + worker selection by `runtime.temporal`; `apply_control` SANA routing; web config + control fields |
| T8 | SANA live-control panel — backend dropdown + `#sanaLive` fieldset (HOT live, WARM "resyncs", COLD restart) |
| T9 | Temporal bench path + A6000 numbers |

Non-GPU suite: **110 passed** (FLUX path unchanged). GPU paths validated via smokes.

### Knob safety classes (live control)

- **HOT** (apply live, per chunk): `step`, `flow_shift`, `motion_score`, `seed`
- **WARM** (trigger `reset_state()` — brief reseed/flash): `prompt`, `num_cached_blocks`, `sink_token`
- **COLD** (require a run restart): `height`/`width`, `backend`, `cfg_scale` (streaming is pinned 1.0)

---

## A6000 performance (BF16 + fused GDN)

Generation throughput; output FPS vs the model's 16 fps native cadence. Real-time-capable at ≤512².

| Resolution | Steps | Chunk p50 | Output FPS | VRAM |
|---|---|---|---|---|
| 384×640 | 4 | 780 ms | ~22 fps | 17 GB |
| 384×640 | 2 | 623 ms | ~27 fps | 17 GB |
| 512×512 | 4 | 804 ms | ~21 fps | 27 GB |
| 704×1280 (native) | 4 | — | ~9 fps (perf sweep) | — |

720p live is below real-time at step-4 BF16 (the Blackwell-only FP4 speedup is unavailable on
Ampere). Use the existing `in_res` lever to run live at ≤512².

---

## Environment setup (reproducible)

The SANA backend runs in a **separate `.venv-sana`** (its `transformers==4.57.3` pin would clash
with the FLUX stack). Full recipe in `requirements-sana-win.txt`; automated by `install.bat --sana`:

1. `external/Sana` = `NVlabs/Sana` cloned at the pinned SHA (`manifest-sana.yaml` → `sana_code.revision`)
2. `.venv-sana`: torch 2.11.0+cu128, `triton-windows==3.6.0.post26`, `transformers==4.57.3`,
   diffusers, accelerate, timm, pytz, qwen-vl-utils, ftfy, `flash-linear-attention`+`fla-core`
   (both `--no-deps`), `mmcv==1.7.2` (pure-python, `--no-build-isolation`), fastapi+uvicorn,
   and `streamforge` editable (`pip install -e . --no-deps`)
3. A no-op `scripts/sana_shims/fcntl.py` shim (a Unix-only import `builder.py` drags in)
4. `_bootstrap.ensure_sana_importable()` injects `external/Sana` + the shim + the required env
   (`USE_CHUNKWISE_GDN=1`, `DISABLE_FLASH_ATTN=1`, `DISABLE_XFORMERS=1`) at import.

Weights (~10 GB: DiT 4.52 GB + Gemma + LTX-2 VAE) auto-download to the HF cache on first run;
`scripts/download_sana_models.py` verifies/pins them.

---

## Usage

```bat
install.bat --sana          :: one-time: clone SANA, build .venv-sana, install deps
startup.bat --sana          :: launch the console under .venv-sana
```

In the console: set **Backend → SANA-Streaming (temporal)**, pick `SANA_FAST`/`SANA_BALANCED`,
choose a ≤512² aspect, Validate, Start. The **SANA Live Control** panel tunes steps / flow shift /
motion / seed live; cached-blocks / sink-token / prompt "resync" (rebuild the GDN state).

---

## Caveats / known issues

- **VAE chunk seams:** `push_frame` VAE-encodes each ~17-frame window independently, so the LTX-2
  causal VAE may seam at chunk boundaries. The DiT path is bit-exact across chunks; the VAE is the
  residual risk. Verify visually; a future improvement is a stateful framewise VAE stream.
- **Live browser verification of the SANA panel** (Chrome plugin) was not run — the UI is
  implemented + syntax/structure-verified and the API layer is unit-tested, but a live `.venv-sana`
  server + browser pass is the remaining acceptance.
- **Two venvs:** the SANA backend only works when StreamForge runs under `.venv-sana`; the FLUX
  backend continues to run under `.venv`.

---

## Key files

```
src/streamforge/diffusion/sana/_bootstrap.py        path/shim/env injection
src/streamforge/diffusion/sana/engine.py            SanaStreamingEngine (incremental driver)
src/streamforge/diffusion/runtime_sana_streaming.py SanaStreamingRuntime + validate_dims
src/streamforge/worker_temporal.py                  QueueFrameBuffer + TemporalInferenceWorker
src/streamforge/sana_control.py                     SanaControl/SanaLiveControl + presets
manifest-sana.yaml                                  pinned artifact SHAs
requirements-sana-win.txt                           env recipe
scripts/sana_recon.md                               recon findings
scripts/sana_gate_triton.py / sana_gate_sdpa.py     Phase-0 gates
scripts/sana_smoke.py                               engine continuity + tensors-out [gpu]
scripts/sana_runtime_smoke.py                       runtime end-to-end [gpu]
scripts/sana_bench.py                               temporal bench [gpu]
scripts/sana_shims/fcntl.py                         Windows no-op shim
```

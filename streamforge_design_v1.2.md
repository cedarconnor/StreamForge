# StreamForge — Real-Time FLUX Live-Restyle Engine

**Design Document v1.2**
**Author:** Cedar Connor
**Status:** Draft for review
**Target hardware:** NVIDIA RTX A6000 (48 GB), single-GPU; scalable to 5090 / RTX 6000 Blackwell
**Primary use case:** Live restyling of a camera/render feed at show framerate, output to a media server (Resolume, disguise, TouchDesigner, Watchout) for large-format projection.

**v1.2 changes (prior-art pass):** Added **Nunchaku / SVDQuant INT4** as a first-class A6000 acceleration path (§5.1) — the one route that sidesteps the FP8 ceiling on Ampere, so the Phase-4 bake-off now tests three paths, not two. Adopted a **two-axis control scheme** (ref_strength × text_magnitude, with named presets) replacing the single strength knob (§4.2, §7.5). Folded concrete **dependency-pinning gotchas** into the manifest (§4.1.1). Added an offline **asset-prep workflow** for style-LoRA baking and VAE debiasing (§7.6). Added the **4B structure-adherence caveat** to risks. Prior-art sources catalogued in Appendix B.

**v1.1 changes:** Repositioned around a *stable output clock with adaptive AI cadence* rather than "FLUX every frame under 33 ms." Added RealtimeClock / InferenceWorker / Governor decoupling (§6), a generated-vs-output cadence model (§3.1), an elevated benchmark-harness subsystem (§5.2), an explicit TensorRT engine list and fallback path (§5.1), a version-locked model manifest (§4.1), a color-management output stage (§8.6), and cheaper temporal-coherence options (§9). Moved 9B-KV to an appendix.

---

## 1. Purpose and scope

StreamForge is a self-hosted, open-stack alternative to closed live-AI tools like Cutback's CKLive and Daydream's Scope. It takes a live video input, restyles it through a step-distilled FLUX model, and hands the result to a media server over **NDI** (network) or **Spout/Syphon** (same-machine, zero-copy GPU texture share).

The design goal is not "fastest FLUX" bragging rights. It is **a stable show-rate output clock while running step-distilled FLUX img2img at the highest cadence the hardware can sustain, with the best achievable image quality, on hardware Cedar already owns, with clean commercial licensing and full pipeline control.** On the A6000, the supported resolution and whether every input frame receives a unique diffusion pass are **Phase 2 benchmark outputs, not assumptions** (§3.1, §5.2).

In scope:
- Live img2img restyle (camera/SDI/NDI/Spout input → restyled output)
- Region-masked regeneration (brush / segmentation)
- Dual-reference conditioning (content + style)
- Temporal coherence stabilization
- Audio-reactive parameter control (MusiCue integration)
- NDI and Spout/Syphon output to a downstream media server

Out of scope (v1):
- Temporally-coherent *generated* video (autoregressive models — see §11)
- Node-graph authoring GUI (config-driven first; GUI is a later milestone)
- Multi-GPU pipeline parallelism (architected for, not built in v1)

---

## 2. Design principles

1. **The output clock is sacred; the AI cadence is adaptive.** The media server must receive a rock-stable 30/50 fps signal, every frame, forever. The diffusion engine runs *under* that clock at whatever cadence the hardware sustains, and the output stage always has a frame to emit (fresh AI result, warped prior, or held frame). A stutter in a two-hour show is a failure even when average FPS looks fine — so the show clock never blocks on inference.
2. **The texture never leaves the GPU until output.** Input upload and output download are the only host↔device copies. Everything between — preprocess, conditioning, denoise, decode, composite — stays resident in VRAM.
3. **Compile everything ahead of time.** Eager PyTorch does not hit framerate. Fixed shapes, fixed step counts, TensorRT/FP8, CUDA graphs.
4. **Quality is a tunable, not a constant.** A single "strength" knob trades source fidelity against stylization, live, per-frame.
5. **Commercially clean by default.** Apache-2.0 / permissive model weights as the default path; non-commercial models are opt-in and flagged.
6. **Fail soft.** If a frame can't make deadline, pass the previous output (or the raw input) rather than stall the show.

---

## 3. System overview

```
                         ┌─────────────────────────────────────────────┐
                         │              StreamForge runtime             │
                         │           (single CUDA context)              │
 ┌──────────┐  upload    │  ┌─────────┐   ┌──────────┐   ┌───────────┐  │  download   ┌──────────────┐
 │  INPUT   │───────────▶│  │ Pre-    │──▶│ Diffusion│──▶│ Composite │──┼────────────▶│   OUTPUT     │
 │ camera / │  (1 copy)  │  │ process │   │ core     │   │ + post    │  │  (1 copy)   │ NDI / Spout  │
 │ SDI/NDI/ │            │  │ (VAE enc│   │ (4-step  │   │ (mask     │  │             │ / Syphon     │
 │ Spout    │            │  │ +control│   │ FLUX)    │   │ blend,    │  │             │              │
 └──────────┘            │  │ +cond)  │   │          │   │ stabilize)│  │             └──────┬───────┘
                         │  └─────────┘   └──────────┘   └───────────┘  │                    │
                         │        ▲             ▲              ▲        │                    ▼
                         │        └─────────────┴──────────────┘        │            ┌──────────────┐
                         │              control bus (per frame)         │            │ MEDIA SERVER │
                         │   strength · prompt/embeds · mask · seed     │            │ Resolume /   │
                         │              ▲                               │            │ disguise /   │
                         └──────────────┼───────────────────────────────┘            │ TouchDesigner│
                                        │                                            └──────────────┘
                            ┌───────────┴────────────┐
                            │  MusiCue / OSC / MIDI  │
                            │  audio-reactive control │
                            └─────────────────────────┘
```

The runtime is a **single CPU orchestration thread** issuing asynchronous work across multiple **CUDA streams**, with the GPU providing the actual parallelism. This matches the "single thread, parallel nodes" pattern observed in CKLive and is the sane design — multiple CPU threads contending for one CUDA context creates more problems than it solves.

### 3.1 Output cadence vs AI cadence (the central model)

The single most important design decision: **the output clock and the AI generation cadence are decoupled.** Pipelining (§6.2) hides I/O and improves utilization, but it does **not** speed up the diffusion core — if a denoise pass takes 80 ms, the engine produces ~12.5 *new* AI frames per second no matter how well the stages overlap. The slowest stage still gates fresh-frame production. So instead of forcing every input frame through diffusion, StreamForge guarantees a stable output clock and lets the AI fill it adaptively.

| Mode | Meaning | Realism |
|---|---|---|
| **True N-fps inference** | Every input frame gets a unique diffusion pass. Hardest; only at low res / Blackwell. | Aspirational |
| **Stable output / lower AI cadence** | Output is always 30/50 fps; AI updates at 10–20 fps with frame hold, warped prior output, or latent reuse filling the gaps. | **v1 design target** |
| **Fallback passthrough** | Raw input or previous output fills missed frames. Show-safe but visually obvious if frequent. | Degraded-mode safety net |

v1 is designed around a **fixed output clock** with **adaptive AI cadence**. This is far more show-realistic on the A6000 than promising a unique FLUX pass every frame, and it degrades gracefully under load instead of stalling. The architecture that enforces this split is in §6.1.

---

## 4. The diffusion core

### 4.1 Model selection

The default path is **commercially clean (Apache-2.0)**. Non-commercial models are out of the v1 streaming scope and live in Appendix A so that lenders, clients, and partners are never confused about whether the default system is billable.

| Model | License | Steps | VRAM | Role |
|---|---|---|---|---|
| **FLUX.2 [klein] 4B-distilled** | Apache-2.0 | 4 (fixed) | ~13 GB | **Default.** Commercially clean, FLUX-class quality, fits A6000 with huge headroom. |
| SDXL-Turbo / SD-Turbo | permissive | 1–2 | ~7 GB | Fallback when 4B can't make cadence at target resolution; lower quality. |

(FLUX.2 [klein] 9B and 9B-KV are **non-commercial** and not part of the live-streaming path — see Appendix A.)

### 4.1.1 Known-good model manifest (version-locked)

ComfyUI, Diffusers, the BFL repo, and GGUF variants diverge in encoder sizes, tensor shapes, and scheduler implementations. Community reports already flag text-encoder compatibility issues for Klein (4B reportedly pairs with a Qwen3-4B encoder, 9B with Qwen3-8B). **Pin every component** and treat the manifest as a build artifact checked into the repo:

```yaml
model:
  transformer:   black-forest-labs/FLUX.2-klein-4B   # exact revision/commit
  text_encoder:  <exact Qwen3 encoder checkpoint + revision>
  tokenizer:     <exact tokenizer revision>
  vae:           <exact TAESD / AutoencoderTiny checkpoint>
  scheduler:     <exact implementation + revision>
  precision:     fp16            # A6000: fp16/int8; FP8 only on Ada/Blackwell
  steps:         4
  license:       Apache-2.0
```

Validate the full stack end-to-end before any compilation work — a mismatched encoder revision will silently shift tensor shapes and break TensorRT export later.

**Known dependency gotchas (from working Klein pipelines — Appendix B):**
- Pin `transformers==4.56.1`. Newer versions break the text encoder's FP8 Triton kernels.
- The BFL `flux2` package installs under the egg name `flux` (not `flux2`); install with `--no-deps` to avoid its `torch==2.8.0` pin clobbering your CUDA-matched torch build.
- Seed handling: `torch.randint(0, 2**63)` overflows — clamp seed range to `2**31`.
- FLUX.2 VAE (`ae.safetensors`, ~336 MB) is shared across all FLUX.2 variants; the Klein 4B pairs with a Qwen3-4B encoder, 9B with Qwen3-8B. Don't cross them.

### 4.2 The img2img loop and the two-axis control scheme

1. **Encode** input frame → latent (tiny VAE, see §4.3).
2. **Add noise** to the latent at a controlled denoise `strength` ∈ [0,1].
3. **Denoise** 1–4 steps, conditioned on style prompt embeddings and/or reference image embeddings.
4. **Decode** latent → RGB (tiny VAE).
5. Hand to composite/post stage.

Rather than a single "strength" knob, live behavior is governed by **two orthogonal axes** (a scheme borrowed from working FLUX.2-klein realtime pipelines — see Appendix B, tedngai/realtimesketch):

- **`ref_strength`** — how closely the output follows the *input frame* (its structure/content). Higher = more source fidelity.
- **`text_magnitude`** — how strongly the *prompt/style* asserts itself. Higher = more stylization.

These two are the show-operator's primary instrument, and they map cleanly onto MusiCue-driven parameter pairs (§7.5). Ship named presets so an operator isn't dialing raw numbers mid-show:

| Preset | ref_strength | text_magnitude | Live feel |
|---|---|---|---|
| PRESERVE | high | low | Faithful to the input; subtle restyle |
| SUBTLE | high-ish | mid | Light stylization |
| BALANCED | mid | mid-high | Equal input/style weight |
| FOLLOW | low | high | Style dominates the input |
| FORCE | very low | highest | Near-full hallucination, input as loose guide |

(Exact numeric values are a Phase-3 tuning output, since they interact with step count and the specific Klein checkpoint.)

### 4.3 Tiny VAE

Swap the full FLUX VAE for **TAESD** (`AutoencoderTiny`). At framerate the full VAE decode is a disproportionate share of per-frame cost; TAESD is the standard real-time trick. Quality loss is minor at projection resolution and well worth the latency. Both encode and decode use the tiny VAE.

### 4.4 Conditioning

- **Text/style prompt:** pre-compute embeddings whenever the prompt changes (not per frame). At 4 steps the text encoder (Qwen3 for Klein) is too expensive to run every frame; cache the embeddings and only recompute on prompt change.
- **Dual reference (content + style):** one reference image conditions structure, one conditions style (IP-Adapter-style image embedding). Pre-compute reference embeddings on load/change. This is where the Klein KV-cache trick legitimately applies — references are fixed, so cache their KV pairs once.

---

## 5. The compilation layer (where cadence is won)

### 5.1 What gets compiled, and the fallback ladder

"Compile everything" is the strategy, but the design must say *exactly* what becomes an engine and where the fallback boundaries are. FLUX.2 transformer export is non-trivial: attention, RoPE, custom ops, dynamic conditioning shapes, and scheduler glue all complicate a static engine. CUDA graphs require stable memory addresses and static shapes, so dynamic masks, prompt swaps, and reference changes need either careful graph invalidation or separate compiled paths.

Explicit engine list:

```text
Engine 0: VAE encode (TAESD)              — base graph
Engine 1: FLUX denoise x N steps          — base graph, fixed resolution + token shapes
Engine 2: VAE decode (TAESD)              — base graph
Engine 3: mask / composite / post kernels — base graph
Optional Engine 4: segmenter (mask brush)
Optional Engine 5: depth / pose / control adapter
Optional Engine 6: optical flow (RAFT / SEA-RAFT)
```

Optional engines (4–6) are **not** part of the base graph; they're loaded only when a show uses them, since each one eats frame budget. TAESD encode and decode are benchmarked and compiled separately from the transformer.

Per-engine fallback ladder:

```text
TensorRT          (primary — production)
torch.compile     (secondary — when a TRT export is unstable)
eager PyTorch     (debug only — never in a show)
```

| Technique | What it buys | Notes |
|---|---|---|
| **Nunchaku / SVDQuant (W4A16 INT4)** | **Large throughput on Ampere without FP8** | **The key A6000 lever** — see §5.1.1. INT4 weights + low-rank correction; runs FLUX where FP8 isn't available. |
| **TensorRT** per engine | 2–5× over eager on Ada/Blackwell | Fixed shapes, fused kernels. Each engine compiled independently. |
| **INT8 / FP16** | Throughput on A6000 | A6000 is Ampere: FP16 + INT8. **No native FP8.** |
| **FP8 quantization** | Large throughput, marginal quality loss | **Ada/Blackwell only** — not the A6000. |
| **CUDA graphs** | Eliminates per-kernel launch overhead | Needs stable addresses + static shapes; invalidate on resolution/mask-topology change. |
| **Static everything** | Required for TensorRT | Fixed resolution, batch, step count. Each resolution = a separate prebuilt engine set; recompiling mid-show is not an option. |

#### 5.1.1 Nunchaku / SVDQuant — the A6000's way around the FP8 ceiling

The single most important addition in v1.2. The earlier framing treated the A6000's lack of native FP8 as a hard throughput ceiling that only Blackwell could lift. That's not the whole picture: **Nunchaku** (MIT Han Lab's SVDQuant inference engine — the same acceleration lineage as StreamDiffusion) runs FLUX transformers at **4-bit weights (W4A16)** with a low-rank high-precision correction branch that preserves quality. Working repos (Appendix B, leeguandong/RealtimeFlux) already run FLUX in real time on consumer GPUs this way, including a quantized text encoder (`--use-qencoder`) to cut VRAM further.

Why this matters specifically for StreamForge on the A6000:
- It delivers FP8-class throughput on **Ampere**, which has no FP8 — so it directly lifts the resolution/cadence envelope that §5's tier table leaves as "benchmark-driven."
- INT4 weights shrink the transformer's memory footprint, leaving more of the 48 GB for larger TensorRT engines, longer pipelines, and optional engines (segmenter/depth/flow).
- It's a quality/speed knob, not a cliff: SVDQuant's low-rank branch is what keeps INT4 from looking like INT4.

This makes the **Phase 4 compile bake-off a three-way test, not two-way**:

```text
Path A: TensorRT (FP16/INT8)        — most mature, the safe baseline
Path B: Nunchaku / SVDQuant (INT4)  — likely the A6000 throughput winner
Path C: torch.compile               — fallback when an export is unstable
```

The go/no-go gate is no longer "can TensorRT hit cadence at resolution X" — it's "which of A/B gets the best cadence×quality at the show's resolution, and does INT4's quality hold at projection scale." Bench both against the same harness (§5.2) and the same manifest. Caveat: Nunchaku must support the exact Klein checkpoint/encoder you pin (§4.1.1) — verify kernel coverage for FLUX.2-klein-4B before committing, since SVDQuant kernels are model-architecture-specific.

**Hardware reality check (corrected):** the A6000 is Ampere — FP16/INT8, no native FP8. The achievable resolution and AI cadence are **benchmark outputs, not assumptions**. Plan against tiers and let Phase 2 pick:

| Tier | Resolution (incl. 16:9 / 2:1) | AI cadence | Output cadence | Purpose |
|---|---|---|---|---|
| Safe A6000 dev | 512×512 / 640×360 | 30 fps target | 30 fps | First real-time gate |
| A6000 show candidate | 768×432 / 768×768 | **benchmark-driven** | 30 fps | Likely practical target |
| Blackwell target | 1024+ | 30–50 fps target | 30–50 fps | Higher-res show mode |

Note that for media-server output, **rectangular 16:9 and 2:1** formats usually matter more than square 768×768; compile engines for the show's actual aspect ratio.

### 5.2 Benchmark harness (a required subsystem, not a milestone)

Average FPS is useless for shows. The harness is its own deliverable and gates everything after Phase 2. It reports:

| Metric | Why it matters |
|---|---|
| p50 / p95 / p99 stage latency | Tail latency, not the mean, decides show-safety |
| Worst-frame latency over 2 h | Catches thermal throttling, memory fragmentation, driver hitches |
| VRAM high-water mark | Prevents surprise OOM during cue/reference swaps |
| Missed-deadline count | Direct show-safety metric |
| Output jitter | Matters more than mean throughput |
| Frame-repeat count | Indicates how hard the governor is degrading |
| Prompt / reference swap latency | Bounds how responsive live control can be |
| NDI / Spout end-to-end latency | Needed to set the cue-bus delay (§6.2) |

Run the harness across **four input paths** — synthetic, Spout, NDI, and real capture — because each has different upload/sync overhead and the real capture path is the one the show actually uses.

---

## 6. Pipeline architecture (the hard part)

### 6.0 Clock / worker / governor decoupling

The diffusion loop must **not** own the show clock. Three cooperating components enforce the cadence model from §3.1:

```text
RealtimeClock
  - owns the output cadence (30 / 50 fps), never blocks on inference
  - each tick, emits the best available frame:
      latest completed AI frame > warped prior frame > held frame > raw input
InferenceWorker
  - runs opportunistically under budget, may skip input frames
  - reports per-stage timings to the Governor
Governor
  - adjusts step count (4→2→1), resolution tier, strength, optional engines
  - selects fallback policy and frame-fill strategy under sustained pressure
```

This is far more robust than making the denoise loop responsible for the clock: the media server always gets a stable signal, and the AI degrades smoothly (fewer fresh frames, more warps/holds) instead of stalling.

### 6.1 GPU-resident dataflow

A single CUDA context. Tensors are handed pointer-to-pointer between stages. The input upload (camera/capture → VRAM) and the final output download (VRAM → NDI/Spout surface) are the only host↔device transfers. Everything in between stays in VRAM.

For input, prefer a capture card or source that can DMA directly to GPU memory (GPUDirect, or NDI/Spout receive straight into a GPU texture — see §8). For output, Spout/Syphon keep the frame on the GPU entirely (zero readback); NDI requires a download + encode (see §8.3).

### 6.2 Staged pipeline parallelism

The naïve sequential loop (capture → preprocess → infer → post → output, one frame at a time) leaves the GPU idle during capture and I/O. Instead, stage the pipeline across CUDA streams so that while frame *N* denoises, frame *N+1* preprocesses and frame *N−1* decodes/outputs:

```
        t0        t1        t2        t3        t4
Stream A  cap(0)    cap(1)    cap(2)    cap(3)
Stream B            pre(0)    pre(1)    pre(2)    pre(3)
Stream C                      inf(0)    inf(1)    inf(2)
Stream D                                post(0)   post(1)
Stream E                                          out(0)
```

This trades a fixed latency of ~3–4 frames (≈ 100–130 ms at 30 fps) for sustained throughput. **For a live show that fixed latency is acceptable and predictable** — delay the audio/lighting cue bus by the same amount to keep the show in sync. Predictable latency is fine; jitter is not.

**Critical caveat:** pipelining hides I/O and raises utilization, but it does **not** speed up the diffusion core. The slowest stage still has to finish within the frame interval for a unique result every frame — if denoise takes 80 ms, fresh-frame production is ~12.5 fps regardless of overlap. That is exactly why the output clock is decoupled (§6.0): pipelining maximizes how many fresh AI frames you get, and the RealtimeClock fills the remainder with warps/holds to keep the output at 30/50 fps.

### 6.3 Governor policy (detail)

The Governor (§6.0) tracks per-stage p95/p99 timing and VRAM, and under sustained pressure escalates a fixed ladder: smooth `strength`/prompt interpolation → increase warp/hold fill ratio → drop step count (4→2→1) → drop resolution tier → degraded passthrough. It never lets the RealtimeClock starve. The escalation order is configurable per show so an operator can decide, e.g., whether to sacrifice resolution or AI cadence first. This is the difference between a demo and a show tool.

---

## 7. Effects / control nodes

These make StreamForge a *tool* rather than a single-effect demo. Each maps directly onto Cedar's existing ComfyUI node experience.

### 7.1 Region-masked regeneration (brush)
A hand-painted mask or a fast segmenter (lightweight SAM-class model at framerate, compiled separately) defines where diffusion applies. Compositing happens in latent space, blended at the VAE boundary so seams don't crawl frame-to-frame.

### 7.2 Structure conditioning (depth / pose lock)
Depth estimation or pose extraction → ControlNet-style conditioning to lock structure while restyling. **Cost warning:** ControlNet adds compute that may not fit alongside a 4-step budget. Use a distilled/lightweight control adapter, and treat structure lock as a per-show opt-in that may force a lower base resolution.

**4B structure-adherence caveat (Appendix B):** working Klein pipelines report that the *distilled* klein-4B follows a structure/sketch input only weakly — good prompt-driven restyle, poor at holding an input's geometry — while the 9B-base (non-commercial, ~28 steps) holds structure well. Implication: for pure *restyle* this is fine (you often *want* the model to reinterpret). But if a show needs tight structure-lock on the commercial-clean 4B, don't rely on the base model's native adherence — budget for a real control adapter, and validate adherence at Phase 1 before designing a cue around it. This is also an argument for motion-vector/structure conditioning coming from the render engine (§9) rather than from the diffusion model's own input fidelity.

### 7.3 Dual reference (content + style)
See §4.4. KV-cached reference embeddings, no per-frame recompute, no offline bake.

### 7.4 Temporal coherence (critical — §9)
Stabilization module sits in the composite/post stage.

### 7.5 Audio-reactive control (MusiCue)
Cedar's existing MusiCue work plugs in directly here. MusiCue's typed musical keyframes (stems, onsets, structure) drive node parameters per frame over the control bus: the two control axes `ref_strength` and `text_magnitude` (§4.2), mask intensity, and seed jitter. The two-axis scheme is a natural MusiCue target — e.g. map energy/onset density to `text_magnitude` (stylization swells with intensity) and section structure to `ref_strength` (verses stay faithful, choruses dissolve into style). This is the offline-to-live bridge — MusiCue already produces exactly the structured control signal this needs.

### 7.6 Offline asset prep (style LoRAs + VAE debiasing)
Not part of the live runtime, but a quality multiplier prepared ahead of a show (Appendix B, comfyUI-Realtime-Lora):

- **Bake show looks into a style LoRA.** Train a FLUX-klein-4B style LoRA (Musubi-Tuner; ~13 GB, runs on the A6000) on reference art for a given show, then load it into the inference pipeline. This is a stronger, more controllable aesthetic path than prompt-only styling, and it pairs with Cedar's existing Wan 2.2 / QWEN 360° LoRA experience. Keep training to the **base** (undistilled) 4B and confirm the trained LoRA applies cleanly to the distilled inference checkpoint.
- **VAE debiasing for color cast.** Per-conv debiasing of the FLUX VAE can correct color/saturation bias at the tensor level — relevant to the §8.6 color risk. **Caveat:** these tools operate on the *full* FLUX VAE; if the live runtime uses TAESD (§4.3), debiasing doesn't transfer to the tiny VAE. Treat it as an asset-prep / reference-render tool, not part of the live path, or validate whether an equivalent correction can be folded into the ColorPipeline.

Both are pre-show prep, run offline, and produce artifacts (a LoRA file, a corrected VAE/reference) the live runtime consumes — they never touch the frame deadline.

---

## 8. Output integration: NDI vs Spout/Syphon

This is the core of getting StreamForge's output into a media server, and the two protocols solve different problems.

### 8.1 Decision matrix

| | **Spout (Win) / Syphon (mac)** | **NDI** |
|---|---|---|
| Transport | Shared GPU memory, same machine only | Network (or localhost), cross-machine |
| Latency | Sub-millisecond, zero-copy | Several ms; encode + network + decode |
| GPU readback | **None** — texture stays on GPU | Required — download, encode (NDI codec) |
| Pixel formats | Up to 32-bit float RGBA (Spout) | Typically 8-bit; high-bandwidth modes exist |
| Cross-machine | No | **Yes** — this is its whole point |
| Platform | Spout=Win (NVIDIA/AMD, not Intel); Syphon=mac, 8-bit RGBA | Win / mac / Linux |
| Best for | Same-box StreamForge → Resolume/TD/MadMapper | StreamForge box → separate media-server box |

**Rule of thumb:** if StreamForge and the media server run on the **same machine**, use **Spout/Syphon** — the texture never leaves the GPU and latency is effectively zero. If they run on **different machines** (common in a real show rack — dedicated AI box feeding a dedicated media-server box), use **NDI**.

### 8.2 Spout/Syphon path (same machine)

Spout shares DirectX/OpenGL textures via shared memory; Syphon does the equivalent on macOS (limited to 8-bit RGBA). The StreamForge output frame is already a GPU texture at the end of the composite stage, so a Spout sender publishes it directly with no readback. Resolume, TouchDesigner (Syphon Spout In TOP), MadMapper, and VDMX all consume Spout/Syphon sources natively.

This is exactly the integration Daydream's Scope shipped for StreamDiffusionV2 — AI-transformed video into Resolume as a live source layer, the frame arriving in the same interval it was generated. It's a proven pattern, not speculation.

Implementation notes:
- **Texture flip:** Spout → Resolume Arena commonly requires a vertical flip. Bake a flip flag into the sender.
- **Format:** Spout supports up to 32-bit float RGBA — useful if downstream wants HDR/linear. Default to 8-bit RGBA for compatibility unless the show needs more.
- **Sender limit:** Spout defaults to 10 active senders on Windows (registry-adjustable). Not a concern for a single output.
- **Platform constraint:** Spout is Windows-only and needs an NVIDIA/AMD GPU (Intel iGPU won't work). If the StreamForge box is Linux, Spout is unavailable — use NDI.

### 8.3 NDI path (cross-machine, or Linux output)

NDI sends frames over the network (or localhost loopback). Unavoidable cost: download the GPU frame to host, encode with the NDI codec, transmit. Add NVENC where possible to offload encoding.

Implementation notes:
- **Python binding:** `ndi-python` (buresu) wraps the NDI SDK for send/receive. SDK install required; mind the NDI SDK license terms before any commercial deployment.
- **Frame sync / clock domains:** NDI senders and receivers do **not** share a clock. The NDI **FrameSync** component converts the push source into a pull source and does time-base correction + jitter handling. A common pattern is to time the receiving end to GPU v-sync and convert the incoming time-base to the GPU's. For a show, the media server's NDI input should use FrameSync to absorb the inevitable crystal-clock drift between the StreamForge box and the media server box. Without it, you accumulate drift over a long show.
- **Bandwidth:** NDI (full) is high-bitrate but compresses well; NDI|HX is lower-bitrate for constrained networks at a quality cost. For a dedicated 10GbE show network, prefer full NDI.
- **Resolution/bit-depth:** standard NDI is 8-bit; if the show needs 10-bit, confirm the SDK path and media-server support.

### 8.4 Hybrid / bridges
`Spout-to-NDI` and `NDI-to-Spout` bridge utilities exist (and a newer `Spout-to-OMT` over the Open Media Transport protocol). These let a same-box Spout pipeline also fan out to a networked NDI consumer when needed. Useful as an escape hatch; adds the NDI encode cost only on the bridged branch.

### 8.5 Recommendation for Cedar's likely rig
Plan for **both**, selected by config:
- **Dev / single-box testing:** Spout (Windows) — zero-latency, immediate Resolume/TD feedback loop.
- **Show rack (AI box → media-server box):** NDI with FrameSync on the receive side, on a dedicated 10GbE network.
Build the output stage as a pluggable sink interface (`SpoutSink`, `SyphonSink`, `NDISink`) so the same pipeline serves either without touching the core.

### 8.6 Color management (a real show requirement)

Large-format projection pipelines are unforgiving about gamma, range, and color space — get this wrong and the AI output looks washed out or crushed on the wall even when it's perfect in the viewer. Add a `ColorPipeline` stage immediately before the sink:

```text
ColorPipeline:
  - linear / sRGB / Rec.709 transform
  - tone map / clamp
  - bit-depth conversion (8-bit / 10-bit / float)
  - alpha: premultiplied vs straight
  - range: legal (16–235) vs full (0–255)
  - test-pattern passthrough mode
```

Critically, include a **known test-pattern mode** that bypasses the AI entirely and emits a standard chart/ramp through the full sink path. This lets the media-server operator confirm the signal path, color, and range *before* the AI is even running — invaluable during a load-in when something looks wrong and you need to isolate whether it's the model or the pipe.

---

## 9. Temporal coherence (the quiet hard problem)

Per-frame img2img **flickers** — each frame is denoised independently, so textures shimmer and details boil between frames. This is the single biggest quality giveaway of live-restyle tools. RAFT/SEA-RAFT optical flow *plus* FLUX may be too expensive to run every frame, so the default stack leans on much cheaper techniques and treats flow as opt-in:

| Technique | Cost | Role |
|---|---|---|
| Fixed noise tensor | very low | **default** — stop re-rolling randomness each frame |
| Latent EMA / blending | low | **default** — blend frame *N* start with *N−1* result; blend factor is a live knob |
| Previous-output color stabilization | low | **default** — clamps shimmer in flat regions |
| Frame-to-frame prompt/strength smoothing | very low | **default** — prevents parameter "pops" on live control moves |
| Motion-vector reuse from render engine | low *if available* | **best case** — see below |
| Optical-flow warp (RAFT / SEA-RAFT) | medium/high | opt-in; reuses Cedar's `ComfyUI_MotionTransfer` code |
| AI-cadence + warped hold frames | medium | the §3.1 fill strategy when FLUX runs 10–20 fps |

**Motion-vector reuse is the sleeper win.** When the input is a *render* feed (Unreal, Notch, disguise, Cinema4D) rather than a camera, the engine can hand you camera motion, depth, and per-pixel motion vectors directly — far cheaper and more accurate than estimating optical flow from pixels. For Cedar's UE5/render-driven Sphere work this is likely the highest-quality, lowest-cost path to coherence and to filling AI-cadence gaps with correct warps. Wire a `motion_vector_in` source alongside the pixel input.

v1 ships the four "default" rows; optical-flow warp and motion-vector reuse are opt-in. The genuinely coherent solution is autoregressive video (§11), but that sacrifices the live-input-restyle use case, which is the whole point of this tool.

---

## 10. Build order (milestones)

Revised so that **latency validation comes before optimization** and the output clock is stable even before the AI is fast.

| Phase | Deliverable | Gate |
|---|---|---|
| **1** | Minimal still-image FLUX.2-klein-4B img2img. Confirm exact model stack (manifest §4.1.1), license, prompt cache, VAE compatibility. **Test 4B structure adherence (§7.2)** — does it hold input geometry, or only restyle? | Correct output on a still; manifest validated; adherence behavior known before cue design. |
| **2** | **Benchmark harness (§5.2) before any optimization** — eager baseline with full per-stage timings. | Harness reporting p50/p95/p99, jitter, VRAM, across input paths. |
| **3** | Minimum viable pipeline: tiny VAE + prompt/reference cache + fixed shapes + two-axis control (§4.2). | Establishes the real MVP latency floor. |
| **4** | **Three-way compile bake-off (§5.1.1):** TensorRT (FP16/INT8) vs Nunchaku/SVDQuant (INT4) vs torch.compile, per engine, same harness + manifest. | Best cadence×quality path chosen; INT4 quality validated at projection scale; **the resolution/cadence go/no-go.** |
| **5** | 30/50 fps **output clock** with AI cadence decoupled (§6.0). Even if AI runs 10–20 fps, the sink is rock-stable. | Stable output clock under variable AI cadence. |
| **6** | **Spout sink first** (same-machine, lower latency, easiest debugging) + ColorPipeline + test-pattern mode. | Live frame appears in Resolume as a source layer; color verified. |
| **7** | Governor: warp/hold fill, adaptive step count, adaptive resolution tier, passthrough fallback (§6.0, §6.3). | No stalls under induced load. |
| **8** | Temporal stabilization: fixed seed + latent EMA + color stabilization (then optional flow / motion-vector reuse). | Flicker visibly suppressed on motion. |
| **9** | NDI sink + receive-side FrameSync + **multi-hour soak test**. | Drift, jitter, and media-server behavior validated over 2 h. |
| **10** | Control nodes only after the core loop is stable: MusiCue, mask brush, depth/pose. | Audio-reactive restyle end-to-end. |
| **later** | Node-graph authoring GUI; multi-GPU pipeline parallelism. | — |

Phases 1–6 are a few focused weeks for someone with Cedar's background, solo, on the A6000. Phases 7–9 harden it into a show tool. The GUI is the long-tail effort that distinguishes a runtime from a product — worth asking whether a specific Sphere cue needs the GUI, or just the runtime driven by config/JSON (Cedar's existing working style).

---

## 11. Relationship to autoregressive video models

StreamForge is deliberately an **img2img live-restyle** engine, not an autoregressive video generator. The trade is explicit:

- **img2img (StreamForge):** restyles a *real* live input, flickers without stabilization, low latency, simple. Right for camera/render-feed restyle — the scenographer use case.
- **Autoregressive video (StreamDiffusionV2, LongLive, Self-Forcing):** temporally coherent *generated* content via KV-cache over prior frames, but hard to drive from a live input and heavier. Right for generated narrative/worlds.

If a future show needs coherent *generated* video rather than restyled input, that's a separate engine built on StreamDiffusionV2 or LongLive — not a StreamForge feature. Keep the two concerns separate.

---

## 12. Open questions / risks

1. **Generated-frame cadence vs output cadence.** The media server must always receive a stable 30/50 fps signal, but the AI may not produce a fresh diffusion result for every input frame at higher resolutions. StreamForge explicitly supports lower AI cadence via frame repeat, latent reuse, optical/motion-vector warping, and smooth parameter interpolation (§3.1, §6.0). The Phase 2/4 benchmarks determine the actual cadence per resolution tier.
2. **A6000 FP8 ceiling — mitigated, not eliminated.** Ampere lacks native FP8 (FP16/INT8 only). **Nunchaku/SVDQuant INT4 (§5.1.1) is the primary mitigation** and likely lifts the envelope substantially, but the actual resolution/cadence must still be *measured* at the Phase 4 three-way bake-off before committing to a show resolution. Blackwell remains the clean path to 1024+/50fps.
3. **Nunchaku kernel coverage for klein-4B.** SVDQuant kernels are architecture-specific; confirm Nunchaku actually supports the pinned FLUX.2-klein-4B + Qwen3-4B stack before betting Phase 4 on the INT4 path. If coverage is missing, fall back to TensorRT FP16/INT8 and re-scope the resolution tier.
4. **4B structure adherence.** The commercial-clean distilled 4B may hold input geometry only weakly (§7.2). Validate at Phase 1; if a show needs tight structure-lock, budget a control adapter or drive structure from render-engine motion vectors rather than the model's native fidelity.
5. **TensorRT export of the FLUX.2 transformer.** Attention, RoPE, custom ops, and dynamic conditioning shapes may resist a clean static engine; the torch.compile fallback (§5.1) and the Nunchaku path (§5.1.1) both hedge this. De-risk early.
6. **ControlNet / structure lock vs budget.** Structure lock (§7.2) may not coexist with a 4-step budget at high resolution. Decide per show; it's an optional engine.
7. **NDI clock drift over long shows.** Must use FrameSync on receive; validate over a multi-hour soak test (Phase 9), not a 5-minute demo.
8. **Recompile friction.** Any resolution/aspect change requires a recompile. Decide show resolution early; keep a small set of pre-built engines for common 16:9 / 2:1 resolutions.
9. **Color management.** Range/gamma/color-space mismatches show up on the wall, not the viewer. Test-pattern mode (§8.6) must work before relying on AI output at load-in. Note VAE debiasing (§7.6) doesn't apply to TAESD.
10. **Model manifest drift.** Encoder/tokenizer/scheduler revisions diverge across ComfyUI/Diffusers/BFL/GGUF; specific pins in §4.1.1. Validate the stack before compiling.
11. **Licensing.** Default 4B path is Apache-2.0 (clean). NDI SDK has its own commercial terms — review before a paid show. The 9B/9B-KV models are non-commercial (Appendix A) — never in a billable pipeline.
12. **Input DMA path.** Getting the capture source straight into GPU memory (vs a CPU round-trip) materially affects the budget; confirm the capture-card / NDI-receive path early.

---

## Appendix A — Non-commercial models (not in the v1 streaming path)

These are documented for completeness and for a possible *offline editing* side workflow, but are **excluded from the live-streaming product** because of licensing and fit. Keeping them out of the main model table (§4.1) avoids any confusion for lenders, clients, or partners about whether the default system is commercially clean.

| Model | License | Why excluded from streaming |
|---|---|---|
| FLUX.2 [klein] 9B | FLUX Non-Commercial License | Non-commercial; too heavy for live cadence on A6000. |
| FLUX.2 [klein] 9B-KV | FLUX Non-Commercial License | Its KV-cache only helps when *reference* images are fixed across many generations. A live feed presents a new input every frame, so the cache gives zero benefit — you pay full 9B cost per frame. Useful only for iterative multi-reference *editing* (fixed references, many variations), which is a different, offline workflow. |

If an offline editing tool is ever built around 9B-KV, it must carry a non-commercial flag and a separate licensing review.

---

## Appendix B — Prior art and what to borrow

Open-source precedents that de-risk StreamForge. None combines all of FLUX.2-klein + Spout/NDI sinks + show-grade governor + soak testing + color management + media-server-first deployment — that combination remains the differentiator — but each contributes pieces.

**Acceleration**
- **Nunchaku / SVDQuant** (MIT Han Lab) — INT4 (W4A16) FLUX inference with low-rank correction. The A6000's route around the FP8 ceiling (§5.1.1). *The single most valuable borrow.*
- **leeguandong/RealtimeFlux** — small demo, but proves FLUX real-time on consumer GPUs via Nunchaku, incl. a quantized text encoder (`--use-qencoder`). Reference for the INT4 path.

**Working FLUX.2-klein realtime pipeline**
- **tedngai/realtimesketch** — complete FLUX.2-klein client/server over WebSocket. Source of the **two-axis control scheme + named presets** (§4.2), the **dependency-pinning gotchas** (§4.1.1), and the **4B structure-adherence caveat** (§7.2). Closest working reference to StreamForge's runtime split.

**Offline asset prep**
- **shootthesound/comfyUI-Realtime-Lora** — Musubi-Tuner FLUX-klein-4B LoRA training + block-level VAE debiasing. Source for §7.6 (style-LoRA baking, color-cast correction). Training/editing tool, not a live path.

**Realtime diffusion architecture / sinks** (from the broader survey)
- **cumulo-autumn/StreamDiffusion** — the foundational realtime pipeline: stream batching, IO queues, prompt cache. Baseline architecture reference.
- **olegchomp/StreamDiffusion-NDI** — NDI + OSC precedent. Reference for `NDISink` + OSC control.
- **olwal/streamdiffusion-spout-service** — Spout + OSC precedent. Reference for `SpoutSink`.
- **olegchomp/TouchDiffusion** — realistic SD-Turbo/LCM benchmark expectations (512² ≈ 55–60 fps on 4090, ~30 on 3090) and TD integration.
- **chenfengxu714/StreamDiffusionV2 / NVlabs/LongLive / open-mmlab/Live2Diff** — the autoregressive-video direction (§11), relevant only if a future generated-video engine is built.

**Don't bother**
- **tensorforger/FluxRT** — weekend prototype (Klein-4B + RIFE interpolation + shared memory). Pattern already covered better by the above.

---

*End of v1.2. Next revision should fold in Phase 4 three-way bake-off numbers (TensorRT vs Nunchaku-INT4 vs torch.compile) on the A6000 to lock the show resolution / aspect / cadence envelope, confirm Nunchaku kernel coverage for the pinned klein-4B stack, and replace the manifest placeholders with pinned revisions.*

# Phase-4 bake-off — interim result (Windows / A6000, 2026-06-05)

Measured at 512², 4 steps, `image=` restyle path, prompt-cached. Baseline = bf16 eager.

| Path | Runtime | infer p50 | VRAM | Verdict |
|---|---|---|---|---|
| — | bf16 eager (baseline) | 1049 ms | 16.7 GB | reference |
| B′ | bitsandbytes 8-bit (LLM.int8) | 1441 ms | 12.8 GB | **slower** (decomp overhead) |
| B′ | bitsandbytes 4-bit NF4 | 1118 ms | 11.1 GB | **slower** (~7%) |
| C | torch.compile (inductor) | — | — | **blocked**: Triton missing on Windows |
| A | TensorRT | not yet tested | — | only Windows-native speedup left |

## Classic img2img mode — IMPLEMENTED + measured (2026-06-05)
Added `EagerRuntime(mode="img2img")`: encode -> noise@strength -> denoise SAME tokens (no
reference concat), validated by a 1.2/255 encode→decode roundtrip.
- **infer p50 = 358 ms** @512² BALANCED (vs 1049 ms edit path) = **~2.9× faster** (~2.8 fps), 16.7 GB.
- Latency scales with strength: BALANCED (denoise 0.69) runs ~2 of 4 steps; faithful presets are
  even faster, FORCE runs ~all 4. So fidelity↑ cadence↑ together at the faithful end.
- Quality tradeoff (validated): low strength = faithful/under-styled; high strength = hallucinates
  (loses structure — the §7.2 classic weakness). Edit path keeps structure+style but costs 3×.
- **Recommendation:** ship BOTH modes; pick per show (edit = structure-critical; img2img = cadence-critical).

## Triton-Windows + compile + INT8/TorchAO (2026-06-05, user-suggested leads)
- **triton-windows 3.6.0** (`pip install "triton-windows<3.7"`) **unblocks torch.compile** on
  this stack (torch 2.11+cu128, Py3.11). "torch.compile just works."
- **bf16 + torch.compile (edit):** steady-state p50 **864 ms** (vs 1049, ~18%). Works cleanly
  (first 1-2 frames pay the one-time compile). VRAM 16.7 GB.
- **INT8 W8A8 dynamic (TorchAO):** ERRORS — `self.size(0) needs to be > 16` (the adaLN/modulation
  linears have M=1; the dynamic-activation kernel rejects them).
- **INT8 weight-only (TorchAO) + compile:** **pathological recompilation** — p50 120 s/frame
  (recompiles ~every frame; torchao tensor-subclass guards fail under dynamo). VRAM 12.8 GB.
  Not usable as-is; needs torch._dynamo recompile tuning (cache_size_limit, mark_dynamic, or the
  torchao-recommended compile mode) — a separate focused effort.
- **vistralis/FLUX.2-klein-4b-INT8-transformer** = ModelOpt INT8 W8A8 as TorchAO safetensors;
  its 2.5× (5090: 1.77→0.72 s) is **with torch.compile** — i.e. it would hit the same Windows
  recompilation issue unless tuned. Same class of problem as above.
- **TensorRT (#4712):** unanswered roadmap request → no native FLUX.2-klein TensorRT yet.

## Conclusions
1. **Quantization (bitsandbytes) is the wrong tool for cadence here.** It reduces VRAM (which
   was never the constraint — 16.7/48 GB) at a latency *cost*. Not a speed win.
2. **The real fast-quant kernels (Nunchaku/SDNQ INT4, torch.compile/inductor) need Triton**,
   which is unavailable on native Windows. They become available under **WSL/Linux**.
3. On native Windows, the only remaining transformer speedup is **TensorRT** (Path A) — heavy,
   with the known FLUX.2 export risk (attention/RoPE/custom ops).
4. **Cheapest Windows-native win is NOT a quant at all:** the **classic img2img mode** (encode→
   noise@strength→denoise SAME tokens, no reference-token concat) removes the measured **431 ms**
   reference overhead (~40%: 1032→~600 ms) AND provides the real `ref_strength` axis, at some
   structure-adherence cost vs the edit path.

## Recommendation for the production cadence path
- **Quick Windows win:** implement the classic img2img mode (~40% faster, adds ref_strength).
- **Real throughput:** either commit to **TensorRT** on Windows, or move the AI box to
  **Linux/WSL** to unlock Triton-based INT4 (Nunchaku/SDNQ) + torch.compile. This is a
  deployment-environment decision (design §5.1.1 / risk #2,#3,#5) — record it before Phase 4 final.
- bitsandbytes stays useful only if a future larger model needs the VRAM headroom.

## FINAL BOTTOM LINE (2026-06-05)
Per-frame latency @512² (lower = better):
| Config | infer p50 | Status |
|---|---|---|
| edit, bf16 | 1049 ms | baseline, robust |
| edit, bf16, compile | 864 ms | works (~18%) |
| **img2img, bf16** | **358 ms** | **robust 2.9× win — production-ready now** |
| edit/img2img, int8+compile | (120 s recompiling) | promising but needs dynamo recompile tuning |

**Production cadence path on Windows today = the classic img2img mode (358 ms, ~2.8 fps).**
With the output clock decoupled, that already drives a stable 30/50 fps to the media server.
**Next perf step (optional):** resolve the int8+compile recompilation (the vistralis/TorchAO
2.5× path) — likely `torch._dynamo.config.cache_size_limit` + `mark_static`/`dynamic=False`, or
run the AI box under **WSL/Linux** where the torchao+compile recipe is well-trodden. Combining
img2img + a working int8+compile would target ~150 ms (~6-7 fps real AI) before any TensorRT.

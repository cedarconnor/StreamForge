# Phase-4 bake-off — interim result (Windows / A6000, 2026-06-05)

Measured at 512², 4 steps, `image=` restyle path, prompt-cached. Baseline = bf16 eager.

| Path | Runtime | infer p50 | VRAM | Verdict |
|---|---|---|---|---|
| — | bf16 eager (baseline) | 1049 ms | 16.7 GB | reference |
| B′ | bitsandbytes 8-bit (LLM.int8) | 1441 ms | 12.8 GB | **slower** (decomp overhead) |
| B′ | bitsandbytes 4-bit NF4 | 1118 ms | 11.1 GB | **slower** (~7%) |
| C | torch.compile (inductor) | — | — | **blocked**: Triton missing on Windows |
| A | TensorRT | not yet tested | — | only Windows-native speedup left |

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

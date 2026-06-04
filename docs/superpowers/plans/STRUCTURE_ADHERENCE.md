# 4B structure adherence (Task 1.3) — preliminary, 2026-06-04

## Method
Restyled frame 0 of the test clip (a person on a couch) via the FLUX.2-klein **native
`image=` edit/reference path** at 4 steps, prompt "vivid oil painting, thick impasto".

## Qualitative result
Structure adherence is **STRONG**, not weak. The output is unmistakably the same subject —
same face, pose, head position, couch background, and lighting — re-rendered in the target
style. See out/src_frame.png vs out/edit_ref.png.

## Why this differs from the design §7.2 caveat
The §7.2 "distilled 4B holds geometry only weakly" caveat referred to **sketch/structure
input** and classic **img2img-noise**. FLUX.2's `image=` is **reference-conditioning via
token concatenation**, which preserves structure far more strongly. For the live-restyle use
case (the whole point of StreamForge), this is the right primitive.

## DECISION (preliminary)
**structure-lock for v1 = NOT NEEDED** for prompt-driven restyle via the `image=` edit path.
A control adapter (design §5.1 Engine 5) is NOT required for v1. Revisit only if a specific
show needs *tighter-than-edit* geometry lock or sketch-driven structure.

## To finalize
- [ ] Quantitative edge-IoU across the control range once EagerRuntime + the two-axis knobs
      are wired (ref_strength via noise-blend, text_magnitude via embedding interp).
- [ ] Confirm adherence holds across motion (multiple frames), not just frame 0.

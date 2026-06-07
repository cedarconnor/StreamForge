# RIFE Frame-Fill — Design Spec

**Date:** 2026-06-07
**Status:** Approved (ready for implementation planning)

## Goal

Turn StreamForge's ~12–14 fresh fps AI restyle into smooth 30/50 fps **output** by synthesizing
the in-between frames via **motion extrapolation** (forward optical-flow warp), instead of repeating
the last AI frame (today's judder). Raises *perceived* frame rate with **zero added latency**.

## Context (why this shape)

The output path is already decoupled (design §6.0): `InferenceWorker` (own thread) restyles source
frames and publishes the latest styled `GpuFrame` to a single-slot `FrameBuffer`; `RealtimeClock`
(emit thread) ticks at 30/50 fps and emits the latest frame, **repeating it when no fresh AI frame
is ready**. `clock.py` already contains an unused `select_fill(ai, warped, held, raw)` scaffold with
a `"warped"` tier — the seam this feature fills.

The TensorRT investigation (see `docs/superpowers/plans/TENSORRT_RESULT.md`) confirmed the
transformer step cannot be made faster on this hardware (memory-bound at batch=1; FP16≈bf16; INT8 no
speedup). So the fresh-frame ceiling (~14 fps @384×224 PRESERVE, 70.7 ms) is fixed, and the
remaining lever for a smoother show is output-side frame-fill.

### Decisions locked during brainstorming
1. **Extrapolation, not interpolation** (user choice: minimum latency). True RIFE interpolation needs
   the *next* frame to fill a gap → adds ~1 AI interval (~70–100 ms) of latency, which was rejected.
   Running RIFE between two *past* styled frames only smooths an already-elapsed gap, so it can't
   smooth the forward edge without waiting. Therefore the engine is **optical-flow forward-warp**,
   **not** the RIFE network. (Bonus: avoids RIFE weights' non-commercial licensing, which would
   clash with the project's commercial-clean constraint. torchvision RAFT is BSD.)
2. **Flow engine: torchvision `raft_small`** (user choice) — learned flow, BSD-licensed, GPU, best
   motion quality. Classical OpenCV DIS is the documented fallback if RAFT load fails.
3. **Motion source: consecutive SOURCE frames.** img2img preserves structure, so source motion ≈
   styled motion; source frames are available immediately (the source runs ahead of the AI), giving
   a valid velocity with no extra latency.
4. **Default `--fill off`** (opt-in for v1; flip to default once proven). Extrapolation cap default
   **~120 ms**. Fill-thread optimization deferred (YAGNI).

## Architecture & data flow

```
SOURCE ──► InferenceWorker (thread)                 RealtimeClock (emit thread, 30/50fps)
            │  restyle(src) ─► styled                │  tick:
            │  RAFT(prev_src,cur_src) ─► velocity     │   fr = FrameBuffer.get_with_freshness()
            │  filler.set_anchor(styled,vel,t) ───────┼──►│  fresh? emit styled (source="ai"); held=styled
            │  FrameBuffer.publish(styled) ───────────┘   │  stale? warped = filler.fill(now)
                                                          │          select_fill(ai,warped,held,raw)
                                                          │          emit warped, else hold held
```

- The **worker** owns motion estimation: it keeps the previous source frame, computes the velocity
  field between it and the current source frame, and hands the **`FrameFiller`** an *anchor* =
  (latest styled tensor, velocity field, timestamp). It also publishes the styled frame to the
  existing `FrameBuffer` (for the clock's freshness signal and fresh-path emit).
- The **clock**, on a stale tick (no fresh AI frame), asks the filler for a frame extrapolated to
  the current time: `warped = warp_forward(anchor_styled, velocity, now − t_anchor)`. `select_fill`
  picks `ai > warped > held`.
- **Zero added latency**: the warp uses already-available data (past source motion + latest styled
  frame); nothing is held back.

### Velocity units & warp direction
- The filler stores the **raw flow** = flow(prev_src → cur_src), i.e. displacement over one
  source-frame interval (pixels per frame), plus the **source fps**. `warp_forward` is the single
  place that converts to displacement: `disp = flow · fps · Δt`, where `Δt = now − t_anchor`
  (wall-clock seconds). This keeps the scaling in one tested function.
- Forward extrapolation: content moves by `+disp`, so rendering samples **backward** by `−disp`
  (`grid_sample`). The sign and magnitude are pinned by `test_warp.py`.

### Resolutions
- Flow is computed at a low **flow-res** (cheap RAFT) on the source frames; the styled anchor is at
  **output res** (restyle upscales internal→source). Default flow-res downscales the source so the
  longest side is ≤ 384 and both dims are divisible by 8 (RAFT requirement). `warp_forward` resizes
  the flow field to the styled tensor's resolution and scales flow magnitude by the resize ratio
  before sampling.

### Threading & placement
- **Flow on the worker thread** (~15 ms at flow-res), at AI rate (~14/s). This nudges the AI rate
  from ~14→~12 fps — an accepted tradeoff (slightly fewer fresh frames in exchange for smooth
  output). Flow only runs when a filler is present.
- **Warp on the clock thread** (~1–2 ms `grid_sample` per stale tick). The clock stays light.
- `FrameFiller` is thread-safe (lock-guarded anchor), the motion-extrapolation analog of
  `FrameBuffer`.

## Components

Each unit is small, single-purpose, and independently testable.

| File | Responsibility | Key interface |
|---|---|---|
| `src/streamforge/fill/__init__.py` | package | — |
| `src/streamforge/fill/warp.py` | pure forward-warp via `grid_sample` | `warp_forward(image_bchw, flow_b2hw, dt: float, fps: float, max_disp: float \| None) -> tensor` |
| `src/streamforge/fill/flow.py` | torchvision `raft_small` wrapper (lazy fp16 eval), classical-DIS fallback | `RaftFlow(flow_hw, device).estimate(prev_bchw, cur_bchw) -> flow_b2hw` |
| `src/streamforge/fill/filler.py` | thread-safe anchor + extrapolation | `FrameFiller(max_extrap_ms, fps).set_anchor(styled, flow, t)` / `.fill(now) -> tensor \| None` (returns `None` past cap; else `warp_forward(styled, flow, now-t, fps, max_disp)`) |
| `src/streamforge/clock.py` *(modify)* | wire `select_fill`; optional filler | `RealtimeClock(fps, fb, emit, filler=None)`; add `filled_count`, `held` |
| `src/streamforge/worker.py` *(modify)* | compute velocity, set anchor, publish | accepts optional `flow` + `filler`; keeps `prev_src` |
| `scripts/live.py` *(modify)* | CLI wiring | `--fill {off,warp}` (default off), `--max-extrap-ms` (default 120) |

`select_fill` tier mapping: `ai` = fresh styled frame; `warped` = filler output; `held` = last
emitted frame (clock keeps it); `raw` = `None` (v1 never emits unstyled source — too jarring).

### `RealtimeClock` tick logic (modified)
```
fr = fb.get_with_freshness()
if fr.is_fresh and fr.value is not None:
    self._held = fr.value
    self.emit(fr.value); return
self.repeat_count += 1                       # unchanged metric
warped = self.filler.fill(now()) if self.filler else None
result = select_fill(ai=None, warped=warped, held=self._held, raw=None)
if result.source == "warped":
    self.filled_count += 1
self.emit(result.value)
```
With `filler=None` the warped branch is always `None` ⇒ behavior is byte-identical to today
(existing clock tests keep passing).

## Error handling / degradation
- **RAFT load failure** → fill disabled, log once, fall back to repeats (today's behavior).
- **Flow NaN/failure** → velocity 0 → warp ≈ identity (≈ held).
- **Extrapolation past `max_extrap_ms`** (AI stalled) → `fill` returns `None` → clock holds last
  frame.
- **Excessive motion** → `warp_forward`'s `max_disp` clamps per-pixel displacement so a long gap
  can't smear the whole frame.

## Testing
- `tests/test_warp.py` — synthetic image + known constant flow ⇒ output shifted by the exact expected
  pixels; identity flow ⇒ unchanged. CPU tensors, no GPU/model.
- `tests/test_filler.py` — `set_anchor` then `fill` within cap returns a warped tensor; beyond cap
  returns `None`; concurrent set/fill is safe (uses a stub warp to stay GPU-free).
- `tests/test_clock_fill.py` — `run_for_ticks` with a stub filler: fresh tick ⇒ emits `ai`; stale
  tick ⇒ emits `warped` and increments `filled_count`; stub returning `None` ⇒ emits `held`.
- RAFT smoke test (GPU, marked slow/skip-if-no-cuda) — two synthetically shifted frames ⇒ flow whose
  mean vector matches the shift sign/scale.
- End-to-end: `scripts/live.py --fill warp --sink null --source file` prints `emitted / fresh_AI /
  filled / repeats / jitter`.

## Success criteria
- At 30 fps out with ~12–14 fps AI, the **majority of stale ticks are filled** (warped), not repeated
  → visibly smooth motion (low `repeats`, high `filled`).
- Output jitter no worse than the current pipeline (Spout readback already ~31 ms; warp adds ~1–2 ms
  — measured via `live.py`, not assumed).
- Graceful hold on AI stall / excessive motion; full test suite green.

## Out of scope (v1)
- Dedicated **fill thread** (pre-render next warped frame) — fast-follow only if clock-thread warp
  regresses jitter.
- GPU motion vectors from render feeds (`Source.Capabilities.has_motion_vectors`) — a future faster
  path than RAFT for sources that expose MVs.
- True RIFE interpolation (rejected: adds latency, weights licensing).
- Making `--fill warp` the default (flip after it's proven on a real show).

# Aspect-Safe Validation Console - Design Spec

**Date:** 2026-06-12
**Status:** Approved UI direction (ready for implementation planning after user review)

## Goal

Fix aspect-ratio distortion in StreamForge's validation and live paths, then add a simplified web
operator console that validates NDI, Spout, or webcam input and shows live input/output previews with
runtime health.

The UI direction is the **Operator Console**: one dense screen where an operator can select a source,
confirm the input signal, see the aspect policy, inspect the live output, and watch freshness/jitter
metrics without stepping through a wizard.

## Current Findings

### Aspect distortion is confirmed

The test clip is 640x480 (4:3), but several saved validation artifacts in `out/clip/` and
`out/src_frame.png` are 512x512. The root cause is explicit square resizing:

- `scripts/restyle_clip.py` decodes the clip frame, converts it to a tensor, then calls
  `torch.nn.functional.interpolate(t, size=(args.res, args.res))`.
- That turns 4:3 source content into 1:1 before the model sees it, so faces and scene geometry are
  stretched in the saved source artifact as well as the output artifact.

The live file-source path preserves source dimensions initially, but the low-internal-resolution
runtime path can still distort:

- `EagerRuntime.restyle()` resizes directly from source shape to fixed `internal_hw`.
- `--in-res 384x224` is a 16:9 internal canvas. It is safe for 16:9 sources, but it distorts 4:3
  input unless the source is first fit/cropped or the internal size is derived from the source ratio.

### Output metadata is underspecified

`SpoutSink` sends the actual tensor width and height to `sendImage()`. `NDISink` currently assigns
`VideoFrameV2.data` and `FourCC`, but does not set explicit `xres`, `yres`, line stride, frame
format, or picture-aspect metadata before sending. That leaves too much behavior to the receiver.

### Input side is not first-class yet

The repo has a clean `Source` abstraction and concrete `SyntheticSource` / `FileSource`, but NDI,
Spout, and webcam are receiver scripts or absent as reusable source classes. The web UI should not
shell out to scripts; it should use source classes through the same pipeline boundary as file input.

### Control path drift

`TwoAxisControl` documents text magnitude as prompt-embedding interpolation/scaling for distilled
FLUX.2, but `EngineParams` only carries `guidance`, and `EagerRuntime` does not consume `guidance`.
This means the current preset UI can expose "style/text strength", but it should be labeled honestly
until prompt-embedding scaling is implemented.

## Aspect Policy

StreamForge should never stretch image content.

The default policy is **preserve source aspect**. If a target canvas has a different aspect ratio,
the system uses **fill and crop**:

- If the source is too tall for the target ratio, scale to fill width and crop top/bottom.
- If the source is too wide for the target ratio, scale to fill height and crop left/right.
- Crop is center-weighted by default, with future room for a vertical bias control if operators need
  to protect faces or stage content.

This policy applies at every boundary where shape changes:

- validation renders and still-image gates,
- runtime downscale to internal diffusion size,
- runtime upscale back to output size,
- web preview thumbnails,
- sink output when an explicit output canvas is configured.

### Internal diffusion size

For performance, StreamForge still supports low internal diffusion resolution. The internal size
must be derived from either:

1. the source frame aspect ratio, or
2. an explicit operator-selected output canvas aspect ratio.

The dimensions are then snapped to the model's required multiple. A 4:3 source should use a 4:3
internal canvas such as 384x288 or 320x240, not 384x224, unless the operator intentionally chooses a
16:9 output canvas and accepts fill/crop.

## Proposed Architecture

### Core media utilities

Create a small, pure module for aspect and frame fitting. This should be independent of the diffusion
runtime and UI so tests can cover it without GPU/model dependencies.

Responsibilities:

- describe source and target sizes,
- compute preserve/fill-crop resize plans,
- apply the plan to Torch tensors,
- return crop metadata for UI display and logs.

The key invariant is: input content is never non-uniformly scaled. Width and height use the same
scale factor, followed by crop if needed.

### Source classes

Add reusable input sources under `src/streamforge/sources/`:

| Source | Purpose | Notes |
|---|---|---|
| `WebcamSource` | Local camera input via OpenCV `VideoCapture(index)` | Fastest MVP validation path. |
| `NDISource` | Receive an NDI stream by name or first available source | Reuse logic from `scripts/ndi_receiver.py`. |
| `SpoutSource` | Receive a Spout sender by name | Reuse logic from `scripts/spout_receiver.py`; Windows-only. |

Each source should report:

- width, height, fps when known,
- last frame age,
- availability/error state,
- capabilities (`has_motion_vectors=False`, `has_depth=False` for v1).

### Pipeline runner service

Extract the `scripts/live.py` orchestration into a reusable service object that can be driven by both
CLI and web API.

Responsibilities:

- hold current configuration,
- start/stop source, runtime, worker, clock, and sink,
- publish status snapshots,
- expose latest input/output preview frames,
- keep metric counters for fresh AI frames, repeats, filled frames, jitter, inference latency, and
  source health.

The CLI should become a thin wrapper around this service rather than a second orchestration path.

### Web API

Use a small FastAPI backend mounted under `src/streamforge/web/`.

Endpoints:

- `GET /api/sources` lists available webcams, NDI sources, and Spout senders.
- `POST /api/validate` opens a selected source briefly and returns dimensions, fps, first-frame
  status, aspect-ratio analysis, and proposed internal/output sizes.
- `POST /api/run/start` starts the pipeline with a selected source, prompt, preset, mode, sink, fps,
  internal size mode, and aspect policy.
- `POST /api/run/stop` stops the current run.
- `GET /api/status` returns source health, frame counters, latency stats, output jitter, aspect
  policy, and current warnings.
- `GET /preview/input.jpg` and `GET /preview/output.jpg` return latest preview JPEGs.
- Optional after MVP: WebSocket `/ws/status` pushes status and frame sequence numbers.

For v1, polling at 2-5 Hz is enough for status. Preview images can refresh independently at a capped
rate, such as 5-10 fps, so the browser does not contend heavily with inference.

### Operator Console UI

The first screen is the actual tool, not a landing page.

Layout:

- Left rail: input type tabs (`NDI`, `Spout`, `Webcam`), source selector, validate button, aspect
  policy selector, output canvas selector, prompt/preset controls, start/stop.
- Center: side-by-side input and output previews with fixed aspect boxes. Each preview should show
  badges for source dimensions, canvas dimensions, crop direction, and frame age.
- Right rail: run state, source health, inference p50/p95, output fps, fresh AI fps, repeats,
  filled frames, jitter, VRAM if available, and warnings.

Visual behavior:

- Show crop overlays when fill/crop is active.
- Show a clear warning if the selected internal size would distort content; the backend should reject
  that configuration unless the user chooses an explicit crop target.
- Do not expose unsupported controls as if they work. Text/style strength should be disabled or
  labeled experimental until prompt-embedding scaling is implemented.

## Data Flow

```text
Input source
  -> frame validation
  -> aspect plan
  -> latest input preview
  -> InferenceWorker + EagerRuntime
  -> optional FrameFiller
  -> RealtimeClock
  -> sink output
  -> latest output preview + status metrics
```

The web UI reads the same status and preview buffers the runner service owns. It should not create a
second source reader or duplicate the model pipeline.

## Error Handling

- Missing webcam, NDI source, or Spout sender: validation returns a structured error and the UI keeps
  Start disabled.
- Source disappears while running: stop the worker, keep the last output preview, set run state to
  degraded/stopped, and report source age/error.
- Unsupported platform for Spout: hide or disable Spout input with a platform warning.
- Invalid target/internal dimensions: reject before runtime initialization and explain the nearest
  valid snapped size.
- Model/runtime initialization failure: surface the exception summary in status; keep UI responsive.
- Browser preview encoding failure: leave pipeline running, mark preview stale, and log the error.

## Testing Strategy

### Pure tests

- Aspect plan tests for 4:3 -> 1:1, 4:3 -> 16:9, 16:9 -> 4:3, same-ratio, and model-snap cases.
- Tensor fit/crop tests using synthetic coordinate grids so distortion is detectable.
- Source metadata tests for `WebcamSource` using a fake `VideoCapture`.
- NDI/Spout source tests with fake receiver objects so dimensions and color conversion are covered
  without requiring external runtimes.
- Runner service state tests for validate/start/stop/status without loading model weights.

### Integration tests

- FastAPI tests with fake sources and fake runtime: validate source, start run, poll status, fetch
  preview JPEG, stop run.
- CLI regression test: `scripts/live.py --source file --sink null --seconds 1` still routes through
  the runner service.
- Aspect artifact test: run the still-frame validation path on the 640x480 clip and assert output
  preview dimensions/crop metadata match the selected policy, with no forced 512x512 stretch.

### Manual validation

- Webcam input: validate dimensions, start preview, confirm input/output boxes show correct ratio.
- NDI input: run a known test sender, validate source discovery, preview, and status.
- Spout input: run a known Spout sender, validate source discovery, preview, and status.
- Output sinks: verify NDI/Spout receivers see the expected dimensions and do not stretch.

## Success Criteria

- No code path performs non-uniform resize on image content.
- A 640x480 source no longer produces stretched square source artifacts.
- An operator can validate NDI, Spout, or webcam input from the web UI before starting inference.
- The UI shows live input and output previews plus source/run health.
- The backend rejects aspect-invalid internal/output configurations instead of silently stretching.
- Existing pure test suite stays green, with new tests covering aspect policy and API behavior.

## Out of Scope For This Slice

- Full authentication or remote multi-user control.
- Zero-copy Spout/NDI preview transport.
- Production packaging as a Windows service.
- New model optimization work beyond preserving the existing fast path.
- Implementing the missing prompt-embedding text magnitude axis, except for honest UI labeling.


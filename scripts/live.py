"""StreamForge live runner (Phase 6) — source -> runtime -> ColorPipeline -> sink,
with the AI cadence decoupled from the output clock.

Test-pattern mode bypasses the AI and emits a standard chart through the full color+sink path,
so the media-server operator can verify the signal path/color/range BEFORE the model runs
(design §8.6).

Examples:
  # 1) verify the pipe in Resolume first (no AI): SMPTE-ish bars at 30fps for 20s
  python scripts/live.py --sink spout --test-pattern --fps 30 --seconds 20

  # 2) live restyle of the test clip into Resolume (img2img = fast path)
  python scripts/live.py --sink spout --source file --clip D:\StreamForge\TestFile\DriveVideo.mp4 \
      --mode img2img --preset BALANCED --prompt "vivid oil painting, thick impasto" --fps 30 --seconds 60
"""
from __future__ import annotations

import argparse
import time

import torch

from streamforge.color import ColorPipeline, make_test_pattern
from streamforge.frame import GpuFrame
from streamforge.metrics import jitter_ms


def build_sink(name: str, flip: bool):
    if name == "null":
        from streamforge.sinks.null_sink import NullSink
        return NullSink()
    if name == "spout":
        from streamforge.sinks.spout_sink import SpoutSink
        return SpoutSink(flip=flip)
    if name == "ndi":
        from streamforge.sinks.ndi_sink import NDISink
        return NDISink()
    raise SystemExit(f"unknown sink {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--sink", choices=["null", "spout", "ndi"], default="spout")
    ap.add_argument("--color", choices=["full", "legal", "off"], default="full")
    ap.add_argument("--no-flip", action="store_true", help="disable Spout vertical flip")
    ap.add_argument("--test-pattern", action="store_true")
    ap.add_argument("--source", choices=["synthetic", "file"], default="file")
    ap.add_argument("--clip", default=r"D:\StreamForge\TestFile\DriveVideo.mp4")
    ap.add_argument("--mode", choices=["edit", "img2img"], default="img2img")
    ap.add_argument("--preset", default="BALANCED")
    ap.add_argument("--prompt", default="vivid oil painting, thick impasto brushstrokes")
    ap.add_argument("--in-res", default="off",
                    help="internal diffusion res WxH (e.g. 384x224, 512x288) — AI runs low-res, "
                         "output upscaled to source res. 'off' = native. Track-A speed lever.")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the transformer (stable at fixed --in-res; ~1.25x)")
    ap.add_argument("--tiny-vae", action="store_true",
                    help="use taef2 tiny VAE (~1.28x; transformer-bound after this)")
    ap.add_argument("--fill", choices=["off", "warp"], default="off",
                    help="output frame-fill: 'warp' = RAFT motion-extrapolation of stale frames")
    ap.add_argument("--max-extrap-ms", type=float, default=120.0,
                    help="cap how far past the last AI frame to extrapolate (ms)")
    ap.add_argument("--flow-max-side", type=int, default=384,
                    help="flow-res longest side for RAFT (smaller = faster)")
    args = ap.parse_args()

    # --- test-pattern mode: no AI, just color + sink (design §8.6) -------------------------
    if args.test_pattern:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        color = None if args.color == "off" else ColorPipeline(range_mode=args.color)
        sink = build_sink(args.sink, flip=not args.no_flip)
        sink.open()
        timestamps: list[float] = []

        def emit(frame) -> None:
            if frame is not None and color is not None:
                frame = frame.with_tensor(color.apply(frame.tensor))
            sink.send(frame)
            timestamps.append(time.perf_counter())

        patt = make_test_pattern(args.res, args.res).to(dev)
        gf = GpuFrame(tensor=patt, seq=0, pts=0.0, width=args.res, height=args.res)
        print(f"TEST PATTERN -> sink={args.sink} color={args.color} flip={not args.no_flip} "
              f"@ {args.fps}fps for {args.seconds}s. Check Resolume for SMPTE-ish bars + ramp.")
        period = 1.0 / args.fps
        end = time.perf_counter() + args.seconds
        nxt = time.perf_counter()
        while time.perf_counter() < end:
            emit(gf)
            nxt += period
            sl = nxt - time.perf_counter()
            if sl > 0:
                time.sleep(sl)
        sink.close()
        print(f"done: emitted={len(timestamps)} jitter={jitter_ms(timestamps):.2f}ms")
        return

    # --- live restyle: source -> worker(runtime) -> framebuffer -> clock -> color -> sink --
    from streamforge.runner import RunnerConfig, StreamForgeRunner

    source_name = args.clip if args.source == "file" else str(args.res)
    config = RunnerConfig(
        source_type=args.source,
        source_name=source_name,
        sink=args.sink,
        fps=args.fps,
        prompt=args.prompt,
        mode=args.mode,
        preset=args.preset,
        in_res=args.in_res,
        color=args.color,
        fill=args.fill,
        max_extrap_ms=args.max_extrap_ms,
        flow_max_side=args.flow_max_side,
        compile_transformer=args.compile,
        tiny_vae=args.tiny_vae,
        spout_flip=not args.no_flip,
    )
    runner = StreamForgeRunner()

    print(f"LIVE: mode={args.mode} preset={args.preset} in_res={args.in_res} compile={args.compile} "
          f"-> sink={args.sink} @ {args.fps}fps for {args.seconds}s. "
          f"Watch the receiver for the restyled '{args.prompt}' layer.")
    runner.start(config)
    try:
        time.sleep(args.seconds)
        status = runner.status()
    finally:
        runner.stop()
    emitted = int(status["emitted"])
    repeats = int(status["repeats"])
    filled = int(status["filled"])
    fresh = int(status["fresh_ai"])
    print(f"done: emitted={emitted} jitter={status['jitter_ms']:.2f}ms "
          f"repeats={repeats} filled={filled} fresh_AI~={fresh} "
          f"(AI~={fresh/args.seconds:.1f}fps)")


if __name__ == "__main__":
    main()

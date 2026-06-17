"""Temporary NDI test sender — broadcasts a moving test pattern so StreamForge's
NDI source has something to receive. Pure NDIlib (no GL context needed).

Run in a separate terminal, then in the console pick Input Type = NDI and set the
Source to match the name below (substring match is fine):

    .venv\\Scripts\\python scripts\\ndi_test_sender.py
    .venv\\Scripts\\python scripts\\ndi_test_sender.py --name StreamForge-Test --width 640 --height 480

Ctrl+C to stop. --frames N sends N frames then exits (used for smoke tests).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import NDIlib as ndi


def make_frame(w: int, h: int, t: float) -> np.ndarray:
    """An animated RGBA gradient with a bouncing white box (clear motion to restyle)."""
    img = np.zeros((h, w, 4), dtype=np.uint8)
    xs = (np.linspace(0, 255, w)[None, :] + t * 60) % 256
    ys = (np.linspace(0, 255, h)[:, None] + t * 30) % 256
    img[..., 0] = xs
    img[..., 1] = ys
    img[..., 2] = (xs + ys) % 256
    img[..., 3] = 255
    bw, bh = min(80, w // 4), min(80, h // 4)
    bx = int((w - bw) * (0.5 + 0.5 * np.sin(t * 1.7)))
    by = int((h - bh) * (0.5 + 0.5 * np.cos(t * 1.3)))
    img[by:by + bh, bx:bx + bw, :3] = 255
    return np.ascontiguousarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="StreamForge-Test")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=0, help="0 = run until Ctrl+C")
    a = ap.parse_args()

    if not ndi.initialize():
        raise SystemExit("NDI runtime failed to initialize")
    sc = ndi.SendCreate()
    sc.ndi_name = a.name
    sender = ndi.send_create(sc)
    if sender is None:
        ndi.destroy()
        raise SystemExit("ndi.send_create failed")

    vf = ndi.VideoFrameV2()
    vf.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBA
    print(f"NDI sending '{a.name}' {a.width}x{a.height} @ {a.fps}fps — Ctrl+C to stop", flush=True)
    dt = 1.0 / a.fps
    t0 = time.time()
    sent = 0
    try:
        while True:
            t = time.time() - t0
            vf.data = make_frame(a.width, a.height, t)
            ndi.send_send_video_v2(sender, vf)
            sent += 1
            if a.frames and sent >= a.frames:
                break
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"NDI sender stopping after {sent} frames", flush=True)
        ndi.send_destroy(sender)
        ndi.destroy()


if __name__ == "__main__":
    main()

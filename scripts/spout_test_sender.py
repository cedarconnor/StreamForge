"""Temporary Spout test sender — shares a moving test pattern as a Spout sender so
StreamForge's Spout source has something to receive. Needs a GL context (glfw).

Run in a separate terminal, then in the console pick Input Type = Spout and set the
Source to the sender name below (exact match):

    .venv\\Scripts\\python scripts\\spout_test_sender.py
    .venv\\Scripts\\python scripts\\spout_test_sender.py --name StreamForge --width 640 --height 480

Ctrl+C to stop. --frames N sends N frames then exits (used for smoke tests).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import glfw
import SpoutGL

GL_RGBA = 0x1908


def make_frame(w: int, h: int, t: float) -> np.ndarray:
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
    ap.add_argument("--name", default="StreamForge")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=0, help="0 = run until Ctrl+C")
    a = ap.parse_args()

    if not glfw.init():
        raise SystemExit("glfw init failed")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # hidden offscreen context
    win = glfw.create_window(64, 64, "spout-test-sender", None, None)
    if not win:
        glfw.terminate()
        raise SystemExit("glfw could not create a GL context/window")
    glfw.make_context_current(win)

    sender = SpoutGL.SpoutSender()
    sender.setSenderName(a.name)
    print(f"Spout sending '{a.name}' {a.width}x{a.height} @ {a.fps}fps — Ctrl+C to stop", flush=True)
    dt = 1.0 / a.fps
    t0 = time.time()
    sent = 0
    try:
        while not glfw.window_should_close(win):
            t = time.time() - t0
            img = make_frame(a.width, a.height, t)
            sender.sendImage(img, a.width, a.height, GL_RGBA, False, 0)
            sender.setFrameSync(a.name)
            glfw.poll_events()
            sent += 1
            if a.frames and sent >= a.frames:
                break
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Spout sender stopping after {sent} frames", flush=True)
        sender.releaseSender()
        glfw.terminate()


if __name__ == "__main__":
    main()

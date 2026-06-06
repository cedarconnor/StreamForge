"""Spout receiver — a zero-install way to view/verify StreamForge's Spout output (no Resolume).

Modes:
  python scripts/spout_receiver.py --list           # list active Spout senders
  python scripts/spout_receiver.py                  # live window (press q/ESC to quit)
  python scripts/spout_receiver.py --save out/rx.png  # grab one frame, save, exit (headless verify)

Run the sender in another terminal (e.g. scripts/live.py --sink spout ...), then run this.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import SpoutGL

GL_RGBA = 0x1908


def _grab(receiver, buffer):
    """One receive attempt -> (buffer, frame_or_None). Reallocates buffer on size change."""
    result = receiver.receiveImage(buffer, GL_RGBA, True, 0)  # bInvert=True undoes the sender flip
    if receiver.isUpdated():
        w, h = receiver.getSenderWidth(), receiver.getSenderHeight()
        return bytearray(w * h * 4), None
    if result and buffer is not None:
        w, h = receiver.getSenderWidth(), receiver.getSenderHeight()
        if w * h * 4 == len(buffer):
            rgba = np.frombuffer(bytes(buffer), np.uint8).reshape((h, w, 4))
            if rgba[..., :3].any():   # skip the empty buffers that arrive before real pixels
                return buffer, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    return buffer, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="StreamForge")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--save", default=None, help="grab one frame to this PNG and exit")
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()

    receiver = SpoutGL.SpoutReceiver()
    receiver.createOpenGL()

    if args.list:
        time.sleep(0.5)
        print("Active Spout senders:", receiver.getSenderList())
        receiver.closeOpenGL()
        return

    receiver.setReceiverName(args.name)
    buffer = None
    deadline = time.time() + args.timeout

    if args.save:
        while time.time() < deadline:
            buffer, frame = _grab(receiver, buffer)
            if frame is not None:
                cv2.imwrite(args.save, frame)
                print(f"received {frame.shape[1]}x{frame.shape[0]} from '{args.name}' -> {args.save}")
                receiver.releaseReceiver(); receiver.closeOpenGL()
                return
            time.sleep(1 / 60)
        print(f"TIMEOUT: no frames from sender '{args.name}'. Is the sender running? Try --list.")
        receiver.closeOpenGL()
        return

    print(f"Receiving '{args.name}' — press q or ESC to quit.")
    while True:
        buffer, frame = _grab(receiver, buffer)
        if frame is not None:
            cv2.imshow("StreamForge Spout receiver", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    receiver.releaseReceiver(); receiver.closeOpenGL()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

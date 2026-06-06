"""NDI receiver — a zero-install way to view/verify StreamForge's NDI output.

Modes:
  python scripts/ndi_receiver.py --list             # list NDI sources on the network
  python scripts/ndi_receiver.py                    # live window (press q/ESC to quit)
  python scripts/ndi_receiver.py --save out/rx.png  # grab one frame, save, exit (headless verify)

(NDI Studio Monitor from the free "NDI Tools" is a GUI alternative.)
Run the sender first: scripts/live.py --sink ndi ...
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import NDIlib as ndi


def _find(finder, name, timeout):
    deadline = time.time() + timeout
    seen = []
    while time.time() < deadline:
        ndi.find_wait_for_sources(finder, 1000)
        seen = ndi.find_get_current_sources(finder)
        for s in seen:
            if name.lower() in s.ndi_name.lower():
                return s, seen
    return None, seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="StreamForge")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--save", default=None)
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()

    if not ndi.initialize():
        raise SystemExit("NDI runtime failed to initialize")
    finder = ndi.find_create_v2()

    if args.list:
        ndi.find_wait_for_sources(finder, 2000)
        srcs = ndi.find_get_current_sources(finder)
        print("NDI sources:", [s.ndi_name for s in srcs])
        ndi.find_destroy(finder); ndi.destroy()
        return

    src, seen = _find(finder, args.name, args.timeout)
    if src is None:
        print(f"source matching '{args.name}' not found. seen: {[s.ndi_name for s in seen]}")
        ndi.find_destroy(finder); ndi.destroy()
        return

    rc = ndi.RecvCreateV3()
    rc.source_to_connect_to = src
    rc.color_format = ndi.RECV_COLOR_FORMAT_RGBX_RGBA
    recv = ndi.recv_create_v3(rc)

    deadline = time.time() + args.timeout
    print(f"connected to '{src.ndi_name}'" + ("" if args.save else " — press q/ESC to quit"))
    try:
        while time.time() < deadline if args.save else True:
            tpe, v, a, m = ndi.recv_capture_v2(recv, 1000)
            if tpe == ndi.FRAME_TYPE_VIDEO:
                bgr = cv2.cvtColor(np.copy(v.data), cv2.COLOR_RGBA2BGR)
                ndi.recv_free_video_v2(recv, v)
                if args.save:
                    cv2.imwrite(args.save, bgr)
                    print(f"received {bgr.shape[1]}x{bgr.shape[0]} from '{src.ndi_name}' -> {args.save}")
                    break
                cv2.imshow("StreamForge NDI receiver", bgr)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            elif tpe == ndi.FRAME_TYPE_AUDIO:
                ndi.recv_free_audio_v2(recv, a)
        else:
            print(f"TIMEOUT: no video from '{args.name}'. Is the sender running?")
    finally:
        ndi.recv_destroy(recv)
        ndi.find_destroy(finder)
        ndi.destroy()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

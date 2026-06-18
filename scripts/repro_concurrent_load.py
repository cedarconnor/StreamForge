"""Verify the SANA double-Start crash is fixed: two SanaStreamingRuntime.load_once() calls racing
concurrently (what happens when Start is clicked again before the first ~30s load finishes and
runner.stop()'s 5s join times out, leaving two worker threads loading at once).

Pre-fix this raced in accelerate/transformers shared init -> Accelerator device=None OR
"Cannot copy out of meta tensor" in the text encoder. With the whole-load lock the two loads
serialize and both succeed. Loads two full stacks (~2x VRAM) briefly. Run under .venv-sana.
"""
from __future__ import annotations

import threading

from streamforge.diffusion.runtime_sana_streaming import SanaStreamingRuntime

results: dict = {}


def load(tag, barrier):
    barrier.wait()  # release both threads together to force maximal overlap
    try:
        rt = SanaStreamingRuntime(internal_hw=(288, 384), default_steps=2, resync_every=0)
        rt.load_once()
        rt.set_prompt("vivid oil painting, thick impasto brushstrokes")
        results[tag] = ("ok", str(rt.device), rt._loaded)
    except Exception as e:
        results[tag] = ("ERR", type(e).__name__ + ": " + str(e), False)


def main():
    barrier = threading.Barrier(2)
    ts = [threading.Thread(target=load, args=(f"t{i}", barrier)) for i in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for k, v in sorted(results.items()):
        print(f"  {k}: {v}")
    ok = len(results) == 2 and all(v[0] == "ok" and v[1] == "cuda" for v in results.values())
    print("PASS — both concurrent loads succeeded" if ok else "FAIL — a concurrent load crashed")


if __name__ == "__main__":
    main()

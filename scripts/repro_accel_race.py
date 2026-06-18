"""Reproduce the SANA Start crash: AttributeError 'NoneType' has no attribute 'type' in
Accelerator.__init__ (accelerator.py:589) when two threads construct Accelerator concurrently
on the FIRST init (the worker-thread race when Start is clicked twice before the first ~30s
model load finishes). Run under .venv-sana.
"""
from __future__ import annotations

import threading

from accelerate import Accelerator
from accelerate.state import AcceleratorState, PartialState


def _reset():
    for cls in (AcceleratorState, PartialState):
        try:
            cls._reset_state()
        except Exception:
            cls._shared_state.clear()


_INIT_LOCK = threading.Lock()


def attempt(barrier, results, tag, use_lock):
    barrier.wait()  # release both threads at once to maximize overlap on the first init
    try:
        if use_lock:
            with _INIT_LOCK:
                a = Accelerator(mixed_precision="bf16")
                d = a.device
        else:
            a = Accelerator(mixed_precision="bf16")
            d = a.device
        results[tag] = ("ok", str(d), d is None)
    except Exception as e:
        results[tag] = ("ERR", type(e).__name__ + ": " + str(e), True)


def _run(use_lock, N=40):
    fails = 0
    for i in range(N):
        _reset()
        results: dict = {}
        barrier = threading.Barrier(2)
        ts = [threading.Thread(target=attempt, args=(barrier, results, f"t{j}", use_lock))
              for j in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        bad = {k: v for k, v in results.items() if v[0] == "ERR" or v[2]}
        if bad:
            fails += 1
            print(f"  iter {i:2d}: RACE -> {results}")
    print(f"{fails}/{N} concurrent first-inits hit device=None / error "
          f"({'WITH lock' if use_lock else 'NO lock'})")


def main():
    print("--- NO lock (reproduce the bug) ---")
    _run(use_lock=False)
    print("--- WITH lock (the fix) ---")
    _run(use_lock=True)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

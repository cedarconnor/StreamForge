"""Task 2.4 / 3.x: run the benchmark harness against a runtime + source.

Eager baseline = the latency floor every later optimization is measured against.
"""
from __future__ import annotations

import argparse
import pathlib

from streamforge.bench.harness import BenchHarness
from streamforge.bench.report import to_table, to_json
from streamforge.sources.synthetic import SyntheticSource


def _build_runtime(name: str):
    if name == "eager":
        from streamforge.diffusion.runtime_eager import EagerRuntime
        return EagerRuntime()
    if name == "eager-compile":
        from streamforge.diffusion.runtime_eager import EagerRuntime
        return EagerRuntime(compile_transformer=True)
    if name in ("eager-8bit", "eager-4bit"):
        from streamforge.diffusion.runtime_eager import EagerRuntime
        return EagerRuntime(quant=name.split("-")[1])
    raise SystemExit(f"unknown runtime {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--runtime", default="eager")
    ap.add_argument("--prompt", default="vivid oil painting, thick impasto brushstrokes")
    args = ap.parse_args()

    src = SyntheticSource(args.res, args.res, args.fps)
    rt = _build_runtime(args.runtime)
    report = BenchHarness(src, rt, frames=args.frames, fps=args.fps, prompt=args.prompt).run()
    print(to_table(report))
    pathlib.Path("out").mkdir(exist_ok=True)
    out = pathlib.Path("out") / f"bench_{args.runtime}_{args.res}.json"
    out.write_text(to_json(report))
    print("wrote", out)


if __name__ == "__main__":
    main()

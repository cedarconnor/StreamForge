"""Run the benchmark harness against a runtime built from orthogonal flags.

Examples:
  python scripts/run_bench.py --mode edit                      # eager baseline
  python scripts/run_bench.py --mode img2img                   # classic img2img (faster)
  python scripts/run_bench.py --mode edit --quant int8ao --compile   # the INT8+TorchAO+compile path
  python scripts/run_bench.py --mode img2img --quant int8ao --compile
"""
from __future__ import annotations

import argparse
import pathlib

from streamforge.bench.harness import BenchHarness
from streamforge.bench.report import to_table, to_json
from streamforge.sources.synthetic import SyntheticSource


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--mode", choices=["edit", "img2img"], default="edit")
    ap.add_argument("--quant", choices=["none", "8bit", "4bit", "int8ao"], default="none")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--prompt", default="vivid oil painting, thick impasto brushstrokes")
    args = ap.parse_args()

    from streamforge.diffusion.runtime_eager import EagerRuntime
    rt = EagerRuntime(
        mode=args.mode,
        quant=None if args.quant == "none" else args.quant,
        compile_transformer=args.compile,
    )

    src = SyntheticSource(args.res, args.res, args.fps)
    report = BenchHarness(src, rt, frames=args.frames, fps=args.fps, prompt=args.prompt).run()
    print(f"[mode={args.mode} quant={args.quant} compile={args.compile} res={args.res}]")
    print(to_table(report))
    pathlib.Path("out").mkdir(exist_ok=True)
    tag = f"{args.mode}_{args.quant}{'_compile' if args.compile else ''}_{args.res}"
    out = pathlib.Path("out") / f"bench_{tag}.json"
    out.write_text(to_json(report))
    print("wrote", out)


if __name__ == "__main__":
    main()

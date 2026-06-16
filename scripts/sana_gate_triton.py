"""Phase-0 Gate 1: prove triton-windows compiles + runs real kernels on the A6000.

`import triton` succeeding is NECESSARY but NOT SUFFICIENT for SANA's own fused GDN
(that is Gate 3). This script exercises three escalating signals:
  1. a trivial hand-written @triton.jit kernel  (does the JIT + C launcher compile?)
  2. an @triton.autotune kernel                 (does autotuning work? closest to fused GDN)
  3. torch.compile lowering to Triton           (the path the existing FLUX engine uses)
"""
from __future__ import annotations

import time

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def _scale_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2.0, mask=mask)


def main() -> None:
    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda:0")
    cap = torch.cuda.get_device_capability(dev)
    print(f"GPU: {torch.cuda.get_device_name(dev)}  sm_{cap[0]}{cap[1]}  triton {triton.__version__}")

    n = 1 << 20
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)

    # 1) trivial JIT kernel
    out = torch.empty_like(x)
    t0 = time.perf_counter()
    _add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    torch.cuda.synchronize()
    assert torch.allclose(out, x + y, atol=1e-5), "add kernel mismatch"
    print(f"  [1/3] @triton.jit add kernel        OK  (first-compile {time.perf_counter()-t0:.2f}s)")

    # 2) autotuned kernel (exercises the autotune machinery fused GDN relies on)
    out2 = torch.empty_like(x)
    t0 = time.perf_counter()
    _scale_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](x, out2, n)
    torch.cuda.synchronize()
    assert torch.allclose(out2, x * 2.0, atol=1e-5), "scale kernel mismatch"
    print(f"  [2/3] @triton.autotune kernel       OK  (first-compile {time.perf_counter()-t0:.2f}s)")

    # 3) torch.compile -> Triton (the FLUX reduce-overhead / int8ao path)
    @torch.compile(mode="reduce-overhead")
    def f(a, b):
        return torch.sin(a) * torch.cos(b) + a * b

    t0 = time.perf_counter()
    r = f(x, y)
    torch.cuda.synchronize()
    ref = torch.sin(x) * torch.cos(y) + x * y
    assert torch.allclose(r, ref, atol=1e-3), "torch.compile result mismatch"
    print(f"  [3/3] torch.compile -> Triton       OK  (first-compile {time.perf_counter()-t0:.2f}s)")

    print("PASS: triton-windows compiles + runs (jit, autotune, torch.compile) on the A6000")


if __name__ == "__main__":
    main()

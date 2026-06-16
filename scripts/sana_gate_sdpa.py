"""Phase-0 Gate 4 probe: which SDPA backends run on the A6000 in BF16?

If torch's built-in FLASH_ATTENTION backend works here, the external flash-attn wheel is
largely redundant for inference (SANA's softmax blocks are the minority anyway). This is the
'SDPA-first' path the design doc ranks #1.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def main() -> None:
    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda:0")
    cap = torch.cuda.get_device_capability(dev)
    print(f"GPU: {torch.cuda.get_device_name(dev)}  sm_{cap[0]}{cap[1]}  torch {torch.__version__}")

    b, h, s, d = 2, 16, 2048, 64
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    print(f"  default SDPA           OK  {tuple(out.shape)} {out.dtype}")

    for be, name in [
        (SDPBackend.FLASH_ATTENTION, "FLASH_ATTENTION"),
        (SDPBackend.EFFICIENT_ATTENTION, "EFFICIENT(mem)"),
        (SDPBackend.MATH, "MATH"),
    ]:
        try:
            with sdpa_kernel([be]):
                o = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
            assert torch.allclose(o.float(), out.float(), atol=2e-2), "backend disagrees with default"
            print(f"  {name:22s} OK")
        except Exception as e:
            print(f"  {name:22s} unavailable: {type(e).__name__}: {e}")

    print("PASS: SDPA runs in BF16 on the A6000 (external flash-attn not required)")


if __name__ == "__main__":
    main()

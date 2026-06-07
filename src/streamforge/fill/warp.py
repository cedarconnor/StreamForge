"""Forward-warp an image along an optical-flow field via grid_sample (pure, testable).

disp = flow * fps * dt, where `fps` is the AI/anchor rate and `flow` is the full inter-anchor
displacement (pixels at flow-res). Content moves by +disp, so we backward-sample by -disp.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def warp_forward(image: torch.Tensor, flow: torch.Tensor, dt: float, fps: float,
                 max_disp: float | None = None) -> torch.Tensor:
    # Warp in fp32: the styled image is bf16 but RAFT flow is fp32; grid_sample needs matched
    # dtypes (and is unreliable in bf16). Cast back to the input dtype on the way out.
    in_dtype = image.dtype
    image = image.float()
    flow = flow.float()
    b, c, h, w = image.shape
    fh, fw = flow.shape[-2], flow.shape[-1]
    if (fh, fw) != (h, w):
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False).clone()
        flow[:, 0] *= w / fw   # x magnitude scales with width ratio
        flow[:, 1] *= h / fh   # y magnitude scales with height ratio
    disp = flow * (fps * dt)
    if max_disp is not None:
        disp = disp.clamp(-max_disp, max_disp)
    yy, xx = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype), indexing="ij")
    src_x = xx[None] - disp[:, 0]   # backward sample
    src_y = yy[None] - disp[:, 1]
    gx = 2.0 * src_x / max(w - 1, 1) - 1.0
    gy = 2.0 * src_y / max(h - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)   # [B,H,W,2]
    out = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.to(in_dtype)

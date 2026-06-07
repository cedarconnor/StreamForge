"""Optical flow via torchvision raft_small (BSD weights). Estimates flow between two source frames
at a low flow-res (longest side <= max_side, dims divisible by 8) to stay cheap; runs on the worker
thread at AI rate. Returns flow in flow-res pixels; warp_forward rescales to the styled image res."""
from __future__ import annotations
import torch
import torch.nn.functional as F


class RaftFlow:
    def __init__(self, device: str = "cuda", max_side: int = 384):
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        self.device = device
        self.max_side = max_side
        self.model = raft_small(weights=Raft_Small_Weights.C_T_V2, progress=False).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        scale = min(1.0, self.max_side / max(h, w))
        fh = max(8, int(round(h * scale / 8)) * 8)
        fw = max(8, int(round(w * scale / 8)) * 8)
        x = F.interpolate(x.to(self.device), size=(fh, fw), mode="bilinear", align_corners=False)
        return x * 2.0 - 1.0   # [0,1] -> [-1,1], RAFT convention

    @torch.no_grad()
    def estimate(self, prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        return self.model(self._prep(prev), self._prep(cur))[-1]   # [1,2,fh,fw]

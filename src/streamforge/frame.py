"""GpuFrame — a frame as a device-resident tensor plus show metadata."""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class GpuFrame:
    """A frame as a device-resident tensor (NCHW float in [0,1]) plus show metadata."""
    tensor: torch.Tensor
    seq: int
    pts: float
    width: int
    height: int
    colorspace: str = "srgb"

    def with_tensor(self, tensor: torch.Tensor) -> "GpuFrame":
        return replace(self, tensor=tensor)

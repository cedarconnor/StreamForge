"""ColorPipeline — the final color stage before the sink (design §8.6).

Large-format projection is unforgiving about gamma, range, and color space. Operates on
NCHW float in [0,1]. Includes a known test pattern that bypasses the AI so an operator can
verify the signal path, color, and range before the model is even running.
"""
from __future__ import annotations

import torch


class ColorPipeline:
    def __init__(self, range_mode: str = "full", tonemap: str = "clamp"):
        self.range_mode = range_mode
        self.tonemap = tonemap

    @staticmethod
    def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    @staticmethod
    def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, 1)
        return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)

    def _range(self, x: torch.Tensor) -> torch.Tensor:
        if self.range_mode == "legal":
            return x * (235 - 16) / 255.0 + 16 / 255.0
        return x

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.tonemap == "clamp":
            x = x.clamp(0, 1)
        return self._range(x).clamp(0, 1)


def make_test_pattern(width: int, height: int) -> torch.Tensor:
    """SMPTE-ish vertical color bars + a luma ramp row — for sink/color verification."""
    bars = torch.tensor(
        [[1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1]]
    ).float()
    img = torch.zeros(3, height, width)
    split = int(height * 0.8)
    seg = max(1, width // bars.shape[0])
    for i in range(bars.shape[0]):
        img[:, :split, i * seg:(i + 1) * seg] = bars[i].view(3, 1, 1)
    ramp = torch.linspace(0, 1, width).view(1, 1, width).expand(3, height - split, width)
    img[:, split:, :] = ramp
    return img[None]

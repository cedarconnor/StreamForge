"""Aspect-ratio planning and tensor fitting utilities.

All resize plans use one uniform scale factor. If source and target ratios differ, content is
cropped after scaling; it is never non-uniformly stretched.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F


class FitMode(str, Enum):
    FILL_CROP = "fill_crop"


@dataclass(frozen=True)
class Size:
    width: int
    height: int

    @property
    def aspect(self) -> float:
        return self.width / self.height


@dataclass(frozen=True)
class FitPlan:
    source: Size
    target: Size
    resized: Size
    crop_left: int
    crop_top: int
    crop_width: int
    crop_height: int
    scale: float
    crop_direction: str


def _validate_size(size: Size, label: str) -> None:
    if size.width <= 0 or size.height <= 0:
        raise ValueError(f"{label} width and height must be positive, got {size.width}x{size.height}")


def plan_fit(source: Size, target: Size, mode: FitMode = FitMode.FILL_CROP) -> FitPlan:
    _validate_size(source, "source")
    _validate_size(target, "target")
    if mode != FitMode.FILL_CROP:
        raise ValueError(f"unsupported fit mode: {mode}")
    scale = max(target.width / source.width, target.height / source.height)
    resized_w = max(target.width, int(round(source.width * scale)))
    resized_h = max(target.height, int(round(source.height * scale)))
    crop_left = max(0, (resized_w - target.width) // 2)
    crop_top = max(0, (resized_h - target.height) // 2)
    if resized_w > target.width:
        direction = "sides"
    elif resized_h > target.height:
        direction = "top_bottom"
    else:
        direction = "none"
    return FitPlan(
        source=source,
        target=target,
        resized=Size(resized_w, resized_h),
        crop_left=crop_left,
        crop_top=crop_top,
        crop_width=target.width,
        crop_height=target.height,
        scale=scale,
        crop_direction=direction,
    )


def fit_tensor(
    image_bchw: torch.Tensor,
    target: Size,
    mode: FitMode = FitMode.FILL_CROP,
    antialias: bool = True,
) -> tuple[torch.Tensor, FitPlan]:
    if image_bchw.ndim != 4:
        raise ValueError(f"expected BCHW tensor, got shape {tuple(image_bchw.shape)}")
    source = Size(width=int(image_bchw.shape[-1]), height=int(image_bchw.shape[-2]))
    plan = plan_fit(source, target, mode)
    resized = F.interpolate(
        image_bchw,
        size=(plan.resized.height, plan.resized.width),
        mode="bilinear",
        align_corners=False,
        antialias=antialias,
    )
    top = plan.crop_top
    left = plan.crop_left
    cropped = resized[:, :, top:top + plan.crop_height, left:left + plan.crop_width]
    return cropped, plan


def snap_to_multiple_for_aspect(source: Size, max_side: int, multiple: int = 16) -> Size:
    _validate_size(source, "source")
    if max_side < multiple:
        raise ValueError(f"max_side must be >= multiple, got max_side={max_side}, multiple={multiple}")
    if source.width >= source.height:
        width = (max_side // multiple) * multiple
        height = max(multiple, round((width / source.aspect) / multiple) * multiple)
    else:
        height = (max_side // multiple) * multiple
        width = max(multiple, round((height * source.aspect) / multiple) * multiple)
    return Size(width=int(width), height=int(height))

import torch

from streamforge.aspect import FitMode, Size, fit_tensor, plan_fit, snap_to_multiple_for_aspect


def test_fill_crop_4x3_into_16x9_crops_top_bottom():
    plan = plan_fit(Size(640, 480), Size(384, 216), FitMode.FILL_CROP)
    assert plan.resized == Size(384, 288)
    assert plan.crop_left == 0
    assert plan.crop_top == 36
    assert plan.crop_width == 384
    assert plan.crop_height == 216
    assert plan.crop_direction == "top_bottom"


def test_fill_crop_16x9_into_4x3_crops_sides():
    plan = plan_fit(Size(1920, 1080), Size(640, 480), FitMode.FILL_CROP)
    assert plan.resized == Size(853, 480)
    assert plan.crop_left == 106
    assert plan.crop_top == 0
    assert plan.crop_width == 640
    assert plan.crop_height == 480
    assert plan.crop_direction == "sides"


def test_same_ratio_has_no_crop():
    plan = plan_fit(Size(640, 480), Size(320, 240), FitMode.FILL_CROP)
    assert plan.resized == Size(320, 240)
    assert plan.crop_left == 0 and plan.crop_top == 0
    assert plan.crop_direction == "none"


def test_fit_tensor_preserves_geometry_by_cropping_not_stretching():
    img = torch.zeros(1, 1, 4, 4)
    img[:, :, 1:3, 1:3] = 1.0
    out, plan = fit_tensor(img, Size(4, 2), FitMode.FILL_CROP)
    assert out.shape == (1, 1, 2, 4)
    assert plan.crop_direction == "top_bottom"
    assert float(out.max()) == 1.0


def test_snap_to_multiple_for_aspect_keeps_4x3_ratio():
    size = snap_to_multiple_for_aspect(Size(640, 480), max_side=384, multiple=16)
    assert size == Size(384, 288)

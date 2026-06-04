import torch

from streamforge.color import ColorPipeline, make_test_pattern


def test_srgb_linear_roundtrip():
    cp = ColorPipeline()
    x = torch.rand(1, 3, 16, 16)
    back = cp.linear_to_srgb(cp.srgb_to_linear(x))
    assert torch.allclose(x, back, atol=1e-4)


def test_legal_range_compresses_black():
    cp = ColorPipeline(range_mode="legal")
    x = torch.zeros(1, 3, 4, 4)        # full black 0.0
    y = cp.apply(x)
    assert y.min() > 0.0               # 0 maps to 16/255


def test_full_range_passes_black_through():
    cp = ColorPipeline(range_mode="full")
    y = cp.apply(torch.zeros(1, 3, 4, 4))
    assert float(y.min()) == 0.0


def test_make_test_pattern_has_structure():
    p = make_test_pattern(64, 64)
    assert p.shape == (1, 3, 64, 64)
    assert p.std() > 0.05

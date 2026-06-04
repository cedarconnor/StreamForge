import torch

from streamforge.sources.synthetic import SyntheticSource


def test_synthetic_source_yields_moving_frames():
    src = SyntheticSource(width=64, height=64, fps=30)
    src.open()
    f0 = src.read()
    f1 = src.read()
    assert f0.width == 64 and f0.tensor.shape == (1, 3, 64, 64)
    assert f0.seq == 0 and f1.seq == 1
    assert not torch.equal(f0.tensor, f1.tensor)   # it moves
    src.close()


def test_synthetic_source_values_in_unit_range():
    src = SyntheticSource(width=16, height=16, fps=30)
    src.open()
    f = src.read()
    assert float(f.tensor.min()) >= 0.0 and float(f.tensor.max()) <= 1.0


def test_synthetic_source_reports_capabilities():
    src = SyntheticSource(width=8, height=8, fps=30)
    assert src.capabilities.has_motion_vectors is False

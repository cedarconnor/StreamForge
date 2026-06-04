import torch

from streamforge.frame import GpuFrame


def test_frame_carries_metadata():
    t = torch.zeros(1, 3, 64, 64)
    f = GpuFrame(tensor=t, seq=5, pts=0.16, width=64, height=64, colorspace="srgb")
    assert f.seq == 5 and f.width == 64 and f.colorspace == "srgb"


def test_frame_with_replaces_tensor_keeps_meta():
    f = GpuFrame(tensor=torch.zeros(1, 3, 8, 8), seq=1, pts=0.0, width=8, height=8, colorspace="srgb")
    t2 = torch.ones(1, 3, 8, 8)
    f2 = f.with_tensor(t2)
    assert f2.seq == 1 and bool((f2.tensor == 1).all())

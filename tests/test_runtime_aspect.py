import torch

from streamforge.control import EngineParams
from streamforge.diffusion.runtime_eager import EagerRuntime


class FakeRuntime(EagerRuntime):
    def __init__(self, internal_hw):
        self.device = "cpu"
        self.internal_hw = self._snap_hw(internal_hw)
        self.fit_plan = None

    def _dispatch(self, image_bchw, params):
        return image_bchw


def test_internal_resize_preserves_source_shape_after_roundtrip():
    rt = FakeRuntime(internal_hw=(224, 384))
    src = torch.rand(1, 3, 480, 640)
    out = rt.restyle(src, EngineParams(denoise_strength=0.5, guidance=1.0))
    assert out.shape == src.shape


def test_internal_resize_records_crop_when_ratio_differs():
    rt = FakeRuntime(internal_hw=(224, 384))
    src = torch.rand(1, 3, 480, 640)
    rt.restyle(src, EngineParams(denoise_strength=0.5, guidance=1.0))
    assert rt.fit_plan.crop_direction == "top_bottom"

import pytest

from streamforge.diffusion.runtime_base import DiffusionRuntime, TemporalDiffusionRuntime


class _Stub(TemporalDiffusionRuntime):
    def set_prompt(self, p):
        pass

    def load_once(self):
        pass

    def reset_state(self):
        pass

    def push_frame(self, x, params=None):
        return []


def test_temporal_flag_and_is_diffusion_runtime():
    r = _Stub()
    assert r.temporal is True
    assert isinstance(r, DiffusionRuntime)


def test_set_mode_is_noop():
    _Stub().set_mode("anything")  # must not raise


def test_restyle_raises():
    with pytest.raises(NotImplementedError):
        _Stub().restyle(None, None)

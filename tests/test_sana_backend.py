import pytest
from fastapi.testclient import TestClient

from streamforge.diffusion.runtime_sana_streaming import SanaStreamingRuntime
from streamforge.runner import RunnerConfig, default_runtime_factory
from streamforge.web.app import RunnerConfigIn, create_app


def test_runnerconfig_defaults_flux():
    assert RunnerConfig().backend == "flux"


def test_backend_field_reaches_config():
    cfg = RunnerConfigIn(backend="sana_streaming", cached_blocks=1, sink_token=False).to_config()
    assert cfg.backend == "sana_streaming"
    assert cfg.cached_blocks == 1
    assert cfg.sink_token is False


def test_factory_returns_temporal_sana_runtime():
    # SanaStreamingRuntime construction is light (model loads lazily in load_once)
    rt = default_runtime_factory(RunnerConfig(backend="sana_streaming"))
    assert isinstance(rt, SanaStreamingRuntime)
    assert getattr(rt, "temporal", False) is True


def test_factory_unknown_backend_raises():
    with pytest.raises(ValueError):
        default_runtime_factory(RunnerConfig(backend="nope"))


def test_control_endpoint_forwards_sana_knobs():
    class _R:
        def __init__(self):
            self.last = None

        def apply_control(self, **kw):
            self.last = kw
            return kw

        def status(self):
            return {"running": False}

    r = _R()
    client = TestClient(create_app(r))
    res = client.post("/api/control", json={"flow_shift": 6.0, "num_cached_blocks": 1})
    assert res.status_code == 200
    assert r.last == {"flow_shift": 6.0, "num_cached_blocks": 1}  # only provided fields forwarded

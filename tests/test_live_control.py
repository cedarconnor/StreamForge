from streamforge.control import TwoAxisControl
from streamforge.runner import LiveControl


def test_params_reflects_initial_preset():
    lc = LiveControl(TwoAxisControl.preset("BALANCED"))
    p = lc.params()
    assert p.steps == 4
    assert 0.0 < p.denoise_strength < 1.0


def test_apply_lowers_denoise_when_ref_strength_raised():
    lc = LiveControl(TwoAxisControl.preset("BALANCED"))
    before = lc.params().denoise_strength
    lc.apply(ref_strength=1.0)
    assert lc.params().denoise_strength < before


def test_apply_updates_text_magnitude_steps_seed():
    lc = LiveControl(TwoAxisControl.preset("BALANCED"))
    lc.apply(text_magnitude=1.2, steps=8, seed=42)
    p = lc.params()
    assert abs(p.text_magnitude - 1.2) < 1e-9
    assert p.steps == 8
    assert p.seed == 42


def test_apply_clamps_steps_to_at_least_one():
    lc = LiveControl(TwoAxisControl.preset("BALANCED"))
    lc.apply(steps=0)
    assert lc.params().steps == 1


def test_as_dict_exposes_axes():
    lc = LiveControl(TwoAxisControl.preset("FOLLOW"))
    d = lc.as_dict()
    assert set(d) == {"ref_strength", "text_magnitude", "steps", "seed"}


import time

import torch

from streamforge.frame import GpuFrame
from streamforge.runner import RunnerConfig, StreamForgeRunner


class _Src:
    def __init__(self):
        self.frame = GpuFrame(torch.zeros(1, 3, 4, 6), 0, 0.0, 6, 4)
        self.reads = 0

    def open(self): pass

    def read(self):
        self.reads += 1
        return self.frame if self.reads <= 3 else None

    def close(self): pass

    def status(self):
        from streamforge.sources.base import SourceStatus
        return SourceStatus(name="stub", width=6, height=4, fps=30.0, available=True)


class _Runtime:
    def __init__(self):
        self.prompt = None
        self.mode = "img2img"
        self.seen_params = []

    def set_prompt(self, prompt): self.prompt = prompt

    def set_mode(self, mode): self.mode = mode

    def restyle(self, tensor, params):
        self.seen_params.append(params)
        return tensor


class _Sink:
    def open(self): pass
    def send(self, f): pass
    def close(self): pass


def _runner():
    return StreamForgeRunner(source_factory=lambda c: _Src(),
                             runtime_factory=lambda c: _Runtime(),
                             sink_factory=lambda c: _Sink())


def test_apply_control_when_idle_is_noop():
    r = _runner()
    assert r.apply_control(ref_strength=0.9) == {}


def test_apply_control_updates_running_params_and_snapshot():
    r = _runner()
    r.start(RunnerConfig(source_type="webcam", source_name="0", preset="BALANCED", fps=5))
    try:
        snap = r.apply_control(ref_strength=1.0, text_magnitude=1.3, steps=6,
                               seed=99, prompt="new style", mode="edit")
        assert snap["steps"] == 6
        assert snap["seed"] == 99
        assert snap["prompt"] == "new style"
        assert snap["mode"] == "edit"
        assert r._runtime.prompt == "new style"
        assert r._runtime.mode == "edit"
        # the live params the worker reads now reflect the update
        assert r._control.params().steps == 6
    finally:
        r.stop()


def test_status_includes_control_block_when_running():
    r = _runner()
    r.start(RunnerConfig(source_type="webcam", source_name="0", preset="FOLLOW", fps=5))
    try:
        time.sleep(0.05)
        ctrl = r.status()["control"]
        assert ctrl is not None
        assert set(ctrl) >= {"ref_strength", "text_magnitude", "steps", "seed", "prompt", "mode"}
    finally:
        r.stop()
    assert r.status()["control"] is None

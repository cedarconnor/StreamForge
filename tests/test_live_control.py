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

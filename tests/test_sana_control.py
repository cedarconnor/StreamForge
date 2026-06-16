from streamforge.sana_control import SanaControl, SanaLiveControl


def test_preset_defaults():
    assert SanaControl.preset("SANA_FAST").step == 2
    assert SanaControl.preset("SANA_FAST").num_cached_blocks == 1
    assert SanaControl.preset("SANA_BALANCED").step == 4
    assert SanaControl.preset("SANA_BALANCED").num_cached_blocks == 2


def test_to_params_clamps():
    p = SanaControl(step=0, motion_score=-5).to_params()
    assert p.step == 1
    assert p.motion_score == 0


def test_hot_update_visible_without_reset():
    lc = SanaLiveControl(SanaControl.preset("SANA_BALANCED"))
    lc.apply(flow_shift=6.0, seed=5, motion_score=3)
    p = lc.params()
    assert p.flow_shift == 6.0
    assert p.seed == 5
    assert p.motion_score == 3
    assert lc.needs_reset is False


def test_warm_update_flags_reset_and_clears():
    lc = SanaLiveControl(SanaControl.preset("SANA_BALANCED"))
    lc.apply(num_cached_blocks=1)
    assert lc.needs_reset is True
    assert lc.params().num_cached_blocks == 1
    lc.clear_reset()
    assert lc.needs_reset is False


def test_as_dict_keys():
    d = SanaLiveControl(SanaControl.preset("SANA_FAST")).as_dict()
    assert set(d) == {"step", "num_cached_blocks", "sink_token", "flow_shift",
                      "motion_score", "seed", "cfg_scale"}

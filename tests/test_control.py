from streamforge.control import EngineParams, TwoAxisControl


def test_preset_names_in_range():
    for name in ["PRESERVE", "SUBTLE", "BALANCED", "FOLLOW", "FORCE"]:
        c = TwoAxisControl.preset(name)
        assert 0.0 <= c.ref_strength <= 1.0
        assert 0.0 <= c.text_magnitude <= 1.0


def test_higher_ref_strength_means_lower_denoise():
    preserve = TwoAxisControl.preset("PRESERVE").to_engine_params()
    force = TwoAxisControl.preset("FORCE").to_engine_params()
    assert preserve.denoise_strength < force.denoise_strength


def test_higher_text_magnitude_means_higher_guidance():
    subtle = TwoAxisControl.preset("SUBTLE").to_engine_params()
    force = TwoAxisControl.preset("FORCE").to_engine_params()
    assert subtle.guidance < force.guidance


def test_interpolation_is_smooth_and_clamped():
    out_of_range = TwoAxisControl(ref_strength=2.0, text_magnitude=-1.0)
    p = out_of_range.to_engine_params()
    assert 0.0 <= p.denoise_strength <= 1.0
    assert p.guidance == 1.0  # text_magnitude clamped to 0 -> GUIDANCE_MIN


def test_preset_ladder_is_monotonic():
    names = ["PRESERVE", "SUBTLE", "BALANCED", "FOLLOW", "FORCE"]
    denoise = [TwoAxisControl.preset(n).to_engine_params().denoise_strength for n in names]
    guidance = [TwoAxisControl.preset(n).to_engine_params().guidance for n in names]
    assert denoise == sorted(denoise)    # increasing: more hallucination toward FORCE
    assert guidance == sorted(guidance)  # increasing: more stylization toward FORCE


def test_engine_params_text_magnitude_defaults_to_one():
    p = EngineParams(denoise_strength=0.5, guidance=2.0)
    assert p.text_magnitude == 1.0


def test_to_engine_params_passes_text_magnitude():
    p = TwoAxisControl(ref_strength=0.5, text_magnitude=0.7).to_engine_params()
    assert abs(p.text_magnitude - 0.7) < 1e-9


def test_text_magnitude_allows_exaggeration_up_to_1_5():
    hi = TwoAxisControl(ref_strength=0.5, text_magnitude=3.0).to_engine_params()
    lo = TwoAxisControl(ref_strength=0.5, text_magnitude=-1.0).to_engine_params()
    assert hi.text_magnitude == 1.5     # clamped to lerp ceiling
    assert lo.text_magnitude == 0.0

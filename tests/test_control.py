from streamforge.control import TwoAxisControl


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

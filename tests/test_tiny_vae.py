"""Pure-logic tests for the taef2 key converter (no GPU / no weights needed).

Locks the subtle bit: the decoder Sequential starts with a param-less Clamp() at index 0, so
decoder keys shift by +1 while encoder keys do not.
"""
from streamforge.diffusion.tiny_vae import _convert_diffusers_sd_to_taesd


def test_encoder_keys_unshifted():
    out = _convert_diffusers_sd_to_taesd({"encoder.layers.0.weight": 1, "encoder.layers.3.bias": 2})
    assert "encoder.0.weight" in out
    assert "encoder.3.bias" in out


def test_decoder_keys_shift_by_one():
    # decoder.layers.0 -> decoder.1 (Clamp() occupies Sequential index 0)
    out = _convert_diffusers_sd_to_taesd({"decoder.layers.0.weight": 1})
    assert "decoder.1.weight" in out
    assert "decoder.0.weight" not in out


def test_nested_block_suffix_preserved():
    out = _convert_diffusers_sd_to_taesd({"decoder.layers.12.conv.0.bias": 1})
    assert "decoder.13.conv.0.bias" in out


def test_values_passed_through():
    out = _convert_diffusers_sd_to_taesd({"encoder.layers.0.weight": "TENSOR"})
    assert out["encoder.0.weight"] == "TENSOR"

import pytest

from streamforge.diffusion.runtime_sana_streaming import validate_dims


def test_dims_ok():
    # /32 spatial, (num_frames-1) % 8 == 0  (vae_stride [8,32,32])
    assert validate_dims(640, 384, 33) == (640, 384, 33)
    assert validate_dims(512, 512, 49) == (512, 512, 49)


def test_width_not_multiple_of_32_rejected():
    with pytest.raises(ValueError):
        validate_dims(641, 384, 33)


def test_height_not_multiple_of_32_rejected():
    with pytest.raises(ValueError):
        validate_dims(640, 385, 33)


def test_bad_frame_count_rejected():
    with pytest.raises(ValueError):
        validate_dims(640, 384, 30)  # (30-1) % 8 != 0

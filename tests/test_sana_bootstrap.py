import os
import sys

import pytest

from streamforge.diffusion.sana import _bootstrap


@pytest.mark.skipif(not _bootstrap._SANA.exists(), reason="SANA checkout not present")
def test_bootstrap_injects_paths_and_env():
    _bootstrap._DONE = False  # allow re-running regardless of prior state
    _bootstrap.ensure_sana_importable()
    assert str(_bootstrap._SANA) in sys.path
    assert str(_bootstrap._SHIMS) in sys.path
    assert os.environ["USE_CHUNKWISE_GDN"] == "1"
    assert os.environ["DISABLE_FLASH_ATTN"] == "1"


def test_root_resolves_to_repo_root():
    # repo root contains both the package src and the external checkout location
    assert (_bootstrap._ROOT / "src" / "streamforge").exists()
    assert _bootstrap._SHIMS.name == "sana_shims"


@pytest.mark.gpu
@pytest.mark.skipif(not _bootstrap._SANA.exists(), reason="SANA checkout not present")
def test_diffusion_importable_in_sana_env():
    """Only meaningful under .venv-sana (needs fla/mmcv/timm/...)."""
    _bootstrap.ensure_sana_importable()
    import diffusion  # noqa: F401

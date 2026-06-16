"""GPU acceptance for the incremental streaming engine (Task 2).

Runs only under .venv-sana with the A6000 (needs the full SANA stack + weights). Excluded from
the default `-m "not gpu"` suite. Asserts the incremental engine matches the batch sampler.
"""
import importlib.util
import pathlib

import pytest

from streamforge.diffusion.sana._bootstrap import _SANA

_SMOKE = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "sana_smoke.py"


@pytest.mark.gpu
@pytest.mark.skipif(not _SANA.exists(), reason="SANA checkout not present")
def test_incremental_matches_batch():
    spec = importlib.util.spec_from_file_location("sana_smoke", _SMOKE)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    r = smoke.run_continuity_check()
    assert r["mae"] < 1e-2, f"continuity MAE {r['mae']} >= 1e-2"
    assert r["decoded_shape"][1] == 3  # [B,3,T,H,W]

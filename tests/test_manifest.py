import pytest
from pydantic import ValidationError

from streamforge.manifest import ModelManifest, LicenseError

VALID = {
    "transformer": {"repo": "black-forest-labs/FLUX.2-klein-4B", "revision": "abc123"},
    "text_encoder": {"repo": "Qwen/Qwen3-4B", "revision": "def456"},
    "tokenizer": {"repo": "Qwen/Qwen3-4B", "revision": "def456"},
    "vae_full": {"repo": "black-forest-labs/FLUX.2-klein-4B", "revision": "abc123",
                 "file": "flux2-vae.safetensors"},
    "vae_tiny": {"repo": "madebyollin/taef2", "revision": "ghi789"},
    "scheduler": "flow_match_power_curve",
    "precision": "bf16",
    "steps": 4,
    "license": "Apache-2.0",
}


def test_loads_valid_manifest():
    m = ModelManifest(**VALID)
    assert m.steps == 4
    assert m.vae_tiny.repo == "madebyollin/taef2"


def test_rejects_unpinned_revision():
    bad = {**VALID, "transformer": {"repo": "black-forest-labs/FLUX.2-klein-4B", "revision": ""}}
    with pytest.raises(ValidationError):
        ModelManifest(**bad)


def test_rejects_placeholder_revision():
    bad = {**VALID, "transformer": {"repo": "black-forest-labs/FLUX.2-klein-4B",
                                    "revision": "FILL_FROM_DOWNLOAD"}}
    with pytest.raises(ValidationError):
        ModelManifest(**bad)


def test_license_gate_blocks_noncommercial():
    bad = {**VALID, "license": "FLUX-Non-Commercial"}
    with pytest.raises(LicenseError):
        ModelManifest(**bad).assert_commercial_clean()

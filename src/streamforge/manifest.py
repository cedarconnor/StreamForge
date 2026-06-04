"""Version-locked model manifest (design §4.1.1).

The manifest is a build artifact: every component is pinned to an exact revision so a
mismatched encoder/tokenizer/scheduler can't silently shift tensor shapes and break
TensorRT export later. It also gates the default path to a commercial-clean license.
"""
from __future__ import annotations

import yaml
from pydantic import BaseModel, field_validator


class LicenseError(Exception):
    """Raised when a manifest declares a non-commercial license on the default path."""


class Pin(BaseModel):
    repo: str
    revision: str
    file: str | None = None

    @field_validator("revision")
    @classmethod
    def revision_must_be_pinned(cls, v: str) -> str:
        if not v or v.strip() in {"", "main", "latest", "FILL_FROM_DOWNLOAD"}:
            raise ValueError("revision must be an exact commit/tag, not blank/main/latest/placeholder")
        return v


COMMERCIAL_CLEAN = {"Apache-2.0", "MIT", "CC0-1.0"}


class ModelManifest(BaseModel):
    transformer: Pin
    text_encoder: Pin
    tokenizer: Pin
    vae_full: Pin
    vae_tiny: Pin
    scheduler: str
    precision: str
    steps: int
    license: str

    def assert_commercial_clean(self) -> None:
        if self.license not in COMMERCIAL_CLEAN:
            raise LicenseError(f"manifest license {self.license!r} is not commercial-clean")

    @classmethod
    def load(cls, path: str) -> "ModelManifest":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(**yaml.safe_load(fh))

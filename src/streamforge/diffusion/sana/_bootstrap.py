"""Make the pinned SANA checkout importable from inside StreamForge.

SANA lives at external/Sana (a pinned reference clone — see manifest-sana.yaml sana_code.revision),
not copied into the package. This injects its path + the Unix `fcntl` no-op shim + the required
env (fused GDN on, flash-attn/xformers off), once, idempotently. Requires the .venv-sana env
(SANA's deps: fla, mmcv, timm, ...) — see requirements-sana-win.txt.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_DONE = False
# src/streamforge/diffusion/sana/_bootstrap.py -> parents[4] == repo root
_ROOT = Path(__file__).resolve().parents[4]
_SANA = _ROOT / "external" / "Sana"
_SHIMS = _ROOT / "scripts" / "sana_shims"


def ensure_sana_importable() -> None:
    """Idempotently put the SANA checkout + fcntl shim on sys.path and set required env."""
    global _DONE
    if _DONE:
        return
    if not _SANA.exists():
        raise RuntimeError(
            f"SANA checkout not found at {_SANA}. Clone NVlabs/Sana there at the SHA in "
            f"manifest-sana.yaml (sana_code.revision) and install requirements-sana-win.txt "
            f"into .venv-sana."
        )
    os.environ.setdefault("USE_CHUNKWISE_GDN", "1")
    os.environ.setdefault("DISABLE_FLASH_ATTN", "1")
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    for p in (str(_SHIMS), str(_SANA)):
        if p not in sys.path:
            sys.path.insert(0, p)
    _DONE = True

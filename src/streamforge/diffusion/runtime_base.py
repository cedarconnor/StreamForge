"""Swappable diffusion backend interface.

The Phase-4 bake-off compares eager / TensorRT / Nunchaku-INT4 behind this single
interface, so the harness, worker, and the rest of the pipeline never change when the
production runtime is chosen.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from streamforge.control import EngineParams

if TYPE_CHECKING:  # avoid importing torch at module import time (keeps pure tests torch-free)
    import torch


class DiffusionRuntime(ABC):
    @abstractmethod
    def set_prompt(self, prompt: str) -> None:
        """Update the style prompt (recomputes + caches its embeddings)."""

    @abstractmethod
    def restyle(self, image_bchw: "torch.Tensor", params: EngineParams) -> "torch.Tensor":
        """Input image in [0,1] NCHW on cuda -> restyled image in [0,1] NCHW on cuda."""

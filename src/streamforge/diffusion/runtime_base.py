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


class TemporalDiffusionRuntime(DiffusionRuntime):
    """Stateful, chunked video-to-video runtime (e.g. SANA-Streaming).

    Unlike the one-frame-in/one-frame-out FLUX path, a temporal backend carries recurrent
    state across frames and emits frames per autoregressive chunk. The TemporalInferenceWorker
    drives push_frame() instead of restyle(); restyle() is therefore N/A here.
    """

    temporal = True

    @abstractmethod
    def load_once(self) -> None:
        """Load model(s) + allocate state. Idempotent (safe to call repeatedly)."""

    @abstractmethod
    def reset_state(self) -> None:
        """Clear recurrent state + KV window (call on prompt change / cut)."""

    @abstractmethod
    def push_frame(self, image_bchw: "torch.Tensor", params=None) -> list:
        """Feed one source frame; return 0..N decoded output frames (a chunk may emit 0
        until the AR block fills, then N at once)."""

    def set_mode(self, mode: str) -> None:
        """SANA has no edit/img2img modes; no-op keeps runner.apply_control() safe."""
        return None

    def restyle(self, image_bchw, params):
        raise NotImplementedError("temporal runtime is chunked; use push_frame via the temporal worker.")

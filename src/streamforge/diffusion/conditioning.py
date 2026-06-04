"""Conditioning caches (design §4.4).

At 4 steps the Qwen3 text encoder is too expensive to run per frame, and references are
fixed, so both prompt embeddings and reference embeddings are cached and recomputed only
on change.
"""
from __future__ import annotations

from typing import Any, Callable


class PromptCache:
    """Caches Qwen3 prompt embeddings; recompute only when the prompt string changes."""

    def __init__(self, encode_fn: Callable[[str], Any]):
        self._encode = encode_fn
        self._key: str | None = None
        self._val: Any = None

    def get(self, prompt: str) -> Any:
        if prompt != self._key:
            self._key, self._val = prompt, self._encode(prompt)
        return self._val


class ReferenceCache:
    """Caches reference-image embeddings; references are fixed, so embed once per reference.

    Keyed on object identity (the reference image object), since reference tensors are large
    and we want a recompute only when the actual reference is swapped.
    """

    def __init__(self, embed_fn: Callable[[Any], Any]):
        self._embed = embed_fn
        self._key: Any = None
        self._val: Any = None

    def get(self, ref: Any) -> Any:
        if ref is not self._key:
            self._key, self._val = ref, self._embed(ref)
        return self._val

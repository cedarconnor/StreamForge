"""SANA-Streaming backend support (Phase-1).

SANA is a pinned *reference checkout* (external/Sana), not file-copied into this package —
its diffusion/ tree is large and deeply interdependent. `_bootstrap.ensure_sana_importable()`
injects the checkout + the fcntl shim + required env so `import diffusion` (the SANA package)
works from inside StreamForge, running under the .venv-sana environment.
"""

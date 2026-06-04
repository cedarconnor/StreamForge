"""Phase-0 gate (Task 0.4): load the full pinned stack and generate one correct still.

This validates the core stack end-to-end (text->image, full VAE) before any compilation or
img2img work. The taef2 tiny-VAE swap is validated separately once vae.py is finalized
against the real taesd.py shipped in models/vae_tiny.

Acceptance: a coherent image (not gray mush); std() > 10 is a crude non-flat sanity check.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from diffusers import Flux2KleinPipeline

from streamforge.manifest import ModelManifest

MODEL_DIR = "models/transformer"
PROMPT = "a neon-lit rainy street at night, cinematic, volumetric light"


def _load_pipe() -> Flux2KleinPipeline:
    """Load from the self-contained klein repo; if its text_encoder isn't bundled, assemble
    it from the separately-downloaded Qwen3 repo (verified against model_index.json)."""
    idx = pathlib.Path(MODEL_DIR) / "model_index.json"
    bundled_text_encoder = True
    if idx.exists():
        keys = json.loads(idx.read_text())
        bundled_text_encoder = "text_encoder" in keys and (pathlib.Path(MODEL_DIR) / "text_encoder").exists()
    if bundled_text_encoder:
        return Flux2KleinPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    from transformers import AutoModel, AutoTokenizer  # assemble path
    te = AutoModel.from_pretrained("models/text_encoder", torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained("models/text_encoder")
    return Flux2KleinPipeline.from_pretrained(
        MODEL_DIR, text_encoder=te, tokenizer=tok, torch_dtype=torch.bfloat16
    )


def main() -> None:
    m = ModelManifest.load("manifest.yaml")
    m.assert_commercial_clean()
    pathlib.Path("out").mkdir(exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    pipe = _load_pipe().to("cuda")
    img = pipe(
        prompt=PROMPT,
        num_inference_steps=m.steps,
        guidance_scale=4.0,
        generator=torch.Generator("cuda").manual_seed(7),
    ).images[0]
    img.save("out/validate_full_vae.png")
    arr = np.asarray(img).astype(np.float32)
    assert arr.std() > 10.0, "image is near-flat — stack is broken"
    print("FULL-VAE still OK:", arr.shape, "std=", round(float(arr.std()), 1))
    print("peak VRAM GB:", round(torch.cuda.max_memory_allocated() / 1e9, 2))


if __name__ == "__main__":
    main()

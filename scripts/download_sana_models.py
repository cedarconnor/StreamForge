"""Verify + pin SANA-Streaming artifacts (DiT, Gemma text encoder, LTX-2 VAE).

Reads manifest-sana.yaml and ensures each repo's PINNED revision is present in the HF cache
(snapshot_download is idempotent and reuses the cache — the streaming runtime loads via the
cache, so no separate models-sana/ copy is needed). Skips the SANA repo's source/ sample MP4s.

Run from .venv-sana. Pass --refresh to re-query latest SHAs and rewrite the manifest.
"""
from __future__ import annotations

import argparse
import pathlib

import yaml
from huggingface_hub import HfApi, snapshot_download

MANIFEST = pathlib.Path("manifest-sana.yaml")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="re-query latest SHAs from HF and rewrite the manifest")
    args = ap.parse_args()

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    jobs = [
        ("sana_repo", ["dit/*"]),
        ("text_encoder", None),
        ("vae", [f"{manifest['vae']['subfolder']}/*"]),
    ]
    for key, patterns in jobs:
        repo = manifest[key]["repo"]
        rev = HfApi().model_info(repo).sha if args.refresh else manifest[key]["revision"]
        print(f"{key}: {repo}@{rev}")
        path = snapshot_download(repo, revision=rev, allow_patterns=patterns)  # uses HF cache
        print(f"  cached at {path}")
        manifest[key]["revision"] = rev

    if args.refresh:
        MANIFEST.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        print("Refreshed SHAs into manifest-sana.yaml")
    print("Done.")


if __name__ == "__main__":
    main()

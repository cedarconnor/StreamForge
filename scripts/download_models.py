"""Download pinned model artifacts and freeze their exact commit hashes into manifest.yaml.

Run once (Task 0.3). Downloads the full repos into models/ and rewrites manifest.yaml with
real commit SHAs so manifest validation passes and the stack is reproducible.

NOTE: this pulls the full FLUX.2-klein-4B + Qwen3-4B + taef2 repos (~20-30 GB). All three
are open (Apache-2.0 / MIT) so no HF login should be required; if a repo is gated, run
`huggingface-cli login` first.
"""
from __future__ import annotations

import pathlib

import yaml
from huggingface_hub import HfApi, snapshot_download

MODELS_DIR = pathlib.Path("models")
REPOS = {
    "transformer": "black-forest-labs/FLUX.2-klein-4B",
    "text_encoder": "Qwen/Qwen3-4B",
    "vae_tiny": "madebyollin/taef2",
}


def latest_commit(repo: str) -> str:
    return HfApi().model_info(repo).sha


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    with open("manifest.yaml", "r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    for key, repo in REPOS.items():
        sha = latest_commit(repo)
        print(f"{key}: {repo}@{sha}")
        snapshot_download(repo, revision=sha, local_dir=str(MODELS_DIR / key))
        manifest[key]["revision"] = sha
    # tokenizer shares the text-encoder repo/commit; full VAE ships in the transformer repo
    manifest["tokenizer"]["revision"] = manifest["text_encoder"]["revision"]
    manifest["vae_full"]["revision"] = manifest["transformer"]["revision"]
    with open("manifest.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False)
    print("Froze revisions into manifest.yaml")


if __name__ == "__main__":
    main()

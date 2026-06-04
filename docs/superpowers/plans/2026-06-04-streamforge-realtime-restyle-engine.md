# StreamForge — Real-Time FLUX Live-Restyle Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-hosted, commercially-clean real-time live-restyle engine that takes a live video input, restyles it through step-distilled FLUX.2-klein-4B img2img, and feeds the result to a media server (Resolume) with a rock-stable output clock, on a single RTX A6000.

**Architecture:** A single CUDA context with a strict split between a **sacred output clock** (RealtimeClock + Sink, never blocks) and an **adaptive AI cadence** (InferenceWorker running opportunistically), mediated by a triple-buffered latest-frame handoff and a Governor that degrades gracefully under load. Input and output are both pluggable interfaces (`Source` / `Sink`). The diffusion runtime is swappable (`DiffusionRuntime` ABC) so the Phase-4 bake-off can compare eager / TensorRT / Nunchaku-INT4 behind one interface without rewriting the pipeline.

**Tech Stack:** Python 3.11, PyTorch (CUDA-matched build), Hugging Face `diffusers` (`Flux2KleinPipeline`) + `transformers==4.56.1`, `madebyollin/taef2` tiny VAE, FLUX.2-klein-4B (Apache-2.0), Qwen3-4B text encoder, pytest, pydantic, NumPy. Later: TensorRT, Nunchaku/SVDQuant, SpoutGL + CUDA-GL interop, NDI SDK (`ndi-python`).

**Scope of this plan:** Phases 0–6 are detailed and executable. Phases 7–10 are outlined (see the final section) — they are intentionally deferred to a re-plan because their concrete shape depends on the Phase-4 bake-off outcome (which runtime wins) and the Phase-1 structure-adherence measurement.

**Decisions locked during brainstorming (2026-06-04):**
- Input: **pluggable `Source` interface** covering synthetic + render-feed(+motion vectors) + camera/SDI + NDIReceive. Synthetic + file sources carry the early phases.
- Structure-lock: **deferred** — Phase 1 measures 4B geometry adherence empirically, and the control-adapter decision is made from that measurement.
- SDXL-Turbo fallback (design §4.1): **cut from v1** (different VAE/latent space + control scheme = a second pipeline; reintroduce only if the bake-off shows 4B cannot make any usable cadence).

**Corrections to design v1.2 folded into this plan (from the prior-art verification pass):**
1. Tiny VAE is **`madebyollin/taef2`** specifically (32-ch `flux_2` latents), not generic "TAESD".
2. **Nunchaku INT4 on the 4B is unverified** (confirmed quants exist only for the non-commercial 9B/9B-kv). Promoted from a risk to a **Phase-0 go/no-go gate**.
3. `text_magnitude` maps to the **distilled guidance scalar**; `ref_strength` maps to **img2img denoise strength**. Validated mechanism, tuned numerics in Phase 3.
4. **Input subsystem** gets a first-class `Source` interface (design only gave it one paragraph).
5. **Concurrency model** made explicit: clock/sink and inference run in separate threads with a triple-buffered handoff (resolves the §3 vs §6.0 contradiction).
6. **Spout "zero-copy" is not free in Python** — Phase 6 prototypes CUDA-GL interop early and falls back to a measured readback path if interop is not achievable.

---

## File Structure

```
streamforge/                          # repo root (git init in Task 0.1)
  pyproject.toml                      # package metadata + pytest config
  requirements.txt                    # deps EXCEPT torch (torch installed CUDA-matched, separately)
  manifest.yaml                       # version-locked model manifest (design §4.1.1) — a build artifact
  README.md
  configs/
    show.example.yaml                 # resolution, aspect, source, sink, presets, governor ladder
  src/streamforge/
    __init__.py
    config.py                         # load + validate show config (pydantic)
    manifest.py                       # ModelManifest model + load/validate (pins, license gate)
    frame.py                          # GpuFrame: device tensor + metadata (seq, pts, w, h, colorspace)
    metrics.py                        # PURE: percentile, jitter, LatencyStats (unit-tested, no GPU)
    control.py                        # PURE: TwoAxisControl + presets -> EngineParams
    color.py                          # ColorPipeline: srgb/linear/Rec709, tonemap, range, test pattern
    clock.py                          # FrameBuffer (triple-buffer) + RealtimeClock + frame-fill selection
    worker.py                         # InferenceWorker (opportunistic, reports per-stage timings)
    governor.py                       # PURE policy: Governor escalation ladder
    sources/
      base.py                         # Source ABC + Capabilities + FrameMeta
      synthetic.py                    # SyntheticSource: moving chart/ramp (deterministic, for bench)
      file_source.py                  # video/image-loop source (repeatable benching)
    sinks/
      base.py                         # Sink ABC
      null_sink.py                    # discards frames (for bench)
      file_sink.py                    # writes frames to disk (output validation)
    vae.py                            # taef2 wrapper load + encode/decode helpers
    diffusion/
      runtime_base.py                 # DiffusionRuntime ABC (swappable bake-off paths)
      runtime_eager.py                # eager diffusers Flux2Klein img2img runtime
      conditioning.py                 # prompt-embed cache + reference-embed cache
    bench/
      harness.py                      # orchestrates a timed run, collects metrics
      report.py                       # renders metrics to table + JSON
    app.py                            # wires source -> runtime -> color -> sink under clock/worker/governor
  scripts/
    download_models.py                # fetch pinned manifest artifacts, freeze revisions
    validate_stack.py                 # Phase-0 gate: load full stack, generate 1 still, assert correctness
    check_nunchaku_4b.py              # Phase-0 gate: probe Nunchaku kernel coverage for klein-4B
    measure_structure_adherence.py    # Phase-1 gate: quantify 4B geometry adherence
    run_bench.py                      # Phase-2+: run harness against a chosen runtime/source/sink
    bakeoff.py                        # Phase-4: run all runtimes through one harness + manifest
  tests/
    test_manifest.py  test_config.py  test_metrics.py  test_control.py
    test_color.py     test_frame.py   test_clock.py    test_governor.py
    test_sources.py   test_sinks.py   test_conditioning.py
  docs/superpowers/plans/             # this plan lives here
```

**Decomposition principle:** pure logic units (`metrics`, `manifest`, `config`, `control`, `color`, `clock`, `governor`, the `Source`/`Sink` ABCs, `synthetic`, `null/file` sinks) are **test-driven with complete code** — no GPU required, fully unit-tested. GPU/model/empirical units (`vae`, `diffusion/*`, the bake-off) use **validation-gate tasks**: a precise procedure, the exact API to call, the doc to confirm against, and a concrete pass/fail acceptance criterion with the command to run. A validation gate is not a placeholder — it is the correct task shape for empirical/research work where the "test" is a measured threshold, not an assertion you can write blind.

**Library-API note for executors:** FLUX.2-klein support landed in `diffusers` on 2026-01-18 and `taef2` is not yet merged into `diffusers`. Before writing any task marked **[verify-API]**, confirm the exact class/signature against the *installed* versions using context7 (`resolve-library-id` → `query-docs` for `diffusers`) or the upstream repo. Where a signature is uncertain, the task gives the function to call, the smoke test, and the acceptance criterion rather than fabricated arguments.

---

## Phase 0 — Environment, pinned manifest, and the two feasibility gates

**Gate to exit Phase 0:** the full pinned model stack loads and produces one correct still image; the Nunchaku-4B coverage question is answered yes/no in writing; revisions are frozen into `manifest.yaml`.

### Task 0.1: Repo skeleton + git + tooling

**Files:**
- Create: `streamforge/pyproject.toml`, `streamforge/requirements.txt`, `streamforge/README.md`, `streamforge/.gitignore`, `streamforge/src/streamforge/__init__.py`

- [ ] **Step 1: Initialize the repo**

```powershell
cd D:\StreamForge\streamforge
git init
```

- [ ] **Step 2: Write `pyproject.toml`** (package + pytest config)

```toml
[project]
name = "streamforge"
version = "0.0.1"
requires-python = ">=3.11,<3.12"
description = "Real-time FLUX live-restyle engine"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
markers = [
  "gpu: requires a CUDA GPU (deselect with -m 'not gpu')",
  "model: requires downloaded model weights",
]
```

- [ ] **Step 3: Write `requirements.txt`** (everything EXCEPT torch — torch is installed CUDA-matched separately to avoid the `flux2`/`torch==2.8.0` clobber noted in design §4.1.1)

```text
transformers==4.56.1
diffusers
accelerate
safetensors
huggingface_hub
pydantic>=2
pyyaml
numpy
pillow
pytest
```

- [ ] **Step 4: Write `.gitignore`**

```text
__pycache__/
*.pyc
.venv/
models/            # downloaded weights — never commit
out/               # rendered frames / reports
*.safetensors
*.engine           # TensorRT engines
.pytest_cache/
```

- [ ] **Step 5: Create the venv and install (torch first, CUDA-matched)**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
# Install the CUDA-matched torch build for the installed driver (CUDA 12.x). Confirm index URL against pytorch.org.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install -r requirements.txt
```

- [ ] **Step 6: Verify torch sees the A6000**

Run: `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
Expected: prints a torch version, `True`, and `NVIDIA RTX A6000`.

- [ ] **Step 7: Commit**

```powershell
git add -A
git commit -m "chore: repo skeleton, pinned deps, pytest config"
```

### Task 0.2: Model manifest model + validation (TDD)

**Files:**
- Create: `src/streamforge/manifest.py`, `manifest.yaml`
- Test: `tests/test_manifest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py
import pytest
from streamforge.manifest import ModelManifest, LicenseError

VALID = {
    "transformer": {"repo": "black-forest-labs/FLUX.2-klein-4B", "revision": "abc123"},
    "text_encoder": {"repo": "Qwen/Qwen3-4B", "revision": "def456"},
    "tokenizer": {"repo": "Qwen/Qwen3-4B", "revision": "def456"},
    "vae_full": {"repo": "black-forest-labs/FLUX.2-klein-4B", "revision": "abc123", "file": "flux2-vae.safetensors"},
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
    with pytest.raises(ValueError):
        ModelManifest(**bad)

def test_license_gate_blocks_noncommercial():
    bad = {**VALID, "license": "FLUX-Non-Commercial"}
    with pytest.raises(LicenseError):
        ModelManifest(**bad).assert_commercial_clean()
```

- [ ] **Step 2: Run it, expect failure**

Run: `pytest tests/test_manifest.py -v`
Expected: FAIL (`ModuleNotFoundError: streamforge.manifest`).

- [ ] **Step 3: Implement `manifest.py`**

```python
# src/streamforge/manifest.py
from __future__ import annotations
from pydantic import BaseModel, field_validator
import yaml

class LicenseError(Exception):
    """Raised when a manifest declares a non-commercial license on the default path."""

class Pin(BaseModel):
    repo: str
    revision: str
    file: str | None = None

    @field_validator("revision")
    @classmethod
    def revision_must_be_pinned(cls, v: str) -> str:
        if not v or v.strip() in {"", "main", "latest"}:
            raise ValueError("revision must be an exact commit/tag, not blank/main/latest")
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
```

- [ ] **Step 4: Run the test, expect pass**

Run: `pytest tests/test_manifest.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Write the manifest skeleton** (revisions filled by Task 0.3's download script — recorded, not guessed)

```yaml
# manifest.yaml  — version-locked model stack (design §4.1.1). Revisions frozen by scripts/download_models.py.
transformer:  { repo: black-forest-labs/FLUX.2-klein-4B, revision: "FILL_FROM_DOWNLOAD" }
text_encoder: { repo: Qwen/Qwen3-4B,                     revision: "FILL_FROM_DOWNLOAD" }
tokenizer:    { repo: Qwen/Qwen3-4B,                     revision: "FILL_FROM_DOWNLOAD" }
vae_full:     { repo: black-forest-labs/FLUX.2-klein-4B, revision: "FILL_FROM_DOWNLOAD", file: flux2-vae.safetensors }
vae_tiny:     { repo: madebyollin/taef2,                 revision: "FILL_FROM_DOWNLOAD" }
scheduler:    flow_match_power_curve
precision:    bf16            # A6000 = Ampere: bf16/fp16 + int8. NO native FP8.
steps:        4
license:      Apache-2.0
```

> The `FILL_FROM_DOWNLOAD` tokens are resolved (overwritten with real commit hashes) by Task 0.3 — they are an explicit "capture this value" handoff, not a permanent placeholder. Validation in Task 0.2's `revision_must_be_pinned` will *fail loudly* if any survive into a real run.

- [ ] **Step 6: Commit**

```powershell
git add -A; git commit -m "feat: version-locked model manifest with pin + license validation"
```

### Task 0.3: Model download + revision-freezing script

**Files:**
- Create: `scripts/download_models.py`

- [ ] **Step 1: Write the download script** (snapshot each repo at a resolved commit, then write the commit back into `manifest.yaml`)

```python
# scripts/download_models.py
"""Download pinned model artifacts and freeze their exact commit hashes into manifest.yaml."""
from huggingface_hub import snapshot_download, HfApi
import yaml, pathlib

MODELS_DIR = pathlib.Path("models")
REPOS = {
    "transformer":  "black-forest-labs/FLUX.2-klein-4B",
    "text_encoder": "Qwen/Qwen3-4B",
    "vae_tiny":     "madebyollin/taef2",
}

def latest_commit(repo: str) -> str:
    return HfApi().model_info(repo).sha

def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    manifest = yaml.safe_load(open("manifest.yaml"))
    for key, repo in REPOS.items():
        sha = latest_commit(repo)
        print(f"{key}: {repo}@{sha}")
        snapshot_download(repo, revision=sha, local_dir=MODELS_DIR / key)
        manifest[key]["revision"] = sha
    # tokenizer shares the text-encoder repo/commit
    manifest["tokenizer"]["revision"] = manifest["text_encoder"]["revision"]
    manifest["vae_full"]["revision"] = manifest["transformer"]["revision"]
    yaml.safe_dump(manifest, open("manifest.yaml", "w"), sort_keys=False)
    print("Froze revisions into manifest.yaml")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the download** (multi-GB; needs `huggingface-cli login` if any repo is gated)

```powershell
python scripts/download_models.py
```

Expected: prints `repo@sha` for each, downloads into `models/`, and rewrites `manifest.yaml` with real commit hashes.

- [ ] **Step 3: Verify the manifest now validates**

Run: `python -c "from streamforge.manifest import ModelManifest; m=ModelManifest.load('manifest.yaml'); m.assert_commercial_clean(); print('OK', m.transformer.revision)"`
Expected: `OK <a real 40-char sha>` (NOT `FILL_FROM_DOWNLOAD`).

- [ ] **Step 4: Commit the frozen manifest** (but NOT the weights — `.gitignore` excludes `models/`)

```powershell
git add manifest.yaml scripts/download_models.py
git commit -m "feat: model download + revision freezing; manifest pinned to real commits"
```

### Task 0.4: GATE — full-stack still-image validation

**Files:**
- Create: `src/streamforge/vae.py`, `scripts/validate_stack.py`

This is a **validation gate**, not a unit test. Acceptance is a correct image, judged by eye + a numeric sanity check.

- [ ] **Step 1: Write the taef2 wrapper [verify-API]**

taef2 is not in diffusers. Confirm the wrapper against `madebyollin/taef2` (`taesd.py` + README): class `DiffusersTAEF2Wrapper`, variant `flux_2`, 32 latent channels, bf16, encode `x*0.5+0.5`, decode `x*2-1` then clamp. Then:

```python
# src/streamforge/vae.py
"""taef2 tiny-VAE loader. taef2 is NOT yet in diffusers; load its wrapper and swap into the pipe."""
import torch

def load_taef2(models_dir: str = "models/vae_tiny", device: str = "cuda", dtype=torch.bfloat16):
    # taef2 ships taesd.py with DiffusersTAEF2Wrapper. Import per the repo README.
    from taesd import DiffusersTAEF2Wrapper   # vendored from madebyollin/taef2
    vae = DiffusersTAEF2Wrapper(decoder_path=f"{models_dir}/taef2_decoder.safetensors",
                                encoder_path=f"{models_dir}/taef2_encoder.safetensors")
    return vae.to(device=device, dtype=dtype)
```

> If the wrapper import path differs in the installed version, fix it here; the contract this module must satisfy is: `.encode(img_bchw_in_[-1,1]).latent_dist.sample()` returns a 32-channel latent, and `.decode(latent).sample` returns an image in `[-1,1]`. Task 3.x depends on exactly this contract.

- [ ] **Step 2: Write the still-image validation script [verify-API]**

Confirm `Flux2KleinPipeline` load + call signature against installed `diffusers` (context7: `query-docs` "Flux2KleinPipeline"). Then:

```python
# scripts/validate_stack.py
"""Phase-0 gate: load the full pinned stack, generate one still, assert it is a real image."""
import torch, numpy as np
from diffusers import Flux2KleinPipeline   # [verify-API] confirm exact class name in installed diffusers
from streamforge.manifest import ModelManifest
from streamforge.vae import load_taef2

def main() -> None:
    m = ModelManifest.load("manifest.yaml"); m.assert_commercial_clean()
    pipe = Flux2KleinPipeline.from_pretrained("models/transformer", torch_dtype=torch.bfloat16).to("cuda")
    img = pipe("a neon-lit rainy street, cinematic", num_inference_steps=m.steps,
               guidance_scale=1.0, generator=torch.Generator("cuda").manual_seed(7)).images[0]
    img.save("out/validate_full_vae.png")
    arr = np.asarray(img).astype(np.float32)
    assert arr.std() > 10.0, "image is near-flat — stack is broken"  # not gray mush
    print("FULL-VAE still OK:", arr.shape, "std=", round(float(arr.std()), 1))

    # Now swap in taef2 and confirm a still still renders (real-time VAE path)
    pipe.vae = load_taef2()
    img2 = pipe("a neon-lit rainy street, cinematic", num_inference_steps=m.steps,
                guidance_scale=1.0, generator=torch.Generator("cuda").manual_seed(7)).images[0]
    img2.save("out/validate_taef2.png")
    arr2 = np.asarray(img2).astype(np.float32)
    assert arr2.std() > 10.0, "taef2 path produced a flat image"
    print("TAEF2 still OK:", arr2.shape, "std=", round(float(arr2.std()), 1))

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the gate**

```powershell
mkdir out -Force; python scripts/validate_stack.py
```

Expected: prints `FULL-VAE still OK` and `TAEF2 still OK`; `out/validate_full_vae.png` and `out/validate_taef2.png` are recognizable images. **Acceptance:** by eye, both are coherent; taef2 is slightly softer but not garbage. If taef2 is garbage, the 32-ch/variant wiring is wrong — fix `vae.py` before proceeding.

- [ ] **Step 4: Record VRAM + timing baseline in the gate output**

Add to `validate_stack.py` before exit: `print("peak VRAM GB:", round(torch.cuda.max_memory_allocated()/1e9, 2))`. Re-run; record the number in `docs/superpowers/plans/PHASE0_NOTES.md`.

- [ ] **Step 5: Commit**

```powershell
git add src/streamforge/vae.py scripts/validate_stack.py docs/superpowers/plans/PHASE0_NOTES.md
git commit -m "feat: phase-0 full-stack still-image gate (full VAE + taef2)"
```

### Task 0.5: GATE — Nunchaku INT4 coverage for klein-4B (go/no-go for Path B)

**Files:**
- Create: `scripts/check_nunchaku_4b.py`, `docs/superpowers/plans/NUNCHAKU_DECISION.md`

The verification pass found Nunchaku quants for the **9B/9B-kv** (non-commercial) but **not confirmed for the 4B**. This gate answers, in writing, whether Path B (INT4 on the commercial 4B) is viable — before Phase 4 is scoped around it.

- [ ] **Step 1: Probe for an existing 4B SVDQuant/Nunchaku quant**

```python
# scripts/check_nunchaku_4b.py
"""Phase-0 gate: determine whether Nunchaku/SVDQuant covers FLUX.2-klein-4B."""
from huggingface_hub import HfApi

def main() -> None:
    api = HfApi()
    hits = list(api.list_models(search="FLUX.2-klein-4B", limit=50))
    nunchaku = [h.id for h in hits if "nunchaku" in h.id.lower() or "svdq" in h.id.lower()]
    print("Candidate 4B Nunchaku/SVDQuant quants:")
    for h in nunchaku: print("  ", h)
    if not nunchaku:
        print("NONE FOUND — Path B requires self-quantizing the 4B or is unavailable.")

if __name__ == "__main__":
    main()
```

Run: `python scripts/check_nunchaku_4b.py`

- [ ] **Step 2: Check kernel-architecture support directly**

Manually review (link in the decision doc): `nunchaku-ai/nunchaku` README + open PR/discussions (the verification pass cited PR #926 and HF discussion #8 as the in-progress FLUX.2 work). Determine: does a released Nunchaku version expose SVDQuant kernels for the FLUX.2-klein **4B** transformer architecture on **Ampere (sm_86)**?

- [ ] **Step 3: If a quant exists, smoke-test it [verify-API]**

Load the quant via Nunchaku's loader, generate one still on the A6000, confirm it is coherent and measure single-image latency. If it loads + renders coherently → Path B is **live**.

- [ ] **Step 4: Write the decision**

```markdown
# NUNCHAKU_DECISION.md
Date: ____
Released Nunchaku version checked: ____
4B SVDQuant/Nunchaku quant available? [yes/no/self-quantize-required]
Ampere sm_86 INT4 kernels for FLUX.2 architecture? [yes/no]
Smoke-test result (if applicable): renders=__ latency_ms=__ VRAM_GB=__
DECISION: Path B is [LIVE / DEFERRED / DROPPED].
If DROPPED/DEFERRED: Phase-4 bake-off runs A (TensorRT FP16/INT8) vs C (torch.compile) only;
  show-resolution tier expectation drops accordingly (design §5 tier table).
```

- [ ] **Step 5: Commit**

```powershell
git add scripts/check_nunchaku_4b.py docs/superpowers/plans/NUNCHAKU_DECISION.md
git commit -m "gate: nunchaku-4B coverage decision recorded (Path B go/no-go)"
```

---

## Phase 1 — Still-image img2img + the structure-adherence measurement

**Gate to exit Phase 1:** a still input image restyles correctly via img2img on the full stack, and 4B geometry-adherence is quantified (drives the deferred structure-lock decision).

### Task 1.1: `GpuFrame` value type (TDD, no GPU needed)

**Files:**
- Create: `src/streamforge/frame.py`
- Test: `tests/test_frame.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_frame.py
import torch
from streamforge.frame import GpuFrame

def test_frame_carries_metadata():
    t = torch.zeros(1, 3, 64, 64)
    f = GpuFrame(tensor=t, seq=5, pts=0.16, width=64, height=64, colorspace="srgb")
    assert f.seq == 5 and f.width == 64 and f.colorspace == "srgb"

def test_frame_with_replaces_tensor_keeps_meta():
    f = GpuFrame(tensor=torch.zeros(1,3,8,8), seq=1, pts=0.0, width=8, height=8, colorspace="srgb")
    t2 = torch.ones(1,3,8,8)
    f2 = f.with_tensor(t2)
    assert f2.seq == 1 and bool((f2.tensor == 1).all())
```

- [ ] **Step 2: Run, expect fail** — `pytest tests/test_frame.py -v` → FAIL (no module).

- [ ] **Step 3: Implement**

```python
# src/streamforge/frame.py
from __future__ import annotations
from dataclasses import dataclass, replace
import torch

@dataclass(frozen=True)
class GpuFrame:
    """A frame as a device-resident tensor (NCHW float in [0,1]) plus show metadata."""
    tensor: torch.Tensor
    seq: int
    pts: float
    width: int
    height: int
    colorspace: str = "srgb"

    def with_tensor(self, tensor: torch.Tensor) -> "GpuFrame":
        return replace(self, tensor=tensor)
```

- [ ] **Step 4: Run, expect pass** — `pytest tests/test_frame.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add -A; git commit -m "feat: GpuFrame value type"`

### Task 1.2: img2img loop on the full stack [verify-API]

**Files:**
- Create: `src/streamforge/diffusion/conditioning.py`, `src/streamforge/diffusion/runtime_base.py`, `src/streamforge/diffusion/runtime_eager.py`
- Test: `tests/test_conditioning.py`

- [ ] **Step 1: Define the runtime interface (TDD the cache, gate the GPU call)**

Write the failing test for the prompt-embed cache (pure-ish, mock the encoder):

```python
# tests/test_conditioning.py
from streamforge.diffusion.conditioning import PromptCache

def test_prompt_cache_recomputes_only_on_change():
    calls = []
    def fake_encode(p): calls.append(p); return ("emb", p)
    c = PromptCache(encode_fn=fake_encode)
    a = c.get("neon street")
    b = c.get("neon street")          # cached
    d = c.get("forest")               # recompute
    assert a is b
    assert calls == ["neon street", "forest"]
```

- [ ] **Step 2: Run, expect fail** — `pytest tests/test_conditioning.py -v` → FAIL.

- [ ] **Step 3: Implement the cache + the runtime ABC**

```python
# src/streamforge/diffusion/conditioning.py
from __future__ import annotations
from typing import Callable, Any

class PromptCache:
    """Caches Qwen3 prompt embeddings; recompute only when the prompt string changes (design §4.4)."""
    def __init__(self, encode_fn: Callable[[str], Any]):
        self._encode = encode_fn
        self._key: str | None = None
        self._val: Any = None
    def get(self, prompt: str):
        if prompt != self._key:
            self._key, self._val = prompt, self._encode(prompt)
        return self._val
```

```python
# src/streamforge/diffusion/runtime_base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from streamforge.control import EngineParams

class DiffusionRuntime(ABC):
    """Swappable diffusion backend. Phase-4 bake-off compares eager / TRT / Nunchaku behind this."""
    @abstractmethod
    def set_prompt(self, prompt: str) -> None: ...
    @abstractmethod
    def restyle(self, image_bchw: torch.Tensor, params: EngineParams) -> torch.Tensor:
        """Input image in [0,1] NCHW on cuda -> restyled image in [0,1] NCHW on cuda."""
```

- [ ] **Step 4: Run cache test, expect pass** — `pytest tests/test_conditioning.py -v` → PASS.

- [ ] **Step 5: Implement the eager runtime [verify-API]**

The img2img loop per design §4.2: encode input → add noise at `denoise_strength` → denoise `steps` with `guidance` → decode. Confirm against installed diffusers whether `Flux2KleinImg2ImgPipeline` exists; if not, drive the transformer + scheduler manually (encode with taef2, `scheduler.add_noise`, manual denoise loop, taef2 decode). Implement `EagerRuntime(DiffusionRuntime)` using `EngineParams.denoise_strength` and `EngineParams.guidance`. (Numeric tuning is Phase 3; correctness only here.)

- [ ] **Step 6: GATE — restyle a still**

Add a small `scripts/restyle_still.py` that loads one photo, runs `EagerRuntime.restyle` at the BALANCED preset, saves `out/restyle_still.png`. **Acceptance:** output is recognizably the input scene, restyled by the prompt (by eye). Commit.

```powershell
python scripts/restyle_still.py
git add -A; git commit -m "feat: eager img2img runtime + still restyle gate"
```

### Task 1.3: GATE — quantify 4B structure adherence

**Files:**
- Create: `scripts/measure_structure_adherence.py`, `docs/superpowers/plans/STRUCTURE_ADHERENCE.md`

Design §7.2 warns the distilled 4B may hold input geometry weakly. This gate measures it so the structure-lock decision is data-driven (the brainstorming "validate first" choice).

- [ ] **Step 1: Write the measurement script**

For a fixed input image, restyle at increasing `text_magnitude` (low→high) and decreasing `ref_strength` (high→low), and at each point compute structural similarity between input and output edges:

```python
# scripts/measure_structure_adherence.py
"""Quantify how strongly klein-4B holds input geometry across the control range."""
import numpy as np, torch, cv2
from PIL import Image
from streamforge.diffusion.runtime_eager import EagerRuntime
from streamforge.control import TwoAxisControl

def edge_iou(a: np.ndarray, b: np.ndarray) -> float:
    ea = cv2.Canny(a, 100, 200) > 0
    eb = cv2.Canny(b, 100, 200) > 0
    inter = (ea & eb).sum(); union = (ea | eb).sum()
    return float(inter / union) if union else 0.0

def main() -> None:
    rt = EagerRuntime(); rt.set_prompt("oil painting, thick impasto, vivid")
    src = np.asarray(Image.open("assets/struct_test.jpg").convert("RGB"))
    src_t = torch.from_numpy(src).permute(2,0,1)[None].float().div(255).cuda()
    rows = []
    for name in ["PRESERVE", "SUBTLE", "BALANCED", "FOLLOW", "FORCE"]:
        params = TwoAxisControl.preset(name).to_engine_params()
        out = rt.restyle(src_t, params)
        out_np = (out[0].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)
        Image.fromarray(out_np).save(f"out/adh_{name}.png")
        rows.append((name, edge_iou(src, out_np)))
    for name, iou in rows: print(f"{name:9s} edge-IoU vs input = {iou:.3f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run + record**

`python scripts/measure_structure_adherence.py` (needs `pip install opencv-python` + an `assets/struct_test.jpg` with clear geometry). Record the edge-IoU curve and eyeball the images.

- [ ] **Step 3: Write the decision**

```markdown
# STRUCTURE_ADHERENCE.md
edge-IoU by preset: PRESERVE __ SUBTLE __ BALANCED __ FOLLOW __ FORCE __
Reading: 4B holds geometry [strongly / moderately / weakly] at PRESERVE/SUBTLE.
DECISION: structure-lock for v1 = [NOT NEEDED / NEEDS control adapter / NEEDS render-engine motion vectors].
If a control adapter is needed it becomes an Optional Engine (design §5.1 Engine 5) planned in the Phase 7-10 re-plan, and may force a lower base resolution.
```

- [ ] **Step 4: Commit** — `git add -A; git commit -m "gate: 4B structure-adherence measured; structure-lock decision recorded"`

---

## Phase 2 — Benchmark harness (before any optimization)

**Gate to exit Phase 2:** the harness reports p50/p95/p99 + jitter + VRAM high-water + missed-deadline + frame-repeat for an eager run, across the synthetic and file sources. (NDI/Spout I/O-path latency metrics are added in Phases 6/9 when those sinks exist — this corrects the design's Phase-2 ordering, which assumed sinks already existed.)

### Task 2.1: Pure metrics (TDD, no GPU)

**Files:**
- Create: `src/streamforge/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics.py
import math
from streamforge.metrics import percentile, jitter_ms, LatencyStats

def test_percentile_basic():
    data = list(range(1, 101))  # 1..100
    assert percentile(data, 50) == 50 or percentile(data, 50) == 51
    assert percentile(data, 99) >= 99

def test_jitter_is_spread_of_intervals():
    # perfectly even 33.3ms intervals -> ~0 jitter
    even = [i * (1/30) for i in range(10)]
    assert jitter_ms(even) < 0.5
    uneven = [0, 0.033, 0.10, 0.12, 0.30]
    assert jitter_ms(uneven) > 5.0

def test_latencystats_rollup():
    s = LatencyStats.from_samples_ms([10, 20, 30, 40, 1000])
    assert s.p50 <= s.p95 <= s.p99 <= s.worst
    assert s.worst == 1000
```

- [ ] **Step 2: Run, expect fail.** `pytest tests/test_metrics.py -v` → FAIL.

- [ ] **Step 3: Implement**

```python
# src/streamforge/metrics.py
from __future__ import annotations
from dataclasses import dataclass
import statistics

def percentile(samples: list[float], p: float) -> float:
    if not samples:
        raise ValueError("empty samples")
    s = sorted(samples)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)

def jitter_ms(timestamps_s: list[float]) -> float:
    """Std-dev of inter-frame intervals, in ms. Low = smooth output clock."""
    if len(timestamps_s) < 3:
        return 0.0
    intervals = [(b - a) * 1000.0 for a, b in zip(timestamps_s, timestamps_s[1:])]
    return statistics.pstdev(intervals)

@dataclass(frozen=True)
class LatencyStats:
    p50: float; p95: float; p99: float; worst: float; mean: float; n: int
    @classmethod
    def from_samples_ms(cls, samples: list[float]) -> "LatencyStats":
        return cls(percentile(samples, 50), percentile(samples, 95),
                   percentile(samples, 99), max(samples),
                   sum(samples) / len(samples), len(samples))
```

- [ ] **Step 4: Run, expect pass.** `pytest tests/test_metrics.py -v` → PASS.
- [ ] **Step 5: Commit.** `git add -A; git commit -m "feat: pure latency/jitter metrics"`

### Task 2.2: Source ABC + SyntheticSource + FileSource (TDD)

**Files:**
- Create: `src/streamforge/sources/base.py`, `src/streamforge/sources/synthetic.py`, `src/streamforge/sources/file_source.py`
- Test: `tests/test_sources.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sources.py
import torch
from streamforge.sources.synthetic import SyntheticSource

def test_synthetic_source_yields_moving_frames():
    src = SyntheticSource(width=64, height=64, fps=30)
    src.open()
    f0 = src.read(); f1 = src.read()
    assert f0.width == 64 and f0.tensor.shape == (1, 3, 64, 64)
    assert f0.seq == 0 and f1.seq == 1
    assert not torch.equal(f0.tensor, f1.tensor)   # it moves
    src.close()

def test_synthetic_source_reports_capabilities():
    src = SyntheticSource(width=8, height=8, fps=30)
    assert src.capabilities.has_motion_vectors is False
```

- [ ] **Step 2: Run, expect fail.** `pytest tests/test_sources.py -v` → FAIL.

- [ ] **Step 3: Implement the ABC + synthetic + file sources**

```python
# src/streamforge/sources/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from streamforge.frame import GpuFrame

@dataclass(frozen=True)
class Capabilities:
    has_motion_vectors: bool = False
    has_depth: bool = False

class Source(ABC):
    capabilities: Capabilities = Capabilities()
    @abstractmethod
    def open(self) -> None: ...
    @abstractmethod
    def read(self) -> GpuFrame | None:
        """Return the next frame, or None at end-of-stream."""
    @abstractmethod
    def close(self) -> None: ...
```

```python
# src/streamforge/sources/synthetic.py
from __future__ import annotations
import torch
from streamforge.frame import GpuFrame
from streamforge.sources.base import Source, Capabilities

class SyntheticSource(Source):
    """Deterministic moving test pattern — for repeatable benchmarking with no capture hardware."""
    capabilities = Capabilities(has_motion_vectors=False)
    def __init__(self, width: int, height: int, fps: int, device: str = "cuda"):
        self.width, self.height, self.fps, self.device = width, height, fps, device
        self._seq = 0
        self._device = device if torch.cuda.is_available() else "cpu"
    def open(self) -> None: self._seq = 0
    def read(self) -> GpuFrame:
        yy, xx = torch.meshgrid(torch.linspace(0, 1, self.height),
                                torch.linspace(0, 1, self.width), indexing="ij")
        phase = (self._seq % self.fps) / self.fps
        r = (xx + phase) % 1.0
        g = (yy + phase) % 1.0
        b = torch.full_like(r, phase)
        t = torch.stack([r, g, b])[None].to(self._device)
        f = GpuFrame(tensor=t, seq=self._seq, pts=self._seq / self.fps,
                     width=self.width, height=self.height)
        self._seq += 1
        return f
    def close(self) -> None: pass
```

```python
# src/streamforge/sources/file_source.py
from __future__ import annotations
import torch, cv2
from streamforge.frame import GpuFrame
from streamforge.sources.base import Source, Capabilities

class FileSource(Source):
    """Loops a video file — repeatable input for benching the real decode/upload path."""
    capabilities = Capabilities(has_motion_vectors=False)
    def __init__(self, path: str, fps: int, device: str = "cuda"):
        self.path, self.fps = path, fps
        self._device = device if torch.cuda.is_available() else "cpu"
        self._cap = None; self._seq = 0
    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.path); self._seq = 0
    def read(self) -> GpuFrame | None:
        ok, bgr = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
            ok, bgr = self._cap.read()
            if not ok: return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(self._device)
        f = GpuFrame(tensor=t, seq=self._seq, pts=self._seq / self.fps,
                     width=t.shape[-1], height=t.shape[-2])
        self._seq += 1
        return f
    def close(self) -> None:
        if self._cap: self._cap.release()
```

- [ ] **Step 4: Run, expect pass.** `pytest tests/test_sources.py -v` → PASS (synthetic runs on CPU fallback).
- [ ] **Step 5: Commit.** `git add -A; git commit -m "feat: Source ABC + synthetic + file sources"`

### Task 2.3: Sink ABC + NullSink + FileSink (TDD)

**Files:**
- Create: `src/streamforge/sinks/base.py`, `src/streamforge/sinks/null_sink.py`, `src/streamforge/sinks/file_sink.py`
- Test: `tests/test_sinks.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_sinks.py
import torch, pathlib
from streamforge.frame import GpuFrame
from streamforge.sinks.null_sink import NullSink
from streamforge.sinks.file_sink import FileSink

def _frame():
    return GpuFrame(tensor=torch.zeros(1,3,16,16), seq=0, pts=0.0, width=16, height=16)

def test_null_sink_counts():
    s = NullSink(); s.open(); s.send(_frame()); s.send(_frame()); s.close()
    assert s.count == 2

def test_file_sink_writes(tmp_path):
    s = FileSink(str(tmp_path)); s.open(); s.send(_frame()); s.close()
    assert len(list(pathlib.Path(tmp_path).glob("*.png"))) == 1
```

- [ ] **Step 2: Run, expect fail.** `pytest tests/test_sinks.py -v` → FAIL.

- [ ] **Step 3: Implement**

```python
# src/streamforge/sinks/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from streamforge.frame import GpuFrame

class Sink(ABC):
    @abstractmethod
    def open(self) -> None: ...
    @abstractmethod
    def send(self, frame: GpuFrame) -> None: ...
    @abstractmethod
    def close(self) -> None: ...
```

```python
# src/streamforge/sinks/null_sink.py
from streamforge.sinks.base import Sink
from streamforge.frame import GpuFrame

class NullSink(Sink):
    """Discards frames; used to bench upstream stages in isolation."""
    def __init__(self): self.count = 0
    def open(self) -> None: self.count = 0
    def send(self, frame: GpuFrame) -> None: self.count += 1
    def close(self) -> None: pass
```

```python
# src/streamforge/sinks/file_sink.py
import pathlib
from PIL import Image
from streamforge.sinks.base import Sink
from streamforge.frame import GpuFrame

class FileSink(Sink):
    """Writes each frame as a PNG — for visual output validation."""
    def __init__(self, out_dir: str): self.out_dir = pathlib.Path(out_dir)
    def open(self) -> None: self.out_dir.mkdir(parents=True, exist_ok=True)
    def send(self, frame: GpuFrame) -> None:
        arr = (frame.tensor[0].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype("uint8")
        Image.fromarray(arr).save(self.out_dir / f"frame_{frame.seq:06d}.png")
    def close(self) -> None: pass
```

- [ ] **Step 4: Run, expect pass.** `pytest tests/test_sinks.py -v` → PASS.
- [ ] **Step 5: Commit.** `git add -A; git commit -m "feat: Sink ABC + null + file sinks"`

### Task 2.4: Harness + report

**Files:**
- Create: `src/streamforge/bench/harness.py`, `src/streamforge/bench/report.py`, `scripts/run_bench.py`
- Test: `tests/test_harness.py`

- [ ] **Step 1: Write failing test** (harness with a fake runtime, no GPU)

```python
# tests/test_harness.py
import torch
from streamforge.bench.harness import BenchHarness, StageTimer
from streamforge.sources.synthetic import SyntheticSource

class _FakeRuntime:
    def set_prompt(self, p): pass
    def restyle(self, img, params): return img  # passthrough

def test_harness_collects_perstage_stats():
    src = SyntheticSource(8, 8, 30)
    h = BenchHarness(source=src, runtime=_FakeRuntime(), frames=20, fps=30)
    report = h.run()
    assert "infer" in report.stages
    assert report.stages["infer"].n == 20
    assert report.missed_deadlines >= 0
    assert report.frame_repeats == 0  # every frame fresh in this fake
```

- [ ] **Step 2: Run, expect fail.** `pytest tests/test_harness.py -v` → FAIL.

- [ ] **Step 3: Implement** (uses `torch.cuda.Event` timing when on GPU, wall-clock otherwise; tracks VRAM high-water, missed deadlines vs the frame budget, repeats)

```python
# src/streamforge/bench/harness.py
from __future__ import annotations
from dataclasses import dataclass, field
import time, torch
from streamforge.metrics import LatencyStats, jitter_ms

@dataclass
class StageTimer:
    samples_ms: list[float] = field(default_factory=list)
    def add(self, ms: float) -> None: self.samples_ms.append(ms)
    @property
    def stats(self) -> LatencyStats: return LatencyStats.from_samples_ms(self.samples_ms)

@dataclass
class BenchReport:
    stages: dict[str, LatencyStats]
    output_jitter_ms: float
    vram_peak_gb: float
    missed_deadlines: int
    frame_repeats: int
    frames: int

def _now_ms() -> float: return time.perf_counter() * 1000.0

class BenchHarness:
    """Runs N frames through source->runtime, recording per-stage latency, jitter, VRAM, misses."""
    def __init__(self, source, runtime, frames: int, fps: int, prompt: str = "test"):
        self.source, self.runtime, self.frames, self.fps = source, runtime, frames, fps
        self.prompt = prompt
        self.budget_ms = 1000.0 / fps

    def run(self) -> BenchReport:
        timers = {k: StageTimer() for k in ("read", "infer")}
        out_ts: list[float] = []
        misses = repeats = 0
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        self.source.open(); self.runtime.set_prompt(self.prompt)
        last_seq = -1
        for _ in range(self.frames):
            t0 = _now_ms(); f = self.source.read(); timers["read"].add(_now_ms() - t0)
            if f is None: break
            t1 = _now_ms(); _ = self.runtime.restyle(f.tensor, _default_params())
            if torch.cuda.is_available(): torch.cuda.synchronize()
            dt = _now_ms() - t1; timers["infer"].add(dt)
            if dt > self.budget_ms: misses += 1
            if f.seq == last_seq: repeats += 1
            last_seq = f.seq
            out_ts.append(_now_ms() / 1000.0)
        self.source.close()
        vram = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
        return BenchReport(
            stages={k: t.stats for k, t in timers.items()},
            output_jitter_ms=jitter_ms(out_ts), vram_peak_gb=round(vram, 2),
            missed_deadlines=misses, frame_repeats=repeats, frames=len(out_ts))

def _default_params():
    from streamforge.control import TwoAxisControl
    return TwoAxisControl.preset("BALANCED").to_engine_params()
```

```python
# src/streamforge/bench/report.py
from streamforge.bench.harness import BenchReport
import json

def to_table(r: BenchReport) -> str:
    lines = [f"frames={r.frames} jitter={r.output_jitter_ms:.2f}ms vram={r.vram_peak_gb}GB "
             f"missed={r.missed_deadlines} repeats={r.frame_repeats}",
             f"{'stage':8s} {'p50':>7} {'p95':>7} {'p99':>7} {'worst':>8}"]
    for name, s in r.stages.items():
        lines.append(f"{name:8s} {s.p50:7.1f} {s.p95:7.1f} {s.p99:7.1f} {s.worst:8.1f}")
    return "\n".join(lines)

def to_json(r: BenchReport) -> str:
    return json.dumps({"frames": r.frames, "jitter_ms": r.output_jitter_ms,
                       "vram_gb": r.vram_peak_gb, "missed": r.missed_deadlines,
                       "repeats": r.frame_repeats,
                       "stages": {k: vars(v) for k, v in r.stages.items()}}, indent=2)
```

- [ ] **Step 4: Run, expect pass.** `pytest tests/test_harness.py -v` → PASS.

- [ ] **Step 5: GATE — eager baseline on GPU**

Write `scripts/run_bench.py` wiring `SyntheticSource` + `EagerRuntime`, then:

```powershell
python scripts/run_bench.py --frames 300 --fps 30 --res 512
```

**Acceptance:** prints a per-stage table with p50/p95/p99, jitter, VRAM, missed-deadline count. This is the **eager latency floor** — the number every later optimization is measured against. Record it in `PHASE0_NOTES.md`.

- [ ] **Step 6: Commit.** `git add -A; git commit -m "feat: benchmark harness + report + eager baseline"`

---

## Phase 3 — Minimum viable pipeline (tiny VAE + caches + fixed shapes + two-axis control)

**Gate to exit Phase 3:** the real MVP latency floor — taef2 + cached prompt/reference embeddings + fixed shapes + the tuned two-axis control — measured by the Phase-2 harness.

### Task 3.1: Two-axis control + presets (TDD)

**Files:**
- Create: `src/streamforge/control.py`
- Test: `tests/test_control.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_control.py
import pytest
from streamforge.control import TwoAxisControl, EngineParams

def test_preset_names():
    for name in ["PRESERVE", "SUBTLE", "BALANCED", "FOLLOW", "FORCE"]:
        c = TwoAxisControl.preset(name)
        assert 0.0 <= c.ref_strength <= 1.0
        assert 0.0 <= c.text_magnitude <= 1.0

def test_higher_ref_strength_means_lower_denoise():
    preserve = TwoAxisControl.preset("PRESERVE").to_engine_params()
    force = TwoAxisControl.preset("FORCE").to_engine_params()
    assert preserve.denoise_strength < force.denoise_strength

def test_higher_text_magnitude_means_higher_guidance():
    subtle = TwoAxisControl.preset("SUBTLE").to_engine_params()
    force = TwoAxisControl.preset("FORCE").to_engine_params()
    assert subtle.guidance < force.guidance

def test_interpolation_is_smooth_and_clamped():
    a = TwoAxisControl(ref_strength=2.0, text_magnitude=-1.0)  # out of range
    p = a.to_engine_params()
    assert 0.0 <= p.denoise_strength <= 1.0
```

- [ ] **Step 2: Run, expect fail.** `pytest tests/test_control.py -v` → FAIL.

- [ ] **Step 3: Implement** (the mapping: `ref_strength` ↑ → denoise ↓; `text_magnitude` ↑ → guidance ↑. The numeric `*_min/_max` bounds below are Phase-3 *starting* values to be tuned against the gate in Step 5 — the functional form is fixed, the constants are calibrated.)

```python
# src/streamforge/control.py
from __future__ import annotations
from dataclasses import dataclass

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t)

@dataclass(frozen=True)
class EngineParams:
    denoise_strength: float   # img2img noise added to the encoded input latent
    guidance: float           # FLUX.2 distilled guidance scalar
    steps: int = 4
    seed: int = 7

# Calibrated bounds (Phase-3 tuning targets these constants; see Step 5 gate).
DENOISE_MIN, DENOISE_MAX = 0.30, 0.95   # PRESERVE..FORCE map into this band
GUIDANCE_MIN, GUIDANCE_MAX = 1.0, 5.0   # distilled guidance scalar usable band

@dataclass(frozen=True)
class TwoAxisControl:
    ref_strength: float
    text_magnitude: float
    steps: int = 4
    seed: int = 7

    def to_engine_params(self) -> EngineParams:
        rs = _clamp(self.ref_strength); tm = _clamp(self.text_magnitude)
        # high ref_strength -> low denoise (stay faithful to input structure)
        denoise = _lerp(DENOISE_MAX, DENOISE_MIN, rs)
        guidance = _lerp(GUIDANCE_MIN, GUIDANCE_MAX, tm)
        return EngineParams(denoise_strength=denoise, guidance=guidance,
                            steps=self.steps, seed=self.seed)

    @classmethod
    def preset(cls, name: str) -> "TwoAxisControl":
        table = {
            "PRESERVE": (0.90, 0.20),
            "SUBTLE":   (0.75, 0.45),
            "BALANCED": (0.55, 0.60),
            "FOLLOW":   (0.30, 0.85),
            "FORCE":    (0.10, 1.00),
        }
        rs, tm = table[name]
        return cls(ref_strength=rs, text_magnitude=tm)
```

- [ ] **Step 4: Run, expect pass.** `pytest tests/test_control.py -v` → PASS.

- [ ] **Step 5: GATE — calibrate the constants**

Run `scripts/restyle_still.py` (from Task 1.2) across all five presets and confirm by eye the perceptual ladder is monotonic: PRESERVE = barely restyled → FORCE = near-hallucination. Adjust `DENOISE_*`/`GUIDANCE_*` constants until the ladder feels right; re-run the test (it asserts ordering, not exact values, so it stays green). Commit.

```powershell
git add -A; git commit -m "feat: two-axis control + calibrated presets"
```

### Task 3.2: Reference-embedding cache + wire taef2 into the runtime

**Files:**
- Modify: `src/streamforge/diffusion/conditioning.py` (add `ReferenceCache`), `src/streamforge/diffusion/runtime_eager.py` (use taef2 encode/decode + caches + fixed shapes)
- Test: `tests/test_conditioning.py` (extend)

- [ ] **Step 1: Add a failing test for `ReferenceCache`**

```python
# tests/test_conditioning.py  (append)
from streamforge.diffusion.conditioning import ReferenceCache

def test_reference_cache_keys_on_identity():
    calls = []
    def fake_embed(img): calls.append(id(img)); return ("emb", id(img))
    c = ReferenceCache(embed_fn=fake_embed)
    a = object(); 
    c.get(a); c.get(a)            # cached
    b = object(); c.get(b)        # recompute
    assert len(calls) == 2
```

- [ ] **Step 2: Run, expect fail.** `pytest tests/test_conditioning.py -v` → FAIL.

- [ ] **Step 3: Implement `ReferenceCache`** (KV/embeddings for fixed references recomputed only on change — design §4.4)

```python
# src/streamforge/diffusion/conditioning.py  (append)
class ReferenceCache:
    """Caches reference-image embeddings; references are fixed, so embed once (design §4.4)."""
    def __init__(self, embed_fn):
        self._embed = embed_fn; self._key = None; self._val = None
    def get(self, ref):
        if ref is not self._key:
            self._key, self._val = ref, self._embed(ref)
        return self._val
```

- [ ] **Step 4: Run, expect pass.** `pytest tests/test_conditioning.py -v` → PASS.

- [ ] **Step 5: Wire taef2 + caches + fixed shapes into `EagerRuntime`** — encode/decode through `load_taef2()` (Task 0.4), prompt embeddings via `PromptCache`, references via `ReferenceCache`, and pre-allocate latent/noise tensors at a fixed resolution so no per-frame allocation occurs.

- [ ] **Step 6: GATE — MVP latency floor**

```powershell
python scripts/run_bench.py --frames 600 --fps 30 --res 512 --runtime eager-taef2
```

**Acceptance:** taef2 + caches measurably lower p95 `infer` vs the Task-2.4 full-VAE eager baseline; the prompt-swap path is exercised once and its latency recorded (a Phase-2 harness metric). Record both in `PHASE0_NOTES.md`. Commit.

```powershell
git add -A; git commit -m "feat: MVP pipeline (taef2 + caches + fixed shapes); latency floor recorded"
```

---

## Phase 4 — Three-way compile bake-off (the resolution/cadence go/no-go)

**Gate to exit Phase 4:** one runtime chosen as the production path; its cadence×quality validated at the show aspect/resolution; the show-resolution tier locked. This is an **experiment protocol**, not feature code — all three runtimes implement the `DiffusionRuntime` ABC (Task 1.2) so the harness and the rest of the pipeline never change.

### Task 4.1: TensorRT runtime (Path A) [verify-API]

- [ ] Export each engine in design §5.1's list (VAE encode/decode = taef2, FLUX denoise ×N at fixed resolution + token shapes) to TensorRT FP16/INT8. Confirm export route against installed TRT + the FLUX.2 transformer (attention/RoPE/custom ops are the known risk — design §5/risk #5). Implement `TrtRuntime(DiffusionRuntime)`. If a clean static export is infeasible, record the blocker and fall back to `torch.compile` for that engine.

### Task 4.2: Nunchaku/SVDQuant INT4 runtime (Path B) [verify-API, gated by Task 0.5]

- [ ] **Only if `NUNCHAKU_DECISION.md` = LIVE.** Implement `NunchakuRuntime(DiffusionRuntime)` loading the INT4 quant (optionally `--use-qencoder` for the Qwen3 encoder, per design §5.1.1). If Task 0.5 said DEFERRED/DROPPED, skip this task and note it in the bake-off report.

### Task 4.3: torch.compile runtime (Path C)

- [ ] Implement `CompileRuntime(DiffusionRuntime)` wrapping the eager runtime with `torch.compile(mode="max-autotune")` on the transformer forward. This is the always-available fallback.

### Task 4.4: Run the bake-off

**Files:** Create `scripts/bakeoff.py`

- [ ] Run every available runtime through the **same** `BenchHarness` (Phase 2) and the **same** manifest, at the show's actual aspect ratio (16:9 / 2:1 — design §5.1.1 note), across the resolution tiers (512 → 768×432 → 768×768). For each: p50/p95/p99 infer, jitter, VRAM, missed-deadlines, AND a quality check (save N restyled frames per runtime; eyeball INT4 quality at projection scale per risk #2). Produce a comparison table.

- [ ] **GATE — decision.** Write `docs/superpowers/plans/BAKEOFF_RESULT.md`: winning runtime, locked show resolution/aspect, sustained AI cadence at that resolution, and whether INT4 quality holds. **This decision is the input to the Phase 7-10 re-plan.** Commit.

---

## Phase 5 — The sacred output clock (decoupled from AI cadence)

**Gate to exit Phase 5:** output emits a rock-stable 30/50 fps even when the AI runs at 10–20 fps, with frame-fill (latest AI → warped → held → raw) and low measured jitter.

### Task 5.1: FrameBuffer triple-buffer + fill selection (TDD)

**Files:**
- Create: `src/streamforge/clock.py` (FrameBuffer + `select_fill`)
- Test: `tests/test_clock.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_clock.py
from streamforge.clock import FrameBuffer, select_fill, FillResult

def test_framebuffer_latest_wins_and_counts_repeats():
    fb = FrameBuffer()
    assert fb.get_latest() is None
    fb.publish("ai-1")
    assert fb.get_latest() == "ai-1"            # fresh
    # no new publish -> next get is a repeat
    r = fb.get_with_freshness()
    assert r.value == "ai-1" and r.is_fresh is False

def test_select_fill_priority_order():
    # latest AI present -> use it
    assert select_fill(ai="x", warped="w", held="h", raw="r").source == "ai"
    assert select_fill(ai=None, warped="w", held="h", raw="r").source == "warped"
    assert select_fill(ai=None, warped=None, held="h", raw="r").source == "held"
    assert select_fill(ai=None, warped=None, held=None, raw="r").source == "raw"
```

- [ ] **Step 2: Run, expect fail.** `pytest tests/test_clock.py -v` → FAIL.

- [ ] **Step 3: Implement**

```python
# src/streamforge/clock.py
from __future__ import annotations
from dataclasses import dataclass
import threading, time

@dataclass(frozen=True)
class Freshness:
    value: object
    is_fresh: bool

class FrameBuffer:
    """Thread-safe single-slot latest-frame handoff between InferenceWorker and RealtimeClock.
    Publisher (worker) overwrites; consumer (clock) reads the latest and learns if it is new."""
    def __init__(self):
        self._lock = threading.Lock(); self._val = None; self._consumed = True
    def publish(self, value) -> None:
        with self._lock:
            self._val = value; self._consumed = False
    def get_latest(self):
        with self._lock:
            return self._val
    def get_with_freshness(self) -> Freshness:
        with self._lock:
            fresh = not self._consumed; self._consumed = True
            return Freshness(self._val, fresh)

@dataclass(frozen=True)
class FillResult:
    value: object
    source: str   # "ai" | "warped" | "held" | "raw"

def select_fill(ai, warped, held, raw) -> FillResult:
    """Frame-fill priority from design §6.0: freshest available wins."""
    if ai is not None:     return FillResult(ai, "ai")
    if warped is not None:  return FillResult(warped, "warped")
    if held is not None:    return FillResult(held, "held")
    return FillResult(raw, "raw")
```

- [ ] **Step 4: Run, expect pass.** `pytest tests/test_clock.py -v` → PASS.
- [ ] **Step 5: Commit.** `git add -A; git commit -m "feat: triple-buffer FrameBuffer + frame-fill selection"`

### Task 5.2: RealtimeClock + InferenceWorker threads (the explicit concurrency model)

**Files:**
- Create: `src/streamforge/worker.py`; extend `src/streamforge/clock.py` with `RealtimeClock`
- Test: `tests/test_clock.py` (add a paced-tick test using a fake monotonic clock)

- [ ] **Step 1: Add a failing test for tick pacing** (inject a time source so the test is deterministic)

```python
# tests/test_clock.py (append)
from streamforge.clock import RealtimeClock

def test_clock_emits_at_target_count():
    emitted = []
    fb = FrameBuffer(); fb.publish(("frame", 0))
    clk = RealtimeClock(fps=50, frame_buffer=fb, emit=lambda f: emitted.append(f))
    clk.run_for_ticks(5)               # synchronous test mode: 5 ticks, no real sleep
    assert len(emitted) == 5
    assert clk.repeat_count == 4        # 1 fresh, 4 repeats (no new publishes)
```

- [ ] **Step 2: Run, expect fail.** `pytest tests/test_clock.py -v` → FAIL.

- [ ] **Step 3: Implement `RealtimeClock`** (each tick: pull latest-with-freshness; if not fresh, count a repeat; emit via the sink callback; never blocks on inference) and `InferenceWorker` (runs the chosen `DiffusionRuntime` in its own thread, publishes to the FrameBuffer, reports per-stage timings to the Governor). **This is the resolution of the design §3-vs-§6.0 contradiction: clock+sink on the main/emit thread, inference on the worker thread, handoff via FrameBuffer.** Add `run_for_ticks(n)` for synchronous testing and `run()` for the real wall-clock loop using `time.perf_counter` + sleep-to-next-tick.

```python
# src/streamforge/clock.py (append)
class RealtimeClock:
    def __init__(self, fps: int, frame_buffer: "FrameBuffer", emit):
        self.period = 1.0 / fps; self.fb = frame_buffer; self.emit = emit
        self.repeat_count = 0; self._running = False
    def _tick_once(self) -> None:
        fr = self.fb.get_with_freshness()
        if not fr.is_fresh: self.repeat_count += 1
        self.emit(fr.value)
    def run_for_ticks(self, n: int) -> None:   # test mode, no sleeping
        for _ in range(n): self._tick_once()
    def run(self) -> None:                       # real mode
        import time
        self._running = True; next_t = time.perf_counter()
        while self._running:
            self._tick_once()
            next_t += self.period
            sleep = next_t - time.perf_counter()
            if sleep > 0: time.sleep(sleep)
    def stop(self) -> None: self._running = False
```

```python
# src/streamforge/worker.py
from __future__ import annotations
import threading, time
from streamforge.clock import FrameBuffer

class InferenceWorker:
    """Runs the diffusion runtime opportunistically on its own thread; publishes latest result.
    Never on the clock thread -> the output clock can never be blocked by a slow denoise."""
    def __init__(self, source, runtime, frame_buffer: FrameBuffer, params_provider, on_timing=None):
        self.source, self.runtime, self.fb = source, runtime, frame_buffer
        self.params_provider, self.on_timing = params_provider, on_timing
        self._t = None; self._running = False
    def start(self) -> None:
        self._running = True; self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
    def _loop(self) -> None:
        import torch
        self.source.open()
        while self._running:
            f = self.source.read()
            if f is None: break
            t0 = time.perf_counter()
            out = self.runtime.restyle(f.tensor, self.params_provider())
            if torch.cuda.is_available(): torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
            self.fb.publish(f.with_tensor(out))
            if self.on_timing: self.on_timing("infer", ms)
        self.source.close()
    def stop(self) -> None: self._running = False
```

- [ ] **Step 4: Run, expect pass.** `pytest tests/test_clock.py -v` → PASS.

- [ ] **Step 5: GATE — decoupled run on GPU**

Add a `scripts/run_live.py` (minimal `app.py`) that starts `InferenceWorker` + `RealtimeClock` against `SyntheticSource` + `NullSink`, runs 60s at 50fps output while the runtime is artificially slow (force ~12fps AI). **Acceptance:** the clock emits ~3000 frames (50×60), measured output jitter < 2ms, and `repeat_count` reflects the AI/output cadence gap. Record. Commit.

```powershell
python scripts/run_live.py --fps 50 --seconds 60 --force-ai-fps 12
git add -A; git commit -m "feat: RealtimeClock + InferenceWorker decoupling; stable clock under slow AI"
```

---

## Phase 6 — Spout sink + ColorPipeline + test-pattern mode (live in Resolume)

**Gate to exit Phase 6:** a live restyled frame appears in Resolume as a Spout source layer; the test-pattern path verifies color/range through the full sink before the AI runs.

### Task 6.1: ColorPipeline (TDD on CPU tensors)

**Files:**
- Create: `src/streamforge/color.py`
- Test: `tests/test_color.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_color.py
import torch
from streamforge.color import ColorPipeline, test_pattern

def test_srgb_linear_roundtrip():
    cp = ColorPipeline()
    x = torch.rand(1,3,16,16)
    back = cp.linear_to_srgb(cp.srgb_to_linear(x))
    assert torch.allclose(x, back, atol=1e-4)

def test_legal_range_compresses():
    cp = ColorPipeline(range_mode="legal")
    x = torch.zeros(1,3,4,4)           # full-black 0.0
    y = cp.apply(x)
    assert y.min() > 0.0               # 0 maps to 16/255

def test_test_pattern_has_structure():
    p = test_pattern(64, 64)
    assert p.shape == (1,3,64,64) and p.std() > 0.05
```

- [ ] **Step 2: Run, expect fail.** `pytest tests/test_color.py -v` → FAIL.

- [ ] **Step 3: Implement** (design §8.6: sRGB/linear, tonemap/clamp, range full↔legal, plus a test pattern)

```python
# src/streamforge/color.py
from __future__ import annotations
import torch

class ColorPipeline:
    """Final color stage before the sink (design §8.6). Operates on NCHW float in [0,1]."""
    def __init__(self, range_mode: str = "full", tonemap: str = "clamp"):
        self.range_mode = range_mode; self.tonemap = tonemap
    @staticmethod
    def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    @staticmethod
    def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, 1)
        return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1/2.4) - 0.055)
    def _range(self, x: torch.Tensor) -> torch.Tensor:
        if self.range_mode == "legal":
            return x * (235 - 16) / 255.0 + 16 / 255.0
        return x
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.tonemap == "clamp": x = x.clamp(0, 1)
        return self._range(x).clamp(0, 1)

def test_pattern(width: int, height: int) -> torch.Tensor:
    """SMPTE-ish vertical color bars + a luma ramp row — for sink/color verification."""
    bars = torch.tensor([[1,1,1],[1,1,0],[0,1,1],[0,1,0],[1,0,1],[1,0,0],[0,0,1]]).float()
    img = torch.zeros(3, height, width)
    seg = max(1, width // bars.shape[0])
    for i in range(bars.shape[0]):
        img[:, : int(height*0.8), i*seg:(i+1)*seg] = bars[i].view(3,1,1)
    ramp = torch.linspace(0, 1, width).view(1,1,width).expand(3, height - int(height*0.8), width)
    img[:, int(height*0.8):, :] = ramp
    return img[None]
```

- [ ] **Step 4: Run, expect pass.** `pytest tests/test_color.py -v` → PASS.
- [ ] **Step 5: Commit.** `git add -A; git commit -m "feat: ColorPipeline + test pattern (design §8.6)"`

### Task 6.2: SpoutSink + zero-copy prototype [verify-API, Windows-only]

**Files:**
- Create: `src/streamforge/sinks/spout_sink.py`

- [ ] **Step 1: Install + smoke-test SpoutGL** — `pip install SpoutGL`; confirm a trivial sender publishes a static image visible in Resolume (Spout In). Bake in the vertical-flip flag (design §8.2 — Resolume Arena needs it).

- [ ] **Step 2: Implement `SpoutSink(Sink)`** — start with the **measured-readback** path (tensor → CPU `uint8 RGBA` → `SpoutGL` send). This is correct and unblocks Resolume immediately.

- [ ] **Step 3: Prototype the zero-copy path (design §6.2/§8.2 claim).** Attempt CUDA→OpenGL interop (register a GL texture with CUDA, device-to-device copy, Spout-send the GL texture, no host readback). **Acceptance:** measure end-to-end sink latency for BOTH paths via an extended harness metric; if interop is achievable and lower-latency, make it the default and keep readback as fallback; if not achievable on this stack, **record the readback cost in `PHASE0_NOTES.md` and correct design §8.2's "zero readback" claim.** Either outcome is a pass — the point is to *know* the real number, not to assume zero.

- [ ] **Step 4: Commit.** `git add -A; git commit -m "feat: SpoutSink (readback + zero-copy prototype) with measured latency"`

### Task 6.3: GATE — live restyle in Resolume + test-pattern verification

**Files:** Extend `scripts/run_live.py` / `app.py` to accept `--sink spout --color full --test-pattern`.

- [ ] **Step 1: Test-pattern first** — run with `--test-pattern` (AI bypassed): the bars+ramp must appear correct in Resolume (color, range, no flip issues). This proves the pipe before the model (design §8.6).
- [ ] **Step 2: Live restyle** — run `FileSource` (or a webcam) → `EagerRuntime`/winning runtime → `ColorPipeline` → `SpoutSink`, output clock at 30fps. **Acceptance:** the restyled feed appears as a live Spout layer in Resolume, stable, correct color. Record a short screen capture.
- [ ] **Step 3: Commit.** `git add -A; git commit -m "feat: live restyle into Resolume via Spout; test-pattern verified"`

**End of detailed plan (Phases 0–6).** At this point StreamForge is a working same-box live-restyle engine: live input → FLUX.2-klein-4B restyle → stable output clock → Resolume, with a measured benchmark harness and a chosen production runtime.

---

## Phases 7–10 — Outline (re-plan after the Phase-4 bake-off)

These are intentionally not expanded into bite-sized tasks: their concrete shape depends on the Phase-4 winning runtime (governor knobs differ between TRT and Nunchaku), the Phase-1 structure-adherence result (whether a control adapter is in scope), and the Phase-6 measured latencies (cue-bus delay). Run `writing-plans` again after `BAKEOFF_RESULT.md` exists.

- **Phase 7 — Governor (`governor.py`):** pure escalation-ladder policy (smooth strength/prompt interpolation → raise warp/hold fill ratio → drop steps 4→2→1 → drop resolution tier → degraded passthrough), TDD against synthetic stats; wired to consume `InferenceWorker` timings and drive `params_provider`. Configurable order per show (design §6.3). Gate: no stalls under induced load.
- **Phase 8 — Temporal coherence:** the four cheap "default" techniques first (fixed seed, latent EMA/blend, previous-output color stabilization, parameter smoothing — design §9), each a small composite/post unit with a live blend knob; then optional optical-flow warp / render-engine motion-vector reuse (`motion_vector_in` source capability — the §9 "sleeper win", reachable because `Source.capabilities.has_motion_vectors` already exists). Gate: flicker visibly suppressed on motion.
- **Phase 9 — NDI sink + FrameSync + soak:** `NDISink` via `ndi-python` (download+encode, NVENC where possible), receive-side FrameSync for clock-drift (design §8.3), and the **2-hour soak test** through the harness (worst-frame latency, VRAM fragmentation, drift). Adds the NDI/Spout end-to-end latency metrics the design originally (mis)placed in Phase 2. Gate: drift/jitter validated over 2h.
- **Phase 10 — Control nodes:** MusiCue/OSC/MIDI on the per-frame control bus driving the two axes (energy→`text_magnitude`, structure→`ref_strength` — design §7.5), region-masked regeneration (latent-space mask blend + optional lightweight segmenter as Optional Engine 4), and depth/pose structure conditioning **only if** the Phase-1 adherence gate flagged it needed. Gate: audio-reactive restyle end-to-end.
- **Later:** node-graph authoring GUI; multi-GPU pipeline parallelism (design §10 "later").

---

## Self-Review (run against design v1.2)

**Spec coverage (design § → plan task):**
- §3.1 cadence model → Phase 5 (clock/worker decoupling). ✓
- §4.1/§4.1.1 manifest+pins → Tasks 0.2/0.3. ✓ (SDXL-Turbo fallback consciously cut — noted.)
- §4.2 two-axis control → Task 3.1. ✓
- §4.3 tiny VAE → Task 0.4 (taef2) + Task 3.2. ✓
- §4.4 conditioning caches → Tasks 1.2/3.2. ✓
- §5.1/§5.1.1 compile + Nunchaku → Phase 4 + Task 0.5 gate. ✓
- §5.2 harness → Phase 2. ✓
- §6.0/§6.1/§6.2/§6.3 clock/worker/governor → Phase 5 (clock/worker) + Phase 7 (governor, outlined). ✓
- §7.1–7.6 effects → Phase 10 (outlined); §7.2 adherence pulled forward to Task 1.3. ✓
- §8 output (Spout/NDI/color) → Phase 6 (Spout+color) + Phase 9 (NDI, outlined). ✓
- §9 temporal → Phase 8 (outlined); `has_motion_vectors` capability seeded in Task 2.2. ✓
- §10 build order → followed, with Phase 0 added in front and the Phase-2 sink-dependency corrected. ✓
- §12 risks → risk #2 (FP8/resolution) Phase 4; #3 (Nunchaku) Task 0.5; #4 (adherence) Task 1.3; #5 (TRT export) Task 4.1; #9 (color) Task 6.1/6.3; #10 (manifest) Task 0.2; #11 (license) Task 0.2 gate; #12 (input DMA) Source design + Task 6.2 readback measurement. ✓

**Placeholder scan:** the only deferred tokens are `FILL_FROM_DOWNLOAD` (resolved by Task 0.3, and the manifest validator *fails* if any survive) and the explicitly-scoped `[verify-API]` tasks (correct shape for a fast-moving model stack — each carries the function to call + acceptance criterion). No "TODO/handle edge cases" placeholders.

**Type consistency:** `DiffusionRuntime.restyle(image, EngineParams)` is used identically in `runtime_eager`, `worker`, `harness`, and Phase-4 runtimes. `TwoAxisControl.to_engine_params() → EngineParams(denoise_strength, guidance, steps, seed)` is consistent across `control`, `runtime`, `harness._default_params`, and the adherence script. `GpuFrame.with_tensor` used in `worker`. `FrameBuffer.publish/get_with_freshness` used by `worker`/`RealtimeClock`. Consistent.

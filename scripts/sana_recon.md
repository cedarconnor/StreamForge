# SANA-Streaming Phase-0 Recon (Task 0)

Source: shallow clone of `NVlabs/Sana` @ `external/Sana` (read-only reference, gitignored). Captured 2026-06-16.
Purpose: replace the design doc's "verify before pinning" unknowns with facts that feed Tasks 1–5.

## 1. Entry point + CLI

**Streaming V2V script:** `external/Sana/inference_video_scripts/v2v/inference_sana_streaming.py`
(There is also a general `inference_sana_video.py` and `wm/inference_sana_wm_streaming.py` — ignore; the live path is the v2v one.)

CLI flags (argparse, lines 145–170):

| flag | default | notes |
|---|---|---|
| `--mode` | `long_streaming` | the live AR path (vs `bidirectional_short` = offline quality, out of scope) |
| `--config` | None → `configs/sana_streaming/sana_streaming_2b_720p.yaml` | doc's guess was correct |
| `--model_path` | None → `hf://Efficient-Large-Model/SANA-Streaming/dit/sana_streaming_ar.pth` | `hf://` URI → **auto-downloads** |
| `--prompt` | **required** | |
| `--video_path` | **required** | local path or `hf://` |
| `--output_dir` / `--output_name` | req / `output.mp4` | |
| `--num_frames` | None → `969` (long_streaming) | |
| `--height` / `--width` | `704` / `1280` | **defaults to 720p — start LOWER on A6000** |
| `--fps` | `16` | |
| `--step` | None → `4` | |
| `--cfg_scale` | None → `1.0` | streaming is distilled; keep 1.0 |
| `--flow_shift` | None → config `inference_flow_shift: 8.0` | |
| `--seed` | `0` | (StreamForge default is 7) |
| `--negative_prompt` | a long default | mostly for bidirectional quality mode |
| `--motion_score` | None | optional |
| `--num_cached_blocks` | `2` | GDN/KV state window |
| `--sink_token` | `True` | attention-sink stability |

## 2. Dimension rules (for Phase-1 `validate_dims`) — CONCRETE

From `vae_stride: [8, 32, 32]` (config) and the latent math (script lines 305–307):
```
latent_t = (num_frames - 1) // 8 + 1      # temporal stride 8  -> frames best as 8k+1
latent_h = height // 32                    # height MUST be divisible by 32
latent_w = width  // 32                    # width  MUST be divisible by 32
```
So `validate_dims`: reject/snap H or W not divisible by **32**; frames align to **8k+1**. (Doc guessed "likely 32" for spatial — confirmed; temporal stride 8 is new.)

## 3. Import graph (the §3.1 question) — RESOLVED

- **`fla` IS REQUIRED** (contra the doc's "only if used"). The streaming GDN forward uses `_cached_gdn_forward_triton` (`sana_gdn_blocks.py:1238`) from `diffusion/model/ops/fused_streaming.py`, which does `from fla.modules import ShortConvolution` (`fused_streaming.py:71`). → Task 1 must install `fla`. PyPI candidates: **`flash-linear-attention`** (provides the `fla` namespace) or **`fla-core`** — install whichever provides `import fla`; verify after install.
- **`flash_attn` is OPTIONAL and fully guarded.** `diffusion/utils/import_utils.py` try/excepts `flash_attn.cute` → `flash_attn_interface` → `flash_attn` → `None`. `sana_v2v_attn_blocks.py:50` honors a **`DISABLE_FLASH_ATTN=1`** env switch and falls back to `F.scaled_dot_product_attention` (lines 1010, 1132). The streaming sampler even hides a hard diffusers `import flash_attn` (`self_forcing_flow_euler_sampler.py`). → **Task 4 needs no monkeypatch; just run with flash_attn absent (optionally set `DISABLE_FLASH_ATTN=1`).**
- **GDN fused switch:** the script sets `os.environ.setdefault("USE_CHUNKWISE_GDN", "1")` and `DISABLE_XFORMERS=1` (script lines 23–24). Doc's `USE_CHUNKWISE_GDN` name + default confirmed. Other knobs in `sana_gdn_blocks.py`: `GDN_DISABLE_COMPILE`, `SANA_USE_FLEX_ATTENTION`, `SANA_WM_SDPA_D112_DIRECT`.
- **xformers excluded correctly** — the script disables it itself.

## 4. Artifacts / manifest values

- **DiT:** `Efficient-Large-Model/SANA-Streaming` → `dit/sana_streaming_ar.pth` (~4.52 GB, BF16). Loaded via `hf://` + `tools.download.find_model` (auto-fetch to HF cache).
- **Text encoder:** `gemma-2-2b-it` resolves to **`Efficient-Large-Model/gemma-2-2b-it`** (`builder.py:76`, the Sana mirror — not `google/...`, so likely ungated). Loaded `AutoModelForCausalLM` in BF16. `model_max_length: 300`, `y_norm: true`, uses a `chi_prompt` enhancement prefix.
- **VAE — ⚠ AMBIGUITY TO RESOLVE BEFORE PINNING:** config says `vae_type: LTX2VAE_chunk_tile`, `vae_pretrained: Lightricks/LTX-2`. But the SANA-Streaming HF repo also ships `ltx2_causal_vae_0516/`. Confirm which the streaming path actually loads (check `get_vae()` in `builder.py` against this config) before writing `manifest-sana.yaml`. `Lightricks/LTX-2` may be large and/or gated.
- **Model class:** `SanaMSVideoV2V_2000M_P1_D20` (2B), `attn_type: V2VStateCachedBiGDNAttention`, `softmax_ratio: 0.25` (25% softmax blocks; rest linear GDN), `fp32_attention: true`, `vae_latent_dim: 128`.

## 5. Refinements to the Phase-0 plan ordering

- **`fla` → add to `requirements-sana-win.txt` (Task 1), not optional.**
- **Gate 3 can auto-download.** Because `--model_path`/`--video_path` accept `hf://`, the simplest Gate-3 run fetches weights itself. Suggested order: Task 1 (env + `fla`) → **Task 3 run the script** (proves fused GDN + auto-fetches) → Task 2 (formalize `download_sana_models.py` + freeze `manifest-sana.yaml` for reproducibility) → Task 5 (in-process tensors). The plan currently lists download as Task 2 before Task 3; either works, but auto-download means Task 2 is about *pinning*, not *fetching*.
- **Start resolution well below 704×1280** on the A6000 for the first runs (e.g. 384×640 or 512×512, both /32-divisible) to isolate "does it run" from "is it real-time."
- **`seed` default differs** (SANA 0 vs StreamForge 7) — align in Phase-1 presets.

## 6. Status

Task 0 (recon) COMPLETE. Gate 1 (Triton) PASSED. Gate 4 (attention) pre-resolved to SDPA (no FA2 wheel for our stack; SANA falls back to SDPA natively).
**Next real go/no-go = Gate 3:** install `fla`, run `inference_sana_streaming.py --mode long_streaming` with `USE_CHUNKWISE_GDN=1`, confirm a valid clip → proves SANA's own fused GDN Triton kernels compile/run on the A6000.

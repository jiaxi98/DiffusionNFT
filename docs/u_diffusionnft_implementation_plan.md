# U-DiffusionNFT Implementation Plan (Dense Reward Map)

## 0. Scope and goal
- Implement the plan described in `docs/u_diffusionnft_appendix.md` without changing the core DiffusionNFT structure (implicit positive/negative policy + forward-process SL loss).
- Replace scalar optimality probability `r(x0,c)` with a **location-conditional** scalar `r(x0,U,c)` derived from a dense reward map.
- This document is the design/implementation blueprint; code changes should follow it.

## 1. Current training data flow (baseline)
- Sampling: `pipeline_with_logprob` produces `(images, latents)` using the **old** adapter. Rewards are computed from **images** and logged as scalars.
- Training: `latents_clean` is used as `x0`. Rewards are aggregated into scalar advantages, mapped into `r in [0,1]`, and used to mix positive/negative losses.
- Key locations:
  - Sampling & reward: `scripts/train_nft_sd3.py` (reward_fn, `samples_data_list`)
  - Loss: `scripts/train_nft_sd3.py` (policy loss + KL)
  - Reward API: `flow_grpo/rewards.py`

## 2. Target algorithm summary (U-DiffusionNFT)
- Each image has a dense reward map `r_map(x0,c)` (or raw map `R_raw`).
- Sample locations `U` from a distribution `p(U)` (uniform by default).
- Use `r(x0,U,c) = r_map(x0,c)[U]` as the **location-conditional** optimality probability in the same objective as Eq.(5).
- Training loss uses location samples:
  - `L = E[ r(U)*||v_theta^+[U]-v[U]||^2 + (1-r(U))*||v_theta^-[U]-v[U]||^2 ]`

## 3. Representation choices (explicit decisions needed)
### 3.1 Reward-map resolution and alignment
- **Decision**: define `U` on the latent spatial grid (H_lat x W_lat) to avoid VAE coupling.
- **Deferred**: how image-space rewards are converted into latent-grid rewards (postpone until after first implementation pass).
- Store maps at the chosen grid resolution to keep memory bounded.

### 3.2 Reward-map normalization (per-prompt group)
- For each prompt group of size `K` (already produced by `DistributedKRepeatSampler`):
  - Compute elementwise mean `mu_c(u)`.
  - `R_norm = R_raw - mu_c(u)`.
  - `r_map = 0.5 + 0.5 * clip(R_norm / Z_c, -1, 1)`.
- This mirrors the scalar normalization in the paper but is applied per location `u`.

### 3.3 Backward compatibility
- Keep the scalar reward path unchanged when `reward_map.enabled = False`.
- If `reward_map.enabled = True`, bypass scalar-advantage logic for loss computation (still log scalar summaries from the map).

## 4. File-by-file implementation plan

### 4.1 `config/base.py`
Add a new config group (names can be adjusted):
- `config.reward_map.enabled` (bool)
- `config.reward_map.map_resolution` or `config.reward_map.grid` (e.g., `(H_lat, W_lat)`)
- `config.reward_map.name` (dense-map function name)
- `config.reward_map.num_samples` (m locations per image)
- `config.reward_map.loss_scale` (optional scale factor for map loss)
- `config.reward_map.normalize` (bool)
- `config.reward_map.Z_c` (normalization scale; scalar or per-prompt)
- `config.reward_map.sample_distribution` (`"uniform"` or custom)

### 4.2 `config/nft.py`
- Add a new config variant enabling reward maps (e.g., `sd3_geneval_dense_reward`), or extend existing configs with overrides.

### 4.3 `flow_grpo/reward_maps.py`
Add a separate registry for dense reward maps (independent from scalar rewards):
- `get_reward_map_fn(device, config.reward_map)` returns a callable that outputs a latent-grid map.
- This keeps `flow_grpo/rewards.py` strictly scalar, matching the original DiffusionNFT path.

### 4.4 `scripts/train_nft_sd3.py`
**Sampling stage** (data collection):
- Add a placeholder hook to attach a reward map to each sample (source deferred), using `reward_map_fn`.
- Store maps in `samples_data_list` (e.g., `reward_map_raw` or `reward_map`) at latent-grid resolution.
- If memory is an issue, store sampled `(U_j, r_j)` pairs instead of full maps.

**Normalization stage** (after gathering across processes):
- Group by prompt (using existing prompt IDs) and compute per-location mean across the K samples.
- Create `r_map` in `[0,1]` using `Z_c` and clipping.
- Optional: store `r_map` only; discard raw maps to save memory.
- Note: normalization is deferred in the initial code path (expects normalized maps).

**Training stage**:
- Sample `U_1..U_m` per image; gather corresponding `r_j` from `r_map`.
- Gather `v_theta^+[U_j]`, `v_theta^-[U_j]`, and `v[U_j]` from the latent tensors.
- Compute the Monte Carlo loss over `U_j`.
- Keep scalar logging via `r_map.mean()` or a configured aggregation.

**Compatibility**:
- Gate all new logic behind `config.reward_map.enabled` so existing runs are unchanged.

### 4.5 Optional utility module (new)
Create `flow_grpo/reward_map_utils.py` for reusable helpers:
- `downsample_reward_map(R_raw, target_hw)`
- `normalize_reward_map(R_raw, prompt_ids, Z_c)`
- `sample_locations(H, W, m, distribution)`
- `gather_locations(tensor, U)` (for batched indexing)

### 4.6 Evaluation / logging
- `scripts/evaluation.py` can optionally log map statistics (mean, std, sparsity) without changing metrics.

## 5. Acceptance criteria (for later code changes)
- Training runs with `reward_map.enabled=True` without modifying the diffusion model architecture.
- Scalar path remains identical when `reward_map.enabled=False`.
- `r_map` normalization behaves as described in `docs/u_diffusionnft_appendix.md`.

## 6. Deferred items (explicitly postponed)
- How to convert image-space rewards into latent-grid rewards.
- How to generate reward maps from the generation pipeline (image-space reward source).

## 7. TODO checklist (not yet implemented)
- [ ] Implement a concrete reward-map generator (image → latent-grid map) and register it in `flow_grpo/reward_maps.py`.
- [ ] Decide and implement map normalization (`reward_map.normalize=True`) and grouping across prompts.
- [ ] Add a dense-reward config variant in `config/nft.py`.
- [ ] Add a small test or debug path (mock reward map) to validate end-to-end behavior.

## 8. Reward-map construction notes (doc-only; decision pending)
### 8.1 Current SD3.5-medium VAE latent mapping (for reference)
- VAE downsampling factor is **8×** (AutoencoderKL with 4 down blocks → `vae_scale_factor = 8`).
- Latent grid size is `(H/8, W/8)` (e.g., 512→64, 768→96, 1024→128).
- Each latent location corresponds roughly to an **8×8 image patch**, but receptive field is larger due to stacked convs/attention.

### 8.2 Why this matters
- Dense reward maps must be aligned to the **latent grid** for the current training loss.
- We will **postpone** the actual mapping from image-space rewards to latent-space rewards until we decide on the encoder/tokenizer.

### 8.3 Alternative encoders/tokenizers to consider (ViT-style)
These are candidates if we want a clearer patch-to-latent correspondence than a conv VAE:
- **ViT-VQGAN / improved VQGAN**: ViT encoder + decoder, discrete tokens, strong patch alignment.
- **TiTok**: transformer-based image tokenizer producing patch-aligned tokens.
- **HieraTok**: hierarchical ViT tokenizer with spatial token organization.
- **Open-MAGVIT2**: public MAGVIT-v2 tokenizer implementation with transformer-based encoder.
- **RAE-style encoders**: pretrained ViT encoder + learned decoder for continuous latents.

### 8.4 Decision gate
- Choose the encoder/tokenizer **before** implementing image→latent reward-map mapping.

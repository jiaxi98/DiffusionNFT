# Repository Guidelines

## Project Structure & Module Organization
- `flow_grpo/` holds the core training and reward logic, plus patches under `flow_grpo/diffusers_patch/`.
- `scripts/` contains entry points for training and evaluation (e.g., `train_nft_sd3.py`, `evaluation.py`).
- `config/` provides run configurations (`base.py`, `nft.py`), selected via `--config config/nft.py:<name>`.
- `dataset/` includes small metadata/text splits for supported datasets (e.g., `geneval`, `ocr`).
- `reward_ckpts/` is the local cache for downloaded reward model checkpoints.
- `assets/` and `flow_grpo/assets/` store figures and small static assets used by docs or rewards.

## Build, Test, and Development Commands
- `pip install -e .` installs the package in editable mode (requires Python >= 3.10).
- `torchrun --nproc_per_node=8 scripts/train_nft_sd3.py --config config/nft.py:sd3_geneval` runs single-node training for GenEval (adjust GPU count/config as needed).
- `torchrun --nproc_per_node=8 scripts/train_nft_sd3.py --config config/nft.py:sd3_multi_reward` runs multi-reward training.
- `torchrun --nproc_per_node=8 scripts/evaluation.py --checkpoint_path ... --dataset geneval --save_images` evaluates a checkpoint and saves images.

## Coding Style & Naming Conventions
- Python with 4-space indentation; prefer `snake_case` for functions/variables and `CamelCase` for classes.
- Keep config names explicit and consistent with datasets (e.g., `sd3_geneval`, `sd3_multi_reward`).
- Formatting: `black` is listed in `extras_require[dev]`; use default Black settings if you format files.

## Testing Guidelines
- There is no formal test suite in the repo. `pytest` is included for optional tests if you add them.
- For regression checks, prefer running `scripts/evaluation.py` on a small dataset (`geneval`, `ocr`, `pickscore`, or `drawbench`) and verifying outputs.
- Example assets for quick checks live in `flow_grpo/test_cases/`.

## Commit & Pull Request Guidelines
- Existing commit messages are short and imperative (e.g., “Update README.md”). Follow the same style.
- PRs should include: a clear description of changes, the config used, and any training/eval commands run.
- If results change, include sample outputs or metrics and note relevant checkpoints or datasets.

## Configuration & Runtime Notes
- Training uses `torchrun` (not `accelerate`) and expects configs in `config/nft.py`.
- Set `WANDB_API_KEY` and `WANDB_ENTITY` if logging to Weights & Biases.
- Do not commit large checkpoints or downloaded reward files; keep them in `reward_ckpts/`.

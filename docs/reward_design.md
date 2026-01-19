# Reward design

## CLIPScore-based reward maps

We will explore three variants derived from the CLIPScore used in this repo
(the default scorer is `openai/clip-vit-large-patch14` in `flow_grpo/clip_scorer.py`).

### Variant 1: pixel-level gradient map (simple baseline)

$$
R_{\text{pix}}(I, C) = 1 - \mathrm{Normalize}\!\left(\left\lvert \nabla_I \mathrm{CLIPScore}(I, C) \right\rvert\right)
$$

- Compute the gradient of CLIPScore with respect to image pixels.
- Take per-pixel magnitude, then normalize to `[0, 1]` (e.g., min–max or percentile).
- Optional: apply a small Gaussian blur to reduce high-frequency noise.

This is easy to implement but can be noisy; it is mainly a diagnostic baseline.

### Variant 2: Grad-CAM on CLIP image encoder (preferred)

We use the CLIP ViT-L/14 image encoder. Unlike a ResNet, it has no conv feature maps,
so we treat the **patch-token activations** as the feature map.

Concrete choice:
- Use the **last transformer block** patch tokens (before pooling the CLS token).
- For a 224×224 input with patch size 14, the patch grid is 16×16.

Procedure:
1) Forward to get patch token activations `F` of shape `(H_p × W_p, D)`.
2) Backprop CLIPScore to get gradients `G = ∂CLIPScore/∂F`.
3) Compute channel weights by global average over spatial tokens.
4) Weighted sum of `F` with these weights, followed by ReLU.
5) Reshape to `(H_p, W_p)` and upsample to image size.

If we ever switch to a ResNet CLIP, the equivalent is the last conv feature map
(`layer4` output, typically 7×7).

### Variant 3: token–image attention map (ViT-only)

CLIP ViT exposes attention matrices between the CLS token and image patches.
We can aggregate these to form a relevance map.

Concrete plan:
- Extract attention from the **last block** (or attention rollout across blocks).
- Average over heads to get a `(H_p × W_p)` map.
- Optionally **gradient-weight** the attention with `∂CLIPScore/∂attention` (Grad-CAM style).
- Upsample the patch map to image size.

Relation to CLIPScore:
CLIPScore uses the CLS embedding, which is a weighted aggregation of patch tokens.
Attention maps (especially gradient-weighted) provide an attribution of which patches
contribute most to the CLIPScore.

### Summary

- Variant 1: simplest; likely noisy.
- Variant 2: stable and semantically aligned; our primary candidate.
- Variant 3: promising for text localization; depends on attention access.

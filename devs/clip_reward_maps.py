import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="CLIPScore-based reward map variants")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Text prompt for CLIPScore")
    parser.add_argument("--out_dir", default="devs/clip_reward_maps", help="Output directory")
    parser.add_argument(
        "--variants",
        default="pix,gradcam,attn",
        help="Comma-separated list: pix, gradcam, attn",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 (cuda recommended)")
    return parser.parse_args()


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    x = x - x.min()
    denom = x.max().clamp(min=1e-6)
    return x / denom


def save_map(map_tensor: torch.Tensor, out_path: Path, size_hw):
    map_np = map_tensor.detach().cpu().numpy()
    map_np = (map_np * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(map_np)
    img = img.resize((size_hw[1], size_hw[0]), resample=Image.BILINEAR)
    img.save(out_path)


def grid_size(num_patches: int) -> int:
    g = int(num_patches**0.5)
    if g * g != num_patches:
        raise ValueError(f"num_patches={num_patches} is not a perfect square")
    return g


def compute_text_features(model, processor, prompt, device, dtype):
    text_inputs = processor(text=[prompt], padding="max_length", truncation=True, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_features = model.get_text_features(**text_inputs).to(dtype=dtype)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def compute_score_with_tokens(model, pixel_values, text_features, hook_layer=-1):
    captured = {}

    def pre_hook(_module, inputs):
        captured["tokens"] = inputs[0]

    handle = model.vision_model.encoder.layers[hook_layer].register_forward_pre_hook(pre_hook)
    vision_outputs = model.vision_model(
        pixel_values=pixel_values,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )
    handle.remove()

    image_embeds = model.visual_projection(vision_outputs.pooler_output)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    score = (image_embeds * text_features).sum()
    return score, vision_outputs, captured.get("tokens")


def variant_pixel_grad(score, pixel_values):
    grad = torch.autograd.grad(score, pixel_values, retain_graph=True)[0]
    grad_mag = torch.sqrt((grad**2).sum(dim=1, keepdim=False))  # (B,H,W)
    grad_map = normalize_map(grad_mag[0])
    return grad_map


def variant_gradcam(score, tokens):
    if tokens is None:
        raise RuntimeError("Failed to capture token activations for Grad-CAM.")
    patch_tokens = tokens[:, 1:, :]  # drop CLS
    grad = torch.autograd.grad(score, tokens, retain_graph=True)[0]
    grad = grad[:, 1:, :]  # drop CLS grads
    if grad is None:
        raise RuntimeError("Grad-CAM gradients are None; try attention map variant instead.")
    weights = grad.mean(dim=1)  # (B, D)
    cam = (patch_tokens * weights.unsqueeze(1)).sum(dim=-1)  # (B, num_patches)
    cam = F.relu(cam)
    cam = normalize_map(cam[0])
    num_patches = cam.shape[0]
    g = grid_size(num_patches)
    cam = cam.view(g, g)
    return cam


def variant_attention(vision_outputs):
    # Use last layer attention, average heads, take CLS->patch attention
    attn = vision_outputs.attentions[-1]  # (B, heads, seq, seq)
    attn = attn.mean(dim=1)  # (B, seq, seq)
    attn_map = attn[:, 0, 1:]  # CLS to patches
    attn_map = normalize_map(attn_map[0])
    num_patches = attn_map.shape[0]
    g = grid_size(num_patches)
    attn_map = attn_map.view(g, g)
    return attn_map


def main():
    args = parse_args()
    device = torch.device(args.device)

    image = Image.open(args.image).convert("RGB")
    orig_h, orig_w = image.size[1], image.size[0]

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()

    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    image_inputs = processor(images=image, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device=device, dtype=dtype)
    pixel_values.requires_grad_(True)

    text_features = compute_text_features(model, processor, args.prompt, device, dtype)
    score, vision_outputs, tokens = compute_score_with_tokens(model, pixel_values, text_features)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    if "pix" in variants:
        pix_map = variant_pixel_grad(score, pixel_values)
        save_map(pix_map, out_dir / f"{stem}_pix.png", (orig_h, orig_w))

    if "gradcam" in variants:
        cam_map = variant_gradcam(score, tokens)
        save_map(cam_map, out_dir / f"{stem}_gradcam.png", (orig_h, orig_w))

    if "attn" in variants:
        attn_map = variant_attention(vision_outputs)
        save_map(attn_map, out_dir / f"{stem}_attn.png", (orig_h, orig_w))

    print(f"Saved maps to {out_dir}")


if __name__ == "__main__":
    main()

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Reward map generation (OCR or CLIP)")
    parser.add_argument(
        "--mode",
        choices=["ocr", "clip"],
        default="ocr",
        help="Reward map mode. Default: ocr",
    )
    parser.add_argument(
        "--image",
        default="samples/ocr_checkpoint-60/prompt_000/seed_1.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "A high-fashion runway with a sleek, modern backdrop displaying "
            '"Spring Collection 2024". Models walk confidently on the catwalk, '
            "showcasing vibrant, floral prints and pastel tones, under soft, ambient "
            "lighting that enhances the fresh, spring vibe."
        ),
        help="Prompt text",
    )
    parser.add_argument("--out_dir", default="samples/reward_maps", help="Output directory")

    # OCR options
    parser.add_argument("--bg", type=float, default=0.5, help="Background reward baseline")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR")
    parser.add_argument("--blur", type=int, default=7, help="Gaussian blur kernel size (odd)")

    # CLIP options
    parser.add_argument(
        "--variants",
        default="pix,gradcam,attn",
        help="Comma-separated list: pix, gradcam, attn",
    )
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 (cuda recommended)")
    parser.add_argument(
        "--clip_model",
        default="openai/clip-vit-large-patch14",
        help="CLIP model name",
    )
    return parser.parse_args()


def extract_targets(prompt: str) -> list[str]:
    return re.findall(r'"([^"]+)"', prompt)


def gaussian_blur(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return mask
    if ksize % 2 == 0:
        ksize += 1
    sigma = ksize / 3.0
    ax = np.arange(-(ksize // 2), ksize // 2 + 1)
    kernel = np.exp(-(ax**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    blurred = mask.copy().astype(np.float32)
    blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=blurred)
    blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=blurred)
    return blurred


def save_map(map_array: np.ndarray, out_path: Path, size_hw=None):
    map_np = (map_array * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(map_np)
    if size_hw is not None:
        img = img.resize((size_hw[1], size_hw[0]), resample=Image.BILINEAR)
    img.save(out_path)


def run_ocr(image: Image.Image, prompt: str, args):
    from paddleocr import PaddleOCR
    from Levenshtein import distance as levenshtein_distance

    def norm_edit_distance(a: str, b: str) -> float:
        a = a.lower().strip().replace(" ", "")
        b = b.lower().strip().replace(" ", "")
        if not a and not b:
            return 0.0
        if not a or not b:
            return 1.0
        return levenshtein_distance(a, b) / max(len(a), len(b))

    def best_similarity(text: str, targets: list[str]) -> float:
        if not targets:
            return 0.0
        return max(1.0 - norm_edit_distance(text, t) for t in targets)

    w, h = image.size
    ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=args.gpu, show_log=False)
    result = ocr.ocr(np.array(image), cls=False)

    targets = extract_targets(prompt)
    reward = np.full((h, w), args.bg, dtype=np.float32)

    if result and len(result) > 0 and result[0] is not None:
        lines = result[0]
    else:
        lines = []

    for line in lines:
        if len(line) < 2:
            continue
        box, (text, conf) = line[0], line[1]
        sim = best_similarity(text, targets)
        reward_box = float(conf) * float(sim)

        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        x0, x1 = max(min(xs), 0), min(max(xs), w - 1)
        y0, y1 = max(min(ys), 0), min(max(ys), h - 1)
        if x1 <= x0 or y1 <= y0:
            continue

        mask = np.zeros((h, w), dtype=np.float32)
        mask[y0 : y1 + 1, x0 : x1 + 1] = 1.0
        if args.blur > 1:
            mask = gaussian_blur(mask, args.blur)
        reward = reward + mask * (reward_box - reward)

    reward = np.clip(reward, 0.0, 1.0)
    return reward


def run_clip(image: Image.Image, prompt: str, args):
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor

    def normalize_map(x: torch.Tensor) -> torch.Tensor:
        x = x - x.min()
        denom = x.max().clamp(min=1e-6)
        return x / denom

    def grid_size(num_patches: int) -> int:
        g = int(num_patches**0.5)
        if g * g != num_patches:
            raise ValueError(f"num_patches={num_patches} is not a perfect square")
        return g

    def compute_text_features(model, processor, text_prompt, device, dtype):
        text_inputs = processor(text=[text_prompt], padding="max_length", truncation=True, return_tensors="pt")
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
        grad_mag = torch.sqrt((grad**2).sum(dim=1, keepdim=False))
        grad_map = normalize_map(grad_mag[0])
        return grad_map

    def variant_gradcam(score, tokens):
        if tokens is None:
            raise RuntimeError("Failed to capture token activations for Grad-CAM.")
        patch_tokens = tokens[:, 1:, :]
        grad = torch.autograd.grad(score, tokens, retain_graph=True)[0]
        grad = grad[:, 1:, :]
        if grad is None:
            raise RuntimeError("Grad-CAM gradients are None; try attention map variant instead.")
        weights = grad.mean(dim=1)
        cam = (patch_tokens * weights.unsqueeze(1)).sum(dim=-1)
        cam = F.relu(cam)
        cam = normalize_map(cam[0])
        num_patches = cam.shape[0]
        g = grid_size(num_patches)
        cam = cam.view(g, g)
        return cam

    def variant_attention(vision_outputs):
        attn = vision_outputs.attentions[-1]
        attn = attn.mean(dim=1)
        attn_map = attn[:, 0, 1:]
        attn_map = normalize_map(attn_map[0])
        num_patches = attn_map.shape[0]
        g = grid_size(num_patches)
        attn_map = attn_map.view(g, g)
        return attn_map

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model.eval()

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if args.fp16 and device.type == "cuda":
        dtype = torch.float16

    image_inputs = processor(images=image, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device=device, dtype=dtype)
    pixel_values.requires_grad_(True)

    text_features = compute_text_features(model, processor, prompt, device, dtype)
    score, vision_outputs, tokens = compute_score_with_tokens(model, pixel_values, text_features)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    maps = {}
    if "pix" in variants:
        maps["pix"] = variant_pixel_grad(score, pixel_values)
    if "gradcam" in variants:
        maps["gradcam"] = variant_gradcam(score, tokens)
    if "attn" in variants:
        maps["attn"] = variant_attention(vision_outputs)
    return maps


def main():
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    if args.mode == "ocr":
        reward = run_ocr(image, args.prompt, args)
        out_path = out_dir / f"{stem}_ocr.png"
        save_map(reward, out_path)
        print(f"Saved OCR reward map to {out_path}")
    else:
        maps = run_clip(image, args.prompt, args)
        for name, map_tensor in maps.items():
            out_path = out_dir / f"{stem}_clip_{name}.png"
            save_map(map_tensor.detach().cpu().numpy(), out_path, size_hw=image.size[::-1])
        print(f"Saved CLIP reward maps to {out_dir}")


if __name__ == "__main__":
    main()

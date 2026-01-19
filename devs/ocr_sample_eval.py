#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

# Avoid model hoster check delays for PaddleOCR
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR
from Levenshtein import distance as levenshtein_distance

from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model, PeftModel


def extract_target_text(prompt: str) -> str:
    match = re.search(r'"([^"]+)"', prompt)
    if match:
        return match.group(1)
    return prompt


def ocr_score(ocr: PaddleOCR, image: Image.Image, prompt: str) -> tuple[float, str]:
    target = extract_target_text(prompt).replace(" ", "").lower()
    result = ocr.ocr(np.array(image.convert("RGB")), cls=False)
    lines = result[0] if result and isinstance(result[0], list) else result
    recognized_parts = []
    if lines:
        for line in lines:
            if not line or len(line) < 2:
                continue
            text_info = line[1]
            if not text_info:
                continue
            recognized_parts.append(text_info[0])
    recognized = "".join(recognized_parts).replace(" ", "").lower()

    if len(target) == 0:
        return 0.0, recognized

    if target in recognized:
        dist = 0
    else:
        dist = levenshtein_distance(recognized, target)
    if dist > len(target):
        dist = len(target)
    score = 1 - dist / len(target)
    return score, recognized


def load_pipeline(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
    target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    ]
    transformer_lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )

    if checkpoint_path:
        lora_path = os.path.join(checkpoint_path, "lora")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA directory not found at {lora_path}")
        pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
        pipeline.transformer.load_adapter(lora_path, adapter_name="default", is_trainable=False)
    else:
        raise ValueError("checkpoint_path is required for local evaluation.")

    pipeline.transformer.eval()
    pipeline.transformer.to(device, dtype=dtype)
    pipeline.vae.to(device, dtype=dtype)
    pipeline.text_encoder.to(device, dtype=dtype)
    pipeline.text_encoder_2.to(device, dtype=dtype)
    pipeline.text_encoder_3.to(device, dtype=dtype)
    pipeline.safety_checker = None
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Generate OCR samples and score with PaddleOCR.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--dataset_file", default="dataset/ocr/test.txt")
    parser.add_argument("--num_prompts", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output_dir", default="samples/ocr_checkpoint-60")
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    prompts = prompts[: args.num_prompts]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 else torch.float32

    pipeline = load_pipeline(args.checkpoint_path, device, dtype)
    ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False)

    rows = []
    for idx, prompt in enumerate(prompts):
        prompt_dir = output_dir / f"prompt_{idx:03d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                image = pipeline(
                    [prompt],
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    output_type="pil",
                    height=args.resolution,
                    width=args.resolution,
                    generator=generator,
                )[0][0]
            img_path = prompt_dir / f"seed_{seed}.jpg"
            image.save(img_path)

            score, recognized = ocr_score(ocr, image, prompt)
            rows.append(
                {
                    "prompt_id": idx,
                    "prompt": prompt,
                    "seed": seed,
                    "image_path": str(img_path),
                    "recognized": recognized,
                    "score": score,
                }
            )

    # Write report
    report_path = output_dir / "report.md"
    scores = [r["score"] for r in rows]
    mean_score = float(np.mean(scores)) if scores else 0.0

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# OCR Sample Evaluation Report\n\n")
        f.write(f"- checkpoint: {args.checkpoint_path}\n")
        f.write(f"- dataset: {args.dataset_file}\n")
        f.write(f"- prompts: {len(prompts)}\n")
        f.write(f"- seeds per prompt: {len(seeds)} ({args.seeds})\n")
        f.write(f"- steps: {args.num_inference_steps}\n")
        f.write(f"- guidance_scale: {args.guidance_scale}\n")
        f.write(f"- resolution: {args.resolution}\n")
        f.write(f"- mean_score: {mean_score:.4f}\n\n")

        f.write("| prompt_id | seed | score | recognized | image_path |\n")
        f.write("|---:|---:|---:|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['prompt_id']} | {r['seed']} | {r['score']:.4f} | "
                f"{r['recognized']} | {r['image_path']} |\n"
            )

    print(f"Saved {len(rows)} images to {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

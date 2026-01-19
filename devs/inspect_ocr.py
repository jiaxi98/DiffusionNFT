#!/usr/bin/env python3
import argparse
import json
import os
import pprint

os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


def main():
    parser = argparse.ArgumentParser(description="Inspect PaddleOCR output structure.")
    parser.add_argument(
        "--image",
        default="flow_grpo/test_cases/hello world.jpg",
        help="Path to an input image.",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR.")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    # PaddleOCR v3 removes show_log; keep args minimal for compatibility.
    # PaddleOCR 2.x API (recommended in this repo)
    ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=args.gpu)
    img = Image.open(args.image).convert("RGB")
    img_np = np.array(img)

    try:
        result = ocr.ocr(img_np, cls=False)
    except RuntimeError as e:
        if args.gpu and "cudnn" in str(e).lower():
            print("GPU failed (cuDNN missing). Retrying on CPU...")
            ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False)
            result = ocr.ocr(img_np, cls=False)
        else:
            raise
    print("type(result):", type(result))
    try:
        print("len(result):", len(result))
    except Exception:
        pass
    pprint.pprint(result, width=120)
    breakpoint()

    # Try to summarize lines in a robust way across PaddleOCR versions
    lines = None
    if isinstance(result, list) and result:
        if isinstance(result[0], list) and result[0] and isinstance(result[0][0], (list, tuple)):
            # Common: result = [ [line1, line2, ...] ] for single image
            lines = result[0]
        else:
            # Some versions return lines directly
            lines = result

    if lines is not None:
        print("lines:", len(lines))
        for i, line in enumerate(lines):
            try:
                box, (text, score) = line
                print(f"{i}: text={text!r} score={score:.4f} box={box}")
            except Exception:
                print(f"{i}: {line}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

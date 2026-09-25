"""
Predict spice class from a single image.

Usage:
    python scripts/eval/predict.py <image_path>
    python scripts/eval/predict.py D:\SpiceNet\test\turmeric\turmeric_001.jpg
"""
import sys, os
_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from src.features import extract_all
from src.model import load_checkpoint


def load_image_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        return np.array(im, dtype=np.uint8)


def preprocess(img_rgb: np.ndarray) -> torch.Tensor:
    resize = int(config.IMG_SIZE * 256 / 224)
    tfm = A.Compose([
        A.Resize(resize, resize),
        A.CenterCrop(config.IMG_SIZE, config.IMG_SIZE),
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])
    return tfm(image=img_rgb)["image"].unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--ckpt", default=str(config.CHECKPOINT_DIR / "best.pth"))
    parser.add_argument("--topk", type=int, default=3, help="Show top-K predictions")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: image not found: {args.image}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {args.ckpt}...")
    model, _, _, _ = load_checkpoint(args.ckpt, device)
    model.eval()

    # Load + preprocess image
    print(f"Reading image: {args.image}")
    img = load_image_rgb(args.image)
    x   = preprocess(img).to(device)

    # Extract texture + color features
    tex_np, col_np = extract_all(img)
    tex = torch.from_numpy(tex_np).unsqueeze(0).float().to(device)
    col = torch.from_numpy(col_np).unsqueeze(0).float().to(device)

    # Predict (fusion mode)
    with torch.no_grad():
        logits, _ = model.forward_fusion(x, tex, col)
        probs = F.softmax(logits, dim=1).cpu().squeeze().numpy()

    # Sort and show top-K
    top_indices = probs.argsort()[::-1][:args.topk]

    print("\n" + "=" * 50)
    print(f"  Prediction for: {os.path.basename(args.image)}")
    print("=" * 50)

    for rank, idx in enumerate(top_indices, 1):
        bar = "#" * int(probs[idx] * 40)
        marker = " <<<" if rank == 1 else ""
        print(f"  {rank}. {config.CLASSES[idx]:<14} {probs[idx]*100:6.2f}%  {bar}{marker}")

    print("=" * 50)
    print(f"  Final answer: {config.CLASSES[top_indices[0]].upper()}")
    print("=" * 50)


if __name__ == "__main__":
    main()

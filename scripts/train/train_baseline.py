"""
Train a modern timm baseline on SpiceNet-Bench unified benchmark.

Examples:
  python scripts/train/train_baseline.py --model swinv2_tiny_window16_256
  python scripts/train/train_baseline.py --model convnextv2_tiny.fcmae_ft_in22k_in1k
  python scripts/train/train_baseline.py --model maxvit_tiny_tf_224
  python scripts/train/train_baseline.py --model efficientnet_b4
  python scripts/train/train_baseline.py --model resnet50

All use ImageNet-pretrained weights from timm.
Saves checkpoint to outputs/checkpoints/baseline_<model>/best.pth
Saves metrics to outputs/baseline_<model>_metrics.json
"""
import sys, os
_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import timm

import config
from src.dataset import get_dataloaders, load_manifest_splits
from src.utils import set_seed


MANIFEST = Path(_base) / "outputs" / "unified_benchmark.json"


def _make_scheduler(optimizer, warmup, total, min_lr):
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=max(1, total - warmup), eta_min=min_lr),
        ],
        milestones=[warmup],
    )


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    tl, tc, tt = 0.0, 0, 0
    for imgs, tex, col, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        tl += criterion(logits, labels).item() * imgs.size(0)
        tc += (logits.argmax(1) == labels).sum().item()
        tt += imgs.size(0)
    return tl / tt, tc / tt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="resnet50",
                        help="timm model name (e.g., swinv2_tiny_window16_256)")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input size — must match the timm model's default")
    parser.add_argument("--epochs",  type=int, default=25)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--wd",      type=float, default=1e-4)
    parser.add_argument("--warmup",  type=int, default=3)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--strong_aug", action="store_true")
    parser.add_argument("--smoke",   action="store_true")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    _, classes = load_manifest_splits(MANIFEST)
    NUM_CLASSES = len(classes)
    print(f"SpiceNet-Bench: {NUM_CLASSES} classes")

    if args.smoke:
        args.epochs = 1
        args.warmup = 0
        print("** SMOKE MODE **")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Override img_size for transformer models if needed ───────────────
    if "swinv2" in args.model and args.img_size != 256:
        print(f"Note: Swin V2 prefers 256 — using {args.img_size} as requested")
    if "256" in args.model and args.img_size == 224:
        args.img_size = 256
        print(f"Inferred img_size=256 from model name")

    # Patch config.IMG_SIZE for transform — careful: this affects the dataset
    config.IMG_SIZE = args.img_size

    train_loader, val_loader, test_loader, _, _ = get_dataloaders(
        multimodal=False,
        manifest_path=str(MANIFEST),
        batch_size=args.batch,
        strong_aug=args.strong_aug,
    )
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")
    print(f"batches per train epoch: {len(train_loader)} batch={args.batch} img={args.img_size}")

    # ── Build model ──────────────────────────────────────────────────────
    model = timm.create_model(args.model, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model} params: {n_params:,}")

    # ── Training ─────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = _make_scheduler(optimizer, args.warmup, args.epochs, 1e-6)

    safe = args.model.replace("/", "_").replace(".", "_")
    ckpt_dir = config.CHECKPOINT_DIR / f"baseline_{safe}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val, patience, history = 0.0, 0, {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tl, tc, tt = 0.0, 0, 0

        for imgs, tex, col, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * imgs.size(0)
            tc += (logits.argmax(1) == labels).sum().item()
            tt += imgs.size(0)

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tl / tt)
        history["train_acc"].append(tc / tt)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        print(f"Ep {epoch:03d}/{args.epochs} | "
              f"tr_loss {tl/tt:.4f} tr_acc {tc/tt:.4f} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | "
              f"lr {lr:.2e} | {time.time()-t0:.1f}s")

        if val_acc > best_val:
            best_val, patience = val_acc, 0
            torch.save({
                "model_state": model.state_dict(),
                "model_name": args.model, "img_size": args.img_size,
                "epoch": epoch, "best_val_acc": best_val,
                "config": vars(args), "history": history,
            }, ckpt_dir / "best.pth")
            print(f"  --> Best: {best_val:.4f}")
        else:
            patience += 1
            if patience >= 8:
                print(f"  Early stop at epoch {epoch}")
                break

    # ── Final test ────────────────────────────────────────────────────────
    print("\nLoading best checkpoint for test...")
    ck = torch.load(ckpt_dir / "best.pth", map_location=device)
    model.load_state_dict(ck["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc:  {test_acc:.4f}")

    metrics = {
        "model": args.model, "img_size": args.img_size,
        "n_params": n_params,
        "test_acc": test_acc, "test_loss": test_loss,
        "best_val_acc": best_val, "epochs_trained": epoch,
        "config": vars(args),
    }
    out_path = config.OUTPUT_DIR / f"baseline_{safe}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

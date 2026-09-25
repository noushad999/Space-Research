"""
Train any timm backbone on one source's 8-class overlap split (image-only,
strong-aug, early-stopped) for the cross-source benchmark. Mirrors the
SpiceFusionNet Phase-1 protocol so the comparison is fair.

    python scripts/train/train_backbone_overlap.py --model resnet50 \
        --manifest outputs/manifest_overlap_ss.json --suffix bench_resnet50_ss --seed 42
"""
import sys, os, argparse, time
from pathlib import Path

_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm

import config
from src.dataset import get_dataloaders, load_manifest_splits
from src.utils import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="timm model name")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--strong-aug", action="store_true", dest="strong_aug")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, classes = load_manifest_splits(args.manifest)
    n = len(classes)
    tr, val, _, _, _ = get_dataloaders(multimodal=False, manifest_path=args.manifest,
                                       strong_aug=args.strong_aug)
    print(f"{args.model} | classes={n} | train={len(tr.dataset)} val={len(val.dataset)} | device={device}", flush=True)

    model = timm.create_model(args.model, pretrained=True, num_classes=n).to(device)
    epochs = 1 if args.smoke else args.epochs
    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    ckdir = config.CHECKPOINT_DIR / args.suffix
    ckdir.mkdir(parents=True, exist_ok=True)
    best, patience = 0.0, 0
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time()
        for imgs, tex, col, labels in tr:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = crit(model(imgs), labels)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); c = t = 0
        with torch.no_grad():
            for imgs, tex, col, labels in val:
                pred = model(imgs.to(device)).argmax(1).cpu()
                c += (pred == labels).sum().item(); t += labels.size(0)
        acc = c / t; sched.step()
        print(f"  ep {ep:02d}/{epochs} | val {acc:.4f} | {time.time()-t0:.0f}s", flush=True)
        if acc > best:
            best, patience = acc, 0
            torch.save({"model_state": model.state_dict(), "model_name": args.model,
                        "best_val_acc": best, "classes": classes}, ckdir / "best.pth")
        else:
            patience += 1
            if patience >= 8 and not args.smoke:
                print("  early stop", flush=True); break
    print(f"done {args.model} {args.suffix} | best val {best:.4f}", flush=True)


if __name__ == "__main__":
    main()

"""
Train SpiceFusionNet on the unified SpiceNet-Bench benchmark.

Usage:
  python scripts/train/train_unified.py                  # full 3-phase, fusion mode
  python scripts/train/train_unified.py --phase 1        # Phase 1 only (faster, image-only)
  python scripts/train/train_unified.py --strong-aug     # use anti-shortcut training augmentation
  python scripts/train/train_unified.py --smoke          # 1-epoch sanity check (no checkpoints)

Outputs go to outputs/checkpoints/unified/.
Test metrics saved to outputs/unified_*_test_metrics.json after training.
"""
import sys, os
_base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _base)

import argparse
import json
from pathlib import Path

import torch

import config
from src.dataset import get_dataloaders, load_manifest_splits
from src.model import SpiceFusionNet
from src.trainer import PhaseTrainer
from src.utils import set_seed, plot_training_curves


_DEFAULT_MANIFEST = Path(_base) / "outputs" / "unified_benchmark.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",      type=int, default=0,   help="0=all phases, 1/2/3=single phase")
    parser.add_argument("--strong-aug", action="store_true", dest="strong_aug")
    parser.add_argument("--smoke",      action="store_true", help="1-epoch smoke test")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--batch",      type=int, default=None,
                        help="Override BATCH_SIZE (default 32 from config)")
    parser.add_argument("--manifest",   default=None,
                        help="Path to manifest JSON (default outputs/unified_benchmark.json)")
    parser.add_argument("--suffix",     default="unified",
                        help="Checkpoint subdir name (e.g. 'overlap_ss')")
    args = parser.parse_args()

    if args.batch is not None:
        config.BATCH_SIZE = args.batch
        print(f"Batch size override: {args.batch}")

    # ── Patch num_classes for unified benchmark ──────────────────────────
    manifest_path = args.manifest or str(_DEFAULT_MANIFEST)
    _, classes = load_manifest_splits(manifest_path)
    NUM_CLASSES = len(classes)
    print(f"Unified benchmark: {NUM_CLASSES} classes")
    print(f"  {classes}")

    # Monkey-patch config (cheaper than threading num_classes everywhere)
    config.NUM_CLASSES = NUM_CLASSES
    config.CLASSES     = classes

    if args.smoke:
        config.P1_EPOCHS = 1
        config.P2_EPOCHS = 1
        config.P3_EPOCHS = 1
        config.PATIENCE = 99
        print("** SMOKE MODE ** — 1 epoch per phase, no early stopping")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_dir = config.CHECKPOINT_DIR / args.suffix
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1 & 2 don't use texture/color features — disable multimodal loader
    # until Phase 3 actually needs them. Saves ~5x DataLoader CPU time.
    multimodal = (args.phase == 3)
    print(f"Multimodal features: {multimodal} (will auto-enable at Phase 3 if needed)")
    print(f"Strong augmentation: {args.strong_aug}")

    train_loader, val_loader, test_loader, _, _ = get_dataloaders(
        multimodal=multimodal,
        manifest_path=manifest_path,
        strong_aug=args.strong_aug,
    )
    print(f"  train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")
    print(f"  batches per train epoch: {len(train_loader)}")

    model = SpiceFusionNet(num_classes=NUM_CLASSES)
    total = sum(p.numel() for p in model.parameters())
    print(f"SpiceFusionNet params: {total:,}")

    trainer = PhaseTrainer(model, device, ckpt_dir)

    run_phases = [args.phase] if args.phase in (1, 2, 3) else [1, 2, 3]
    histories = {}
    for phase in run_phases:
        if phase == 1:
            histories["p1"] = trainer.phase1(train_loader, val_loader)
            plot_training_curves(histories["p1"], config.OUTPUT_DIR, prefix="unified_p1")
        elif phase == 2:
            trainer.phase2(train_loader)
        elif phase == 3:
            # Phase 3 needs multimodal data
            if not multimodal:
                print("Switching to multimodal loaders for Phase 3...")
                train_loader, val_loader, test_loader, _, _ = get_dataloaders(
                    multimodal=True, manifest_path=manifest_path,
                    strong_aug=args.strong_aug)
                multimodal = True
            histories["p3"] = trainer.phase3(train_loader, val_loader)
            plot_training_curves(histories["p3"], config.OUTPUT_DIR, prefix="unified_p3")

    print("\nTraining complete. Run evaluate.py on the unified ckpt for full metrics.")


if __name__ == "__main__":
    main()
